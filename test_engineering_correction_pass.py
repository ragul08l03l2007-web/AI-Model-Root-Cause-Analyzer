# test_engineering_correction_pass.py
import json
import numpy as np
import pandas as pd

from analysis.model_analyzer import analyze_model


def test_regression_dataset_a_weather_power():
    print("\n--------------------------------------------------")
    print(">> TEST A: Realistic Regression (Weather -> Power Output)")
    print("--------------------------------------------------")
    np.random.seed(42)
    n = 120
    temp = np.random.uniform(15, 35, n)
    humidity = np.random.uniform(30, 90, n)
    pressure = np.random.uniform(990, 1030, n)
    wind_speed = np.random.uniform(0, 25, n)
    solar_radiation = np.random.uniform(100, 1000, n)
    
    # Power output function with noise
    power_output = 5.0 * temp + 0.8 * solar_radiation - 1.2 * humidity + 0.5 * wind_speed + np.random.normal(0, 15, n)
    
    df = pd.DataFrame({
        "temperature": temp,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "solar_radiation": solar_radiation,
        "power_output": power_output
    })
    
    res = analyze_model(df, "power_output", analysis_type="regression")
    
    assert res["task_type"] == "regression"
    assert "model_performance" in res
    assert "mae" in res["model_performance"]
    assert "r2" in res["model_performance"]
    
    # Model selection status check
    comp = res["model_comparison"]
    selected_count = sum(1 for m in comp if m["selection_status"] == "Selected")
    unselected_count = sum(1 for m in comp if m["selection_status"] == "Unselected")
    assert selected_count == 1, f"Expected exactly 1 Selected model, got {selected_count}"
    assert unselected_count == len(comp) - 1
    
    # Verify recommendations do not recommend C or alpha if Linear Regression was selected
    selected_m = res["selected_model"]
    recs = res["recommendations"]
    print(f"Selected Model: {selected_m}")
    print(f"Recommendations count: {len(recs)}")
    for r in recs:
        print(f" - {r}")
        if "Linear Regression" in selected_m:
            assert "lower C" not in r and "higher alpha" not in r, f"Invalid LinearRegression recommendation: {r}"
            
    # Verify regression error summary is NOT generic "No major error detected."
    main_err = res["error_analysis"]["main_error"]
    print(f"Main Error Summary: {main_err}")
    assert main_err != "No major error detected."
    
    # Verify no classification terms exist in payload
    assert res.get("class_count") is None
    assert "minority_class" not in res.get("target_profile", {})
    print("   [PASSED] Test A verified.")


def test_multiclass_dataset_b():
    print("\n--------------------------------------------------")
    print(">> TEST B: Multiclass Classification (A / B / C)")
    print("--------------------------------------------------")
    np.random.seed(123)
    n = 150
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    targets = np.random.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    
    df = pd.DataFrame({"feat_1": x1, "feat_2": x2, "grade": targets})
    res = analyze_model(df, "grade", analysis_type="classification")
    
    assert res["task_type"] == "classification"
    assert res["class_count"] == 3
    dist_labels = [d["label"] for d in res["class_distribution_list"]]
    assert set(dist_labels) == {"A", "B", "C"}
    print(f"Classes: {dist_labels}")
    print("   [PASSED] Test B verified.")


def test_binary_dataset_c_approved_rejected():
    print("\n--------------------------------------------------")
    print(">> TEST C: Binary Classification (Approved / Rejected)")
    print("--------------------------------------------------")
    np.random.seed(99)
    n = 100
    income = np.random.uniform(20000, 100000, n)
    score = np.random.uniform(300, 850, n)
    decisions = ["Approved" if (s > 600 or inc > 60000) else "Rejected" for s, inc in zip(score, income)]
    
    df = pd.DataFrame({"annual_salary": income, "credit_rating": score, "decision": decisions})
    res = analyze_model(df, "decision", analysis_type="classification")
    
    assert res["task_type"] == "classification"
    assert res["class_count"] == 2
    assert set(res["class_names"]) == {"Approved", "Rejected"}
    print(f"Binary Classes: {res['class_names']}")
    print("   [PASSED] Test C verified.")


def test_regression_dataset_d_zero_targets():
    print("\n--------------------------------------------------")
    print(">> TEST D: Regression with Zero-Valued Targets (Percentage Error Safety)")
    print("--------------------------------------------------")
    np.random.seed(456)
    n = 50
    x = np.random.uniform(-5, 5, n)
    # Target contains exact zeros
    y = np.maximum(0.0, 2.0 * x + np.random.normal(0, 1, n))
    
    df = pd.DataFrame({"predictor_x": x, "target_y": y})
    res = analyze_model(df, "target_y", analysis_type="regression")
    
    pred_errors = res["prediction_errors"]
    assert len(pred_errors) > 0
    
    # Check percentage error calculations
    for pe in pred_errors:
        act = pe["actual"]
        abs_err = pe["absolute_error"]
        pct_err = pe.get("percentage_error")
        pct_disp = pe.get("percentage_error_display")
        
        if abs(act) < 1e-9:
            assert pct_disp == "N/A (actual=0)"
        else:
            expected_pct = (abs_err / abs(act)) * 100.0
            assert abs(pct_err - expected_pct) < 0.01, f"Math mismatch: got {pct_err}, expected {expected_pct}"
            assert "%" in pct_disp
            
    print("Sample Prediction Errors:")
    for pe in pred_errors[:4]:
        print(f"  Row {pe['row']}: Actual={pe['actual']}, Pred={pe['predicted']}, AbsErr={pe['absolute_error']}, PctErr={pe['percentage_error_display']}")
        
    print("   [PASSED] Test D verified.")


def test_regression_dataset_e_heteroscedasticity():
    print("\n--------------------------------------------------")
    print(">> TEST E: Regression with Strong Heteroscedasticity")
    print("--------------------------------------------------")
    np.random.seed(789)
    n = 150
    x = np.random.uniform(5, 50, n)
    # Variance expands strongly with x
    noise = np.random.normal(0, 0.1 * (x ** 2), n)
    y = 2.5 * x + noise
    
    df = pd.DataFrame({"feature_scale": x, "measurement": y})
    res = analyze_model(df, "measurement", analysis_type="regression")
    
    het_corr = res["error_analysis"].get("heteroscedasticity_correlation", 0.0)
    print(f"Heteroscedasticity Correlation: r = {het_corr:+.4f}")
    assert abs(het_corr) >= 0.20, f"Expected notable heteroscedasticity correlation, got {het_corr}"
    
    # Verify main_error describes residual variance
    main_err = res["error_analysis"]["main_error"]
    print(f"Diagnostic Summary: {main_err}")
    assert "Residual" in main_err or "variance" in main_err or "MAE" in main_err
    print("   [PASSED] Test E verified.")


def test_regression_dataset_f_small_sample():
    print("\n--------------------------------------------------")
    print(">> TEST F: Small Regression Dataset (Sample Uncertainty)")
    print("--------------------------------------------------")
    np.random.seed(321)
    n = 16
    df = pd.DataFrame({
        "var_a": np.random.randn(n),
        "var_b": np.random.randn(n),
        "target_val": np.random.randn(n) * 10
    })
    
    res = analyze_model(df, "target_val", analysis_type="regression")
    assert res["dataset_info"]["usable_rows"] == 16
    
    # Confidence should be low/moderate due to small sample
    candidates = res["root_causes_structured"]
    for c in candidates:
        print(f"Candidate: {c['title']} | Conf: {c['confidence']} ({c['confidence_score']:.2f})")
        assert c["confidence_score"] <= 0.85, f"Confidence too high ({c['confidence_score']}) for N=16"
        
    print("   [PASSED] Test F verified.")


def test_regression_dataset_g_categorical_predictors():
    print("\n--------------------------------------------------")
    print(">> TEST G: Regression with Categorical Predictors")
    print("--------------------------------------------------")
    np.random.seed(654)
    n = 100
    category = np.random.choice(["Urban", "Suburban", "Rural"], size=n)
    sqft = np.random.uniform(500, 3500, n)
    # Price depends on category + sqft
    price_base = {"Urban": 200000, "Suburban": 150000, "Rural": 100000}
    price = [price_base[c] + 120.0 * s + np.random.normal(0, 15000) for c, s in zip(category, sqft)]
    
    df = pd.DataFrame({"location_type": category, "square_footage": sqft, "home_price": price})
    res = analyze_model(df, "home_price", analysis_type="regression")
    
    assert res["task_type"] == "regression"
    assert res["model_performance"]["r2"] > 0.70
    assert len(res["feature_relationships"]) >= 2
    print(f"Model R²: {res['model_performance']['r2']:.4f}")
    print("   [PASSED] Test G verified.")


def test_traceability_and_canonical_data_quality():
    print("\n--------------------------------------------------")
    print(">> TEST H: Traceability & Canonical Data Quality")
    print("--------------------------------------------------")
    np.random.seed(111)
    df = pd.DataFrame({
        "num1": [1.0, np.nan, 3.0, 4.0, 5.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 50.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
        "target": [2.0, 4.0, 6.0, 8.0, 10.0, 10.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]
    })
    
    res = analyze_model(df, "target", analysis_type="regression")
    
    # 1. Canonical data quality consistency
    dq = res["data_quality"]
    d_info = res["dataset_info"]
    risk = res["risk"]
    
    assert dq["total_missing_cells"] == d_info["missing_feature_values"]
    assert dq["duplicate_rows"] == d_info["duplicate_rows"]
    
    # 2. Traceability: Candidates have evidence_ids
    for rc in res["root_causes_structured"]:
        assert len(rc.get("evidence_ids", [])) > 0, f"Missing evidence_ids in {rc['title']}"
        
    # 3. Traceability: Risk drivers have evidence_ids, risk_driver_id, domain, reason
    for rd in risk.get("drivers", []):
        assert len(rd.get("evidence_ids", [])) > 0, f"Missing evidence_ids in risk driver {rd['issue']}"
        assert "risk_driver_id" in rd, f"Missing risk_driver_id in {rd}"
        assert "domain" in rd, f"Missing domain in {rd}"
        assert "observed_evidence" in rd, f"Missing observed_evidence in {rd}"
        assert "reason" in rd, f"Missing reason in {rd}"
        
    # 4. Model Selection & Comparison completeness
    for m in res["model_comparison"]:
        assert "model_id" in m, f"Missing model_id in {m}"
        assert "selection_score" in m, f"Missing selection_score in {m}"
        assert "selection_criterion" in m, f"Missing selection_criterion in {m}"
        assert m["selection_status"] in ("Selected", "Unselected")

    # 5. Traceability: Structured recommendations have evidence_ids
    for rec_obj in res.get("recommendations_structured", []):
        assert "action" in rec_obj
        assert "evidence_ids" in rec_obj
        assert "compatible_models" in rec_obj
        assert "objective" in rec_obj
        assert "reason" in rec_obj
        
    # 6. Warnings populated
    warnings = res.get("warnings", [])
    print(f"Active Warnings count: {len(warnings)}")
    for w in warnings:
        print(f" - {w}")
    assert len(warnings) > 0, "Expected operational warnings for dataset with missing and duplicate records"
    
    print("   [PASSED] Test H verified.")


if __name__ == "__main__":
    test_regression_dataset_a_weather_power()
    test_multiclass_dataset_b()
    test_binary_dataset_c_approved_rejected()
    test_regression_dataset_d_zero_targets()
    test_regression_dataset_e_heteroscedasticity()
    test_regression_dataset_f_small_sample()
    test_regression_dataset_g_categorical_predictors()
    test_traceability_and_canonical_data_quality()
    print("\n==================================================")
    print("ALL 8 ENGINEERING CORRECTION ACCEPTANCE TESTS PASSED 100% GREEN!")
    print("==================================================")
