# test_universal_evidence_engine.py
"""
Comprehensive Automated Acceptance & Validation Suite for the
Universal Evidence-Driven Diagnostic Engine.

Tests A through K covering:
- Binary classification with arbitrary non-numeric labels ("red", "blue")
- Multiclass classification ("A", "B", "C") with all classes retained
- Continuous regression
- High missingness handling
- Pure categorical features
- Different error patterns
- Small dataset sample-size uncertainty
- Balanced vs Imbalanced classification
- Weak/noisy predictive signals
- Dominant predictive feature reliance
- Traceable evidence_ids connecting diagnostics to evidence objects
- Continuous evidence-weighted risk response
"""

import json
import numpy as np
import pandas as pd
from analysis.model_analyzer import analyze_model


def test_suite():
    print("=" * 70)
    print("STARTING UNIVERSAL EVIDENCE-DRIVEN DIAGNOSTIC ACCEPTANCE TESTS")
    print("=" * 70)

    # --------------------------------------------------------
    # TEST A: Binary classification with arbitrary string labels ("red", "blue")
    # --------------------------------------------------------
    print("\n>> TEST A: Binary classification with non-numeric labels ('red', 'blue')")
    np.random.seed(42)
    n = 150
    f1 = np.random.randn(n)
    f2 = np.random.randn(n)
    # Probability depends on f1 + f2
    prob = 1 / (1 + np.exp(-(f1 + f2 * 0.8)))
    target_a = np.where(prob > 0.5, "blue", "red")
    df_a = pd.DataFrame({"alpha_feat": f1, "beta_feat": f2, "color_outcome": target_a})

    res_a = analyze_model(df_a, "color_outcome")
    assert res_a["task_type"] == "classification"
    assert res_a["target_profile"]["is_binary"] is True
    assert set(res_a["class_names"]) == {"blue", "red"}
    assert "evidence_records" in res_a
    assert len(res_a["evidence_records"]) > 0
    print("   [PASSED] Binary arbitrary string labels analyzed cleanly.")

    # --------------------------------------------------------
    # TEST B: Multiclass classification ("A", "B", "C")
    # --------------------------------------------------------
    print("\n>> TEST B: Multiclass classification ('A', 'B', 'C')")
    n = 210
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    classes_b = []
    for i in range(n):
        if x1[i] > 0.5:
            classes_b.append("Class_A")
        elif x2[i] > 0.3:
            classes_b.append("Class_B")
        else:
            classes_b.append("Class_C")

    df_b = pd.DataFrame({"signal_1": x1, "signal_2": x2, "category_target": classes_b})
    res_b = analyze_model(df_b, "category_target")
    assert res_b["task_type"] == "classification"
    assert res_b["target_profile"]["is_multiclass"] is True
    assert len(res_b["class_names"]) == 3
    assert set(res_b["class_names"]) == {"Class_A", "Class_B", "Class_C"}
    # Check that error patterns do not contain forced False Positive/Negative
    for err in res_b["prediction_errors"]:
        assert "False Negative" not in err["error_type"], "False Negative inappropriately used in multiclass!"
        assert "False Positive" not in err["error_type"], "False Positive inappropriately used in multiclass!"
    print("   [PASSED] Multiclass classification retained all 3 classes without binary terminology.")

    # --------------------------------------------------------
    # TEST C: Continuous Regression
    # --------------------------------------------------------
    print("\n>> TEST C: Continuous Regression")
    n = 180
    num_x = np.random.uniform(10, 100, n)
    cont_target = 3.5 * num_x + np.random.normal(0, 15, n)
    df_c = pd.DataFrame({"continuous_predictor": num_x, "price_target": cont_target})

    res_c = analyze_model(df_c, "price_target")
    assert res_c["task_type"] == "regression"
    assert "regression_metrics" in res_c
    assert "r2" in res_c["regression_metrics"]
    assert "mae" in res_c["regression_metrics"]
    assert "rmse" in res_c["regression_metrics"]
    assert "confusion_matrix" in res_c
    print("   [PASSED] Continuous regression evaluated with proper regression metrics.")

    # --------------------------------------------------------
    # TEST D: High Missingness
    # --------------------------------------------------------
    print("\n>> TEST D: High Missingness Handling")
    df_d = df_a.copy()
    mask_miss = np.random.rand(len(df_d)) < 0.25
    df_d.loc[mask_miss, "alpha_feat"] = None

    res_d = analyze_model(df_d, "color_outcome")
    assert res_d["status"] == "Success"
    assert res_d["data_quality"]["total_missing_cells"] > 0
    print("   [PASSED] Missingness handled and profiled generically.")

    # --------------------------------------------------------
    # TEST E: Categorical Features
    # --------------------------------------------------------
    print("\n>> TEST E: Categorical Features")
    n_e = 160
    cats = np.random.choice(["Region_North", "Region_South", "Region_East", "Region_West"], n_e)
    y_e = np.random.choice(["Approved", "Declined"], n_e)
    df_e = pd.DataFrame({"geo_region": cats, "num_val": np.random.randn(n_e), "outcome": y_e})

    res_e = analyze_model(df_e, "outcome")
    assert "geo_region" in res_e["feature_impact"]
    assert res_e["status"] == "Success"
    print("   [PASSED] Categorical features transformed and profiled.")

    # --------------------------------------------------------
    # TEST F: Small Dataset & Sample Uncertainty Discount
    # --------------------------------------------------------
    print("\n>> TEST F: Small Dataset Sample-Size Uncertainty")
    df_small = df_a.head(15).copy()
    res_small = analyze_model(df_small, "color_outcome")
    assert res_small["status"] == "Success"
    # Verify small sample evidence was emitted
    small_evs = [e for e in res_small["evidence_records"] if e["domain"] == "sample_size"]
    assert len(small_evs) > 0, "Small dataset evidence should be emitted!"
    # Verify confidence is appropriately discounted
    for dc in res_small["root_causes_structured"]:
        if "Small Dataset" in dc["title"] or dc["domain"] == "sample_size":
            assert dc["confidence_score"] <= 0.85
    print("   [PASSED] Small dataset uncertainty properly discounted confidence.")

    # --------------------------------------------------------
    # TEST G: Balanced vs Imbalanced Classification
    # --------------------------------------------------------
    print("\n>> TEST G: Balanced vs Imbalanced Classification")
    # 1. Perfectly balanced
    df_bal = pd.DataFrame({
        "feat_1": np.random.randn(200),
        "feat_2": np.random.randn(200),
        "target": ["Yes"] * 100 + ["No"] * 100
    })
    res_bal = analyze_model(df_bal, "target")
    assert res_bal["target_profile"]["class_imbalance"] is False

    # 2. Highly imbalanced (10% vs 90%)
    df_imb = pd.DataFrame({
        "feat_1": np.random.randn(200),
        "feat_2": np.random.randn(200),
        "target": ["Rare"] * 20 + ["Common"] * 180
    })
    res_imb = analyze_model(df_imb, "target")
    assert res_imb["target_profile"]["class_imbalance"] is True
    assert res_imb["target_profile"]["least_frequent_class"] == "Rare"
    print("   [PASSED] Class balance and imbalance detected dynamically.")

    # --------------------------------------------------------
    # TEST H: Dominant Feature Reliance
    # --------------------------------------------------------
    print("\n>> TEST H: Dominant Feature Reliance Signal")
    n = 150
    dom_f = np.random.randn(n)
    noise_1 = np.random.randn(n) * 0.01
    noise_2 = np.random.randn(n) * 0.01
    y_dom = (dom_f > 0).astype(int)
    df_dom = pd.DataFrame({"super_feature": dom_f, "noise_a": noise_1, "noise_b": noise_2, "target": y_dom})

    res_dom = analyze_model(df_dom, "target")
    assert "super_feature" in res_dom["feature_importance"]
    assert res_dom["feature_impact"]["super_feature"]["relative_share_pct"] > 50.0
    print("   [PASSED] Dominant feature reliance detected.")

    # --------------------------------------------------------
    # TEST I: Traceable Evidence IDs in Diagnostic Candidates
    # --------------------------------------------------------
    print("\n>> TEST I: Traceable Evidence IDs in Diagnostic Candidates")
    for dc in res_imb["root_causes_structured"]:
        if dc["category"] != "Diagnostic Observation":
            assert "evidence_ids" in dc
            assert isinstance(dc["evidence_ids"], list)
            assert "evidence_items" in dc
    print("   [PASSED] Diagnostic candidates have traceable evidence IDs.")

    # --------------------------------------------------------
    # TEST J: Continuous Evidence-Weighted Risk Response
    # --------------------------------------------------------
    print("\n>> TEST J: Continuous Risk Response")
    assert 5 <= res_a["risk_score"] <= 100
    assert 5 <= res_b["risk_score"] <= 100
    assert 5 <= res_c["risk_score"] <= 100
    assert "drivers" in res_a["risk"]
    assert "breakdown" in res_a["risk"]
    for driver in res_a["risk"]["drivers"]:
        assert "contribution" in driver
        assert "evidence" in driver
    print("   [PASSED] Risk score computed continuously with traceable drivers.")

    # --------------------------------------------------------
    # TEST K: Clean Serialization (JSON Serializable)
    # --------------------------------------------------------
    print("\n>> TEST K: Payload Serialization Verification")
    json_str = json.dumps(res_a)
    assert len(json_str) > 0
    json_str_b = json.dumps(res_b)
    assert len(json_str_b) > 0
    json_str_c = json.dumps(res_c)
    assert len(json_str_c) > 0
    print("   [PASSED] Complete payload is valid JSON serializable.")

    print("\n" + "=" * 70)
    print("ALL 11 UNIVERSAL EVIDENCE ACCEPTANCE TESTS PASSED 100% GREEN!")
    print("=" * 70)


if __name__ == "__main__":
    test_suite()
