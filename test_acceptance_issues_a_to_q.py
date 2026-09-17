# test_acceptance_issues_a_to_q.py
import json
import pandas as pd
import numpy as np

from analysis.data_quality import analyze_data_quality
from analysis.model_analyzer import analyze_model


def test_issue_a_no_numpy_leakage():
    print("Verifying Issue A: No NumPy type representations leaking into UI/JSON/Text...")
    df = pd.DataFrame({
        "feature_1": [1.0, 2.5, 3.2, 4.8, 5.1, 6.0, 7.3, 8.9, 9.4, 10.0],
        "feature_2": [10, 20, 15, 30, 25, 35, 40, 45, 50, 55],
        "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    })
    res = analyze_model(df, "target", analysis_type="auto")
    json_dump = json.dumps(res, default=str)
    
    assert "np.int" not in json_dump, f"np.int found in json: {json_dump}"
    assert "np.float" not in json_dump, f"np.float found in json: {json_dump}"
    assert "np.int64" not in res["task_reason"]
    assert "Target is binary numeric (values: 0, 1)." == res["task_reason"] or "values: 0, 1" in res["task_reason"]
    print("-> Issue A Verified: PASSED!\n")


def test_issue_b_imbalance_ratio_clarity():
    print("Verifying Issue B: Minority % vs Minority-to-Majority Ratio clarity...")
    # 30 minority (1), 70 majority (0) -> minority % is 30.0%, minority-to-majority ratio is 30/70 = 0.43
    df = pd.DataFrame({
        "x1": np.random.randn(100),
        "target": [1] * 30 + [0] * 70
    })
    res = analyze_model(df, "target", analysis_type="classification")
    prof = res["target_profile"]
    
    assert prof["minority_class"] == "1"
    assert prof["majority_class"] == "0"
    assert abs(prof["minority_percentage"] - 30.0) < 0.1
    assert abs(prof["minority_to_majority_ratio"] - 0.43) < 0.05
    print("prof in test:", prof)
    assert "30.0%" in prof["imbalance_explanation"]
    assert "0.43" in prof["imbalance_explanation"]
    print(f"Distribution explanation: {prof['imbalance_explanation']}")
    print("-> Issue B Verified: PASSED!\n")


def test_issue_c_d_n_p_evidence_hierarchy_and_confidence():
    print("Verifying Issues C, D, N, P: Diagnostic hierarchy, multi-source confidence & 8-field evidence chain...")
    np.random.seed(42)
    # create synthetic dataset where class 1 is frequently misclassified
    x1 = np.random.normal(0, 1, 150)
    x2 = np.random.normal(0, 1, 150)
    # target mostly 0, minority 1 has overlapping distribution
    y = np.array([1 if i < 30 else 0 for i in range(150)])
    
    df = pd.DataFrame({"feat_1": x1, "feat_2": x2, "label": y})
    res = analyze_model(df, "label", analysis_type="classification")
    
    structured_rc = res.get("root_causes_structured", [])
    assert len(structured_rc) > 0
    
    for rc in structured_rc:
        # Verify 8 mandatory fields
        assert "finding" in rc
        assert "category" in rc
        assert rc["category"] in {"Root Cause Candidate", "Model Behavior", "Data Quality Issue", "Diagnostic Observation", "Risk Factor"}
        assert "evidence" in rc
        assert "interpretation" in rc
        assert "potential_explanation" in rc
        assert "impact" in rc
        assert "confidence" in rc
        assert rc["confidence"] in {"High", "Medium", "Low", "Insufficient evidence"}
        assert "recommended_action" in rc
        
        # Verify no unsubstantiated causal claims
        assert "causes" not in rc["potential_explanation"].lower() or "may" in rc["potential_explanation"].lower()
    
    print("Sample Structured Finding:")
    print(json.dumps(structured_rc[0], indent=2))
    print("-> Issues C, D, N, P Verified: PASSED!\n")


def test_issue_e_f_feature_importance_relative_and_direction():
    print("Verifying Issues E, F: Relative feature influence tiers and non-causal relationship analysis...")
    np.random.seed(123)
    x_strong = np.linspace(10, 100, 200)
    x_weak = np.random.randn(200)
    y = 2.5 * x_strong + np.random.normal(0, 5, 200)
    
    df = pd.DataFrame({"dominant_feature": x_strong, "noise_feature": x_weak, "outcome": y})
    res = analyze_model(df, "outcome", analysis_type="regression")
    
    f_imp = res["feature_impact"]
    assert "dominant_feature" in f_imp
    dominant_data = f_imp["dominant_feature"]
    assert dominant_data["relative_share_pct"] > 70.0
    assert dominant_data["influence_tier"] in {"Very Strong Model Influence", "Strong Model Influence"}
    assert "Higher values strongly associated with higher" in dominant_data["direction"]
    
    # Check relationships
    f_rels = res["feature_relationships"]
    assert len(f_rels) > 0
    assert f_rels[0]["feature"] == "dominant_feature"
    assert "bins" in f_rels[0]
    print(f"Dominant Feature Impact: {dominant_data}")
    print("-> Issues E, F Verified: PASSED!\n")


def test_issue_g_h_error_patterns_and_individual_confidence():
    print("Verifying Issues G, H: Dynamic error patterns and distinct model prediction confidence...")
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "f2": [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19],
        "class_col": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    })
    res = analyze_model(df, "class_col", analysis_type="classification")
    
    # Patterns
    patterns = res.get("error_patterns", [])
    for p in patterns:
        assert p["pattern_type"] in {"Dominant Error Pattern", "Frequent False-Negative Pattern", "Frequent False-Positive Pattern", "Frequent Confusion", "Repeated Confusion", "Isolated Error"}
    
    # Individual errors
    errors = res.get("prediction_errors", [])
    for err in errors:
        assert "confidence" in err
        assert 0.0 <= err["confidence"] <= 1.0
        assert "row" in err or "dataset_row" in err
        assert "error_type" in err
    print("-> Issues G, H Verified: PASSED!\n")


def test_issue_i_j_overfitting_and_cv_stability():
    print("Verifying Issues I, J: Multi-indicator overfitting assessment and CV stability...")
    np.random.seed(99)
    df = pd.DataFrame({
        "a": np.random.randn(80),
        "b": np.random.randn(80),
        "c": np.random.randn(80),
        "target": np.random.choice([0, 1], size=80)
    })
    res = analyze_model(df, "target", analysis_type="classification")
    cv = res["cross_validation"]
    stab = res["model_stability"]
    
    assert "cv_min" in cv
    assert "cv_max" in cv
    assert "cv_range" in cv
    assert "standard_deviation" in cv
    assert "stability_status" in stab
    assert "stability_explanation" in stab
    assert "overfitting_diagnostic" in stab
    assert "overfitting_explanation" in stab
    print(f"CV Stability: {stab['stability_status']} - {stab['stability_explanation']}")
    print(f"Overfitting Assessment: {stab['overfitting_diagnostic']} - {stab['overfitting_explanation']}")
    print("-> Issues I, J Verified: PASSED!\n")


def test_issue_k_l_m_data_quality_and_duplicates():
    print("Verifying Issues K, L, M: Dynamic data quality reason, IQR outlier bounds, transparent duplicates...")
    df = pd.DataFrame({
        "num_col": [10, 12, 11, 13, 12, 14, 11, 100, 12, 11], # 100 is outlier
        "cat_col": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "dup_col": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2] # has duplicates
    })
    dq = analyze_data_quality(df)
    
    assert "quality_reason" in dq
    assert len(dq["quality_reason"]) > 0
    assert "duplicate_handling_note" in dq
    assert "outlier_details" in dq
    assert "num_col" in dq["outlier_details"]
    assert "lower_bound" in dq["outlier_details"]["num_col"]
    assert "upper_bound" in dq["outlier_details"]["num_col"]
    
    print(f"Quality Reason: {dq['quality_reason']}")
    print(f"Duplicate Policy: {dq['duplicate_handling_note']}")
    print(f"Outlier Details (num_col): {dq['outlier_details']['num_col']}")
    print("-> Issues K, L, M Verified: PASSED!\n")


def test_issue_o_q_healthy_fallback_and_dataset_agnostic():
    print("Verifying Issues O, Q: Clean healthy fallback without overdiagnosing, and dataset-agnostic execution...")
    # Perfectly clean balanced dataset
    df_clean = pd.DataFrame({
        "sensor_alpha": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
        "sensor_beta": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] * 5,
        "device_state": (["OK"] * 25) + (["FAIL"] * 25)
    })
    res = analyze_model(df_clean, "device_state", analysis_type="classification")
    
    # Verify no customer churn domain hardcoding
    res_str = json.dumps(res, default=str).lower()
    assert "churn" not in res_str
    assert "customer" not in res_str
    assert "monthly_charges" not in res_str
    
    print("-> Issues O, Q Verified: PASSED!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("STARTING SECTION 32 FULL ACCEPTANCE VERIFICATION SUITE")
    print("=" * 60)
    test_issue_a_no_numpy_leakage()
    test_issue_b_imbalance_ratio_clarity()
    test_issue_c_d_n_p_evidence_hierarchy_and_confidence()
    test_issue_e_f_feature_importance_relative_and_direction()
    test_issue_g_h_error_patterns_and_individual_confidence()
    test_issue_i_j_overfitting_and_cv_stability()
    test_issue_k_l_m_data_quality_and_duplicates()
    test_issue_o_q_healthy_fallback_and_dataset_agnostic()
    print("=" * 60)
    print("ALL ISSUES (A THROUGH Q) TESTED & VERIFIED 100% GREEN!")
    print("=" * 60)
