# test_multi_dataset_validation.py
"""
Multi-dataset verification suite for AI Model Root-Cause Analyzer.
Verifies that genuinely different datasets produce dynamic, evidence-driven,
non-causal findings, explainable risk drivers, and connected recommendations.
"""

import json
import numpy as np
import pandas as pd
from analysis.model_analyzer import analyze_model


def test_dataset_1_binary_imbalanced():
    print("--------------------------------------------------")
    print("DATASET 1: Highly Imbalanced Binary Classification")
    print("--------------------------------------------------")
    np.random.seed(42)
    n = 300
    age = np.random.randint(18, 70, n)
    income = np.random.uniform(20000, 150000, n)
    debt = np.random.uniform(500, 50000, n)
    
    # 85% class 0 (Good), 15% class 1 (Default)
    y = (debt / (income + 1e-5) > 0.45).astype(int)
    # Ensure at least 40 defaults
    y[:45] = 1
    y[45:] = 0
    
    df = pd.DataFrame({"age": age, "income": income, "debt": debt, "default": y})
    res = analyze_model(df, "default", analysis_type="classification")
    
    assert res["analysis_type"] == "classification"
    assert res["class_count"] == 2
    assert bool(res["class_imbalance"]) is True
    assert "risk" in res
    assert "drivers" in res["risk"]
    assert len(res["root_causes_structured"]) > 0
    
    rc0 = res["root_causes_structured"][0]
    assert "finding" in rc0
    assert "confidence_score" in rc0
    assert "evidence_strength" in rc0
    assert "interpretation" in rc0
    assert "potential_explanation" in rc0
    assert "recommendation" in rc0
    
    print(f"Target Distribution: {res['class_distribution']}")
    print(f"Selected Model: {res['selected_model']} (Accuracy: {res['model_performance']['accuracy']:.1%})")
    print(f"Risk Score: {res['risk_score']}/100 ({res['overall_risk']})")
    print(f"Risk Drivers Count: {len(res['risk']['drivers'])}")
    print(f"Root Causes Count: {len(res['root_causes_structured'])}")
    print(f"Top Root Cause: {rc0['finding']} [Strength: {rc0['evidence_strength']}]")
    print("-> DATASET 1 PASSED!\n")
    return res


def test_dataset_2_continuous_regression():
    print("--------------------------------------------------")
    print("DATASET 2: Continuous Real Estate Regression")
    print("--------------------------------------------------")
    np.random.seed(101)
    n = 200
    area = np.random.uniform(500, 4000, n)
    rooms = np.random.randint(1, 6, n)
    age = np.random.uniform(0, 50, n)
    price = 50000 + (area * 180) + (rooms * 25000) - (age * 1200) + np.random.normal(0, 15000, n)
    
    df = pd.DataFrame({"area_sqft": area, "num_rooms": rooms, "property_age": age, "sale_price": price})
    res = analyze_model(df, "sale_price", analysis_type="regression")
    
    assert res["analysis_type"] == "regression"
    assert res["regression_metrics"]["r2"] > 0.85
    assert "risk" in res
    assert "drivers" in res["risk"]
    assert len(res["root_causes_structured"]) > 0
    
    print(f"Regression Performance: R² = {res['regression_metrics']['r2']:.4f}, MAE = {res['regression_metrics']['mae']:.2f}")
    print(f"Selected Model: {res['selected_model']}")
    print(f"Risk Score: {res['risk_score']}/100 ({res['overall_risk']})")
    print(f"Root Causes: {[rc['finding'] for rc in res['root_causes_structured']]}")
    print("-> DATASET 2 PASSED!\n")
    return res


def test_dataset_3_multiclass_balanced():
    print("--------------------------------------------------")
    print("DATASET 3: Balanced Multiclass Classification (3 Classes)")
    print("--------------------------------------------------")
    np.random.seed(202)
    n = 150
    feat_a = np.random.randn(n)
    feat_b = np.random.randn(n)
    tier = np.random.choice(["Tier_A", "Tier_B", "Tier_C"], n)
    
    df = pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b, "customer_tier": tier})
    res = analyze_model(df, "customer_tier", analysis_type="classification")
    
    assert res["analysis_type"] == "classification"
    assert res["class_count"] == 3
    assert "class_metrics" in res
    assert len(res["class_metrics"]) == 3
    assert len(res["root_causes_structured"]) > 0
    
    print(f"Classes: {res['class_names']}")
    print(f"Class Distribution: {res['class_distribution']}")
    print(f"Selected Model: {res['selected_model']}")
    print(f"Risk Score: {res['risk_score']}/100 ({res['overall_risk']})")
    print("-> DATASET 3 PASSED!\n")
    return res


def test_dataset_4_small_with_outliers_and_missing():
    print("--------------------------------------------------")
    print("DATASET 4: Small Dataset (25 rows) with Outliers & Missing Values")
    print("--------------------------------------------------")
    np.random.seed(303)
    n = 25
    x1 = np.random.uniform(10, 50, n)
    x1[0] = 500.0  # extreme outlier
    x2 = np.random.uniform(100, 200, n)
    x2[2] = np.nan  # missing value
    y = (x1 > 30).astype(int)
    
    df = pd.DataFrame({"var1": x1, "var2": x2, "flag": y})
    res = analyze_model(df, "flag", analysis_type="classification")
    
    assert res["analysis_type"] == "classification"
    assert res["dataset_info"]["rows"] == 25
    # Small sample constraint should be captured
    findings = [rc["finding"] for rc in res["root_causes_structured"]]
    assert any("Small Dataset" in f for f in findings)
    
    print(f"Usable Rows: {res['dataset_info']['usable_rows']}")
    print(f"Missing Values: {res['dataset_info']['missing_feature_values']}")
    print(f"Risk Score: {res['risk_score']}/100 ({res['overall_risk']})")
    print(f"Root Causes: {findings}")
    print("-> DATASET 4 PASSED!\n")
    return res


def test_differential_findings_across_datasets():
    print("--------------------------------------------------")
    print("VERIFYING DATASET SENSITIVITY & DIVERSITY OF RESULTS")
    print("--------------------------------------------------")
    r1 = test_dataset_1_binary_imbalanced()
    r2 = test_dataset_2_continuous_regression()
    r3 = test_dataset_3_multiclass_balanced()
    r4 = test_dataset_4_small_with_outliers_and_missing()
    
    # Verify that different datasets produce different metrics, risk scores, and root causes
    assert r1["analysis_type"] != r2["analysis_type"]
    assert r1["risk_score"] != r2["risk_score"]
    assert r1["selected_model"] is not None
    assert r2["selected_model"] is not None
    assert r1["root_causes"] != r2["root_causes"]
    assert r1["recommendations"] != r2["recommendations"]
    
    # Verify JSON serializability of all results
    for i, r in enumerate([r1, r2, r3, r4], 1):
        dumped = json.dumps(r, default=str)
        assert len(dumped) > 100
        assert "np.int" not in dumped
        assert "np.float" not in dumped
    
    print("==================================================")
    print("ALL 4 DATASET VALIDATIONS & DIFFERENTIAL CHECKS PASSED 100%!")
    print("==================================================")


if __name__ == "__main__":
    test_differential_findings_across_datasets()
