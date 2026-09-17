# test_universal_datasets.py
"""
Validates that the AI Model Root-Cause Analyzer dynamically adapts to arbitrary,
structurally diverse datasets across all 11 scenarios (A through K) defined in Section 36:
  A. Binary classification
  B. Multiclass classification (4 classes)
  C. Continuous regression
  D. Heavy missingness dataset (>40% missing in key features)
  E. Balanced classification
  F. Highly imbalanced classification
  G. Small dataset (12 rows)
  H. Dataset with irrelevant unique identifier
  I. Dataset with categorical and string variables
  J. Dataset with strong predictive feature / potential leakage
  K. Dataset with noisy features and low signal
"""

import json
import numpy as np
import pandas as pd
from analysis.model_analyzer import analyze_model
from analysis.data_quality import analyze_data_quality


def run_universal_dataset_tests():
    print("=" * 70)
    print("RUNNING UNIVERSAL DATASET SENSITIVITY TEST SUITE (SCENARIOS A - K)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Scenario A: Binary Classification
    # -------------------------------------------------------------------------
    print("\n[Scenario A] Binary Classification...")
    np.random.seed(42)
    n = 120
    df_a = pd.DataFrame({
        "feature_1": np.random.normal(0, 1, n),
        "feature_2": np.random.normal(2, 1, n),
        "target_col": np.random.choice(["Pass", "Fail"], size=n)
    })
    res_a = analyze_model(df_a, "target_col", analysis_type="auto")
    assert res_a["task_type"] == "classification"
    assert res_a["class_count"] == 2
    assert "Pass" in res_a["class_distribution"] and "Fail" in res_a["class_distribution"]
    print(f"-> PASSED: Task={res_a['task_type']}, Classes={res_a['class_names']}, Selected={res_a['selected_model']}")

    # -------------------------------------------------------------------------
    # Scenario B: Multiclass Classification (4 classes)
    # -------------------------------------------------------------------------
    print("\n[Scenario B] Multiclass Classification (4 classes)...")
    np.random.seed(101)
    n = 160
    df_b = pd.DataFrame({
        "speed": np.random.uniform(10, 100, n),
        "temperature": np.random.uniform(20, 80, n),
        "pressure": np.random.uniform(1, 5, n),
        "state": np.random.choice(["Normal", "Warning", "Critical", "Maintenance"], size=n)
    })
    res_b = analyze_model(df_b, "state", analysis_type="auto")
    assert res_b["task_type"] == "classification"
    assert res_b["class_count"] == 4
    assert len(res_b["confusion_matrix"]) == 4
    print(f"-> PASSED: Task={res_b['task_type']}, Classes={res_b['class_names']}, CM={len(res_b['confusion_matrix'])}x{len(res_b['confusion_matrix'])}")

    # -------------------------------------------------------------------------
    # Scenario C: Continuous Regression
    # -------------------------------------------------------------------------
    print("\n[Scenario C] Continuous Regression...")
    np.random.seed(202)
    n = 150
    x1 = np.linspace(1, 50, n)
    x2 = np.random.normal(0, 2, n)
    y_reg = 10.0 + 3.2 * x1 - 1.5 * x2 + np.random.normal(0, 1.0, n)
    df_c = pd.DataFrame({"engine_rpm": x1, "oil_temp": x2, "fuel_consumption": y_reg})
    res_c = analyze_model(df_c, "fuel_consumption", analysis_type="auto")
    assert res_c["task_type"] == "regression"
    assert "r2" in res_c["model_performance"]
    assert res_c["model_performance"]["r2"] > 0.85
    assert len(res_c["prediction_errors"]) > 0
    print(f"-> PASSED: Task={res_c['task_type']}, R2={res_c['model_performance']['r2']:.4f}, MAE={res_c['model_performance']['mae']:.4f}")

    # -------------------------------------------------------------------------
    # Scenario D: Heavy Missingness Dataset
    # -------------------------------------------------------------------------
    print("\n[Scenario D] Heavy Missingness Dataset (>40% missing)...")
    np.random.seed(303)
    n = 100
    feat_missing = [val if np.random.rand() > 0.45 else None for val in np.random.normal(5, 2, n)]
    df_d = pd.DataFrame({
        "sensor_a": feat_missing,
        "sensor_b": np.random.normal(10, 3, n),
        "target_status": np.random.choice([0, 1], size=n)
    })
    dq_d = analyze_data_quality(df_d)
    res_d = analyze_model(df_d, "target_status", analysis_type="classification")
    assert dq_d["missing_values"]["sensor_a"] > 35
    assert dq_d["quality_score"] < 95
    assert any("missing" in w.lower() for w in dq_d["warnings"])
    print(f"-> PASSED: Detected {dq_d['missing_values']['sensor_a']}% missing cells; Quality Score={dq_d['quality_score']}/100")

    # -------------------------------------------------------------------------
    # Scenario E: Balanced Classification
    # -------------------------------------------------------------------------
    print("\n[Scenario E] Balanced Classification...")
    np.random.seed(404)
    n = 100
    df_e = pd.DataFrame({
        "v1": np.random.normal(0, 1, n),
        "v2": np.random.normal(0, 1, n),
        "target": ["Class_A"] * 50 + ["Class_B"] * 50
    })
    res_e = analyze_model(df_e, "target", analysis_type="classification")
    prof_e = res_e["target_profile"]
    assert prof_e["class_imbalance"] is False
    assert abs(prof_e["minority_to_majority_ratio"] - 1.0) < 0.05
    print(f"-> PASSED: Imbalance Ratio={prof_e['minority_to_majority_ratio']:.2f}, Has Imbalance={prof_e['class_imbalance']}")

    # -------------------------------------------------------------------------
    # Scenario F: Highly Imbalanced Classification
    # -------------------------------------------------------------------------
    print("\n[Scenario F] Highly Imbalanced Classification (90/10)...")
    np.random.seed(505)
    n = 150
    df_f = pd.DataFrame({
        "feature_x": np.random.randn(n),
        "feature_y": np.random.randn(n),
        "anomaly": [1] * 15 + [0] * 135
    })
    res_f = analyze_model(df_f, "anomaly", analysis_type="classification")
    prof_f = res_f["target_profile"]
    assert prof_f["class_imbalance"] is True
    assert prof_f["minority_percentage"] < 15.0
    assert prof_f["minority_class"] == "1"
    print(f"-> PASSED: Minority={prof_f['minority_class']} ({prof_f['minority_percentage']}%), Ratio={prof_f['minority_to_majority_ratio']}")

    # -------------------------------------------------------------------------
    # Scenario G: Small Dataset (12 rows)
    # -------------------------------------------------------------------------
    print("\n[Scenario G] Small Dataset (12 rows)...")
    df_g = pd.DataFrame({
        "param_1": [1.1, 2.3, 3.5, 4.2, 5.1, 6.0, 7.2, 8.4, 9.1, 10.0, 11.2, 12.5],
        "param_2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        "outcome": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    })
    res_g = analyze_model(df_g, "outcome", analysis_type="auto")
    assert res_g["task_type"] == "classification"
    assert res_g["dataset_info"]["usable_rows"] == 12
    # Small sample constraint finding should be present
    rc_titles = [rc["finding"] for rc in res_g["root_causes_structured"]]
    assert any("Small Dataset" in t for t in rc_titles)
    print(f"-> PASSED: Executed without crashing on 12 rows; Root Causes={rc_titles}")

    # -------------------------------------------------------------------------
    # Scenario H: Irrelevant Identifier Column
    # -------------------------------------------------------------------------
    print("\n[Scenario H] Dataset with Irrelevant Unique Identifier...")
    n = 80
    df_h = pd.DataFrame({
        "transaction_uuid": [f"TXN_{i:05d}" for i in range(n)],
        "amount": np.random.uniform(5, 500, n),
        "is_fraud": np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    })
    res_h = analyze_model(df_h, "is_fraud", analysis_type="classification")
    assert "transaction_uuid" in res_h["excluded_features"]
    assert "transaction_uuid" not in res_h["feature_importance"]
    print(f"-> PASSED: Correctly excluded identifier '{list(res_h['excluded_features'].keys())[0]}'")

    # -------------------------------------------------------------------------
    # Scenario I: Categorical & String Variables
    # -------------------------------------------------------------------------
    print("\n[Scenario I] Dataset with Categorical & String Variables...")
    n = 100
    df_i = pd.DataFrame({
        "device_type": np.random.choice(["Desktop", "Mobile", "Tablet"], size=n),
        "browser": np.random.choice(["Chrome", "Safari", "Edge", "Firefox"], size=n),
        "duration": np.random.uniform(10, 600, n),
        "converted": np.random.choice([True, False], size=n)
    })
    res_i = analyze_model(df_i, "converted", analysis_type="auto")
    assert res_i["task_type"] == "classification"
    assert "device_type" in res_i["feature_impact"]
    assert "browser" in res_i["feature_impact"]
    print(f"-> PASSED: Categorical variables successfully encoded; Top Features={list(res_i['feature_impact'].keys())[:3]}")

    # -------------------------------------------------------------------------
    # Scenario J: Strong Predictive Feature / Potential Leakage
    # -------------------------------------------------------------------------
    print("\n[Scenario J] Dataset with Strong Predictive Feature / Leakage...")
    n = 100
    y_vals = np.linspace(10, 500, n)
    df_j = pd.DataFrame({
        "leak_proxy": y_vals * 1.0,  # 100% correlated proxy
        "aux_feature": np.random.randn(n),
        "target_metric": y_vals
    })
    res_j = analyze_model(df_j, "target_metric", analysis_type="regression")
    assert len(res_j["leakage_findings"]) > 0
    leak_names = [l["feature"] for l in res_j["leakage_findings"]]
    assert "leak_proxy" in leak_names
    print(f"-> PASSED: Correctly detected target leakage in '{leak_names}' (Severity: {res_j['leakage_findings'][0]['severity']})")

    # -------------------------------------------------------------------------
    # Scenario K: Noisy Features with Weak Signal
    # -------------------------------------------------------------------------
    print("\n[Scenario K] Dataset with Pure Random Noise...")
    np.random.seed(999)
    n = 120
    df_k = pd.DataFrame({
        "noise_1": np.random.randn(n),
        "noise_2": np.random.randn(n),
        "noise_3": np.random.randn(n),
        "random_target": np.random.randn(n)
    })
    res_k = analyze_model(df_k, "random_target", analysis_type="regression")
    assert res_k["task_type"] == "regression"
    assert res_k["model_performance"]["r2"] < 0.20  # Low R2 on pure noise
    # Risk engine should identify explanatory power deficit
    driver_issues = [d["issue"] for d in res_k["risk"]["drivers"]]
    assert any("Explanatory Power" in issue for issue in driver_issues)
    print(f"-> PASSED: Model achieved low R2 ({res_k['model_performance']['r2']:.4f}); Risk Driver={driver_issues[0]}")

    print("\n" + "=" * 70)
    print("ALL 11 UNIVERSAL DATASET SCENARIOS (A - K) PASSED 100% GREEN!")
    print("=" * 70)


if __name__ == "__main__":
    run_universal_dataset_tests()
