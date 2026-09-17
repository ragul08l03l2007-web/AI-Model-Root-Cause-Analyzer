# test_generalization.py

import json
import numpy as np
import pandas as pd

from analysis.data_quality import analyze_data_quality
from analysis.model_analyzer import analyze_model


def test_binary_classification():
    print("--------------------------------------------------")
    print("TEST 1: Customer Churn / Credit Risk (Binary Classification)")
    print("--------------------------------------------------")
    np.random.seed(101)
    n = 150
    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "tenure_months": np.random.randint(1, 72, size=n),
        "monthly_charges": np.random.uniform(20.0, 120.0, size=n),
        "contract_type": np.random.choice(["Month-to-Month", "One Year", "Two Year"], size=n),
        "payment_method": np.random.choice(["Electronic", "Mailed", "Bank Transfer"], size=n),
        "support_tickets": np.random.poisson(1.5, size=n),
        "constant_col": ["FixedValue"] * n,
        "notes": [f"Customer account notes record #{i} with long description text" for i in range(n)],
        "signup_date": pd.date_range("2021-01-01", periods=n, freq="W").strftime("%Y-%m-%d"),
    })
    # Target churn correlated with monthly_charges and tenure
    score = (df["monthly_charges"] / 100.0) - (df["tenure_months"] / 40.0) + (df["support_tickets"] * 0.3)
    prob = 1.0 / (1.0 + np.exp(-score))
    df["churn"] = (np.random.uniform(size=n) < prob).astype(int)

    dq = analyze_data_quality(df)
    res = analyze_model(df, "churn", analysis_type="auto")

    assert res["task_type"] == "classification", f"Expected classification, got {res['task_type']}"
    assert "churn" not in res["feature_importance"], "Target column leaked into feature importance!"
    assert "constant_col" in res["excluded_features"], "Constant column was not excluded!"
    assert "notes" in res["excluded_features"], "Text column was not excluded!"
    assert len(res["root_causes_structured"]) > 0, "No root causes produced!"
    assert len(res["recommendations"]) > 0, "No recommendations produced!"
    assert len(res["priority_findings"]) > 0, "No priority findings produced!"

    # Verify JSON serializability with zero errors
    json_str = json.dumps(res, indent=2)
    assert "np.int" not in json_str, "NumPy type leaked into JSON!"
    assert "np.float" not in json_str, "NumPy type leaked into JSON!"

    print(f"Task Type: {res['task_type']} ({res['task_reason']})")
    print(f"Selected Model: {res['selected_model']}")
    print(f"Accuracy: {res['model_performance']['accuracy']:.1%}, F1: {res['model_performance']['f1_score']:.1%}")
    print(f"Cross-Validation: {res['cross_validation']['average_score']:.1%}")
    print(f"Excluded Features: {list(res['excluded_features'].keys())}")
    print(f"Cross-Model Status: {res['cross_model_status']}")
    print(f"Discovered Segments: {len(res['segment_analysis'])}")
    print(f"Root Causes: {len(res['root_causes_structured'])}")
    print("-> Test 1 PASSED!\n")


def test_regression_house_prices():
    print("--------------------------------------------------")
    print("TEST 2: Real Estate Price Estimation (Continuous Regression)")
    print("--------------------------------------------------")
    np.random.seed(202)
    n = 120
    df = pd.DataFrame({
        "property_id": [f"PROP_{i}" for i in range(n)],
        "sqft": np.random.uniform(600, 3500, size=n),
        "bedrooms": np.random.choice([1, 2, 3, 4, 5], size=n),
        "bathrooms": np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0], size=n),
        "neighborhood": np.random.choice(["Downtown", "Suburbs", "Rural", "Uptown"], size=n),
        "year_built": np.random.randint(1970, 2023, size=n),
    })
    # Target price
    df["price"] = (
        100000
        + df["sqft"] * 180.0
        + df["bedrooms"] * 15000
        + (df["year_built"] - 1970) * 1200
        + np.random.normal(0, 25000, size=n)
    )

    dq = analyze_data_quality(df)
    res = analyze_model(df, "price", analysis_type="auto")

    assert res["task_type"] == "regression", f"Expected regression, got {res['task_type']}"
    assert "price" not in res["feature_importance"], "Target column leaked into feature importance!"
    assert res["model_performance"]["r2"] is not None, "Missing R2 score!"
    assert len(res["prediction_errors"]) > 0, "No prediction errors analyzed!"

    # Verify JSON serialization
    json_str = json.dumps(res, indent=2)
    assert "np.int" not in json_str, "NumPy type leaked into JSON!"
    assert "np.float" not in json_str, "NumPy type leaked into JSON!"

    print(f"Task Type: {res['task_type']} ({res['task_reason']})")
    print(f"Selected Model: {res['selected_model']}")
    print(f"MAE: ${res['regression_metrics']['mae']:,.2f}, R²: {res['regression_metrics']['r2']:.4f}")
    print(f"CV R²: {res['cross_validation']['average_score']:.4f}")
    print(f"Feature Impact: {list(res['feature_impact'].keys())[:3]}")
    print(f"Discovered Segments: {len(res['segment_analysis'])}")
    print(f"Root Causes: {len(res['root_causes_structured'])}")
    print("-> Test 2 PASSED!\n")


def test_multiclass_classification():
    print("--------------------------------------------------")
    print("TEST 3: Multiclass Customer Tier (3 Classes)")
    print("--------------------------------------------------")
    np.random.seed(303)
    n = 180
    df = pd.DataFrame({
        "engagement_score": np.random.uniform(10, 100, size=n),
        "purchase_frequency": np.random.poisson(4, size=n),
        "app_sessions": np.random.randint(1, 50, size=n),
        "tier": np.random.choice(["Bronze", "Silver", "Gold"], size=n, p=[0.5, 0.35, 0.15])
    })

    dq = analyze_data_quality(df)
    res = analyze_model(df, "tier", analysis_type="auto")

    assert res["task_type"] == "classification"
    assert res["class_count"] == 3
    assert len(res["confusion_matrix"]) == 3

    json_str = json.dumps(res, indent=2)
    assert "np.int" not in json_str
    assert "np.float" not in json_str

    print(f"Task Type: {res['task_type']} (Classes: {res['class_names']})")
    print(f"Class Distribution: {res['class_distribution']}")
    print(f"Per-Class Metrics: {list(res['class_metrics'].keys())}")
    print(f"Selected Model: {res['selected_model']}")
    print("-> Test 3 PASSED!\n")


def test_edge_cases_and_small_dataset():
    print("--------------------------------------------------")
    print("TEST 4: Small Dataset (15 rows) with Missing Data & Near Constants")
    print("--------------------------------------------------")
    df = pd.DataFrame({
        "feat_a": [1, 2, 3, None, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "feat_b": ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "B"], # near constant
        "feat_c": [10.5, 12.0, 14.5, 11.0, 13.0, 15.0, 16.5, 17.0, 18.5, 19.0, 20.5, 21.0, 22.5, 23.0, 24.5],
        "target": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    })

    dq = analyze_data_quality(df)
    res = analyze_model(df, "target", analysis_type="auto")

    assert res["task_type"] == "classification"
    assert res["model_performance"]["accuracy"] is not None
    assert "target" in str(res["task_reason"]).lower() or "discrete" in str(res["task_reason"]).lower()
    
    json_str = json.dumps(res, indent=2)
    assert "np.int" not in json_str
    assert "np.float" not in json_str

    print(f"Auto Detect Reason: {res['task_reason']}")
    print(f"Model: {res['selected_model']}, Accuracy: {res['model_performance']['accuracy']:.1%}")
    print(f"Data Quality Score: {dq['quality_score']}/100 ({dq['overall_quality']})")
    print("-> Test 4 PASSED!\n")


if __name__ == "__main__":
    test_binary_classification()
    test_regression_house_prices()
    test_multiclass_classification()
    test_edge_cases_and_small_dataset()
    print("ALL 4 GENERALIZATION & DIAGNOSTIC TESTS PASSED SUCCESSFULLY!")
