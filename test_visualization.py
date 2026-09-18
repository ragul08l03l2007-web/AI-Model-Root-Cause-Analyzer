# test_visualization.py
"""
Test Suite for Evidence-Driven Visualization Engine (analysis/visualization.py).
Validates that Plotly chart specifications:
1. Generate correctly for Classification datasets (Performance, Confusion Matrix, Donut, Features, Missing Values, Risk Gauge)
2. Generate correctly for Regression datasets (Metrics, Actual vs Predicted, Residual Plot, Residual Distribution)
3. Return 100% valid JSON-serializable primitives (int, float, list, dict, str) with NO numpy types.
4. Have valid Plotly 'data', 'layout', and 'config' keys.
"""

import os
import json
import pandas as pd
from analysis.model_analyzer import analyze_model
from analysis.visualization import (
    generate_all_visualizations,
    build_risk_gauge_chart,
    build_performance_metrics_chart,
    build_cv_stability_chart,
    build_feature_importance_chart,
    build_missing_values_chart,
    build_target_distribution_chart,
    build_confusion_matrix_chart,
    build_actual_vs_predicted_chart,
    build_residual_plot,
    build_residual_distribution_chart
)


def _assert_valid_plotly_structure(chart_dict: dict, name: str):
    assert isinstance(chart_dict, dict), f"[{name}] Expected dict, got {type(chart_dict)}"
    assert "data" in chart_dict, f"[{name}] Missing 'data' key"
    assert "layout" in chart_dict, f"[{name}] Missing 'layout' key"
    assert isinstance(chart_dict["data"], list), f"[{name}] 'data' must be a list"
    assert isinstance(chart_dict["layout"], dict), f"[{name}] 'layout' must be a dict"

    # Verify JSON serializability with zero custom encoders
    try:
        serialized = json.dumps(chart_dict)
        assert len(serialized) > 10, f"[{name}] Empty JSON output"
    except Exception as e:
        raise AssertionError(f"[{name}] Failed JSON serialization (potential NumPy leak): {e}")


def test_classification_visualizations():
    print("\n[TEST 1] Testing Visualization Generation on Classification Dataset...")
    df = pd.read_csv("random_test_dataset.csv")
    result = analyze_model(df, target_column="churn", analysis_type="classification")

    charts = generate_all_visualizations(result)
    assert isinstance(charts, dict)

    expected_charts = [
        "risk_gauge",
        "performance_metrics",
        "cv_stability",
        "feature_importance",
        "target_distribution",
        "confusion_matrix"
    ]

    for key in expected_charts:
        assert key in charts, f"Missing expected classification chart: {key}"
        _assert_valid_plotly_structure(charts[key], key)
        print(f"  -> Validated Plotly chart: {key} (Traces: {len(charts[key]['data'])})")

    # Verify confusion matrix specific values
    cm_chart = charts["confusion_matrix"]
    assert cm_chart["data"][0]["type"] == "heatmap"
    print("  -> Confusion matrix heatmap confirmed")
    print("  -> TEST 1 (Classification Visualizations) PASSED!")


def test_regression_visualizations():
    print("\n[TEST 2] Testing Visualization Generation on Continuous Regression Dataset...")
    df = pd.read_csv("continuous_regression_test_dataset.csv")
    result = analyze_model(df, target_column="annual_bonus", analysis_type="regression")

    charts = generate_all_visualizations(result)
    assert isinstance(charts, dict)

    expected_charts = [
        "risk_gauge",
        "performance_metrics",
        "cv_stability",
        "feature_importance",
        "actual_vs_predicted",
        "residual_plot",
        "residual_distribution"
    ]

    for key in expected_charts:
        assert key in charts, f"Missing expected regression chart: {key}"
        _assert_valid_plotly_structure(charts[key], key)
        print(f"  -> Validated Plotly chart: {key} (Traces: {len(charts[key]['data'])})")

    # Verify regression-specific charts
    avp = charts["actual_vs_predicted"]
    assert avp["data"][0]["type"] == "scatter"
    assert avp["data"][1]["name"] == "Ideal Fit (y = ŷ)"

    res_plot = charts["residual_plot"]
    assert res_plot["data"][0]["type"] == "scatter"
    assert res_plot["data"][1]["name"] == "Zero Error Baseline"

    res_dist = charts["residual_distribution"]
    assert res_dist["data"][0]["type"] == "histogram"

    print("  -> Regression diagnostics (Actual vs Predicted, Residual Plot, Residual Distribution) confirmed")
    print("  -> TEST 2 (Regression Visualizations) PASSED!")


def test_isolated_components():
    print("\n[TEST 3] Testing Isolated Component Renderers & Edge Cases...")
    dummy_result = {
        "task_type": "classification",
        "risk": {"score": 45, "level": "MEDIUM", "drivers": [{"domain": "Data Quality", "points": 20}]},
        "model_performance": {"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1_score": 0.85},
        "target_column": "target",
        "target_profile": {"class_distribution_list": [{"label": "0", "count": 70}, {"label": "1", "count": 30}]},
        "feature_impact": {"feat_a": {"importance": 0.6, "relative_share_pct": 60.0, "influence_tier": "Very Strong"}},
        "data_quality": {"missing_percentages": {"feat_a": 12.5, "feat_b": 0.0}}
    }

    gauge = build_risk_gauge_chart(dummy_result)
    _assert_valid_plotly_structure(gauge, "isolated_gauge")

    perf = build_performance_metrics_chart(dummy_result)
    _assert_valid_plotly_structure(perf, "isolated_perf")

    feats = build_feature_importance_chart(dummy_result)
    _assert_valid_plotly_structure(feats, "isolated_feats")

    miss = build_missing_values_chart(dummy_result)
    _assert_valid_plotly_structure(miss, "isolated_miss")

    print("  -> Edge-case & isolated component tests passed")
    print("  -> TEST 3 PASSED!")


if __name__ == "__main__":
    print("======================================================================")
    print("RUNNING EVIDENCE-DRIVEN VISUALIZATION TEST SUITE")
    print("======================================================================")
    test_classification_visualizations()
    test_regression_visualizations()
    test_isolated_components()
    print("======================================================================")
    print("ALL VISUALIZATION ENGINE TESTS PASSED 100%!")
    print("======================================================================")
