# analysis/visualization.py
"""
AI Model Root-Cause Analyzer: Evidence-Driven Visualization Engine.
Transforms deterministic ML diagnostic outputs, model metrics, data-quality profiles,
feature influences, and prediction errors into interactive Plotly chart specifications.

Architecture:
- Pure Python & standard library JSON serializers (zero NumPy type leaks).
- Produces Plotly data/layout/config dictionary specifications for direct browser rendering.
- Covers:
  1. Risk Severity & Health Gauge + Domain Factor Breakdown
  2. Model Performance Metrics & Cross-Validation Fold Stability
  3. Feature Importance Ranking & Relative Share Tiers
  4. Data Quality (Missing Values %, Outliers, Class/Target Distributions)
  5. Error Diagnostics (Confusion Matrix Heatmap for Classification)
  6. Regression Diagnostics (Actual vs Predicted, Residual Plots, Residual Distributions)
  7. Empirical Feature-Target Relationships
"""

from typing import Dict, List, Any, Optional
import math
from analysis.evidence import safe_primitive


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Sanitizes numbers to float without NaN/Inf."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 4)
    except (ValueError, TypeError):
        return default


def _default_layout(title: str, height: int = 340, dark_mode: bool = False) -> Dict[str, Any]:
    """Generates a clean, modern, responsive Plotly layout."""
    font_color = "#0f172a"
    grid_color = "#e2e8f0"
    paper_bg = "rgba(0,0,0,0)"
    plot_bg = "rgba(0,0,0,0)"

    return {
        "title": {
            "text": f"<b>{title}</b>",
            "font": {"size": 15, "color": font_color, "family": "system-ui, -apple-system, sans-serif"},
            "x": 0.02,
            "xanchor": "left"
        },
        "margin": {"l": 50, "r": 30, "t": 50, "b": 45},
        "height": height,
        "autosize": True,
        "paper_bgcolor": paper_bg,
        "plot_bgcolor": plot_bg,
        "font": {"family": "system-ui, -apple-system, sans-serif", "color": font_color, "size": 12},
        "xaxis": {
            "showgrid": True,
            "gridcolor": grid_color,
            "zeroline": False,
            "tickfont": {"size": 11, "color": "#475569"}
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": grid_color,
            "zeroline": False,
            "tickfont": {"size": 11, "color": "#475569"}
        },
        "hoverlabel": {
            "bgcolor": "#1e293b",
            "font": {"color": "#ffffff", "family": "system-ui, sans-serif", "size": 12},
            "bordercolor": "#334155"
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 11}
        }
    }


def _default_config() -> Dict[str, Any]:
    """Default Plotly interactive configuration."""
    return {
        "responsive": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "hoverClosestCartesian", "hoverCompareCartesian"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "model_diagnostic_chart",
            "height": 500,
            "width": 800,
            "scale": 2
        }
    }


# ==============================================================================
# 1. RISK & HEALTH METERS
# ==============================================================================

def build_risk_gauge_chart(result: Dict[str, Any]) -> Dict[str, Any]:
    """Builds a semi-circular Plotly risk gauge (0 to 100)."""
    risk_info = result.get("risk", result.get("overall_risk", {}))
    if isinstance(risk_info, dict):
        score = _safe_float(risk_info.get("score", risk_info.get("risk_score", result.get("risk_score", 0))))
        level = str(risk_info.get("level", risk_info.get("risk_level", result.get("overall_risk", "LOW")))).upper()
    else:
        score = _safe_float(result.get("risk_score", 0))
        level = str(result.get("overall_risk", "LOW")).upper()

    bar_color = "#10b981" if score < 30 else ("#f59e0b" if score < 60 else "#ef4444")

    data = [{
        "type": "indicator",
        "mode": "gauge+number",
        "value": score,
        "title": {"text": f"<b>{level} RISK</b><br><span style='font-size:12px;color:#64748b;'>Operational Vulnerability Index</span>", "font": {"size": 14}},
        "number": {"suffix": "/100", "font": {"size": 28, "color": bar_color, "family": "system-ui, sans-serif"}},
        "gauge": {
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8", "tickvals": [0, 30, 60, 100]},
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": "#f1f5f9",
            "borderwidth": 1,
            "bordercolor": "#cbd5e1",
            "steps": [
                {"range": [0, 30], "color": "rgba(16, 185, 129, 0.15)"},
                {"range": [30, 60], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [60, 100], "color": "rgba(239, 68, 68, 0.15)"}
            ],
            "threshold": {
                "line": {"color": "#0f172a", "width": 3},
                "thickness": 0.8,
                "value": score
            }
        }
    }]

    layout = {
        "margin": {"l": 30, "r": 30, "t": 40, "b": 20},
        "height": 220,
        "autosize": True,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "system-ui, sans-serif"}
    }

    return safe_primitive({"data": data, "layout": layout, "config": _default_config()})


def build_risk_breakdown_chart(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Builds a horizontal bar chart showing points contributed by each risk domain."""
    risk_info = result.get("risk", result.get("overall_risk", {}))
    drivers = []
    if isinstance(risk_info, dict):
        drivers = risk_info.get("drivers", [])

    if not drivers:
        root_causes = result.get("root_causes_structured", [])
        for rc in root_causes:
            sev = rc.get("severity", "MEDIUM")
            pts = 30 if sev == "CRITICAL" else (20 if sev == "HIGH" else (10 if sev == "MEDIUM" else 5))
            drivers.append({
                "domain": rc.get("category", rc.get("domain", "General")),
                "issue": rc.get("finding", rc.get("root_cause", "Diagnostic Finding")),
                "points": pts,
                "severity": sev
            })

    if not drivers:
        return None

    domain_pts: Dict[str, float] = {}
    for d in drivers:
        dom = d.get("domain", "General").title()
        pts = float(d.get("points", 10))
        domain_pts[dom] = domain_pts.get(dom, 0.0) + pts

    domains = list(domain_pts.keys())
    points = [domain_pts[k] for k in domains]

    colors = ["#4f46e5", "#0ea5e9", "#f59e0b", "#ef4444", "#10b981", "#8b5cf6"]

    data = [{
        "type": "bar",
        "y": domains,
        "x": points,
        "orientation": "h",
        "marker": {"color": colors[:len(domains)]},
        "text": [f"+{p:.0f} pts" for p in points],
        "textposition": "auto",
        "hovertemplate": "<b>%{y}</b><br>Risk Contribution: +%{x} pts<extra></extra>"
    }]

    layout = _default_layout("Risk Score Drivers by Domain", height=220)
    layout["xaxis"]["title"] = "Risk Points Contributed"
    layout["margin"]["l"] = 120

    return safe_primitive({"data": data, "layout": layout, "config": _default_config()})


# ==============================================================================
# 2. MODEL PERFORMANCE & CROSS-VALIDATION
# ==============================================================================

def build_performance_metrics_chart(result: Dict[str, Any]) -> Dict[str, Any]:
    """Builds a bar chart of evaluated model performance metrics."""
    task_type = result.get("task_type", "classification")
    selected_model = result.get("selected_model", "Selected Model")

    if task_type == "classification":
        perf = result.get("model_performance", {})
        metric_names = ["Accuracy", "Balanced Acc", "Precision", "Recall", "F1 Score"]
        metric_keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score"]
        values = []
        for k in metric_keys:
            v = perf.get(k, perf.get(f"weighted_{k}", 0.0))
            values.append(round(_safe_float(v) * 100, 2))

        data = [{
            "type": "bar",
            "x": metric_names,
            "y": values,
            "marker": {
                "color": ["#3b82f6", "#06b6d4", "#10b981", "#8b5cf6", "#6366f1"],
                "line": {"color": "#1e40af", "width": 1}
            },
            "text": [f"{v:.1f}%" for v in values],
            "textposition": "outside",
            "hovertemplate": "<b>%{x}</b>: %{y:.2f}%<extra></extra>"
        }]

        layout = _default_layout(f"Model Performance Metrics ({selected_model})", height=320)
        layout["yaxis"]["range"] = [0, 105]
        layout["yaxis"]["title"] = "Score (%)"

    else:
        reg = result.get("regression_metrics", result.get("model_performance", {}))
        metric_names = ["R² Score", "Explained Var", "RMSE", "MAE", "MedAE"]
        r2 = _safe_float(reg.get("r2_score", reg.get("r2", 0.0)))
        exp_var = _safe_float(reg.get("explained_variance", 0.0))
        rmse = _safe_float(reg.get("rmse", reg.get("root_mean_squared_error", 0.0)))
        mae = _safe_float(reg.get("mae", reg.get("mean_absolute_error", 0.0)))
        medae = _safe_float(reg.get("median_absolute_error", 0.0))

        raw_values = [r2, exp_var, rmse, mae, medae]

        data = [{
            "type": "bar",
            "x": metric_names,
            "y": raw_values,
            "marker": {
                "color": ["#10b981" if r2 > 0.5 else "#f59e0b", "#3b82f6", "#ef4444", "#f97316", "#8b5cf6"],
                "line": {"color": "#334155", "width": 1}
            },
            "text": [f"{v:.3f}" for v in raw_values],
            "textposition": "outside",
            "hovertemplate": "<b>%{x}</b>: %{y:.4f}<extra></extra>"
        }]

        layout = _default_layout(f"Regression Evaluation Metrics ({selected_model})", height=320)
        layout["yaxis"]["title"] = "Metric Value"

    return safe_primitive({"data": data, "layout": layout, "config": _default_config()})


def build_cv_stability_chart(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Builds a cross-validation fold stability chart with mean baseline."""
    cv = result.get("cross_validation", {})
    scores = cv.get("scores", cv.get("fold_scores", []))
    if not scores:
        return None

    fold_labels = [f"Fold {i+1}" for i in range(len(scores))]
    scores_pct = [round(_safe_float(s) * (100 if result.get("task_type") == "classification" else 1.0), 2) for s in scores]
    mean_val = round(_safe_float(cv.get("mean", sum(scores)/len(scores))) * (100 if result.get("task_type") == "classification" else 1.0), 2)

    data = [
        {
            "type": "bar",
            "x": fold_labels,
            "y": scores_pct,
            "name": "Fold Score",
            "marker": {"color": "#6366f1", "opacity": 0.85},
            "text": [f"{v:.2f}" for v in scores_pct],
            "textposition": "auto",
            "hovertemplate": "<b>%{x}</b>: %{y:.2f}<extra></extra>"
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": fold_labels,
            "y": [mean_val] * len(fold_labels),
            "name": f"Mean CV ({mean_val:.2f})",
            "line": {"color": "#ef4444", "width": 2.5, "dash": "dash"},
            "hovertemplate": f"<b>Mean Score</b>: {mean_val:.2f}<extra></extra>"
        }
    ]

    unit = "%" if result.get("task_type") == "classification" else "Score"
    layout = _default_layout(f"Cross-Validation Fold Stability ({cv.get('stability', 'STABLE')} Stability)", height=320)
    layout["yaxis"]["title"] = f"Validation Score ({unit})"
    if result.get("task_type") == "classification":
        min_s = max(0, min(scores_pct) - 10)
        max_s = min(105, max(scores_pct) + 10)
        layout["yaxis"]["range"] = [min_s, max_s]

    return safe_primitive({"data": data, "layout": layout, "config": _default_config()})


# ==============================================================================
# 3. FEATURE IMPACT & IMPORTANCE
# ==============================================================================

def build_feature_importance_chart(result: Dict[str, Any], max_features: int = 10) -> Optional[Dict[str, Any]]:
    """Builds a sorted horizontal bar chart of feature importances with color tiers."""
    feature_impact = result.get("feature_impact", result.get("feature_importance", {}))
    if not feature_impact or not isinstance(feature_impact, dict):
        return None

    target_col = str(result.get("target_column", ""))

    items = []
    for feat, data in feature_impact.items():
        if str(feat) == target_col:
            continue
        if isinstance(data, dict):
            imp = _safe_float(data.get("importance", 0.0))
            share = _safe_float(data.get("relative_share_pct", 0.0))
            tier = data.get("influence_tier", "Moderate Model Influence")
            direct = data.get("direction", "")
        else:
            imp = _safe_float(data)
            share = 0.0
            tier = "Moderate Model Influence"
            direct = ""
        items.append({"feature": str(feat), "importance": imp, "share": share, "tier": tier, "direction": direct})

    if not items:
        return None

    items.sort(key=lambda x: x["importance"], reverse=True)
    items = items[:max_features]
    items.reverse()  # For bottom-up horizontal bar ordering

    feats = [x["feature"] for x in items]
    shares = [x["share"] if x["share"] > 0 else x["importance"] for x in items]

    color_map = {
        "Very Strong": "#4f46e5",
        "Strong": "#0ea5e9",
        "Moderate": "#3b82f6",
        "Low": "#94a3b8"
    }

    colors = []
    for x in items:
        matched = False
        for k, col in color_map.items():
            if k in x["tier"]:
                colors.append(col)
                matched = True
                break
        if not matched:
            colors.append("#6366f1")

    data = [{
        "type": "bar",
        "y": feats,
        "x": shares,
        "orientation": "h",
        "marker": {"color": colors},
        "text": [f"{s:.1f}%" if x["share"] > 0 else f"{s:.4f}" for s, x in zip(shares, items)],
        "textposition": "auto",
        "customdata": [[x["tier"], x["direction"]] for x in items],
        "hovertemplate": "<b>%{y}</b><br>Share: %{x:.2f}%<br>Tier: %{customdata[0]}<br>%{customdata[1]}<extra></extra>"
    }]

    layout = _default_layout(f"Top {len(items)} Predictive Feature Influence", height=max(280, len(items) * 28))
    layout["xaxis"]["title"] = "Relative Importance Share (%)" if any(x["share"] > 0 for x in items) else "Importance Score"
    layout["margin"]["l"] = 140

    return safe_primitive({"data": data, "layout": layout, "config": _default_config()})


# ==============================================================================
# 4. DATA QUALITY & TARGET PROFILE
# ==============================================================================

def build_missing_values_chart(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Builds a bar chart showing missing value percentage by column."""
    quality = result.get("data_quality", {})
    missing_pcts = quality.get("missing_percentages", {})
    if not missing_pcts:
        missing_vals = quality.get("missing_values", {})
        total_rows = result.get("dataset_info", {}).get("rows", 100) or 100
        missing_pcts = {k: (_safe_float(v) / total_rows) * 100 for k, v in missing_vals.items()}

    if not missing_pcts:
        return None

    # Filter or sort columns
    sorted_cols = sorted(missing_pcts.items(), key=lambda x: x[1], reverse=True)[:15]
    cols = [k for k, _ in sorted_cols]
    vals = [round(_safe_float(v), 2) for _, v in sorted_cols]

    colors = ["#ef4444" if v >= 20 else ("#f59e0b" if v >= 5 else "#10b981") for v in vals]

    data = [
        {
            "type": "bar",
            "x": cols,
            "y": vals,
            "marker": {"color": colors},
            "text": [f"{v:.1f}%" for v in vals],
            "textposition": "auto",
            "hovertemplate": "<b>%{x}</b>: %{y:.2f}% Missing<extra></extra>"
        }
    ]

    layout = _default_layout("Missing Values Percentage by Column", height=300)
    layout["yaxis"]["title"] = "Missing (%)"
    layout["yaxis"]["range"] = [0, max(25, max(vals) + 5) if vals else 25]

    # Threshold shapes
    layout["shapes"] = [
        {
            "type": "line",
            "x0": -0.5,
            "x1": len(cols) - 0.5,
            "y0": 5,
            "y1": 5,
            "line": {"color": "#f59e0b", "width": 1.5, "dash": "dot"}
        },
        {
            "type": "line",
            "x0": -0.5,
            "x1": len(cols) - 0.5,
            "y0": 20,
            "y1": 20,
            "line": {"color": "#ef4444", "width": 1.5, "dash": "dot"}
        }
    ]

    return safe_primitive({"data": data, "layout": layout, "config": _default_config()})


def build_target_distribution_chart(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Builds class distribution donut for classification or histogram for regression."""
    task_type = result.get("task_type", "classification")
    target_col = result.get("target_column", "Target")

    if task_type == "classification":
        target_profile = result.get("target_profile", result.get("target_analysis", {}))
        dist_list = target_profile.get("class_distribution_list", [])

        if not dist_list:
            raw_dist = target_profile.get("class_distribution", result.get("class_distribution", {}))
            dist_list = [{"label": str(k), "count": int(v), "percentage": 0.0} for k, v in raw_dist.items()]

        if not dist_list:
            return None

        labels = [str(x.get("label", "")) for x in dist_list]
        counts = [int(x.get("count", 0)) for x in dist_list]

        palette = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4"]

        data = [{
            "type": "pie",
            "labels": labels,
            "values": counts,
            "hole": 0.55,
            "marker": {"colors": palette[:len(labels)]},
            "textinfo": "label+percent",
            "hovertemplate": "<b>Class %{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"
        }]

        imbalance = target_profile.get("class_imbalance", False)
        sub_text = "Class Imbalance Detected" if imbalance else "Balanced Distribution"
        layout = _default_layout(f"Target Class Distribution: '{target_col}'", height=300)
        layout["annotations"] = [{
            "text": f"<b>{len(labels)} Classes</b><br><span style='font-size:11px;color:#64748b;'>{sub_text}</span>",
            "showarrow": False,
            "font": {"size": 12, "color": "#0f172a"}
        }]

        return safe_primitive({"data": data, "layout": layout, "config": _default_config()})

    else:
        # Regression target summary stats
        target_profile = result.get("target_profile", result.get("target_analysis", {}))
        mean_v = _safe_float(target_profile.get("mean", 0.0))
        median_v = _safe_float(target_profile.get("median", 0.0))
        min_v = _safe_float(target_profile.get("min", 0.0))
        max_v = _safe_float(target_profile.get("max", 0.0))
        q1 = _safe_float(target_profile.get("q1", 0.0))
        q3 = _safe_float(target_profile.get("q3", 0.0))

        data = [{
            "type": "box",
            "name": str(target_col),
            "q1": [q1],
            "median": [median_v],
            "q3": [q3],
            "mean": [mean_v],
            "lowerfence": [min_v],
            "upperfence": [max_v],
            "marker": {"color": "#3b82f6"},
            "boxpoints": False,
            "hovertemplate": f"<b>{target_col}</b><br>Mean: {mean_v:.2f}<br>Median: {median_v:.2f}<br>Range: [{min_v:.2f}, {max_v:.2f}]<extra></extra>"
        }]

        layout = _default_layout(f"Continuous Target Spread: '{target_col}'", height=280)
        layout["yaxis"]["title"] = f"{target_col} Value Range"

        return safe_primitive({"data": data, "layout": layout, "config": _default_config()})


# ==============================================================================
# 5. ERROR DIAGNOSTICS & CONFUSION MATRIX
# ==============================================================================

def build_confusion_matrix_chart(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Builds an interactive 2D Confusion Matrix Heatmap for classification."""
    if result.get("task_type") != "classification":
        return None

    cm_data = result.get("confusion_matrix", {})
    if isinstance(cm_data, list):
        matrix = cm_data
        classes = [str(c) for c in result.get("class_names", [])]
    elif isinstance(cm_data, dict):
        matrix = cm_data.get("matrix", [])
        classes = [str(c) for c in cm_data.get("classes", result.get("class_names", []))]
    else:
        matrix = []
        classes = []

    if not classes and matrix:
        classes = [f"Class {i}" for i in range(len(matrix))]

    if not matrix or not isinstance(matrix, list) or len(matrix) < 2:
        return None

    # Calculate total and text annotations
    text_matrix = []
    total_samples = sum(sum(row) for row in matrix) or 1

    for i, row in enumerate(matrix):
        row_text = []
        for j, val in enumerate(row):
            pct = (val / total_samples) * 100
            status = "Correct" if i == j else "Error"
            row_text.append(f"{val} ({pct:.1f}%)<br>[{status}]")
        text_matrix.append(row_text)

    data = [{
        "type": "heatmap",
        "z": matrix,
        "x": [f"Pred: {c}" for c in classes],
        "y": [f"Actual: {c}" for c in classes],
        "text": text_matrix,
        "texttemplate": "%{text}",
        "colorscale": [
            [0.0, "#eff6ff"],
            [0.3, "#bfdbfe"],
            [0.7, "#3b82f6"],
            [1.0, "#1e3a8a"]
        ],
        "showscale": True,
        "hovertemplate": "<b>%{y}</b><br><b>%{x}</b><br>Observations: %{z}<extra></extra>"
    }]

    layout = _default_layout("Confusion Matrix (Actual vs Predicted)", height=340)
    layout["xaxis"]["title"] = "Predicted Class"
    layout["yaxis"]["title"] = "Actual True Class"
    layout["yaxis"]["autorange"] = "reversed"

    return safe_primitive({"data": data, "layout": layout, "config": _default_config()})


# ==============================================================================
# 6. REGRESSION DIAGNOSTICS (ACTUAL VS PREDICTED & RESIDUALS)
# ==============================================================================

def build_actual_vs_predicted_chart(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Builds Actual vs Predicted scatter plot with 45-degree reference line."""
    if result.get("task_type") != "regression":
        return None

    pred_errors = result.get("prediction_errors", [])
    if not pred_errors or not isinstance(pred_errors, list):
        return None

    actuals = [_safe_float(p.get("actual", 0.0)) for p in pred_errors]
    predicteds = [_safe_float(p.get("predicted", 0.0)) for p in pred_errors]

    if not actuals or not predicteds:
        return None

    min_val = min(min(actuals), min(predicteds))
    max_val = max(max(actuals), max(predicteds))
    padding = (max_val - min_val) * 0.05
    line_min = min_val - padding
    line_max = max_val + padding

    data = [
        {
            "type": "scatter",
            "mode": "markers",
            "x": actuals,
            "y": predicteds,
            "name": "Predictions",
            "marker": {
                "color": "#3b82f6",
                "size": 7,
                "opacity": 0.75,
                "line": {"color": "#1e40af", "width": 1}
            },
            "hovertemplate": "Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>"
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": [line_min, line_max],
            "y": [line_min, line_max],
            "name": "Ideal Fit (y = ŷ)",
            "line": {"color": "#ef4444", "width": 2, "dash": "dash"},
            "hoverinfo": "none"
        }
    ]

    layout = _default_layout("Actual vs. Predicted Values", height=340)
    layout["xaxis"]["title"] = "Actual True Value"
    layout["yaxis"]["title"] = "Model Predicted Value"

    return safe_primitive({"data": data, "layout": layout, "config": _default_config()})


def build_residual_plot(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Builds a Residuals vs Predicted scatter plot with zero-error baseline."""
    if result.get("task_type") != "regression":
        return None

    pred_errors = result.get("prediction_errors", [])
    if not pred_errors or not isinstance(pred_errors, list):
        return None

    predicteds = [_safe_float(p.get("predicted", 0.0)) for p in pred_errors]
    residuals = [_safe_float(p.get("residual", p.get("actual", 0.0) - p.get("predicted", 0.0))) for p in pred_errors]

    if not predicteds or not residuals:
        return None

    min_p = min(predicteds)
    max_p = max(predicteds)

    colors = ["#ef4444" if abs(r) > (2 * (sum(abs(x) for x in residuals)/len(residuals))) else "#3b82f6" for r in residuals]

    data = [
        {
            "type": "scatter",
            "mode": "markers",
            "x": predicteds,
            "y": residuals,
            "name": "Residuals",
            "marker": {
                "color": colors,
                "size": 7,
                "opacity": 0.8
            },
            "hovertemplate": "Predicted: %{x:.3f}<br>Residual (y - ŷ): %{y:.3f}<extra></extra>"
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": [min_p, max_p],
            "y": [0, 0],
            "name": "Zero Error Baseline",
            "line": {"color": "#0f172a", "width": 2, "dash": "dash"},
            "hoverinfo": "none"
        }
    ]

    layout = _default_layout("Residual Plot (Residuals vs Predicted)", height=340)
    layout["xaxis"]["title"] = "Predicted Value (ŷ)"
    layout["yaxis"]["title"] = "Residual Error (Actual - Predicted)"

    return safe_primitive({"data": data, "layout": layout, "config": _default_config()})


def build_residual_distribution_chart(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Builds a histogram of regression residuals to check normality."""
    if result.get("task_type") != "regression":
        return None

    pred_errors = result.get("prediction_errors", [])
    if not pred_errors or not isinstance(pred_errors, list):
        return None

    residuals = [_safe_float(p.get("residual", p.get("actual", 0.0) - p.get("predicted", 0.0))) for p in pred_errors]
    if not residuals:
        return None

    data = [{
        "type": "histogram",
        "x": residuals,
        "nbinsx": 15,
        "marker": {
            "color": "#8b5cf6",
            "line": {"color": "#6d28d9", "width": 1}
        },
        "hovertemplate": "Residual Bin: %{x}<br>Count: %{y}<extra></extra>"
    }]

    layout = _default_layout("Residual Error Distribution", height=300)
    layout["xaxis"]["title"] = "Residual (Actual - Predicted)"
    layout["yaxis"]["title"] = "Sample Count"

    return safe_primitive({"data": data, "layout": layout, "config": _default_config()})


# ==============================================================================
# 7. EMPIRICAL FEATURE RELATIONSHIPS
# ==============================================================================

def build_feature_relationships_chart(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Builds relationship charts for the dominant feature."""
    feature_rel = result.get("feature_relationships", [])
    if not feature_rel or not isinstance(feature_rel, list):
        return None

    # Pick the top relationship
    top_rel = feature_rel[0]
    feat_name = top_rel.get("feature_name", "Top Feature")
    cat_summary = top_rel.get("category_summary", [])
    task_type = result.get("task_type", "classification")

    if cat_summary and isinstance(cat_summary, list):
        cats = [str(c.get("category", "")) for c in cat_summary]
        samples = [int(c.get("sample_count", 0)) for c in cat_summary]

        if task_type == "regression":
            means = [_safe_float(c.get("mean_target", 0.0)) for c in cat_summary]
            data = [{
                "type": "bar",
                "x": cats,
                "y": means,
                "marker": {"color": "#0ea5e9"},
                "text": [f"{m:.2f}" for m in means],
                "textposition": "auto",
                "hovertemplate": "<b>%{x}</b><br>Mean Target: %{y:.2f}<extra></extra>"
            }]
            layout = _default_layout(f"Mean Target Value by '{feat_name}' Category", height=300)
            layout["yaxis"]["title"] = "Mean Target Value"
        else:
            pcts = [_safe_float(c.get("dominant_class_pct", 0.0)) for c in cat_summary]
            dom_classes = [str(c.get("dominant_class", "")) for c in cat_summary]
            data = [{
                "type": "bar",
                "x": cats,
                "y": pcts,
                "marker": {"color": "#6366f1"},
                "text": [f"{d} ({p:.1f}%)" for d, p in zip(dom_classes, pcts)],
                "textposition": "auto",
                "hovertemplate": "<b>%{x}</b><br>Dominant Class: %{text}<extra></extra>"
            }]
            layout = _default_layout(f"Dominant Target Rate by '{feat_name}' Category", height=300)
            layout["yaxis"]["title"] = "Dominant Class Share (%)"
            layout["yaxis"]["range"] = [0, 105]

        return safe_primitive({"data": data, "layout": layout, "config": _default_config()})

    return None


# ==============================================================================
# MASTER GENERATOR
# ==============================================================================

def generate_all_visualizations(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coordinates and produces all evidence-driven Plotly chart specifications.
    Returns a dictionary of chart structures ready for direct client-side Plotly.newPlot() calls.
    """
    charts: Dict[str, Any] = {}

    # 1. Risk Meters
    charts["risk_gauge"] = build_risk_gauge_chart(result)
    risk_breakdown = build_risk_breakdown_chart(result)
    if risk_breakdown:
        charts["risk_breakdown"] = risk_breakdown

    # 2. Performance & Stability
    charts["performance_metrics"] = build_performance_metrics_chart(result)
    cv_chart = build_cv_stability_chart(result)
    if cv_chart:
        charts["cv_stability"] = cv_chart

    # 3. Feature Importance
    feat_chart = build_feature_importance_chart(result)
    if feat_chart:
        charts["feature_importance"] = feat_chart

    # 4. Data Quality & Target Distribution
    miss_chart = build_missing_values_chart(result)
    if miss_chart:
        charts["data_quality_missing"] = miss_chart

    target_chart = build_target_distribution_chart(result)
    if target_chart:
        charts["target_distribution"] = target_chart

    # 5. Error & Diagnostics
    if result.get("task_type") == "classification":
        cm_chart = build_confusion_matrix_chart(result)
        if cm_chart:
            charts["confusion_matrix"] = cm_chart
    else:
        avp_chart = build_actual_vs_predicted_chart(result)
        if avp_chart:
            charts["actual_vs_predicted"] = avp_chart

        res_chart = build_residual_plot(result)
        if res_chart:
            charts["residual_plot"] = res_chart

        res_dist = build_residual_distribution_chart(result)
        if res_dist:
            charts["residual_distribution"] = res_dist

    # 6. Relationships
    rel_chart = build_feature_relationships_chart(result)
    if rel_chart:
        charts["feature_relationships"] = rel_chart

    return safe_primitive(charts)
