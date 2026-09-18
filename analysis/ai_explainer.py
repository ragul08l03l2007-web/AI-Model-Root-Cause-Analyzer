# analysis/ai_explainer.py
"""
AI Explainer & Remediation Engine.
Extracts sanitized, compact statistical evidence from deterministic diagnostic runs
and leverages pluggable AI providers (Gemini, OpenAI, Ollama, Offline) to generate:
1. Executive Root-Cause Summaries & Deep-Dives (Stage 1)
2. Targeted Python Remediation & Fix Pipelines (Stage 2)
3. Evidence-Grounded Diagnostic Copilot Conversations (Stage 3)
"""

from typing import Dict, List, Any, Optional
import json
from analysis.evidence import safe_primitive
from analysis.ai_providers import BaseAIProvider, get_ai_provider, OfflineDeterministicProvider


def extract_evidence_payload(
    model_result: Dict[str, Any],
    data_quality: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extracts and sanitizes a compact, high-signal statistical evidence dictionary
    from the full model_analyzer output.
    Ensures complete JSON-serializability without heavy matrix dumps.
    """
    dq = data_quality or model_result.get("data_quality", {})
    task_type = str(model_result.get("task_type", "classification")).lower()

    # 1. Dataset Profile
    num_feats = len(dq.get("numeric_features", []))
    cat_feats = len(dq.get("categorical_features", []))
    tot_feats = num_feats + cat_feats if (num_feats + cat_feats) > 0 else len(model_result.get("feature_impact", {}))

    missing_dict = dq.get("missing_values", {})
    tot_missing = sum(missing_dict.values()) if isinstance(missing_dict, dict) else int(missing_dict or 0)

    outliers_dict = dq.get("outliers", {})
    tot_outliers = sum(outliers_dict.values()) if isinstance(outliers_dict, dict) else int(outliers_dict or 0)

    dataset_summary = {
        "total_rows": int(dq.get("total_rows", model_result.get("train_rows", 0) + model_result.get("test_rows", 0))),
        "total_features": tot_feats,
        "quality_score": int(dq.get("quality_score", 100)),
        "quality_status": str(dq.get("overall_quality", "GOOD")),
        "missing_values": tot_missing,
        "duplicate_rows": int(dq.get("duplicate_rows", 0)),
        "potential_outliers": tot_outliers
    }

    # 2. Target Profile
    tp_raw = model_result.get("target_profile", {})
    target_profile = {
        "target_column": str(tp_raw.get("target_column", "target")),
        "target_type": str(tp_raw.get("target_type", task_type))
    }
    if task_type == "classification":
        target_profile["class_distribution"] = tp_raw.get("class_distribution", {})
        if "minority_to_majority_ratio" in tp_raw:
            target_profile["imbalance_ratio"] = float(tp_raw.get("minority_to_majority_ratio", 1.0))
    else:
        for k in ("mean", "median", "std", "min", "max"):
            if k in tp_raw:
                target_profile[k] = round(float(tp_raw[k]), 4)

    # 3. Model Performance & Selection
    perf = model_result.get("model_performance", {})
    cv = model_result.get("cross_validation", {})
    model_sel = model_result.get("model_selection", {})
    selected_name = str(model_sel.get("selected_model", model_result.get("selected_model", "Selected Model")))

    metrics_cleaned = {}
    if task_type == "regression":
        reg_m = model_result.get("regression_metrics", {})
        for k in ("r2", "rmse", "mae", "mse"):
            val = reg_m.get(k, perf.get(k))
            if val is not None:
                metrics_cleaned[k] = round(float(val), 4)
    else:
        for k in ("accuracy", "f1_score", "precision", "recall"):
            val = perf.get(k)
            if val is not None:
                metrics_cleaned[k] = round(float(val), 4)

    selected_model_info = {
        "name": selected_name,
        "metrics": metrics_cleaned,
        "cross_validation": {
            "cv_mean": round(float(cv.get("average_score", 0.0)), 4),
            "cv_std": round(float(cv.get("standard_deviation", 0.0)), 4)
        }
    }

    # 4. Model Stability
    stab_raw = model_result.get("model_stability", {})
    model_stability = {
        "stability_status": str(stab_raw.get("stability_status", "STABLE")),
        "overfitting_diagnostic": str(stab_raw.get("overfitting_diagnostic", "None")),
        "explanation": str(stab_raw.get("overfitting_explanation", ""))
    }

    # 5. Structured Root Causes
    root_causes_cleaned = []
    for rc in model_result.get("root_causes_structured", []):
        root_causes_cleaned.append({
            "category": str(rc.get("category", "General")),
            "finding": str(rc.get("finding", rc.get("root_cause", ""))),
            "severity": str(rc.get("severity", "MEDIUM")),
            "confidence": str(rc.get("confidence", "Medium")),
            "evidence": str(rc.get("evidence", "")),
            "interpretation": str(rc.get("interpretation", "")),
            "potential_explanation": str(rc.get("potential_explanation", "")),
            "impact": str(rc.get("impact", "")),
            "recommended_action": str(rc.get("recommended_action", ""))
        })

    # If structured is empty, fall back to string list
    if not root_causes_cleaned and model_result.get("root_causes"):
        for rc_str in model_result.get("root_causes", []):
            root_causes_cleaned.append({
                "category": "Detected Issue",
                "finding": str(rc_str),
                "severity": "MEDIUM",
                "confidence": "Medium",
                "evidence": "Detected during diagnostic pipeline run.",
                "interpretation": str(rc_str),
                "potential_explanation": "",
                "impact": "May degrade generalization performance.",
                "recommended_action": "Inspect and sanitize feature inputs."
            })

    # 6. Feature Importance (Top 6)
    feat_raw = model_result.get("feature_impact", {})
    feat_cleaned = {}
    for feat_name, feat_val in list(feat_raw.items())[:6]:
        if isinstance(feat_val, dict):
            feat_cleaned[feat_name] = {
                "importance": round(float(feat_val.get("importance", 0.0)), 4),
                "share_pct": round(float(feat_val.get("relative_share_pct", 0.0)), 2),
                "direction": str(feat_val.get("direction", ""))
            }
        else:
            feat_cleaned[feat_name] = round(float(feat_val), 4)

    # 7. Overall Risk
    overall_risk = {
        "risk_level": str(model_result.get("overall_risk", "LOW")),
        "risk_score": int(model_result.get("risk_score", 0))
    }

    # 8. Verification & Closed-Loop Remediation

    verif_exp = model_result.get("verification_engine", {}).get("candidate_experiments", [])
    if not verif_exp:
        verif_exp = model_result.get("verification_experiments", [])
    remed_sim = model_result.get("verification_engine", {}).get("remediation_simulation", {})
    if not remed_sim:
        remed_sim = model_result.get("remediation_simulation", {})

    verification_summary = []
    for exp in verif_exp:
        verification_summary.append({
            "candidate": exp.get("candidate_feature"),
            "metric": exp.get("metric_name"),
            "baseline": exp.get("baseline_metric"),
            "ablation_delta": exp.get("ablation_delta"),
            "permutation_delta": exp.get("permutation_delta"),
            "perturbed_metric": exp.get("perturbed_metric"),
            "noise_delta": exp.get("noise_delta"),
            "prediction_flip_rate_pct": exp.get("prediction_flip_rate_pct"),
            "noise_sensitivity": exp.get("noise_sensitivity"),
            "control_feature": exp.get("control_feature"),
            "control_delta": exp.get("control_delta"),
            "evidence_ratings": exp.get("evidence_ratings", {}),
            "score_decomposition": exp.get("score_decomposition", {}),
            "verdict": exp.get("verdict"),
            "evidence_score": exp.get("evidence_score")
        })

    completed_experiments = []
    if verif_exp:
        completed_experiments.extend([
            "retrained_feature_ablation",
            "test_time_permutation",
            "measurement_stability",
            "control_feature_test"
        ])
    if remed_sim and remed_sim.get("status") == "Success":
        completed_experiments.append("closed_loop_remediation")

    payload = {
        "task_type": task_type,
        "dataset_summary": dataset_summary,
        "target_profile": target_profile,
        "selected_model": selected_model_info,
        "model_stability": model_stability,
        "root_causes": root_causes_cleaned,
        "feature_importance": feat_cleaned,
        "overall_risk": overall_risk,
        "completed_experiments": completed_experiments,
        "completed_verification_experiments": completed_experiments,
        "verification_experiments": verification_summary,
        "remediation_simulation": {
            "resolution_verdict": remed_sim.get("resolution_verdict", "None"),
            "test_metric_delta": remed_sim.get("deltas", {}).get("test_metric_delta", 0.0),
            "generalization_gap_reduction": remed_sim.get("deltas", {}).get("generalization_gap_reduction", 0.0)
        } if remed_sim else {}
    }

    return safe_primitive(payload)



class AIExplainer:
    """
    High-level orchestrator that bridges deterministic ML diagnostic evidence
    with pluggable AI providers (Gemini, OpenAI, Ollama, Offline).
    """

    def __init__(self, provider: Optional[BaseAIProvider] = None):
        self.provider = provider or get_ai_provider()

    def set_provider(self, provider: BaseAIProvider) -> None:
        """Dynamically swap the active AI provider."""
        self.provider = provider

    def get_active_provider_name(self) -> str:
        """Returns the name of the currently configured AI provider."""
        return self.provider.provider_name

    def explain(
        self,
        model_result: Dict[str, Any],
        data_quality: Optional[Dict[str, Any]] = None,
        provider: Optional[BaseAIProvider] = None
    ) -> Dict[str, Any]:
        """
        Stage 1: Generates a complete, structured executive root-cause report
        derived strictly from the deterministic evidence payload.
        """
        active_prov = provider or self.provider
        payload = extract_evidence_payload(model_result, data_quality)
        return active_prov.generate_explanation(payload)

    def generate_fix_script(
        self,
        model_result: Dict[str, Any],
        data_quality: Optional[Dict[str, Any]] = None,
        provider: Optional[BaseAIProvider] = None
    ) -> str:
        """
        Stage 2: Generates an actionable, runnable Python remediation pipeline
        tailored specifically to the detected issues.
        """
        active_prov = provider or self.provider
        payload = extract_evidence_payload(model_result, data_quality)
        return active_prov.generate_fix(payload)

    def ask_copilot(
        self,
        question: str,
        model_result: Dict[str, Any],
        data_quality: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        provider: Optional[BaseAIProvider] = None
    ) -> str:
        """
        Stage 3: Interactive Copilot Q&A grounded in the diagnostic evidence payload.
        """
        active_prov = provider or self.provider
        payload = extract_evidence_payload(model_result, data_quality)
        return active_prov.chat(payload, question, chat_history=chat_history)
