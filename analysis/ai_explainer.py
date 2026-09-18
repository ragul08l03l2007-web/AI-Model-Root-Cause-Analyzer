# analysis/ai_explainer.py
"""
AI Explainer, Interactive Copilot & Automated Remediation Engine.

Extracts comprehensive, screen-aligned statistical evidence, feature-level data
catalogs, mathematical calculations, and evidence graph lineages from diagnostic runs.
Leverages pluggable AI providers (Gemini, OpenAI, Ollama, Offline) to generate:
1. Executive Root-Cause Summaries & Deep-Dives (Stage 1)
2. Targeted Python Remediation & Fix Pipelines (Stage 2)
3. Universal Feature-Aware Diagnostic Copilot Conversations (Stage 3)
"""

from typing import Dict, List, Any, Optional
import json
from analysis.evidence import safe_primitive
from analysis.ai_providers import BaseAIProvider, get_ai_provider, OfflineDeterministicProvider


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        f = float(val)
        return round(f, 4)
    except (ValueError, TypeError):
        return default


def extract_evidence_payload(
    model_result: Dict[str, Any],
    data_quality: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extracts and sanitizes a comprehensive, high-signal statistical evidence dictionary
    from the full model_analyzer output. Indexes every feature, on-screen calculation,
    and experimental trial for interactive Q&A.
    """
    dq = data_quality or model_result.get("data_quality", {})
    task_type = str(model_result.get("task_type", "classification")).lower()

    # 1. Dataset Profile
    num_feats_list = dq.get("numeric_features", [])
    cat_feats_list = dq.get("categorical_features", [])
    tot_feats = len(num_feats_list) + len(cat_feats_list)
    if tot_feats == 0:
        tot_feats = len(model_result.get("feature_impact", {}))

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
        "potential_outliers": tot_outliers,
        "numeric_features": num_feats_list,
        "categorical_features": cat_feats_list,
    }

    # 2. Target Profile
    tp_raw = model_result.get("target_profile", {})
    target_profile = {
        "target_column": str(tp_raw.get("target_column", model_result.get("target_column", "target"))),
        "target_type": str(tp_raw.get("target_type", task_type))
    }
    if task_type == "classification":
        target_profile["class_distribution"] = tp_raw.get("class_distribution", {})
        if "minority_to_majority_ratio" in tp_raw:
            target_profile["imbalance_ratio"] = float(tp_raw.get("minority_to_majority_ratio", 1.0))
        target_profile["class_count"] = len(tp_raw.get("class_distribution", {}))
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
        for k in ("r2", "rmse", "mae", "mse", "explained_variance"):
            val = reg_m.get(k, perf.get(k))
            if val is not None:
                metrics_cleaned[k] = round(float(val), 4)
    else:
        for k in ("accuracy", "balanced_accuracy", "f1_score", "precision", "recall"):
            val = perf.get(k)
            if val is not None:
                metrics_cleaned[k] = round(float(val), 4)

    selected_model_info = {
        "name": selected_name,
        "metrics": metrics_cleaned,
        "cross_validation": {
            "cv_mean": round(float(cv.get("average_score", 0.0)), 4),
            "cv_std": round(float(cv.get("standard_deviation", 0.0)), 4),
            "cv_folds": cv.get("folds_count", 5),
            "stability_status": str(cv.get("stability_status", "STABLE")),
        }
    }

    # 4. Model Stability & Overfitting
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

    # 6. Experimental Verification & Closed-Loop Remediation
    verif_exp = model_result.get("verification_engine", {}).get("candidate_experiments", [])
    if not verif_exp:
        verif_exp = model_result.get("verification_experiments", [])
    remed_sim = model_result.get("verification_engine", {}).get("remediation_simulation", {})
    if not remed_sim:
        remed_sim = model_result.get("remediation_simulation", {})

    evidence_graph = model_result.get("evidence_graph", model_result.get("verification_engine", {}).get("evidence_graph", {}))

    verification_summary = []
    exp_by_feature = {}
    for exp in verif_exp:
        cand_feat = exp.get("candidate_feature") or exp.get("candidate")
        exp_dict = {
            "candidate": cand_feat,
            "candidate_feature": cand_feat,
            "metric": exp.get("metric_name"),
            "baseline": exp.get("baseline_metric"),
            "ablated_metric": exp.get("ablated_metric"),
            "ablation_delta": exp.get("ablation_delta"),
            "ablation_delta_pct": exp.get("ablation_delta_pct"),
            "permuted_metric": exp.get("permuted_metric"),
            "permutation_delta": exp.get("permutation_delta"),
            "permutation_delta_pct": exp.get("permutation_delta_pct"),
            "perturbed_metric": exp.get("perturbed_metric"),
            "noise_delta": exp.get("noise_delta"),
            "prediction_flip_rate_pct": exp.get("prediction_flip_rate_pct"),
            "noise_sensitivity": exp.get("noise_sensitivity"),
            "control_feature": exp.get("control_feature"),
            "control_delta": exp.get("control_delta"),
            "control_specificity_ratio": exp.get("control_specificity_ratio"),
            "evidence_ratings": exp.get("evidence_ratings", {}),
            "score_decomposition": exp.get("score_decomposition", {}),
            "verdict": exp.get("verdict"),
            "evidence_score": exp.get("evidence_score"),
            "summary": exp.get("summary", "")
        }
        verification_summary.append(exp_dict)
        if cand_feat:
            exp_by_feature[cand_feat] = exp_dict

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

    # 7. Complete Feature Catalogue (Every feature in dataset with statistics & experiments)
    feat_raw = model_result.get("feature_impact", {})
    raw_rel = model_result.get("feature_relationships", [])
    feat_relationships = {}
    if isinstance(raw_rel, list):
        for r in raw_rel:
            if isinstance(r, dict) and "feature" in r:
                feat_relationships[r["feature"]] = r
    elif isinstance(raw_rel, dict):
        feat_relationships = raw_rel
    subgroup_errors = model_result.get("segment_analysis", [])

    feature_catalogue = {}
    # Combine features from feature_impact, data_quality, and dataset columns
    all_known_features = list(feat_raw.keys())
    for f in num_feats_list + cat_feats_list:
        if f not in all_known_features:
            all_known_features.append(f)

    for feat_name in all_known_features:
        feat_val = feat_raw.get(feat_name, {})
        if isinstance(feat_val, dict):
            imp_val = _safe_float(feat_val.get("importance", 0.0))
            share_pct = _safe_float(feat_val.get("relative_share_pct", 0.0))
            tier = str(feat_val.get("influence_tier", "Moderate"))
            dir_str = str(feat_val.get("direction", ""))
        else:
            imp_val = _safe_float(feat_val)
            share_pct = 0.0
            tier = "Moderate"
            dir_str = ""

        # Data quality per feature
        feat_missing = missing_dict.get(feat_name, 0) if isinstance(missing_dict, dict) else 0
        feat_outliers = outliers_dict.get(feat_name, 0) if isinstance(outliers_dict, dict) else 0
        feat_type = "numeric" if feat_name in num_feats_list else ("categorical" if feat_name in cat_feats_list else "unknown")

        # Inferred empirical relationship
        rel_info = feat_relationships.get(feat_name, {})

        # Subgroup involvement
        involved_segments = [s for s in subgroup_errors if s.get("feature") == feat_name or feat_name in str(s.get("segment_area", ""))]

        # Verification experiment info
        exp_info = exp_by_feature.get(feat_name)

        feature_catalogue[feat_name] = {
            "name": feat_name,
            "data_type": feat_type,
            "importance": imp_val,
            "relative_share_pct": share_pct,
            "influence_tier": tier,
            "observed_direction": dir_str or rel_info.get("observed_direction", "N/A"),
            "missing_count": int(feat_missing),
            "outliers_count": int(feat_outliers),
            "is_experimentally_tested": exp_info is not None,
            "verification": exp_info or {
                "verdict": "UNTESTED",
                "evidence_score": 0,
                "note": "Controlled counterfactual experiments were not conducted for this feature."
            },
            "subgroup_error_segments_count": len(involved_segments)
        }

    # 8. Top Feature Importance Summary (for compact prompts)
    feat_cleaned = {}
    for feat_name, data in list(feature_catalogue.items())[:6]:
        feat_cleaned[feat_name] = {
            "importance": data["importance"],
            "share_pct": data["relative_share_pct"],
            "direction": data["observed_direction"]
        }

    # 9. Overall Risk & On-Screen Calculations Glossary
    risk_info = model_result.get("risk", {})
    risk_level = str(risk_info.get("level", model_result.get("overall_risk", "LOW")))
    risk_score = int(risk_info.get("score", model_result.get("risk_score", 0)))
    risk_drivers = risk_info.get("drivers", [])

    calculation_glossary = {
        "balanced_accuracy": {
            "metric_name": "Balanced Accuracy",
            "formula": "mean(class_recalls) = sum(recall_per_class) / n_classes",
            "value": metrics_cleaned.get("balanced_accuracy"),
            "role": "Measures overall class-balanced performance, unaffected by majority class prevalence."
        },
        "accuracy": {
            "metric_name": "Accuracy",
            "formula": "correct_predictions / total_predictions",
            "value": metrics_cleaned.get("accuracy"),
        },
        "weighted_f1": {
            "metric_name": "Weighted F1 Score",
            "formula": "sum(f1_score_per_class * class_support) / total_support",
            "value": metrics_cleaned.get("f1_score"),
        },
        "cross_validation": {
            "cv_mean": selected_model_info["cross_validation"]["cv_mean"],
            "cv_std": selected_model_info["cross_validation"]["cv_std"],
            "stability_status": selected_model_info["cross_validation"]["stability_status"],
            "formula": "5-Fold Cross Validation generalization estimate across distinct dataset partitions."
        },
        "risk_score": {
            "score": risk_score,
            "level": risk_level,
            "formula": "Continuous weighted sum across Data Quality, Model Performance, Overfitting Gap, Class Imbalance, and Single-Feature Reliance.",
            "drivers": risk_drivers
        },
        "evidence_score_formula": {
            "formula": "Ablation (35 pts max) + Permutation (35 pts max) + Control Specificity (15 pts max) + Measurement Stability (10 pts max) + Consistency (5 pts max) = 100 pts max",
            "thresholds": ">= 70 -> VERIFIED MODEL RELIANCE; < 30 -> NO MEASURABLE MODEL RELIANCE; 30-69 -> INCONCLUSIVE"
        },
        "remediation_simulation": {
            "resolution_verdict": remed_sim.get("resolution_verdict", "None"),
            "test_metric_delta": remed_sim.get("deltas", {}).get("test_metric_delta", 0.0),
            "generalization_gap_reduction": remed_sim.get("deltas", {}).get("generalization_gap_reduction", 0.0),
            "baseline": remed_sim.get("baseline", {}),
            "remediated": remed_sim.get("remediated", {}),
        } if remed_sim else {}
    }

    payload = {
        "task_type": task_type,
        "dataset_summary": dataset_summary,
        "target_profile": target_profile,
        "selected_model": selected_model_info,
        "model_stability": model_stability,
        "root_causes": root_causes_cleaned,
        "feature_importance": feat_cleaned,
        "feature_catalogue": feature_catalogue,
        "calculation_glossary": calculation_glossary,
        "overall_risk": {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "drivers": risk_drivers
        },
        "completed_experiments": completed_experiments,
        "completed_verification_experiments": completed_experiments,
        "verification_experiments": verification_summary,
        "remediation_simulation": calculation_glossary["remediation_simulation"],
        "evidence_graph_metadata": evidence_graph.get("metadata", {})
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
        Stage 3: Interactive Copilot Q&A grounded in the diagnostic evidence payload,
        feature catalogue, and calculation glossary.
        """
        active_prov = provider or self.provider
        payload = extract_evidence_payload(model_result, data_quality)
        return active_prov.chat(payload, question, chat_history=chat_history)
