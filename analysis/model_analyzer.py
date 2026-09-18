# analysis/model_analyzer.py
"""
AI Model Root-Cause Analyzer: General-purpose, evidence-driven diagnostic orchestrator.
Coordinates profiling, universal target analysis, preprocessing, multi-model evaluation,
prediction-level error analysis, feature relationship discovery, canonical evidence collection,
dynamic candidate grouping, continuous confidence estimation, and evidence-weighted risk assessment.
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score

from analysis.evidence import DiagnosticEvidence, EvidenceBus, safe_primitive
from analysis.profiler import DataProfiler
from analysis.target_analyzer import TargetAnalyzer
from analysis.preprocessor import PreprocessingEngine
from analysis.model_engine import ModelEngine
from analysis.error_analyzer import ErrorAnalyzer
from analysis.feature_analyzer import FeatureAnalyzer
from analysis.leakage_detector import LeakageDetector
from analysis.diagnostic_engine import (
    DiagnosticEngine,
    ConfidenceEngine,
    RiskEngine,
    RecommendationEngine,
)
from analysis.verification_engine import VerificationEngine



def _safe_val(val: Any) -> Any:
    """Backward-compatible helper for native Python type serialization."""
    return safe_primitive(val)


def _clean_dict(d: Any) -> Any:
    """Backward-compatible helper to recursively sanitize dictionary types."""
    return safe_primitive(d)


def _make_one_hot_encoder():
    """Backward-compatible encoder factory."""
    from analysis.preprocessor import _make_one_hot_encoder as m_ohe
    return m_ohe()


def analyze_model(
    df: pd.DataFrame,
    target_column: str,
    analysis_type: str = "auto",
    task_type: Optional[str] = None,
    mode: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main dynamic entrypoint to profile, preprocess, train, validate, and diagnose
    machine learning model behavior on any tabular dataset.
    """
    if task_type is not None:
        analysis_type = task_type
    elif mode is not None:
        analysis_type = mode

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("The provided dataset is empty or invalid.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

    total_rows = len(df)
    total_columns = len(df.columns)
    evidence_bus = EvidenceBus()

    # --------------------------------------------------------
    # 1. Target Validation & Target Analysis
    # --------------------------------------------------------
    data = df.copy()
    target_missing = int(data[target_column].isna().sum())
    data = data.dropna(subset=[target_column]).reset_index(drop=True)

    if len(data) < 2:
        raise ValueError("Dataset has fewer than 2 valid rows after removing missing target rows.")

    y_raw = data[target_column]
    unique_target_count = y_raw.nunique()

    if unique_target_count < 2:
        clean_u = [safe_primitive(v) for v in y_raw.unique()]
        raise ValueError(
            f"The target column '{target_column}' contains only {unique_target_count} unique value "
            f"({clean_u}). At least two distinct target values are required for machine learning."
        )

    resolved_task_type, task_reason, target_profile, target_evs = TargetAnalyzer.analyze_target(
        y_raw=y_raw,
        target_column=target_column,
        explicit_mode=analysis_type
    )
    for ev in target_evs:
        evidence_bus.emit(ev)

    # --------------------------------------------------------
    # 2. Target Leakage & Feature Preparation
    # --------------------------------------------------------
    leakage_findings, leakage_evs = LeakageDetector.detect_leakage(
        df=data,
        target_column=target_column,
        task_type=resolved_task_type
    )
    for ev in leakage_evs:
        evidence_bus.emit(ev)

    X_df, feature_names, numeric_columns, categorical_columns, excluded_features = (
        PreprocessingEngine.prepare_features(data, target_column)
    )

    # --------------------------------------------------------
    # 3. Comprehensive Data Quality Profile
    # --------------------------------------------------------
    data_quality = DataProfiler.profile_dataset(data, target_column)
    missing_cells_cnt = int(data_quality.get("total_missing_cells", 0))
    dup_rows_cnt = int(data_quality.get("duplicate_rows", 0))
    if missing_cells_cnt > 0:
        evidence_bus.emit(DiagnosticEvidence(
            evidence_id="missing_values_detected",
            signal_name=f"Missing Predictor Values ({missing_cells_cnt} cells)",
            domain="data_integrity",
            metric="total_missing_cells",
            observed_value=missing_cells_cnt,
            baseline_value=0,
            magnitude=min(1.0, missing_cells_cnt / max(1, len(data))),
            direction="missingness",
            sample_support=missing_cells_cnt,
            affected_population=f"{missing_cells_cnt} cells across feature columns",
            signal_type="structural",
            strength="HIGH" if missing_cells_cnt >= 20 else "MODERATE",
            reliability=1.0,
            model_scope="dataset",
            context=f"Dataset contains {missing_cells_cnt} missing cell(s)."
        ))
    if dup_rows_cnt > 0:
        evidence_bus.emit(DiagnosticEvidence(
            evidence_id="duplicate_rows_detected",
            signal_name=f"Duplicate Rows ({dup_rows_cnt} rows)",
            domain="data_integrity",
            metric="duplicate_row_count",
            observed_value=dup_rows_cnt,
            baseline_value=0,
            magnitude=min(1.0, dup_rows_cnt / max(1, len(data))),
            direction="redundancy",
            sample_support=dup_rows_cnt,
            affected_population=f"{dup_rows_cnt} duplicate observation rows",
            signal_type="structural",
            strength="HIGH" if dup_rows_cnt >= 10 else "MODERATE",
            reliability=1.0,
            model_scope="dataset",
            context=f"Dataset contains {dup_rows_cnt} duplicate row(s)."
        ))

    # --------------------------------------------------------
    # 4. Pipeline Construction & Feature Transformation
    # --------------------------------------------------------
    preprocessor = PreprocessingEngine.build_column_transformer(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns
    )

    X_transformed = preprocessor.fit_transform(X_df)

    # Extract transformed feature names
    transformed_feature_names = []
    try:
        transformed_feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        for col in numeric_columns:
            transformed_feature_names.append(f"num__{col}")
        for col in categorical_columns:
            try:
                cats = preprocessor.named_transformers_["cat"].named_steps["onehot"].categories_
                for c_idx, cat_col in enumerate(categorical_columns):
                    for cat_val in cats[c_idx]:
                        transformed_feature_names.append(f"cat__{cat_col}_{cat_val}")
            except Exception:
                transformed_feature_names.append(f"cat__{col}")

    if len(transformed_feature_names) != X_transformed.shape[1]:
        transformed_feature_names = [f"f_{i}" for i in range(X_transformed.shape[1])]

    # --------------------------------------------------------
    # 5. Target Encoding & Train/Test Split
    # --------------------------------------------------------
    if resolved_task_type == "classification":
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_raw.astype(str))
        classes = list(label_encoder.classes_)
    else:
        label_encoder = None
        y_encoded = np.asarray(pd.to_numeric(y_raw, errors="coerce").fillna(0.0), dtype=float)
        classes = []

    n_samples = len(data)
    stratify_labels = None

    if resolved_task_type == "classification":
        u_cls, counts = np.unique(y_encoded, return_counts=True)
        min_c = int(np.min(counts)) if len(counts) > 0 else 0
        n_classes = len(u_cls)
        raw_test_n = max(1, int(round(n_samples * 0.20)))

        if min_c >= 2 and n_samples >= (2 * n_classes) and raw_test_n >= n_classes and (n_samples - raw_test_n) >= n_classes:
            stratify_labels = y_encoded
            test_size = raw_test_n
        else:
            stratify_labels = None
            test_size = max(1, min(n_samples - 1, int(round(n_samples * 0.20))))
    else:
        test_size = max(1, min(n_samples - 1, int(round(n_samples * 0.20))))

    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=42,
        stratify=stratify_labels
    )

    X_train = X_transformed[train_idx]
    y_train = y_encoded[train_idx]
    X_test = X_transformed[test_idx]
    y_test = y_encoded[test_idx]

    X_test_raw = X_df.iloc[test_idx].reset_index(drop=True)
    y_test_raw = y_raw.iloc[test_idx].reset_index(drop=True)

    # --------------------------------------------------------
    # 6. Multi-Model Training, CV & Model Selection
    # --------------------------------------------------------
    model_eval_res, model_evs = ModelEngine.evaluate_and_select_model(
        task_type=resolved_task_type,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_full=X_transformed,
        y_full=y_encoded
    )
    for ev in model_evs:
        evidence_bus.emit(ev)

    selected_model = model_eval_res["selected_model"]
    selected_model_name = model_eval_res["selected_model_name"]
    model_selection = model_eval_res["model_selection"]
    model_comparison = model_eval_res["model_comparison"]
    model_stability = model_eval_res["model_stability"]
    cross_validation = model_eval_res["cross_validation"]
    fitted_models = model_eval_res["fitted_models"]
    selected_meta = model_eval_res["selected_meta"]

    # --------------------------------------------------------
    # 7. Model Predictions & Evaluation Metrics
    # --------------------------------------------------------
    y_pred = selected_model.predict(X_test)
    y_proba = None
    if resolved_task_type == "classification" and hasattr(selected_model, "predict_proba"):
        try:
            y_proba = selected_model.predict_proba(X_test)
        except Exception:
            y_proba = None

    if resolved_task_type == "classification":
        y_test_orig = [classes[idx] for idx in y_test]
        y_pred_orig = [classes[idx] for idx in y_pred]
        model_performance = {
            "accuracy": selected_meta.get("test_accuracy", selected_meta.get("accuracy", 0.0)),
            "balanced_accuracy": selected_meta.get("test_balanced_accuracy", selected_meta.get("balanced_accuracy", round(float(balanced_accuracy_score(y_test, y_pred)), 4))),
            "precision": selected_meta.get("test_precision", selected_meta.get("precision", 0.0)),
            "recall": selected_meta.get("test_recall", selected_meta.get("recall", 0.0)),
            "f1_score": selected_meta.get("test_f1_score", selected_meta.get("f1_score", 0.0)),
            "selected_model": selected_model_name,
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "feature_count": len(feature_names),
        }
        regression_metrics = {}
    else:
        y_test_orig = y_test
        y_pred_orig = y_pred
        model_performance = {
            "mae": selected_meta["test_mae"],
            "mse": selected_meta["test_mse"],
            "rmse": selected_meta["test_rmse"],
            "r2": selected_meta["test_r2"],
            "selected_model": selected_model_name,
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "feature_count": len(feature_names),
        }
        regression_metrics = {
            "mae": selected_meta["test_mae"],
            "mse": selected_meta["test_mse"],
            "rmse": selected_meta["test_rmse"],
            "r2": selected_meta["test_r2"],
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "feature_count": len(feature_names),
        }

    # --------------------------------------------------------
    # 8. Feature Importance, Direction & Cross-Model Stability
    # --------------------------------------------------------
    feature_impact, feature_relationships, feat_evs = FeatureAnalyzer.analyze_feature_impact(
        model=selected_model,
        transformed_feature_names=transformed_feature_names,
        original_feature_names=feature_names,
        X_df=X_df,
        y_raw=y_raw,
        task_type=resolved_task_type,
        target_profile=target_profile
    )
    for ev in feat_evs:
        evidence_bus.emit(ev)

    feature_importance = {k: v["importance"] for k, v in feature_impact.items()}
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_feature_names = [f for f, _ in sorted_features]

    cross_model_consistency, cross_model_evs = FeatureAnalyzer.evaluate_cross_model_consistency(
        fitted_models=fitted_models,
        transformed_feature_names=transformed_feature_names,
        original_feature_names=feature_names,
        model_predictions=model_eval_res.get("model_predictions", {})
    )
    for ev in cross_model_evs:
        evidence_bus.emit(ev)

    cross_model_status = cross_model_consistency["status"]
    cross_model_agree = (cross_model_consistency.get("consistency_score", 0.5) >= 0.70)

    # --------------------------------------------------------
    # 9. Prediction-Level Error Analysis & Subgroup Discovery
    # --------------------------------------------------------
    if resolved_task_type == "classification":
        err_results, err_evs = ErrorAnalyzer.analyze_classification_errors(
            y_test=np.asarray(y_test_orig),
            y_pred=np.asarray(y_pred_orig),
            y_proba=y_proba,
            X_test_raw=X_test_raw,
            classes=classes,
            target_profile=target_profile,
            top_feature_names=top_feature_names
        )
        total_test_errors = err_results["total_test_errors"]
        class_metrics = err_results["class_metrics"]
        confusion_matrix_dict = err_results["confusion_matrix"]
        error_by_class = err_results["error_by_class"]
        prediction_errors = err_results["prediction_errors"]
        error_patterns = err_results["error_patterns"]
        regression_errors = {}
    else:
        err_results, err_evs = ErrorAnalyzer.analyze_regression_errors(
            y_test=y_test,
            y_pred=y_pred,
            X_test_raw=X_test_raw,
            top_feature_names=top_feature_names
        )
        total_test_errors = len(err_results["prediction_errors"])
        class_metrics = {}
        confusion_matrix_dict = {}
        error_by_class = {}
        prediction_errors = err_results["prediction_errors"]
        error_patterns = []
        regression_errors = err_results

    for ev in err_evs:
        evidence_bus.emit(ev)

    segment_analysis = ErrorAnalyzer.discover_subgroup_patterns(
        task_type=resolved_task_type,
        X_df=X_test_raw,
        y_true=np.asarray(y_test_orig),
        y_pred=np.asarray(y_pred_orig),
        feature_names=feature_names
    )
    if segment_analysis:
        for idx, seg in enumerate(segment_analysis):
            ev_id = f"subgroup_error_segment_{idx+1}"
            seg["evidence_id"] = ev_id
            evidence_bus.emit(DiagnosticEvidence(
                evidence_id=ev_id,
                signal_name=f"Elevated Error Rate in Subgroup ({seg.get('segment_area', '')})",
                domain="error_disparity",
                metric="subgroup_error_rate",
                observed_value=float(seg.get("sample_count", 1)),
                baseline_value=0.0,
                magnitude=0.70,
                direction="error_concentration",
                sample_support=int(seg.get("sample_count", 1)),
                affected_population=str(seg.get("segment_area", "Subgroup feature area")),
                affected_features=[str(seg.get("feature", ""))] if seg.get("feature") else [],
                signal_type="behavioral",
                strength="MODERATE",
                reliability=0.85,
                model_scope="selected_model",
                context=str(seg.get("evidence", ""))
            ))

    # --------------------------------------------------------
    # 10. Candidate Hypotheses Synthesis from Evidence Bus
    # --------------------------------------------------------
    all_evidence = evidence_bus.all()
    cv_std_val = cross_validation.get("standard_deviation", 0.0)

    root_causes_structured = DiagnosticEngine.build_diagnostic_candidates(
        evidence_list=all_evidence,
        task_type=resolved_task_type,
        target_profile=target_profile,
        selected_model_name=selected_model_name,
        total_rows=len(data),
        test_rows=len(X_test),
        cv_std=cv_std_val,
        cross_model_consistent=cross_model_agree
    )

    # --------------------------------------------------------
    # 11. Automated Root-Cause Verification & Closed-Loop Remediation
    # --------------------------------------------------------
    verification_payload = {}
    eval_metric_label = "Weighted F1" if resolved_task_type == "classification" else "R2 Score"
    try:
        verification_payload = VerificationEngine.verify_and_simulate(
            X_df=X_df,
            y_raw=y_raw,
            task_type=resolved_task_type,
            champion_model=selected_model,
            feature_impact=feature_impact,
            diagnostic_candidates=root_causes_structured,
            max_candidates=3,
            target_column=target_column,
            selected_model_name=selected_model_name,
            evaluation_metric=eval_metric_label,
        )
    except Exception as e:
        verification_payload = {
            "status": "Skipped",
            "reason": str(e),
            "candidate_experiments": [],
            "experiments_count": 0,
            "remediation_simulation": {},
            "evidence_graph": {}
        }

    # --------------------------------------------------------
    # 12. Evidence Fusion: Experimental Evidence Outranks Heuristics
    # --------------------------------------------------------
    verif_exps = verification_payload.get("candidate_experiments", [])
    exp_by_feature = {e.get("candidate_feature"): e for e in verif_exps if e.get("candidate_feature")}

    for rc in root_causes_structured:
        # Check if candidate is tied to a feature that was experimentally tested
        tested_feat = None
        for f in rc.get("evidence_items", []):
            for af in f.get("affected_features", []):
                if af in exp_by_feature:
                    tested_feat = af
                    break
        if not tested_feat and rc.get("domain") == "feature_reliance":
            for cand_f in exp_by_feature:
                if cand_f in rc.get("finding", "") or cand_f in rc.get("title", ""):
                    tested_feat = cand_f
                    break

        if tested_feat and tested_feat in exp_by_feature:
            exp_res = exp_by_feature[tested_feat]
            verdict = exp_res.get("verdict", "")
            ev_score = exp_res.get("evidence_score", 0)
            rc["verification_score"] = f"Evidence Score: {ev_score}/100"
            rc["evidence_score"] = ev_score
            rc["verification_evidence"] = exp_res.get("summary", "")

            if verdict == "VERIFIED MODEL RELIANCE":
                rc["diagnostic_status"] = "VERIFIED MODEL RELIANCE"
                rc["verification_status"] = "Verified through controlled experiments"
                rc["finding"] = f"Verified Model Reliance on Predictor '{tested_feat}'"
                rc["interpretation"] = (
                    f"Controlled experiments confirmed the selected model exhibits strong empirical reliance on '{tested_feat}' "
                    f"under the tested dataset, split, and interventions (ablation delta: {exp_res.get('ablation_delta', 0):+.4f}, "
                    f"permutation delta: {exp_res.get('permutation_delta', 0):+.4f})."
                )
            elif verdict == "NO MEASURABLE MODEL RELIANCE":
                rc["diagnostic_status"] = "NO MEASURABLE MODEL RELIANCE"
                rc["verification_status"] = "No measurable model reliance under tested interventions"
                rc["severity"] = "LOW"
                rc["finding"] = f"No Measurable Model Reliance on Predictor '{tested_feat}'"
                rc["interpretation"] = (
                    f"Under the tested split and interventions, removing or permuting '{tested_feat}' produced no measurable "
                    f"change in the selected evaluation metric."
                )
            elif verdict:
                rc["diagnostic_status"] = verdict
                rc["verification_status"] = f"Tested ({verdict})"
        else:
            if rc.get("domain") in {"leakage", "sample_size"}:
                rc["diagnostic_status"] = "SIGNAL DETECTED"
            else:
                rc["diagnostic_status"] = "CANDIDATE HYPOTHESIS"
            rc["verification_status"] = "NOT TESTED"
            rc["verification_score"] = "N/A"
            rc["verification_evidence"] = "Controlled counterfactual experiments were not conducted for this candidate."

    # Sort structured diagnostic findings by severity, evidence strength, and confidence
    sev_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    strength_order = {"HIGH": 3, "MODERATE": 2, "LOW": 1}
    root_causes_structured.sort(
        key=lambda x: (
            sev_order.get(str(x.get("severity", "MEDIUM")).upper(), 2),
            strength_order.get(str(x.get("evidence_strength", "MODERATE")).upper(), 2),
            float(x.get("confidence_score", 0.50))
        ),
        reverse=True
    )

    root_causes_strings = []
    for rc in root_causes_structured:
        if rc["category"] == "Diagnostic Observation" and "Healthy" in rc["finding"]:
            root_causes_strings.append("No obvious data-related root causes detected")
        elif "Target Leakage" in rc["finding"]:
            root_causes_strings.append(f"{rc['root_cause']}")
        else:
            root_causes_strings.append(f"{rc['finding']}")

    priority_findings = []
    for rc in root_causes_structured:
        priority_findings.append({
            "finding": rc["finding"],
            "issue": rc["finding"],
            "title": rc["title"],
            "signal": rc.get("signal", rc["title"]),
            "initial_evidence": rc.get("initial_evidence", rc["evidence"]),
            "hypothesis": rc.get("hypothesis", rc.get("potential_explanation", "")),
            "diagnostic_status": rc.get("diagnostic_status", "SIGNAL DETECTED"),
            "verification_status": rc.get("verification_status", "NOT TESTED"),
            "verification_score": rc.get("verification_score", "N/A"),
            "category": rc["category"],
            "domain": rc.get("domain", ""),
            "evidence": rc["evidence"],
            "reason": rc["evidence"],
            "priority": rc["severity"],
            "severity": rc["severity"],
            "confidence": rc["confidence"],
            "confidence_score": rc["confidence_score"],
            "evidence_strength": rc["evidence_strength"],
            "impact": rc["impact"],
            "evidence_ids": rc.get("evidence_ids", []),
        })

    # --------------------------------------------------------
    # 13. Continuous Evidence-Weighted Risk Calculation
    # --------------------------------------------------------
    class_info = {
        "class_imbalance": target_profile.get("class_imbalance", False),
        "minority_class": target_profile.get("minority_class"),
        "majority_class": target_profile.get("majority_class"),
        "minority_recall": class_metrics.get(str(target_profile.get("minority_class")), {}).get("recall", 1.0) if class_metrics else 1.0,
        "majority_recall": class_metrics.get(str(target_profile.get("majority_class")), {}).get("recall", 1.0) if class_metrics else 1.0,
        "imbalance_ratio": target_profile.get("minority_to_majority_ratio", 1.0),
    }

    data_info = {
        "total_rows": len(data),
        "total_missing_cells": data_quality.get("total_missing_cells", 0),
        "duplicate_rows": data_quality.get("duplicate_rows", 0),
        "outlier_columns": [c for c, cnt in data_quality.get("outliers", {}).items() if cnt > 0],
    }

    risk_dict = RiskEngine.compute_risk(
        task_type=resolved_task_type,
        performance=model_performance,
        stability=model_stability,
        class_info=class_info,
        data_info=data_info,
        leakage_findings=leakage_findings,
        error_segments=segment_analysis,
        diagnostic_candidates=root_causes_structured
    )

    risk_score = risk_dict["score"]
    overall_risk = risk_dict["level"]

    # --------------------------------------------------------
    # 14. Model-Aware Actionable Recommendations
    # --------------------------------------------------------
    recommendations, recommendations_structured = RecommendationEngine.generate_recommendations(
        task_type=resolved_task_type,
        candidates=root_causes_structured,
        data_quality=data_quality,
        selected_model_name=selected_model_name
    )

    # --------------------------------------------------------
    # 15. Operational Evidence-Driven Warnings Construction
    # --------------------------------------------------------
    warnings_list = []
    for rc in root_causes_structured:
        sev = rc.get("severity", "MEDIUM")
        if sev in ("HIGH", "CRITICAL"):
            warnings_list.append(f"[{sev}] {rc['finding']}: {rc['evidence']}")
        elif sev == "MEDIUM" and rc.get("category") != "Diagnostic Observation":
            warnings_list.append(f"[MEDIUM] {rc['finding']}: {rc['evidence']}")

    for w in data_quality.get("warnings", []):
        if "No major data-quality problems" not in w and w not in warnings_list:
            warnings_list.append(f"[Data Quality] {w}")

    if target_profile.get("binary_target_compatibility_note"):
        warnings_list.append(f"[Target Semantic Caveat] {target_profile['binary_target_compatibility_note']}")

    if resolved_task_type == "regression":
        top_rc = root_causes_structured[0] if root_causes_structured else None
        res_summary = err_results.get("main_error", "")
        if top_rc and top_rc.get("severity") in ("HIGH", "CRITICAL"):
            main_diagnostic_summary = f"Significant model issue detected: {top_rc.get('finding', '')}. {top_rc.get('evidence', '')}"
        elif top_rc and top_rc.get("severity") == "MEDIUM" and top_rc.get("category") != "Diagnostic Observation":
            main_diagnostic_summary = f"{top_rc.get('finding', '')}: {top_rc.get('evidence', '')}"
        elif res_summary and "No dominant systematic" not in res_summary:
            main_diagnostic_summary = res_summary
        else:
            main_diagnostic_summary = "No major regression issue detected across evaluated partitions."
    else:
        main_diagnostic_summary = err_results.get("main_error", "No major error pattern detected.")

    # --------------------------------------------------------
    # 14. Complete Structured Analysis Payload
    # --------------------------------------------------------
    dataset_info = {
        "rows": total_rows,
        "total_rows": total_rows,
        "columns": total_columns,
        "total_columns": total_columns,
        "usable_rows": len(data),
        "usable_features": len(feature_names),
        "usable_predictors": len(feature_names),
        "missing_cells": int(data_quality.get("total_missing_cells", 0)),
        "missing_feature_values": int(data_quality.get("total_missing_cells", 0)),
        "duplicate_rows": int(data_quality.get("duplicate_rows", 0)),
    }

    evidence_records_payload = [e.to_dict() for e in all_evidence]

    result_payload = {
        "status": "Success",
        "task_type": resolved_task_type,
        "analysis_type": resolved_task_type,
        "task_reason": task_reason,
        "auto_detection_reason": task_reason,
        "target_column": target_column,
        "target_profile": target_profile,
        "target_analysis": target_profile,
        "binary_target_compatibility_note": target_profile.get("binary_target_compatibility_note"),
        "class_imbalance": bool(target_profile.get("class_imbalance", False)),
        "class_count": len(classes) if resolved_task_type == "classification" else None,
        "class_names": [str(c) for c in classes] if resolved_task_type == "classification" else [],
        "class_distribution": target_profile.get("class_distribution", {}) if resolved_task_type == "classification" else {},
        "class_distribution_list": target_profile.get("class_distribution_list", []) if resolved_task_type == "classification" else [],
        "class_metrics": class_metrics if resolved_task_type == "classification" else {},
        "dataset_info": dataset_info,
        "model_performance": model_performance,
        "regression_metrics": regression_metrics,
        "selected_model": selected_model_name,
        "model_selection": model_selection,
        "model_comparison": model_comparison,
        "cross_validation": cross_validation,
        "model_stability": model_stability,
        "feature_importance": feature_importance,
        "feature_impact": feature_impact,
        "feature_relationships": feature_relationships,
        "cross_model_status": cross_model_status,
        "cross_model_consistency": cross_model_consistency,
        "evidence_records": evidence_records_payload,
        "evidence_count": len(evidence_records_payload),
        "error_analysis": err_results,
        "main_error": main_diagnostic_summary,
        "total_test_errors": total_test_errors,
        "high_confidence_errors_count": err_results.get("high_confidence_errors_count", len(err_results.get("high_confidence_errors", []))),
        "prediction_errors": prediction_errors,
        "error_patterns": error_patterns,
        "confusion_matrix": confusion_matrix_dict,
        "segment_analysis": segment_analysis,
        "excluded_features": excluded_features,
        "leakage_findings": leakage_findings,
        "root_causes_structured": root_causes_structured,
        "root_causes": root_causes_strings,
        "priority_findings": priority_findings,
        "priority_issues": priority_findings,
        "verification_engine": verification_payload,
        "verification_experiments": verification_payload.get("candidate_experiments", []),
        "remediation_simulation": verification_payload.get("remediation_simulation", {}),
        "evidence_graph": verification_payload.get("evidence_graph", {}),
        "risk": risk_dict,
        "risk_assessment": risk_dict,
        "risk_score": risk_score,
        "overall_risk": overall_risk,
        "recommendations": recommendations,
        "recommendations_structured": recommendations_structured,
        "warnings": warnings_list,
        "usable_rows": len(data),
        "usable_features": len(feature_names),
        "total_rows": total_rows,
        "total_columns": total_columns,
        "data_quality": data_quality,
        "dataset_profile": data_quality,
        "limitations": [
            "Observations reflect empirical patterns on provided dataset partitions.",
            "Feature importance measures predictive usefulness in the model for this dataset. It does not prove causal necessity, positive/negative relationship direction, or business importance.",
            "Controlled ablation and permutation experiments measure empirical model reliance under the tested interventions. They do not establish real-world causal relationships.",
            "Confidence intervals widen on smaller sample counts."
        ],
    }

    return safe_primitive(result_payload)