# analysis/error_analyzer.py
"""
ErrorAnalyzer: Universal prediction-level error inspection, confusion transitions,
residual statistics, heteroscedasticity, and dynamic error-to-feature subgroup discovery.
Generically handles binary, multiclass, and regression problems without hardcoded class labels.
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from analysis.evidence import DiagnosticEvidence, safe_primitive


class ErrorAnalyzer:
    """
    Performs empirical prediction-level error inspection, confusion transition discovery,
    residual diagnostics, and dynamic subgroup analysis.
    """

    @classmethod
    def analyze_classification_errors(
        cls,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        X_test_raw: pd.DataFrame,
        classes: List[Any],
        target_profile: Dict[str, Any],
        top_feature_names: List[str]
    ) -> Tuple[Dict[str, Any], List[DiagnosticEvidence]]:
        """
        Detailed classification error inspection with confusion concentration,
        prediction confidence, per-class metrics, and evidence emission.
        """
        classes_str = [str(c) for c in classes]
        n_classes = len(classes_str)
        n_samples = len(y_test)
        evidence_list: List[DiagnosticEvidence] = []

        # 1. Per-Class Performance Metrics
        precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
            y_test, y_pred, labels=classes, zero_division=0
        )
        c_pcts = target_profile.get("class_percentages", {})
        c_dist = target_profile.get("class_distribution", {})
        total_obs = sum(c_dist.values()) if c_dist else len(y_test)

        class_metrics = {}
        for idx, c_name in enumerate(classes_str):
            pct_val = float(c_pcts.get(c_name, (c_dist.get(c_name, 0) / max(1, total_obs) * 100.0) if c_dist else (support_arr[idx] / max(1, n_samples) * 100.0)))
            class_metrics[c_name] = {
                "precision": round(float(precision_arr[idx]), 4),
                "recall": round(float(recall_arr[idx]), 4),
                "f1_score": round(float(f1_arr[idx]), 4),
                "f1": round(float(f1_arr[idx]), 4),
                "support": int(support_arr[idx]),
                "percentage_of_dataset": round(pct_val, 2),
                "dataset_proportion": round(pct_val, 2),
                "dataset_percentage": round(pct_val, 2),
                "proportion": round(pct_val, 2),
            }

        # 2. Dynamic Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        confusion_dict = {}
        for i, act_cls in enumerate(classes_str):
            confusion_dict[act_cls] = {}
            for j, pred_cls in enumerate(classes_str):
                confusion_dict[act_cls][pred_cls] = int(cm[i, j])

        # 3. Individual Prediction Errors & Confusion Transitions
        errors = []
        error_by_class = {c: 0 for c in classes_str}
        confusion_pairs = {}
        test_indices = list(X_test_raw.index)

        if y_proba is not None:
            try:
                max_probas = np.max(y_proba, axis=1)
            except Exception:
                max_probas = np.ones(n_samples) * 0.75
        else:
            max_probas = np.ones(n_samples) * 0.75

        for i in range(n_samples):
            act = y_test[i]
            pred = y_pred[i]
            act_str = str(act)
            pred_str = str(pred)

            if act != pred:
                error_by_class[act_str] = error_by_class.get(act_str, 0) + 1
                pair_key = f"{act_str} -> {pred_str}"
                conf_val = float(max_probas[i]) if i < len(max_probas) else 0.75

                if pair_key not in confusion_pairs:
                    confusion_pairs[pair_key] = {
                        "actual": act_str,
                        "predicted": pred_str,
                        "occurrences": 0,
                        "confidences": [],
                    }
                confusion_pairs[pair_key]["occurrences"] += 1
                confusion_pairs[pair_key]["confidences"].append(conf_val)

                # Generic non-assumptive error description
                err_type = f"Misclassification ({act_str} -> {pred_str})"

                row_idx = test_indices[i] if i < len(test_indices) else i
                feat_dict = {
                    str(col): safe_primitive(X_test_raw.iloc[i][col])
                    for col in X_test_raw.columns[:10]
                }

                conf_level = "High" if conf_val >= 0.75 else ("Medium" if conf_val >= 0.50 else "Low")

                errors.append({
                    "row": int(row_idx) + 1,
                    "dataset_row": int(row_idx) + 1,
                    "actual": act_str,
                    "predicted": pred_str,
                    "actual_class": act_str,
                    "predicted_class": pred_str,
                    "confidence": round(conf_val, 4),
                    "confidence_percentage": round(conf_val * 100, 2),
                    "confidence_level": conf_level,
                    "error_type": err_type,
                    "features": feat_dict,
                    "feature_values": feat_dict,
                    "top_contributing_features": top_feature_names[:3],
                })

        total_test_errors = len(errors)

        # 4. Synthesize Error Patterns
        error_patterns = []
        for pair_key, p_data in sorted(confusion_pairs.items(), key=lambda x: x[1]["occurrences"], reverse=True):
            cnt = p_data["occurrences"]
            pct = round((cnt / total_test_errors) * 100, 1) if total_test_errors > 0 else 0.0
            avg_conf = round(float(np.mean(p_data["confidences"])), 4) if p_data["confidences"] else 0.0

            if cnt == max([p["occurrences"] for p in confusion_pairs.values()], default=0) and pct >= 35.0:
                p_type = "Dominant Error Pattern"
            elif cnt >= 3:
                p_type = "Frequent Confusion Pattern"
            elif cnt >= 2:
                p_type = "Repeated Confusion"
            else:
                p_type = "Isolated Error"

            error_patterns.append({
                "confusion_pair": pair_key,
                "actual": p_data["actual"],
                "predicted": p_data["predicted"],
                "actual_class": p_data["actual"],
                "predicted_class": p_data["predicted"],
                "occurrences": cnt,
                "error_percentage": pct,
                "average_confidence": round(avg_conf * 100, 1),
                "pattern_type": p_type,
            })

            # Emit evidence if dominant confusion transition occurs
            if cnt >= 3 or (pct >= 40.0 and total_test_errors >= 3):
                evidence_list.append(DiagnosticEvidence(
                    evidence_id=f"confusion_pattern_{p_data['actual']}_to_{p_data['predicted']}",
                    signal_name=f"Confusion Transition ({p_data['actual']} -> {p_data['predicted']})",
                    domain="confusion_pattern",
                    metric="error_concentration_share",
                    observed_value=pct,
                    baseline_value=round(100.0 / max(1, len(confusion_pairs)), 1),
                    magnitude=min(1.0, pct / 100.0),
                    direction="misclassification",
                    sample_support=cnt,
                    affected_population=f"Actual '{p_data['actual']}' predicted as '{p_data['predicted']}'",
                    affected_classes=[p_data["actual"], p_data["predicted"]],
                    signal_type="behavioral",
                    strength="HIGH" if pct >= 50.0 else "MODERATE",
                    reliability=min(1.0, cnt / 5.0),
                    model_scope="selected_model",
                    context=f"{cnt} of {total_test_errors} test misclassifications ({pct:.1f}%) were {pair_key} (avg confidence: {avg_conf * 100:.1f}%)."
                ))

        # 5. Evaluate Per-Class Recall Disparities
        recalls = [class_metrics[c]["recall"] for c in classes_str if class_metrics[c]["support"] > 0]
        if recalls:
            min_rec = min(recalls)
            max_rec = max(recalls)
            rec_gap = max_rec - min_rec

            # Find weakest recall class
            weakest_class = min(classes_str, key=lambda c: class_metrics[c]["recall"] if class_metrics[c]["support"] > 0 else 1.0)
            weakest_rec = class_metrics[weakest_class]["recall"]
            weakest_supp = class_metrics[weakest_class]["support"]

            if (rec_gap >= 0.12 or weakest_rec < 0.65) and weakest_supp >= 2:
                evidence_list.append(DiagnosticEvidence(
                    evidence_id=f"recall_disparity_{weakest_class}",
                    signal_name=f"Sensitivity Deficit for Class '{weakest_class}' (Recall: {weakest_rec:.1%})",
                    domain="class_performance",
                    metric="class_recall_disparity",
                    observed_value=round(weakest_rec, 4),
                    baseline_value=round(max_rec, 4),
                    magnitude=min(1.0, max(0.15, rec_gap)),
                    direction="deficit",
                    sample_support=weakest_supp,
                    affected_population=f"Class '{weakest_class}' ({weakest_supp} test samples)",
                    affected_classes=[weakest_class],
                    signal_type="behavioral",
                    strength="HIGH" if weakest_rec < 0.50 else "MODERATE",
                    reliability=min(1.0, weakest_supp / 8.0),
                    model_scope="selected_model",
                    context=f"Class '{weakest_class}' recall is {weakest_rec:.1%} vs {max_rec:.1%} for highest performing class (gap: {rec_gap:+.1%})."
                ))

        # Uncertain & High Confidence prediction tracking
        uncertain_records = [e for e in errors if e["confidence"] < 0.60]
        high_conf_errors = [e for e in errors if e["confidence"] >= 0.80]

        if len(high_conf_errors) >= 3:
            evidence_list.append(DiagnosticEvidence(
                evidence_id="high_confidence_errors_count",
                signal_name=f"Overconfident Misclassifications ({len(high_conf_errors)} of {total_test_errors} errors)",
                domain="confusion_pattern",
                metric="high_confidence_error_count",
                observed_value=len(high_conf_errors),
                baseline_value=0,
                magnitude=min(1.0, len(high_conf_errors) / max(1, total_test_errors)),
                direction="overconfidence",
                sample_support=len(high_conf_errors),
                affected_population=f"{len(high_conf_errors)} of {total_test_errors} test errors misclassified with >= 80% confidence",
                signal_type="behavioral",
                strength="MODERATE",
                reliability=0.85,
                model_scope="selected_model",
                context=f"Across all {total_test_errors} test errors, the model assigned >= 80% confidence to {len(high_conf_errors)} incorrect predictions."
            ))

        # Binary confusion metrics (for backward compatibility if n_classes == 2)
        if n_classes == 2:
            tn = int(cm[0, 0])
            fp = int(cm[0, 1])
            fn = int(cm[1, 0])
            tp = int(cm[1, 1])
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        main_error_str = "No major error pattern detected."
        if error_patterns:
            main_error_str = f"Dominant pattern: {error_patterns[0]['confusion_pair']} ({error_patterns[0]['occurrences']} occurrences)"

        return safe_primitive({
            "total_test_errors": total_test_errors,
            "test_samples": n_samples,
            "error_rate": round(total_test_errors / n_samples, 4) if n_samples > 0 else 0.0,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "main_error": main_error_str,
            "class_metrics": class_metrics,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_dict": confusion_dict,
            "error_by_class": error_by_class,
            "prediction_errors": errors,
            "error_patterns": error_patterns,
            "uncertain_predictions": uncertain_records[:15],
            "high_confidence_errors": high_conf_errors,
            "high_confidence_errors_count": len(high_conf_errors),
            "suspicious_records_count": len(high_conf_errors),
        }), evidence_list

    @classmethod
    def analyze_regression_errors(
        cls,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        X_test_raw: pd.DataFrame,
        top_feature_names: List[str]
    ) -> Tuple[Dict[str, Any], List[DiagnosticEvidence]]:
        """
        Detailed regression error inspection including residuals, heteroscedasticity,
        systematic bias, extreme residuals, and quartile errors.
        """
        y_test = np.asarray(y_test, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        residuals = y_test - y_pred
        abs_residuals = np.abs(residuals)
        n_samples = len(y_test)
        evidence_list: List[DiagnosticEvidence] = []

        mean_res = float(np.mean(residuals)) if n_samples > 0 else 0.0
        std_res = float(np.std(residuals)) if n_samples > 1 else 0.0
        median_res = float(np.median(residuals)) if n_samples > 0 else 0.0
        skew_res = float(pd.Series(residuals).skew()) if std_res > 0 else 0.0

        under_pred_count = int(np.sum(residuals > 0))
        over_pred_count = int(np.sum(residuals < 0))

        # Heteroscedasticity check: correlation between predictions and absolute residuals
        if np.std(y_pred) > 0 and np.std(abs_residuals) > 0:
            het_corr = float(np.corrcoef(y_pred, abs_residuals)[0, 1])
        else:
            het_corr = 0.0

        if abs(het_corr) >= 0.28 and n_samples >= 15:
            direction_desc = "expands with larger predictions" if het_corr > 0 else "contracts with larger predictions"
            evidence_list.append(DiagnosticEvidence(
                evidence_id="regression_heteroscedasticity",
                signal_name=f"Scale-Dependent Residual Variance (r = {het_corr:+.2f})",
                domain="regression_residual",
                metric="heteroscedasticity_correlation",
                observed_value=round(het_corr, 4),
                baseline_value=0.0,
                magnitude=min(1.0, abs(het_corr)),
                direction=direction_desc,
                sample_support=n_samples,
                affected_population="Extreme prediction range",
                signal_type="behavioral",
                strength="HIGH" if abs(het_corr) >= 0.45 else "MODERATE",
                reliability=min(1.0, n_samples / 30.0),
                model_scope="selected_model",
                context=f"Prediction error variance {direction_desc} (correlation between predicted value and error magnitude: {het_corr:+.2f})."
            ))

        # Systematic bias check
        if std_res > 0 and abs(mean_res) / std_res >= 0.25 and n_samples >= 15:
            bias_dir = "under-predicting" if mean_res > 0 else "over-predicting"
            evidence_list.append(DiagnosticEvidence(
                evidence_id="regression_systematic_bias",
                signal_name=f"Systematic Residual Mean Shift ({mean_res:+.4f})",
                domain="regression_residual",
                metric="mean_residual_bias",
                observed_value=round(mean_res, 4),
                baseline_value=0.0,
                magnitude=min(1.0, abs(mean_res) / (2.0 * std_res)),
                direction=bias_dir,
                sample_support=n_samples,
                affected_population="Overall regression test split",
                signal_type="behavioral",
                strength="MODERATE",
                reliability=min(1.0, n_samples / 25.0),
                model_scope="selected_model",
                context=f"Mean residual is non-zero ({mean_res:+.4f}, std: {std_res:.4f}), indicating systematic {bias_dir} tendency."
            ))

        # Adaptive residual concentration across ranges of actual target
        quartile_errors = []
        if n_samples >= 8:
            unique_targets = len(np.unique(y_test))
            n_bins = min(4, max(2, unique_targets // 3))
            try:
                q_cuts = pd.qcut(pd.Series(y_test), q=n_bins, duplicates="drop")
                for interval, grp in pd.Series(abs_residuals).groupby(q_cuts, observed=False):
                    quartile_errors.append({
                        "range": str(interval),
                        "count": int(len(grp)),
                        "mean_abs_error": round(float(grp.mean()), 4),
                        "median_abs_error": round(float(grp.median()), 4),
                    })
            except Exception:
                pass

        # Prediction errors ranked by largest absolute error
        ranked_indices = np.argsort(-abs_residuals)
        prediction_errors = []
        test_indices = list(X_test_raw.index)

        std_cutoff = 2.5 * std_res if std_res > 0 else float("inf")
        extreme_error_count = int(np.sum(abs_residuals > std_cutoff))

        for idx in ranked_indices[:50]:
            act = float(y_test[idx])
            pred = float(y_pred[idx])
            res = float(residuals[idx])
            abs_res = float(abs_residuals[idx])
            row_idx = test_indices[idx] if idx < len(test_indices) else idx

            # Numerically safe percentage error calculation
            if abs(act) >= 1e-9:
                pct_err = round((abs_res / abs(act)) * 100.0, 4)
                pct_err_disp = f"{pct_err:.2f}%"
            else:
                pct_err = None
                pct_err_disp = "N/A (actual=0)"

            feat_dict = {
                str(col): safe_primitive(X_test_raw.iloc[idx][col])
                for col in X_test_raw.columns[:10]
            }

            err_level = "Severe" if abs_res > std_cutoff else ("High" if abs_res > 1.5 * std_res else "Moderate")

            prediction_errors.append({
                "row": int(row_idx) + 1,
                "dataset_row": int(row_idx) + 1,
                "actual": round(act, 4),
                "predicted": round(pred, 4),
                "residual": round(res, 4),
                "absolute_error": round(abs_res, 4),
                "squared_error": round(res ** 2, 4),
                "percentage_error": pct_err,
                "percentage_error_display": pct_err_disp,
                "error_level": err_level,
                "features": feat_dict,
                "feature_values": feat_dict,
                "top_contributing_features": top_feature_names[:3],
            })

        # Synthesize evidence-driven error summary for regression
        target_range = float(np.ptp(y_test)) if n_samples > 1 else 1.0
        target_std = float(np.std(y_test)) if n_samples > 1 else 1.0
        mae_val = float(np.mean(abs_residuals)) if n_samples > 0 else 0.0
        max_abs = float(np.max(abs_residuals)) if n_samples > 0 else 0.0

        if abs(het_corr) >= 0.28 and n_samples >= 15:
            main_error_str = f"Residual variance changes across prediction range (heteroscedasticity correlation r = {het_corr:+.2f})."
        elif std_res > 0 and abs(mean_res) / std_res >= 0.25 and n_samples >= 15:
            bias_dir = "under-predicting" if mean_res > 0 else "over-predicting"
            main_error_str = f"Residuals show systematic {bias_dir} directional bias across evaluation sample (mean shift: {mean_res:+.4f}, std: {std_res:.4f})."
        elif extreme_error_count > 0:
            main_error_str = f"Large absolute residuals are present in a subset of evaluation observations ({extreme_error_count} extreme residual(s) exceeding 2.5 std)."
        elif target_range > 0 and (mae_val / target_range >= 0.25 or (target_std > 0 and std_res / target_std >= 0.70)):
            main_error_str = f"Substantial prediction error magnitude across evaluation partition (MAE: {mae_val:.4f}, max absolute error: {max_abs:.4f}, residual std: {std_res:.4f})."
        elif n_samples > 0 and mae_val > 0:
            main_error_str = f"Continuous residuals distributed with MAE {mae_val:.4f} and residual standard deviation {std_res:.4f}."
        else:
            main_error_str = "No dominant systematic residual pattern was detected."

        return safe_primitive({
            "main_error": main_error_str,
            "mean_residual": round(mean_res, 4),
            "median_residual": round(median_res, 4),
            "std_residual": round(std_res, 4),
            "skewness_residual": round(skew_res, 4),
            "under_predicted_count": under_pred_count,
            "over_predicted_count": over_pred_count,
            "heteroscedasticity_correlation": round(het_corr, 4),
            "quartile_errors": quartile_errors,
            "prediction_errors": prediction_errors,
            "extreme_error_count": extreme_error_count,
        }), evidence_list

    @classmethod
    def discover_subgroup_patterns(
        cls,
        task_type: str,
        X_df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Dynamically discovers feature subgroups with disproportionately elevated error rates.
        Uses adaptive binning without hardcoded quantile failures.
        """
        discovered = []
        n_samples = len(y_true)
        if n_samples < 10:
            return discovered

        if task_type == "classification":
            is_error = (y_true != y_pred).astype(int)
            baseline_error_rate = float(np.mean(is_error))
        else:
            errors = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
            baseline_error_rate = float(np.mean(errors))

        for feat in feature_names[:6]:
            if feat not in X_df.columns:
                continue
            series = X_df[feat]

            if pd.api.types.is_numeric_dtype(series):
                num_series = pd.to_numeric(series, errors="coerce")
                valid_mask = num_series.notna()
                if valid_mask.sum() < 10:
                    continue

                u_vals = num_series[valid_mask].nunique()
                n_bins = min(3, max(2, u_vals // 4)) if u_vals >= 4 else 2
                try:
                    bins = pd.qcut(num_series[valid_mask], q=n_bins, duplicates="drop")
                    for bin_label, grp_idx in num_series[valid_mask].groupby(bins, observed=False).groups.items():
                        if len(grp_idx) < 4:
                            continue
                        if task_type == "classification":
                            grp_err_rate = float(np.mean(is_error[grp_idx]))
                            if grp_err_rate >= (baseline_error_rate + 0.20) and grp_err_rate >= 0.40:
                                discovered.append({
                                    "segment_area": f"'{feat}' in range {bin_label}",
                                    "feature": str(feat),
                                    "sample_count": len(grp_idx),
                                    "error_rate": f"{grp_err_rate:.1%}",
                                    "baseline_error_rate": f"{baseline_error_rate:.1%}",
                                    "evidence": f"Error rate increases to {grp_err_rate:.1%} in this feature range vs {baseline_error_rate:.1%} baseline.",
                                })
                        else:
                            grp_mae = float(np.mean(errors[grp_idx]))
                            if baseline_error_rate > 0 and grp_mae >= (baseline_error_rate * 1.5):
                                discovered.append({
                                    "segment_area": f"'{feat}' in range {bin_label}",
                                    "feature": str(feat),
                                    "sample_count": len(grp_idx),
                                    "error_rate": f"MAE {grp_mae:.4f}",
                                    "baseline_error_rate": f"MAE {baseline_error_rate:.4f}",
                                    "evidence": f"MAE expands to {grp_mae:.4f} in this feature range vs {baseline_error_rate:.4f} overall.",
                                })
                except Exception:
                    pass
            else:
                # Categorical column
                top_cats = series.value_counts().head(4).index
                for cat in top_cats:
                    mask = (series == cat).values
                    if np.sum(mask) < 4:
                        continue
                    if task_type == "classification":
                        grp_err_rate = float(np.mean(is_error[mask]))
                        if grp_err_rate >= (baseline_error_rate + 0.20) and grp_err_rate >= 0.40:
                            discovered.append({
                                "segment_area": f"'{feat}' = '{cat}'",
                                "feature": str(feat),
                                "sample_count": int(np.sum(mask)),
                                "error_rate": f"{grp_err_rate:.1%}",
                                "baseline_error_rate": f"{baseline_error_rate:.1%}",
                                "evidence": f"Error rate reaches {grp_err_rate:.1%} for category '{cat}' vs {baseline_error_rate:.1%} baseline.",
                            })
                    else:
                        grp_mae = float(np.mean(errors[mask]))
                        if baseline_error_rate > 0 and grp_mae >= (baseline_error_rate * 1.5):
                            discovered.append({
                                "segment_area": f"'{feat}' = '{cat}'",
                                "feature": str(feat),
                                "sample_count": int(np.sum(mask)),
                                "error_rate": f"MAE {grp_mae:.4f}",
                                "baseline_error_rate": f"MAE {baseline_error_rate:.4f}",
                                "evidence": f"MAE reaches {grp_mae:.4f} for category '{cat}' vs {baseline_error_rate:.4f} overall.",
                            })

        return safe_primitive(discovered[:5])
