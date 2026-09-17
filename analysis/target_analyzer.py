# analysis/target_analyzer.py
"""
TargetAnalyzer: Universal, dataset-driven target inspection and problem-type detection.
Dynamically handles arbitrary binary, multiclass, and continuous regression targets
without domain-specific, churn-oriented, or hardcoded class assumptions.
"""

from typing import Tuple, Dict, List, Any, Optional
import numpy as np
import pandas as pd
from analysis.evidence import DiagnosticEvidence, safe_primitive


class TargetAnalyzer:
    """
    Inspects target distributions, determines task type (binary, multiclass, regression),
    and extracts statistical target profiles for machine-learning workflows.
    """

    @classmethod
    def analyze_target(
        cls,
        y_raw: pd.Series,
        target_column: str,
        explicit_mode: str = "auto"
    ) -> Tuple[str, str, Dict[str, Any], List[DiagnosticEvidence]]:
        """
        Analyzes the target variable, detects task type, and generates target profile & evidence.
        """
        mode = str(explicit_mode or "auto").strip().lower()
        if mode not in {"classification", "regression", "auto"}:
            mode = "auto"

        clean_y = y_raw.dropna()
        total_valid = len(clean_y)
        unique_vals = clean_y.unique()
        unique_count = len(unique_vals)
        evidence_list: List[DiagnosticEvidence] = []

        # --------------------------------------------------------
        # Problem Type Auto-Detection
        # --------------------------------------------------------
        is_numeric = pd.api.types.is_numeric_dtype(clean_y)
        is_bool = pd.api.types.is_bool_dtype(clean_y) or (
            clean_y.dtype == "object" and set(clean_y.unique()).issubset({True, False, "True", "False"})
        )

        if mode == "classification":
            task_type = "classification"
            task_reason = f"Task type explicitly configured as Classification by user ({unique_count} distinct classes)."
        elif mode == "regression":
            task_type = "regression"
            task_reason = f"Task type explicitly configured as Regression by user ({unique_count} distinct numeric values)."
        else:
            # Automatic heuristic detection
            if not is_numeric or is_bool:
                task_type = "classification"
                task_reason = f"Target contains discrete categorical/boolean values ({unique_count} distinct classes)."
            else:
                sorted_u = sorted(clean_y.unique())
                clean_vals = [safe_primitive(v) for v in sorted_u]
                if unique_count == 2:
                    task_type = "classification"
                    task_reason = f"Target is binary numeric (values: {clean_vals[0]}, {clean_vals[1]})."
                elif unique_count <= 10 and total_valid >= 50 and (unique_count / total_valid) < 0.05:
                    task_type = "classification"
                    task_reason = f"Target contains a discrete set of {unique_count} numeric classes."
                else:
                    u_pct = round((unique_count / total_valid) * 100, 1) if total_valid > 0 else 0.0
                    task_type = "regression"
                    task_reason = (
                        f"Target '{target_column}' is continuous numeric with "
                        f"{unique_count} unique values across {total_valid} observations ({u_pct}% uniqueness)."
                    )

        # --------------------------------------------------------
        # Build Profile & Evidence
        # --------------------------------------------------------
        profile: Dict[str, Any] = {
            "target_column": str(target_column),
            "target_type": task_type,
            "total_observations": total_valid,
            "unique_values_count": unique_count,
        }

        if task_type == "classification":
            counts = clean_y.value_counts(dropna=True)
            class_count = len(counts)
            is_binary = (class_count == 2)
            is_multiclass = (class_count > 2)

            class_distribution_list: List[Dict[str, Any]] = []
            class_distribution_dict: Dict[str, int] = {}
            class_percentages: Dict[str, float] = {}

            for rank, (cls_label, cnt) in enumerate(counts.items(), 1):
                label_str = str(safe_primitive(cls_label))
                prop = (int(cnt) / total_valid) if total_valid > 0 else 0.0
                pct = round(prop * 100, 2)
                class_distribution_dict[label_str] = int(cnt)
                class_percentages[label_str] = pct
                class_distribution_list.append({
                    "label": label_str,
                    "count": int(cnt),
                    "proportion": round(prop, 4),
                    "percentage": pct,
                    "rank_by_frequency": rank,
                })

            # Evaluate imbalance across all classes
            min_count = class_distribution_list[-1]["count"] if class_distribution_list else 0
            max_count = class_distribution_list[0]["count"] if class_distribution_list else 1
            min_class_label = class_distribution_list[-1]["label"] if class_distribution_list else ""
            max_class_label = class_distribution_list[0]["label"] if class_distribution_list else ""

            imbalance_ratio = round(min_count / max_count, 2) if max_count > 0 else 1.0
            min_pct = class_distribution_list[-1]["percentage"] if class_distribution_list else 0.0

            # General entropy-based / ratio-based imbalance check
            has_imbalance = False
            if is_binary:
                has_imbalance = (imbalance_ratio < 0.50 or min_pct < 33.3)
            else:
                # Multiclass imbalance: least frequent class is notably smaller than average expectation
                expected_pct = 100.0 / class_count if class_count > 0 else 100.0
                has_imbalance = (min_pct < (expected_pct * 0.40) or imbalance_ratio < 0.30)

            if has_imbalance:
                imbalance_explanation = (
                    f"Class representation is uneven across {class_count} classes: least frequent class '{min_class_label}' "
                    f"represents {min_pct:.1f}% ({min_count}/{total_valid}) of observations "
                    f"(min-to-max ratio: {imbalance_ratio:.2f}:1)."
                )
                evidence_list.append(DiagnosticEvidence(
                    evidence_id=f"target_imbalance_{target_column}",
                    signal_name=f"Target Class Imbalance ({min_class_label} at {min_pct:.1f}%)",
                    domain="class_performance",
                    metric="class_imbalance_ratio",
                    observed_value=imbalance_ratio,
                    baseline_value=1.0,
                    magnitude=max(0.0, min(1.0, 1.0 - imbalance_ratio)),
                    direction="disparity",
                    sample_support=min_count,
                    affected_population=f"Class '{min_class_label}' ({min_pct:.1f}% share)",
                    affected_classes=[min_class_label],
                    signal_type="structural",
                    strength="HIGH" if imbalance_ratio < 0.25 else "MODERATE",
                    reliability=1.0,
                    model_scope="dataset",
                    context=f"Class '{min_class_label}' has {min_count} observations vs {max_count} for '{max_class_label}'."
                ))
            else:
                imbalance_explanation = (
                    f"Classes are relatively balanced across {class_count} classes "
                    f"(min-to-max ratio: {imbalance_ratio:.2f}:1)."
                )

            profile.update({
                "subtype": "binary" if is_binary else "multiclass",
                "is_binary": is_binary,
                "is_multiclass": is_multiclass,
                "class_count": class_count,
                "class_distribution_list": class_distribution_list,
                "class_distribution": class_distribution_dict,
                "class_percentages": class_percentages,
                "class_names": list(class_distribution_dict.keys()),
                "least_frequent_class": min_class_label,
                "most_frequent_class": max_class_label,
                "least_frequent_count": min_count,
                "most_frequent_count": max_count,
                "least_frequent_percentage": min_pct,
                "imbalance_ratio": imbalance_ratio,
                "class_imbalance": has_imbalance,
                "imbalance_explanation": imbalance_explanation,
                # Backward-compatible keys
                "minority_class": min_class_label,
                "majority_class": max_class_label,
                "minority_count": min_count,
                "majority_count": max_count,
                "minority_percentage": min_pct,
                "minority_to_majority_ratio": imbalance_ratio,
            })

        else:
            # Continuous regression profile
            numeric_y = pd.to_numeric(clean_y, errors="coerce").dropna().astype(float)
            mean_val = float(numeric_y.mean()) if not numeric_y.empty else 0.0
            median_val = float(numeric_y.median()) if not numeric_y.empty else 0.0
            std_val = float(numeric_y.std()) if len(numeric_y) > 1 else 0.0
            var_val = float(numeric_y.var()) if len(numeric_y) > 1 else 0.0
            min_val = float(numeric_y.min()) if not numeric_y.empty else 0.0
            max_val = float(numeric_y.max()) if not numeric_y.empty else 0.0
            skew_val = float(numeric_y.skew()) if std_val > 0 else 0.0

            q25 = float(numeric_y.quantile(0.25)) if not numeric_y.empty else 0.0
            q75 = float(numeric_y.quantile(0.75)) if not numeric_y.empty else 0.0
            iqr_val = q75 - q25

            if abs(skew_val) >= 1.5:
                direction_str = "right-skewed" if skew_val > 0 else "left-skewed"
                evidence_list.append(DiagnosticEvidence(
                    evidence_id=f"target_skewness_{target_column}",
                    signal_name=f"High Target Distribution Skewness ({skew_val:+.2f})",
                    domain="regression_residual",
                    metric="target_skewness",
                    observed_value=round(skew_val, 2),
                    baseline_value=0.0,
                    magnitude=min(1.0, abs(skew_val) / 3.0),
                    direction=direction_str,
                    sample_support=total_valid,
                    affected_population="Continuous target distribution",
                    signal_type="structural",
                    strength="MODERATE",
                    reliability=0.95,
                    model_scope="dataset",
                    context=f"Target exhibits marked {direction_str} skewness (skew = {skew_val:.2f})."
                ))

            # Binary-target regression compatibility check
            compat_note = None
            is_binary_num = False
            distinct_num_vals = []
            if not numeric_y.empty:
                unique_num_vals = sorted(list(numeric_y.unique()))
                if len(unique_num_vals) == 2:
                    is_binary_num = True
                    distinct_num_vals = [int(v) if float(v).is_integer() else float(v) for v in unique_num_vals]
                    distinct_str = " and ".join(str(v) for v in distinct_num_vals)
                    compat_note = (
                        f"Target contains only two distinct numeric values ({distinct_str}). "
                        f"Regression analysis was explicitly selected, so regression metrics are reported as configured. "
                        f"A classification formulation may be more naturally suited for binary target semantics."
                    )
                    evidence_list.append(DiagnosticEvidence(
                        evidence_id=f"binary_target_regression_caveat_{target_column}",
                        signal_name=f"Binary Numeric Target in Regression Mode ({distinct_str})",
                        domain="target_structure",
                        metric="target_distinct_values",
                        observed_value=2,
                        baseline_value=10,
                        magnitude=0.35,
                        direction="discrete_target",
                        sample_support=total_valid,
                        affected_population=f"All {total_valid} observations in target column '{target_column}'",
                        signal_type="structural",
                        strength="MODERATE",
                        reliability=1.0,
                        model_scope="dataset",
                        context=compat_note
                    ))

            profile.update({
                "subtype": "continuous" if not is_binary_num else "binary_numeric",
                "is_binary_numeric": is_binary_num,
                "binary_target_compatibility_note": compat_note,
                "distinct_values": distinct_num_vals,
                "mean": round(mean_val, 4),
                "median": round(median_val, 4),
                "std": round(std_val, 4),
                "variance": round(var_val, 4),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
                "q25": round(q25, 4),
                "q75": round(q75, 4),
                "iqr": round(iqr_val, 4),
                "skewness": round(skew_val, 4),
                "zero_count": int((numeric_y == 0).sum()),
                "negative_count": int((numeric_y < 0).sum()),
            })

        return task_type, task_reason, safe_primitive(profile), evidence_list
