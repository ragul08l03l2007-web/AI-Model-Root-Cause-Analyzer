# analysis/derivations.py
"""
AI Model Root-Cause Analyzer: Mathematical Derivation & Calculation Trace Engine.
Produces step-by-step mathematical proofs, theoretical equations, substituted arithmetic,
and intuitive engineering explanations for all computed features, metrics, and diagnostic outputs.
"""

from typing import Dict, List, Any, Optional
import math
from analysis.evidence import safe_primitive


class DerivationEngine:
    """
    Constructs rigorous mathematical derivation objects for all computed diagnostic outputs.
    Each derivation record includes:
      - title: Human-readable name of the metric / feature
      - formula_latex: Formal mathematical equation (LaTeX format)
      - formula_text: Plain-text readable formula
      - variables: Dictionary of variable definitions and actual substituted values
      - calculation_steps: Ordered list of step-by-step arithmetic operations
      - output_value: Final computed scalar or summary string
      - unit_or_range: Measurement scale / theoretical bounds
      - interpretation: Statistical and machine-learning rationale
    """

    @classmethod
    def generate_all_derivations(
        cls,
        result: Dict[str, Any],
        data_quality: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes complete mathematical derivations across all subsystems of the analyzer.
        """
        task_type = result.get("task_type", "classification")
        target_col = result.get("target_column", "target")
        perf = result.get("model_performance", {})
        target_prof = result.get("target_profile", {})
        cv = result.get("cross_validation", {})
        feat_impact = result.get("feature_impact", {})
        risk = result.get("risk", {})
        dq = data_quality or result.get("data_quality", {})
        err_res = result.get("error_analysis", {})

        derivations = {
            "risk_score": cls.derive_risk_score(risk, result),
            "target_profile": cls.derive_target_profile(target_prof, target_col, task_type),
            "model_performance": cls.derive_model_performance(perf, task_type),
            "generalization": cls.derive_generalization(perf, cv, task_type),
            "cross_validation": cls.derive_cross_validation(cv),
            "features": cls.derive_features(feat_impact, target_col),
            "data_quality": cls.derive_data_quality(dq, result.get("total_rows", 0)),
            "confidence_score": cls.derive_confidence_engine(result),
            "error_analysis": cls.derive_error_analysis(err_res, task_type),
        }

        return safe_primitive(derivations)

    # =========================================================================
    # 1. RISK SCORE DERIVATION
    # =========================================================================
    @classmethod
    def derive_risk_score(cls, risk: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        final_score = int(risk.get("score", result.get("risk_score", 15)))
        breakdown = risk.get("breakdown", {})
        p_perf = float(breakdown.get("performance_gap", 0))
        p_gen = float(breakdown.get("generalization_gap", 0))
        p_dist = float(breakdown.get("distribution_risk", 0))
        p_dq = float(breakdown.get("data_integrity", 0))

        steps = [
            f"1. Base Baseline Floor: S_base = 5.0 (guaranteed diagnostic baseline)",
            f"2. Subsystem 1: Performance Deficit Penalty (Max 35.0): P_perf = {p_perf:.1f} pts",
            f"3. Subsystem 2: Generalization Drop Penalty (Max 30.0): P_gen = {p_gen:.1f} pts",
            f"4. Subsystem 3: Target Imbalance & Leakage Penalty (Max 25.0): P_dist = {p_dist:.1f} pts",
            f"5. Subsystem 4: Data Quality & Hygiene Penalty (Max 25.0): P_integrity = {p_dq:.1f} pts",
            f"6. Summation: Raw_Risk = 5.0 + {p_perf:.1f} + {p_gen:.1f} + {p_dist:.1f} + {p_dq:.1f} = {5.0 + p_perf + p_gen + p_dist + p_dq:.1f}",
            f"7. Boundary Clamping: Final_Risk = min(100, max(5, round({5.0 + p_perf + p_gen + p_dist + p_dq:.1f}))) = {final_score}/100",
        ]

        if final_score >= 65:
            level_note = "HIGH RISK: The model demonstrates multiple compounding failure points (e.g. substantial generalization gap, critical data leakage, severe imbalance)."
        elif final_score >= 35:
            level_note = "MEDIUM RISK: Measurable vulnerabilities detected requiring mitigation before mission-critical production deployment."
        else:
            level_note = "LOW RISK: Healthy diagnostic profile. Model performance and data integrity remain within acceptable statistical bounds."

        return {
            "title": "Overall Production Risk Score",
            "metric_key": "risk_score",
            "formula_latex": r"\text{Risk Score} = \min\left(100, \max\left(5, 5 + P_{\text{perf}} + P_{\text{gen}} + P_{\text{dist}} + P_{\text{integrity}}\right)\right)",
            "formula_text": "Risk Score = min(100, max(5, 5.0 + P_perf + P_gen + P_dist + P_integrity))",
            "variables": {
                "S_base": {"desc": "Baseline diagnostic floor", "val": 5.0},
                "P_perf": {"desc": "Holdout partition performance deficit penalty", "val": p_perf, "max": 35.0},
                "P_gen": {"desc": "Train vs Test / CV generalization gap penalty", "val": p_gen, "max": 30.0},
                "P_dist": {"desc": "Class imbalance & target leakage penalty", "val": p_dist, "max": 25.0},
                "P_integrity": {"desc": "Missing values, duplicate rows & small sample penalty", "val": p_dq, "max": 25.0},
            },
            "calculation_steps": steps,
            "output_value": f"{final_score} / 100 ({risk.get('level', 'LOW')} RISK)",
            "unit_or_range": "0 to 100 (Integer score)",
            "interpretation": level_note,
        }

    # =========================================================================
    # 2. TARGET PROFILE & IMBALANCE DERIVATION
    # =========================================================================
    @classmethod
    def derive_target_profile(cls, target_prof: Dict[str, Any], target_col: str, task_type: str) -> Dict[str, Any]:
        total_obs = target_prof.get("total_observations", target_prof.get("total_rows", 0))

        if task_type == "classification":
            minority_cls = str(target_prof.get("minority_class", "N/A"))
            majority_cls = str(target_prof.get("majority_class", "N/A"))
            min_count = int(target_prof.get("least_frequent_count", target_prof.get("minority_count", 0)))
            maj_count = int(target_prof.get("most_frequent_count", target_prof.get("majority_count", 1)))
            min_pct = float(target_prof.get("minority_percentage", target_prof.get("least_frequent_percentage", 0.0)))
            imb_ratio = float(target_prof.get("minority_to_majority_ratio", target_prof.get("imbalance_ratio", 0.0)))

            steps = [
                f"1. Total Valid Observations: N_total = {total_obs}",
                f"2. Minority Class ('{minority_cls}') Count: N_minority = {min_count}",
                f"3. Majority Class ('{majority_cls}') Count: N_majority = {maj_count}",
                f"4. Minority Representation Percentage: P_min = (N_minority / N_total) × 100% = ({min_count} / {total_obs}) × 100% = {min_pct:.2f}%",
                f"5. Minority-to-Majority Ratio: R_imb = N_minority / N_majority = {min_count} / {maj_count} = {imb_ratio:.2f} : 1",
            ]

            return {
                "title": f"Target Class Distribution & Imbalance ({target_col})",
                "metric_key": "target_imbalance",
                "formula_latex": r"R_{\text{imbalance}} = \frac{N_{\text{minority}}}{N_{\text{majority}}}, \quad P_{\text{minority}}\% = \frac{N_{\text{minority}}}{N_{\text{total}}} \times 100\%",
                "formula_text": "Imbalance Ratio = N_minority / N_majority | Minority % = (N_minority / N_total) * 100",
                "variables": {
                    "N_total": {"desc": "Total observations with non-null target", "val": total_obs},
                    "N_minority": {"desc": f"Least frequent class '{minority_cls}' frequency", "val": min_count},
                    "N_majority": {"desc": f"Most frequent class '{majority_cls}' frequency", "val": maj_count},
                    "P_minority": {"desc": "Minority class proportion percentage", "val": f"{min_pct:.2f}%"},
                    "R_imbalance": {"desc": "Minority to majority frequency ratio", "val": f"{imb_ratio:.2f}:1"},
                },
                "calculation_steps": steps,
                "output_value": f"Minority %: {min_pct:.1f}% | Ratio: {imb_ratio:.2f}:1 (Imbalanced: {bool(target_prof.get('class_imbalance', False))})",
                "unit_or_range": "Ratio: [0.0, 1.0], Percentage: [0%, 50%]",
                "interpretation": f"A ratio of {imb_ratio:.2f}:1 means that for every 1 observation in class '{minority_cls}', there are approximately {1.0 / max(0.001, imb_ratio):.1f} observations in class '{majority_cls}'.",
            }
        else:
            mean_val = float(target_prof.get("mean", 0.0))
            std_val = float(target_prof.get("std", 0.0))
            min_val = float(target_prof.get("min", 0.0))
            max_val = float(target_prof.get("max", 0.0))
            skew_val = float(target_prof.get("skewness", 0.0))

            steps = [
                f"1. Total Continuous Samples: N = {total_obs}",
                f"2. Sample Mean: μ = (1 / N) ∑ y_i = {mean_val:.4f}",
                f"3. Sample Standard Deviation: σ = √[ (1 / (N - 1)) ∑ (y_i - μ)² ] = {std_val:.4f}",
                f"4. Sample Range: [Min, Max] = [{min_val:.4f}, {max_val:.4f}]",
                f"5. Fisher-Pearson Skewness Coefficient: γ_1 = {skew_val:.4f}",
            ]

            return {
                "title": f"Target Continuous Distribution Moments ({target_col})",
                "metric_key": "target_distribution",
                "formula_latex": r"\mu = \frac{1}{N}\sum_{i=1}^N y_i, \quad \sigma = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (y_i - \mu)^2}, \quad \gamma_1 = \frac{\frac{1}{N}\sum(y_i - \mu)^3}{\sigma^3}",
                "formula_text": "Mean = (1/N) * sum(y_i) | Std = sqrt((1/(N-1)) * sum((y_i - Mean)^2)) | Skewness = E[(y - mu)^3] / sigma^3",
                "variables": {
                    "N": {"desc": "Sample size", "val": total_obs},
                    "mu": {"desc": "Empirical Mean", "val": round(mean_val, 4)},
                    "sigma": {"desc": "Empirical Standard Deviation", "val": round(std_val, 4)},
                    "skewness": {"desc": "Third standardized moment (skewness)", "val": round(skew_val, 4)},
                },
                "calculation_steps": steps,
                "output_value": f"Mean = {mean_val:.2f}, Std = {std_val:.2f}, Skewness = {skew_val:.2f}",
                "unit_or_range": f"[{min_val:.2f}, {max_val:.2f}]",
                "interpretation": f"Distribution skewness of {skew_val:.2f} indicates {'approximately symmetric' if abs(skew_val) < 0.5 else ('moderate positive right-tail' if skew_val > 0 else 'moderate negative left-tail')} distribution.",
            }

    # =========================================================================
    # 3. MODEL PERFORMANCE METRICS DERIVATION
    # =========================================================================
    @classmethod
    def derive_model_performance(cls, perf: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        if task_type == "classification":
            acc = float(perf.get("accuracy", 0.0))
            bal_acc = float(perf.get("balanced_accuracy", acc))
            f1 = float(perf.get("f1_macro", perf.get("f1_score", 0.0)))
            prec = float(perf.get("precision_macro", perf.get("precision", 0.0)))
            rec = float(perf.get("recall_macro", perf.get("recall", 0.0)))

            steps = [
                f"1. Standard Accuracy: Acc = (TP + TN) / (TP + TN + FP + FN) = {acc * 100:.2f}%",
                f"2. Balanced Accuracy: BalAcc = (1 / K) ∑ (Recall_k) = {bal_acc * 100:.2f}%",
                f"3. Macro Precision: Prec = (1 / K) ∑ [ TP_k / (TP_k + FP_k) ] = {prec * 100:.2f}%",
                f"4. Macro Recall: Rec = (1 / K) ∑ [ TP_k / (TP_k + FN_k) ] = {rec * 100:.2f}%",
                f"5. Macro F1-Score: F1 = 2 × (Prec × Rec) / (Prec + Rec) = {f1 * 100:.2f}%",
            ]

            return {
                "title": "Classification Performance Metric Derivations",
                "metric_key": "classification_metrics",
                "formula_latex": r"\text{Accuracy} = \frac{\sum \text{Correct}}{N}, \quad \text{BalAcc} = \frac{1}{K}\sum_{k=1}^K \text{Recall}_k, \quad F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}",
                "formula_text": "Accuracy = Correct / Total | Balanced Accuracy = Mean(Recall_k) | F1 = 2 * (Precision * Recall) / (Precision + Recall)",
                "variables": {
                    "Accuracy": {"desc": "Overall fraction of correct predictions", "val": f"{acc * 100:.2f}%"},
                    "Balanced_Accuracy": {"desc": "Unweighted mean of recall across all classes", "val": f"{bal_acc * 100:.2f}%"},
                    "Macro_Precision": {"desc": "Average positive predictive value across classes", "val": f"{prec * 100:.2f}%"},
                    "Macro_Recall": {"desc": "Average true positive rate across classes", "val": f"{rec * 100:.2f}%"},
                    "Macro_F1": {"desc": "Harmonic mean of macro precision and macro recall", "val": f"{f1 * 100:.2f}%"},
                },
                "calculation_steps": steps,
                "output_value": f"Accuracy: {acc * 100:.1f}% | Balanced Accuracy: {bal_acc * 100:.1f}% | F1: {f1 * 100:.1f}%",
                "unit_or_range": "[0.0%, 100.0%]",
                "interpretation": "Balanced accuracy provides an uncompromised evaluation metric robust against majority class bias in imbalanced datasets.",
            }
        else:
            r2 = float(perf.get("r2", perf.get("r2_score", 0.0)))
            rmse = float(perf.get("rmse", 0.0))
            mae = float(perf.get("mae", 0.0))
            mape = float(perf.get("mape", 0.0))

            steps = [
                f"1. Residual Sum of Squares: SS_res = ∑ (y_i - ŷ_i)²",
                f"2. Total Sum of Squares: SS_tot = ∑ (y_i - ȳ)²",
                f"3. Coefficient of Determination: R² = 1 - (SS_res / SS_tot) = {r2:.4f} ({r2 * 100:.2f}% variance explained)",
                f"4. Root Mean Squared Error: RMSE = √[ (1 / n) ∑ (y_i - ŷ_i)² ] = {rmse:.4f}",
                f"5. Mean Absolute Error: MAE = (1 / n) ∑ |y_i - ŷ_i| = {mae:.4f}",
                f"6. Mean Absolute Percentage Error: MAPE = (100% / n) ∑ |(y_i - ŷ_i) / y_i| = {mape:.2f}%",
            ]

            return {
                "title": "Regression Performance Metric Derivations",
                "metric_key": "regression_metrics",
                "formula_latex": r"R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}, \quad \text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}, \quad \text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|",
                "formula_text": "R^2 = 1 - (SS_res / SS_tot) | RMSE = sqrt((1/n) * sum((y - y_hat)^2)) | MAE = (1/n) * sum(|y - y_hat|)",
                "variables": {
                    "R_squared": {"desc": "Proportion of target variance captured by the model", "val": round(r2, 4)},
                    "RMSE": {"desc": "Root mean squared deviation in target units", "val": round(rmse, 4)},
                    "MAE": {"desc": "Mean absolute error magnitude in target units", "val": round(mae, 4)},
                    "MAPE": {"desc": "Mean relative percentage error", "val": f"{mape:.2f}%"},
                },
                "calculation_steps": steps,
                "output_value": f"R² = {r2:.3f} | RMSE = {rmse:.2f} | MAE = {mae:.2f}",
                "unit_or_range": "R²: (-∞, 1.0], RMSE/MAE: [0, +∞)",
                "interpretation": f"The model explains {max(0.0, r2) * 100:.1f}% of total target variability, with an average absolute prediction deviation of {mae:.2f} units.",
            }

    # =========================================================================
    # 4. GENERALIZATION & OVERFITTING GAP DERIVATION
    # =========================================================================
    @classmethod
    def derive_generalization(cls, perf: Dict[str, Any], cv: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        train_score = float(perf.get("train_score", perf.get("train_accuracy", 0.0)))
        test_score = float(perf.get("test_score", perf.get("test_accuracy", perf.get("accuracy", perf.get("r2", 0.0)))))
        cv_mean = float(cv.get("average_score", cv.get("mean", 0.0)))

        gen_gap = max(0.0, train_score - test_score)
        gen_ratio = (train_score / max(0.001, test_score)) if test_score > 0 else 1.0

        metric_name = "Accuracy" if task_type == "classification" else "R²"

        steps = [
            f"1. Training Partition {metric_name}: Score_train = {train_score * 100:.2f}%",
            f"2. Held-Out Test Partition {metric_name}: Score_test = {test_score * 100:.2f}%",
            f"3. Cross-Validation Mean {metric_name}: Score_cv_mean = {cv_mean * 100:.2f}%",
            f"4. Absolute Generalization Gap: Δ_gen = Score_train - Score_test = {train_score * 100:.2f}% - {test_score * 100:.2f}% = {gen_gap * 100:.2f}%",
            f"5. Generalization Overfitting Ratio: Ratio_gen = Score_train / Score_test = {gen_ratio:.2f}",
        ]

        if gen_gap > 0.15:
            diag_note = f"Critical Overfitting: Training performance exceeds evaluation holdout by {gen_gap * 100:.1f}%, indicating significant memory of training noise."
        elif gen_gap > 0.07:
            diag_note = f"Moderate Generalization Gap: {gen_gap * 100:.1f}% drop between train and test. Regularization recommended."
        else:
            diag_note = f"Healthy Generalization: Minimal score deviation ({gen_gap * 100:.1f}%) between partitions."

        return {
            "title": "Generalization & Overfitting Gap Derivation",
            "metric_key": "generalization_gap",
            "formula_latex": r"\Delta_{\text{generalization}} = \text{Score}_{\text{train}} - \text{Score}_{\text{test}}, \quad \text{Ratio}_{\text{gen}} = \frac{\text{Score}_{\text{train}}}{\text{Score}_{\text{test}}}",
            "formula_text": "Generalization Gap = Score_train - Score_test | Overfitting Ratio = Score_train / Score_test",
            "variables": {
                "Score_train": {"desc": f"Training split {metric_name}", "val": f"{train_score * 100:.2f}%"},
                "Score_test": {"desc": f"Independent test holdout {metric_name}", "val": f"{test_score * 100:.2f}%"},
                "Delta_gen": {"desc": "Absolute performance disparity", "val": f"{gen_gap * 100:.2f}%"},
                "Score_cv_mean": {"desc": "Stratified cross-validation mean", "val": f"{cv_mean * 100:.2f}%"},
            },
            "calculation_steps": steps,
            "output_value": f"Gap: {gen_gap * 100:.1f}% (Train: {train_score * 100:.1f}% vs Test: {test_score * 100:.1f}%)",
            "unit_or_range": "[-100%, +100%]",
            "interpretation": diag_note,
        }

    # =========================================================================
    # 5. CROSS-VALIDATION STABILITY DERIVATION
    # =========================================================================
    @classmethod
    def derive_cross_validation(cls, cv: Dict[str, Any]) -> Dict[str, Any]:
        k_folds = int(cv.get("folds_count", cv.get("k", 5)))
        fold_scores = [float(s) for s in cv.get("fold_scores", cv.get("scores", []))]
        mean_s = float(cv.get("average_score", cv.get("mean", 0.0)))
        std_s = float(cv.get("std_score", cv.get("std", 0.0)))
        min_s = float(cv.get("min_score", min(fold_scores) if fold_scores else 0.0))
        max_s = float(cv.get("max_score", max(fold_scores) if fold_scores else 0.0))
        span_s = max_s - min_s

        scores_str = ", ".join(f"{s * 100:.1f}%" for s in fold_scores) if fold_scores else "N/A"

        steps = [
            f"1. Evaluated K-Fold Partition Count: K = {k_folds}",
            f"2. Partition Scores (s_1 ... s_k): [{scores_str}]",
            f"3. Cross-Validation Mean: μ_cv = (1 / K) ∑ s_i = {mean_s * 100:.2f}%",
            f"4. Sample Standard Deviation: σ_cv = √[ (1 / (K - 1)) ∑ (s_i - μ_cv)² ] = {std_s * 100:.2f}%",
            f"5. Fold Performance Range: Span = Max(s_i) - Min(s_i) = {max_s * 100:.1f}% - {min_s * 100:.1f}% = {span_s * 100:.1f}%",
        ]

        if std_s > 0.08 or span_s > 0.15:
            cv_note = f"High Partition Variance (σ = {std_s * 100:.1f}%): Model performance fluctuates widely across folds, indicating sensitivity to subset composition."
        elif std_s > 0.03:
            cv_note = f"Moderate Partition Stability (σ = {std_s * 100:.1f}%): Normal variance expected across {k_folds} folds."
        else:
            cv_note = f"High Partition Stability (σ = {std_s * 100:.1f}%): Consistent predictions across all validation folds."

        return {
            "title": f"{k_folds}-Fold Cross-Validation Stability Derivation",
            "metric_key": "cv_stability",
            "formula_latex": r"\mu_{\text{cv}} = \frac{1}{K}\sum_{i=1}^K s_i, \quad \sigma_{\text{cv}} = \sqrt{\frac{1}{K-1}\sum_{i=1}^K (s_i - \mu_{\text{cv}})^2}, \quad \text{Range} = \max(s_i) - \min(s_i)",
            "formula_text": "Mean = (1/K) * sum(s_i) | Std = sqrt((1/(K-1)) * sum((s_i - Mean)^2)) | Range = Max - Min",
            "variables": {
                "K": {"desc": "Number of cross-validation splits", "val": k_folds},
                "Mean_cv": {"desc": "Average validation score", "val": f"{mean_s * 100:.2f}%"},
                "Std_cv": {"desc": "Fold score standard deviation", "val": f"{std_s * 100:.2f}%"},
                "Range_cv": {"desc": "Score spread (Max - Min)", "val": f"{span_s * 100:.2f}%"},
            },
            "calculation_steps": steps,
            "output_value": f"Mean: {mean_s * 100:.1f}% ± {std_s * 100:.1f}% (Range: {min_s * 100:.1f}% - {max_s * 100:.1f}%)",
            "unit_or_range": "[0%, 100%]",
            "interpretation": cv_note,
        }

    # =========================================================================
    # 6. FEATURE IMPORTANCE, SHARE & CORRELATION DERIVATION
    # =========================================================================
    @classmethod
    def derive_features(cls, feat_impact: Dict[str, Any], target_col: str) -> Dict[str, Any]:
        feature_derivations = {}

        total_imp = sum(float(f.get("importance", 0.0)) for f in feat_impact.values())
        total_imp = max(0.0001, total_imp)

        for feat_name, feat_data in feat_impact.items():
            raw_imp = float(feat_data.get("importance", 0.0))
            share_pct = float(feat_data.get("relative_share_pct", (raw_imp / total_imp) * 100.0))
            corr = float(feat_data.get("correlation", 0.0))
            mi = float(feat_data.get("mutual_info", 0.0))
            tier = feat_data.get("influence_tier", "Moderate Model Influence")

            steps = [
                f"1. Raw Feature Importance Score: I('{feat_name}') = {raw_imp:.4f}",
                f"2. Total Importance Across All Features: ∑ I_j = {total_imp:.4f}",
                f"3. Relative Attribution Share: Share('{feat_name}') = (I / ∑ I_j) × 100% = ({raw_imp:.4f} / {total_imp:.4f}) × 100% = {share_pct:.2f}%",
                f"4. Pearson Correlation with Target ('{target_col}'): r = Cov(X, y) / (σ_X · σ_y) = {corr:+.3f}",
                f"5. Mutual Information Score: I(X; y) = {mi:.4f} bits",
                f"6. Influence Tier Assignment: '{tier}' (Based on relative share threshold {share_pct:.1f}%)",
            ]

            feature_derivations[feat_name] = {
                "title": f"Feature Impact Derivation: '{feat_name}'",
                "feature_name": feat_name,
                "formula_latex": r"\text{Share}_i\% = \frac{I_i}{\sum_{j=1}^M I_j} \times 100\%, \quad r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}",
                "formula_text": "Relative Share % = (Importance_i / Sum(Importance)) * 100 | Pearson r = Cov(X, Y) / (Std_X * Std_Y)",
                "variables": {
                    "Raw_Importance": {"desc": "Gini impurity / Permutation attribution score", "val": round(raw_imp, 4)},
                    "Relative_Share": {"desc": "Normalized share of total model predictive power", "val": f"{share_pct:.2f}%"},
                    "Pearson_r": {"desc": f"Linear correlation with '{target_col}'", "val": f"{corr:+.3f}"},
                    "Mutual_Info": {"desc": "Nonlinear dependency score", "val": round(mi, 4)},
                    "Tier": {"desc": "Assigned diagnostic influence tier", "val": tier},
                },
                "calculation_steps": steps,
                "output_value": f"Share: {share_pct:.1f}% | Correlation: {corr:+.2f} | Tier: {tier}",
                "unit_or_range": "Share: [0%, 100%], Correlation: [-1.0, +1.0]",
                "interpretation": f"'{feat_name}' accounts for {share_pct:.1f}% of model predictive reliance. {feat_data.get('relationship_summary', '')} (Note: Association does not imply causation).",
            }

        return feature_derivations

    # =========================================================================
    # 7. DATA QUALITY & TUKEY IQR DERIVATION
    # =========================================================================
    @classmethod
    def derive_data_quality(cls, dq: Dict[str, Any], total_rows: int) -> Dict[str, Any]:
        tot_cells = dq.get("total_cells", max(1, total_rows * dq.get("total_columns", 1)))
        missing_count = int(dq.get("total_missing_cells", 0))
        dup_count = int(dq.get("duplicate_rows", 0))
        missing_pct = float(dq.get("missing_percentage", (missing_count / max(1, tot_cells)) * 100.0))
        dup_pct = float(dq.get("duplicate_percentage", (dup_count / max(1, total_rows)) * 100.0))

        outliers_dict = dq.get("outliers", {})
        outlier_derivations = {}

        for col, o_info in outliers_dict.items():
            if isinstance(o_info, dict):
                c_out = int(o_info.get("count", 0))
                p_out = float(o_info.get("percentage", 0.0))
                q1 = float(o_info.get("q1", 0.0))
                q3 = float(o_info.get("q3", 0.0))
                iqr = float(o_info.get("iqr", q3 - q1))
                lower = float(o_info.get("lower_bound", q1 - 1.5 * iqr))
                upper = float(o_info.get("upper_bound", q3 + 1.5 * iqr))

                o_steps = [
                    f"1. 25th Percentile (First Quartile): Q_1 = {q1:.3f}",
                    f"2. 75th Percentile (Third Quartile): Q_3 = {q3:.3f}",
                    f"3. Interquartile Range: IQR = Q_3 - Q_1 = {q3:.3f} - {q1:.3f} = {iqr:.3f}",
                    f"4. Lower Fence: Lower_Bound = Q_1 - 1.5 × IQR = {q1:.3f} - 1.5 × {iqr:.3f} = {lower:.3f}",
                    f"5. Upper Fence: Upper_Bound = Q_3 + 1.5 × IQR = {q3:.3f} + 1.5 × {iqr:.3f} = {upper:.3f}",
                    f"6. Outlier Observations Identified: Count(x_i < {lower:.3f} or x_i > {upper:.3f}) = {c_out} ({p_out:.1f}%)",
                ]

                outlier_derivations[col] = {
                    "column": col,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "outlier_count": c_out,
                    "outlier_pct": p_out,
                    "steps": o_steps,
                }

        steps = [
            f"1. Total Cells (Rows × Columns): N_cells = {tot_cells}",
            f"2. Missing Cell Count: N_missing = {missing_count} -> P_missing = ({missing_count} / {tot_cells}) × 100% = {missing_pct:.2f}%",
            f"3. Duplicate Row Count: N_dup = {dup_count} -> P_dup = ({dup_count} / {total_rows}) × 100% = {dup_pct:.2f}%",
            f"4. Columns Evaluated for Tukey IQR Outliers: {len(outlier_derivations)} numerical feature(s)",
        ]

        return {
            "title": "Data Quality & Tukey IQR Outlier Derivations",
            "metric_key": "data_quality_derivation",
            "formula_latex": r"\text{IQR} = Q_3 - Q_1, \quad \text{Lower Bound} = Q_1 - 1.5 \cdot \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \cdot \text{IQR}",
            "formula_text": "IQR = Q3 - Q1 | Lower Bound = Q1 - 1.5 * IQR | Upper Bound = Q3 + 1.5 * IQR",
            "variables": {
                "Total_Cells": {"desc": "Matrix volume (Rows * Columns)", "val": tot_cells},
                "Missing_Cells": {"desc": "Null / NaN count", "val": f"{missing_count} ({missing_pct:.2f}%)"},
                "Duplicate_Rows": {"desc": "Exact duplicate records", "val": f"{dup_count} ({dup_pct:.2f}%)"},
                "Outlier_Columns": {"desc": "Columns containing values beyond IQR fences", "val": len(outlier_derivations)},
            },
            "calculation_steps": steps,
            "outlier_columns": outlier_derivations,
            "output_value": f"Missing: {missing_pct:.1f}% | Duplicates: {dup_pct:.1f}% | Outlier Columns: {len(outlier_derivations)}",
            "unit_or_range": "[0%, 100%]",
            "interpretation": "Tukey 1.5x IQR fence rule detects extreme observations without assuming standard normal distribution.",
        }

    # =========================================================================
    # 8. CONTINUOUS CONFIDENCE ENGINE DERIVATION
    # =========================================================================
    @classmethod
    def derive_confidence_engine(cls, result: Dict[str, Any]) -> Dict[str, Any]:
        total_rows = int(result.get("total_rows", 100))
        cv = result.get("cross_validation", {})
        cv_std = float(cv.get("std_score", cv.get("std", 0.0)))
        cross_model_consistent = bool(result.get("cross_model_consistency", True))

        sample_factor = min(1.0, max(0.40, math.sqrt(total_rows) / 10.0))
        cv_penalty = 0.12 * min(1.0, max(0.0, cv_std / 0.15))
        consensus_bonus = 0.06 if cross_model_consistent else -0.06

        steps = [
            f"1. Total Sample Support: N_samples = {total_rows}",
            f"2. Continuous Sample Factor: F_sample = min(1.0, max(0.40, √{total_rows} / 10.0)) = {sample_factor:.3f}",
            f"3. Base Signal Strength: S_base = 0.35 + 0.35 × (Effect_Magnitude) + Direct_Bonus",
            f"4. Cross-Validation Variance Penalty: P_cv = 0.12 × min(1.0, {cv_std:.3f} / 0.15) = {cv_penalty:.3f}",
            f"5. Multi-Model Consensus Adjustment: Adj_consensus = {consensus_bonus:+.2f} ({'Consistent across estimators' if cross_model_consistent else 'Model-specific artifact'})",
            f"6. Continuous Synthesis: Confidence = round(min(1.0, max(0.10, Raw_Score × F_sample)), 2)",
        ]

        return {
            "title": "Continuous Diagnostic Confidence Engine",
            "metric_key": "confidence_engine",
            "formula_latex": r"\text{Confidence} = \text{round}\left( \min\left(1.0, \max\left(0.10, \left[(0.35 + 0.35 \cdot M + B_{\text{dir}} + B_{\text{sig}}) \cdot R - P_{\text{cv}} \pm \text{Adj}_{\text{consensus}}\right] \cdot \min\left(1.0, \max\left(0.40, \frac{\sqrt{N}}{10}\right)\right) \right)\right), 2\right)",
            "formula_text": "Confidence = round(min(1.0, max(0.10, Base_Score * min(1.0, max(0.40, sqrt(N) / 10)))), 2)",
            "variables": {
                "N_samples": {"desc": "Sample size support", "val": total_rows},
                "F_sample": {"desc": "Sample scaling multiplier", "val": round(sample_factor, 3)},
                "P_cv": {"desc": "Cross-validation instability deduction", "val": round(cv_penalty, 3)},
                "Adj_consensus": {"desc": "Cross-model agreement bonus", "val": round(consensus_bonus, 2)},
            },
            "calculation_steps": steps,
            "output_value": "Continuous Scale: [0.10, 1.00] (Low < 0.48, Medium 0.48-0.71, High >= 0.72)",
            "unit_or_range": "[0.10, 1.00]",
            "interpretation": "Eliminates arbitrary heuristic confidence bins. Small sample sizes or volatile cross-validation folds naturally depress confidence, while large multi-model concordant findings receive high confidence.",
        }

    # =========================================================================
    # 9. ERROR ANALYSIS & PREDICTION UNCERTAINTY DERIVATION
    # =========================================================================
    @classmethod
    def derive_error_analysis(cls, err_res: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        total_errors = int(err_res.get("total_errors", len(err_res.get("prediction_errors", []))))
        high_conf_errs = int(err_res.get("high_confidence_errors_count", len(err_res.get("high_confidence_errors", []))))

        if task_type == "classification":
            steps = [
                f"1. Prediction Error Condition: Error occurs when ŷ_i ≠ y_i",
                f"2. Prediction Confidence Margin: M_i = P(ŷ_(1)) - P(ŷ_(2)) (Difference between top predicted probability and second highest)",
                f"3. High-Confidence Error Criterion: ŷ_i ≠ y_i and P(ŷ_i) ≥ 0.70 (or M_i ≥ 0.40)",
                f"4. Total Observed Test Errors: {total_errors}",
                f"5. High-Confidence / High-Risk Errors: {high_conf_errs} ({high_conf_errs / max(1, total_errors) * 100:.1f}% of all errors)",
            ]

            return {
                "title": "Classification Error & Uncertainty Margin Derivation",
                "metric_key": "error_uncertainty",
                "formula_latex": r"M_i = P(\hat{y}_{(1)}) - P(\hat{y}_{(2)}), \quad e_i = \mathbb{I}(\hat{y}_i \neq y_i)",
                "formula_text": "Confidence Margin = P(Top_Class) - P(Runner_Up_Class) | Error = 1 if y_hat != y else 0",
                "variables": {
                    "Total_Errors": {"desc": "Misclassified test partition samples", "val": total_errors},
                    "High_Confidence_Errors": {"desc": "Errors where model probability >= 70%", "val": high_conf_errs},
                },
                "calculation_steps": steps,
                "output_value": f"Total Errors: {total_errors} (High-Confidence Overconfident Errors: {high_conf_errs})",
                "unit_or_range": "Counts: [0, N_test]",
                "interpretation": "High-confidence errors signify model overconfidence on deceptive feature regions or severe label noise.",
            }
        else:
            steps = [
                f"1. Point Prediction Residual: e_i = y_i - ŷ_i",
                f"2. Absolute Residual Magnitude: |e_i| = |y_i - ŷ_i|",
                f"3. Squared Residual: e_i² = (y_i - ŷ_i)²",
                f"4. Large-Residual Cutoff: |e_i| > 2 × RMSE",
                f"5. Total Test Predictions Evaluated: {total_errors or 'All test observations'}",
            ]

            return {
                "title": "Regression Residual & Deviation Derivation",
                "metric_key": "regression_residuals",
                "formula_latex": r"e_i = y_i - \hat{y}_i, \quad |e_i| = |y_i - \hat{y}_i|, \quad \text{Outlier Residual} = |e_i| > 2 \cdot \text{RMSE}",
                "formula_text": "Residual = y_actual - y_predicted | Absolute Error = |y - y_hat|",
                "variables": {
                    "e_i": {"desc": "Residual difference", "val": "y_i - ŷ_i"},
                    "Total_Samples": {"desc": "Test partition observations", "val": total_errors},
                },
                "calculation_steps": steps,
                "output_value": f"Residual tracking active across evaluation set",
                "unit_or_range": "Residuals in target units",
                "interpretation": "Residual dispersion patterns indicate whether error is homoscedastic (uniform) or heteroscedastic (growing with target magnitude).",
            }
