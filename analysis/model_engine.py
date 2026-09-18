# analysis/model_engine.py
"""
ModelEngine & ValidationEngine: Multi-model training, adaptive cross-validation,
and sample-size-aware generalization assessment across arbitrary classification and regression datasets.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from analysis.evidence import DiagnosticEvidence, safe_primitive


class ModelEngine:
    """
    Manages candidate estimators, adaptive cross-validation, model selection,
    and sample-size-aware generalization analysis.
    """

    @classmethod
    def get_classification_candidates(cls) -> Dict[str, Any]:
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        }

    @classmethod
    def get_regression_candidates(cls) -> Dict[str, Any]:
        return {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        }

    @classmethod
    def determine_cv_strategy(
        cls,
        task_type: str,
        y: np.ndarray,
        n_samples: int
    ) -> Tuple[Any, int]:
        """
        Adapts cross-validation partitioning dynamically to sample size and class frequencies.
        Never blindly requests more folds than the data supports.
        """
        if task_type == "classification":
            unique, counts = np.unique(y, return_counts=True)
            min_class_cnt = int(np.min(counts)) if len(counts) > 0 else 1
            if min_class_cnt >= 2:
                n_splits = max(2, min(5, min_class_cnt))
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            else:
                n_splits = max(2, min(5, n_samples))
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            n_splits = max(2, min(5, n_samples))
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        return cv, n_splits

    @classmethod
    def evaluate_and_select_model(
        cls,
        task_type: str,
        X_train: Any,
        y_train: np.ndarray,
        X_test: Any,
        y_test: np.ndarray,
        X_full: Any,
        y_full: np.ndarray
    ) -> Tuple[Dict[str, Any], List[DiagnosticEvidence]]:
        """
        Trains candidate estimators, evaluates test performance, computes CV stability,
        selects the optimal champion model, and generates generalization evidence.
        """
        candidates = (
            cls.get_classification_candidates()
            if task_type == "classification"
            else cls.get_regression_candidates()
        )

        n_full = len(y_full)
        n_train = len(y_train)
        n_test = len(y_test)
        evidence_list: List[DiagnosticEvidence] = []

        cv_strategy, cv_folds = cls.determine_cv_strategy(task_type, y_full, n_full)
        scoring_metric = "f1_weighted" if task_type == "classification" else "r2"

        comparison_list = []
        fitted_models = {}
        model_predictions = {}
        model_cv_scores = {}

        for name, model_obj in candidates.items():
            model = clone(model_obj)
            try:
                cv_scores = cross_val_score(
                    model, X_full, y_full,
                    cv=cv_strategy,
                    scoring=scoring_metric,
                    error_score="raise"
                )
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))
                model_cv_scores[name] = cv_scores
            except Exception:
                cv_mean = 0.0
                cv_std = 0.0
                cv_scores = np.array([0.0])
                model_cv_scores[name] = cv_scores

            # Fit on training split for held-out evaluation
            try:
                model.fit(X_train, y_train)
                fitted_models[name] = model

                y_pred = model.predict(X_test)
                model_predictions[name] = y_pred

                if task_type == "classification":
                    acc = float(accuracy_score(y_test, y_pred))
                    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
                    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
                    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
                    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

                    comparison_list.append({
                        "model": name,
                        "test_accuracy": round(acc, 4),
                        "test_balanced_accuracy": round(bal_acc, 4),
                        "test_precision": round(prec, 4),
                        "test_recall": round(rec, 4),
                        "test_f1_score": round(f1, 4),
                        "accuracy": round(acc, 4),
                        "balanced_accuracy": round(bal_acc, 4),
                        "f1_score": round(f1, 4),
                        "cv_mean": round(cv_mean, 4),
                        "cv_std": round(cv_std, 4),
                        "cv_scores": [round(float(s), 4) for s in cv_scores],
                        "selection_status": "Candidate",
                    })
                else:
                    mae = float(mean_absolute_error(y_test, y_pred))
                    mse = float(mean_squared_error(y_test, y_pred))
                    rmse = float(np.sqrt(mse))
                    r2 = float(r2_score(y_test, y_pred))

                    comparison_list.append({
                        "model": name,
                        "test_mae": round(mae, 4),
                        "test_mse": round(mse, 4),
                        "test_rmse": round(rmse, 4),
                        "test_r2": round(r2, 4),
                        "mae": round(mae, 4),
                        "r2": round(r2, 4),
                        "cv_mean": round(cv_mean, 4),
                        "cv_std": round(cv_std, 4),
                        "cv_scores": [round(float(s), 4) for s in cv_scores],
                        "selection_status": "Candidate",
                    })

            except Exception as exc:
                comparison_list.append({
                    "model": name,
                    "cv_mean": round(cv_mean, 4),
                    "cv_std": round(cv_std, 4),
                    "selection_status": "Failed",
                    "failure_reason": str(exc),
                })

        valid_candidates = [m for m in comparison_list if m.get("selection_status") != "Failed"]
        if not valid_candidates:
            raise RuntimeError("All candidate machine-learning models failed to train.")

        # Selection strictly based on cross-validation generalization score
        valid_candidates.sort(key=lambda x: (x["cv_mean"], -x["cv_std"]), reverse=True)
        selected_meta = valid_candidates[0]
        selected_model_name = selected_meta["model"]

        if task_type == "classification":
            criterion = f"Highest cross-validation weighted F1 across {cv_folds} folds"
            metric = "weighted_f1"
        else:
            criterion = f"Highest cross-validation R² mean across {cv_folds} folds"
            metric = "r2"

        for item in comparison_list:
            m_name = item["model"]
            item["model_id"] = m_name.lower().replace(" ", "_")
            item["model_name"] = m_name
            item["selection_score"] = round(item.get("cv_mean", 0.0), 4)
            item["selection_criterion"] = criterion

            if m_name == selected_model_name:
                item["selection_status"] = "Selected"
                item["is_selected"] = True
                item["is_best"] = True
            else:
                item["selection_status"] = "Unselected"
                item["is_selected"] = False
                item["is_best"] = False

        selected_model = fitted_models[selected_model_name]
        selected_cv_scores = model_cv_scores[selected_model_name]
        selected_cv_mean = float(np.mean(selected_cv_scores))
        selected_cv_std = float(np.std(selected_cv_scores))
        selected_cv_min = float(np.min(selected_cv_scores))
        selected_cv_max = float(np.max(selected_cv_scores))
        selected_cv_range = float(selected_cv_max - selected_cv_min)

        selected_cv_mean_canonical = round(selected_cv_mean, 4)
        selected_cv_std_canonical = round(selected_cv_std, 4)
        selected_cv_min_canonical = round(selected_cv_min, 4)
        selected_cv_max_canonical = round(selected_cv_max, 4)
        selected_cv_range_canonical = round(selected_cv_range, 4)

        if task_type == "classification":
            explanation = (
                f"Selected because it achieved the highest cross-validation weighted F1 "
                f"({selected_cv_mean_canonical:.1%} ± {selected_cv_std_canonical:.1%}) across all evaluated "
                f"candidate models, providing the strongest generalization estimate."
            )
        else:
            explanation = (
                f"Selected because it achieved the highest cross-validation R² mean "
                f"({selected_cv_mean_canonical:.4f} ± {selected_cv_std_canonical:.4f}) across all evaluated "
                f"candidate models, providing the strongest generalization estimate."
            )

        selected_slug = selected_model_name.lower().replace(" ", "_")
        model_selection = {
            "selected_model_id": selected_slug,
            "selected_model": selected_model_name,
            "selected_model_name": selected_model_name,
            "selection_status": "Selected",
            "criterion": criterion,
            "selection_criterion": criterion,
            "metric": metric,
            "selected_score": selected_cv_mean_canonical,
            "selection_score": selected_cv_mean_canonical,
            "selection_basis": "cross_validation",
            "selection_explanation": explanation,
            "tie_break_used": None,
            "cv_mean": selected_cv_mean_canonical,
            "cv_std": selected_cv_std_canonical,
        }

        # --------------------------------------------------------
        # Sample-Size-Aware Generalization Analysis
        # --------------------------------------------------------
        train_pred = selected_model.predict(X_train)
        is_small_sample = (n_full < 40 or n_test < 12)

        if task_type == "classification":
            train_score = float(accuracy_score(y_train, train_pred))
            eval_score = float(selected_meta["test_accuracy"])
            score_gap = train_score - eval_score
            train_to_cv_gap = train_score - selected_cv_mean_canonical

            if score_gap >= 0.20 and train_to_cv_gap >= 0.15:
                if is_small_sample:
                    overfit_status = "Generalization Divergence (Constrained Sample Size)"
                    overfit_expl = (
                        f"Training accuracy ({train_score:.1%}) notably exceeds held-out test accuracy "
                        f"({eval_score:.1%}) by {score_gap:.1%} and cross-validation mean ({selected_cv_mean_canonical:.1%}) "
                        f"by {train_to_cv_gap:.1%}; evaluation sample size ({n_test} test rows) widens uncertainty."
                    )
                else:
                    overfit_status = "Substantial Training-Evaluation Generalization Gap"
                    overfit_expl = (
                        f"Training accuracy ({train_score:.1%}) substantially exceeds both held-out test accuracy "
                        f"({eval_score:.1%}) by {score_gap:.1%} and cross-validation mean ({selected_cv_mean_canonical:.1%}) "
                        f"by {train_to_cv_gap:.1%}."
                    )
            elif score_gap >= 0.10 or train_to_cv_gap >= 0.10:
                overfit_status = "Moderate Training-Evaluation Performance Gap"
                overfit_expl = (
                    f"Training accuracy ({train_score:.1%}) moderately exceeds test accuracy ({eval_score:.1%}) "
                    f"by {score_gap:.1%}."
                )
            else:
                overfit_status = "No Material Generalization Problem"
                overfit_expl = f"No material generalization problem detected between training ({train_score:.1%}) and evaluation partitions ({eval_score:.1%})."

            if selected_cv_std_canonical > 0.12 or selected_cv_range_canonical > 0.28:
                stab_status = "VARIABLE"
                stab_expl = (
                    f"Cross-validation fold scores exhibit notable variation "
                    f"({selected_cv_min_canonical:.1%} to {selected_cv_max_canonical:.1%}, std: {selected_cv_std_canonical:.1%}), "
                    f"indicating sensitivity to partition boundaries."
                )
            elif selected_cv_std_canonical > 0.05:
                stab_status = "MODERATE"
                stab_expl = (
                    f"Cross-validation fold scores range from {selected_cv_min_canonical:.1%} to {selected_cv_max_canonical:.1%} "
                    f"(mean: {selected_cv_mean_canonical:.1%}, std: {selected_cv_std_canonical:.1%})."
                )
            else:
                stab_status = "STABLE"
                stab_expl = f"Consistent performance across all {cv_folds} cross-validation folds (std: {selected_cv_std_canonical:.1%})."

        else:
            train_score = float(r2_score(y_train, train_pred))
            eval_score = float(selected_meta["test_r2"])
            score_gap = train_score - eval_score
            train_to_cv_gap = train_score - selected_cv_mean_canonical

            if score_gap >= 0.20 and train_to_cv_gap >= 0.15:
                if is_small_sample:
                    overfit_status = "Generalization Divergence (Constrained Sample Size)"
                    overfit_expl = (
                        f"Training R² ({train_score:.4f}) notably exceeds test R² ({eval_score:.4f}) by {score_gap:.4f} "
                        f"and cross-validation mean ({selected_cv_mean_canonical:.4f}) by {train_to_cv_gap:.4f}; "
                        f"evaluation sample size ({n_test} test rows) limits statistical certainty."
                    )
                else:
                    overfit_status = "Substantial Training-Evaluation Generalization Gap"
                    overfit_expl = (
                        f"Training R² ({train_score:.4f}) substantially exceeds held-out test R² "
                        f"({eval_score:.4f}) by {score_gap:.4f} and cross-validation mean ({selected_cv_mean_canonical:.4f}) "
                        f"by {train_to_cv_gap:.4f}."
                    )
            elif score_gap >= 0.10 or train_to_cv_gap >= 0.10:
                overfit_status = "Moderate Training-Evaluation Performance Gap"
                overfit_expl = f"Training R² ({train_score:.4f}) moderately exceeds test R² ({eval_score:.4f}) by {score_gap:.4f}."
            else:
                overfit_status = "No Material Generalization Problem"
                overfit_expl = f"No material generalization problem detected between training ({train_score:.4f}) and evaluation R² ({eval_score:.4f})."

            if selected_cv_std_canonical > 0.15 or selected_cv_range_canonical > 0.35:
                stab_status = "VARIABLE"
                stab_expl = (
                    f"Cross-validation R² exhibits notable variation across folds "
                    f"({selected_cv_min_canonical:.4f} to {selected_cv_max_canonical:.4f}, std: {selected_cv_std_canonical:.4f})."
                )
            elif selected_cv_std_canonical > 0.05:
                stab_status = "MODERATE"
                stab_expl = f"Cross-validation R² ranges from {selected_cv_min_canonical:.4f} to {selected_cv_max_canonical:.4f} (std: {selected_cv_std_canonical:.4f})."
            else:
                stab_status = "STABLE"
                stab_expl = f"Consistent R² across all {cv_folds} folds (std: {selected_cv_std_canonical:.4f})."

        # Emit Generalization Evidence
        if score_gap >= 0.08:
            gap_fmt = f"{score_gap:.1%}" if task_type == "classification" else f"{score_gap:.4f}"
            evidence_list.append(DiagnosticEvidence(
                evidence_id="generalization_score_gap",
                signal_name=f"Training vs Evaluation Gap ({gap_fmt})",
                domain="generalization",
                metric="train_test_score_gap",
                observed_value=round(score_gap, 4),
                baseline_value=0.0,
                magnitude=min(1.0, max(0.15, score_gap / 0.40)),
                direction="gap",
                sample_support=n_full,
                affected_population="Held-out evaluation partition",
                signal_type="behavioral",
                strength="HIGH" if (score_gap >= 0.22 and not is_small_sample) else "MODERATE",
                reliability=0.65 if is_small_sample else 0.90,
                model_scope="selected_model",
                context=overfit_expl
            ))

        # Emit Partition Stability Evidence
        if selected_cv_std_canonical >= 0.05:
            evidence_list.append(DiagnosticEvidence(
                evidence_id="cross_validation_fold_variance",
                signal_name=f"Cross-Validation Partition Variance (std = {selected_cv_std_canonical:.4f})",
                domain="partition_stability",
                metric="cv_fold_std",
                observed_value=selected_cv_std_canonical,
                baseline_value=0.0,
                magnitude=min(1.0, selected_cv_std_canonical / 0.20),
                direction="variation",
                sample_support=n_full,
                affected_population=f"All {cv_folds} cross-validation folds",
                signal_type="structural",
                strength="HIGH" if selected_cv_std_canonical >= 0.12 else "MODERATE",
                reliability=0.90,
                model_scope="selected_model",
                context=stab_expl
            ))

        # Emit Evaluation Performance Deficit Evidence if sub-optimal
        if task_type == "classification":
            acc_eval = float(selected_meta["test_accuracy"])
            f1_eval = float(selected_meta["test_f1_score"])
            if (acc_eval + f1_eval) / 2.0 < 0.90:
                evidence_list.append(DiagnosticEvidence(
                    evidence_id="eval_performance_deficit",
                    signal_name=f"Evaluation Performance Deficit (Acc: {acc_eval:.1%}, F1: {f1_eval:.1%})",
                    domain="performance",
                    metric="eval_performance_metric",
                    observed_value=round((acc_eval + f1_eval) / 2.0, 4),
                    baseline_value=0.90,
                    magnitude=min(1.0, (0.90 - (acc_eval + f1_eval) / 2.0) / 0.90),
                    direction="deficit",
                    sample_support=n_test,
                    affected_population=f"Held-out test partition ({n_test} samples)",
                    signal_type="behavioral",
                    strength="HIGH" if ((acc_eval + f1_eval) / 2.0 < 0.70) else "MODERATE",
                    reliability=0.90,
                    model_scope="selected_model",
                    context=f"Held-out test accuracy is {acc_eval:.1%} and weighted F1 is {f1_eval:.1%}."
                ))
            else:
                evidence_list.append(DiagnosticEvidence(
                    evidence_id="healthy_baseline_classification",
                    signal_name=f"Optimal Baseline Classification Performance (Acc: {acc_eval:.1%}, F1: {f1_eval:.1%})",
                    domain="generalization",
                    metric="eval_performance_metric",
                    observed_value=round((acc_eval + f1_eval) / 2.0, 4),
                    baseline_value=0.90,
                    magnitude=0.10,
                    direction="optimal",
                    sample_support=n_test,
                    affected_population=f"Held-out test partition ({n_test} samples)",
                    signal_type="behavioral",
                    strength="HIGH",
                    reliability=0.95,
                    model_scope="selected_model",
                    context=f"Held-out test accuracy is {acc_eval:.1%} and weighted F1 is {f1_eval:.1%}."
                ))
        else:
            r2_eval = float(selected_meta["test_r2"])
            if r2_eval < 0.85:
                evidence_list.append(DiagnosticEvidence(
                    evidence_id="regression_r2_deficit",
                    signal_name=f"Regression Explanatory Power Deficit (R² = {r2_eval:.4f})",
                    domain="performance",
                    metric="r2_score",
                    observed_value=round(r2_eval, 4),
                    baseline_value=0.85,
                    magnitude=min(1.0, max(0.0, (0.85 - r2_eval) / 0.85)),
                    direction="deficit",
                    sample_support=n_test,
                    affected_population=f"Held-out test partition ({n_test} samples)",
                    signal_type="behavioral",
                    strength="HIGH" if r2_eval < 0.50 else "MODERATE",
                    reliability=0.90,
                    model_scope="selected_model",
                    context=f"Held-out test R² is {r2_eval:.4f} (explains {max(0.0, r2_eval):.1%} of target variance)."
                ))
            else:
                evidence_list.append(DiagnosticEvidence(
                    evidence_id="healthy_baseline_regression",
                    signal_name=f"Optimal Baseline Regression Performance (R² = {r2_eval:.4f})",
                    domain="generalization",
                    metric="r2_score",
                    observed_value=round(r2_eval, 4),
                    baseline_value=0.85,
                    magnitude=0.10,
                    direction="optimal",
                    sample_support=n_test,
                    affected_population=f"Held-out test partition ({n_test} samples)",
                    signal_type="behavioral",
                    strength="HIGH",
                    reliability=0.95,
                    model_scope="selected_model",
                    context=f"Held-out test R² is {r2_eval:.4f} (explains {max(0.0, r2_eval):.1%} of target variance)."
                ))

        # Emit Small Sample Size Evidence
        if n_full < 60:
            evidence_list.append(DiagnosticEvidence(
                evidence_id="small_sample_size_constraint",
                signal_name=f"Constrained Sample Size ({n_full} observations)",
                domain="sample_size",
                metric="total_observation_count",
                observed_value=n_full,
                baseline_value=100,
                magnitude=min(1.0, (60 - n_full) / 60.0),
                direction="limited_sample",
                sample_support=n_full,
                affected_population=f"Entire dataset ({n_full} rows, {n_test} in test split)",
                signal_type="structural",
                strength="HIGH" if n_full < 30 else "MODERATE",
                reliability=1.0,
                model_scope="dataset",
                context=f"Dataset contains {n_full} usable rows ({n_test} in evaluation split), resulting in wider statistical uncertainty intervals."
            ))

        model_stability = {
            "stability_status": stab_status,
            "stability_explanation": stab_expl,
            "overfitting_diagnostic": overfit_status,
            "overfitting_explanation": overfit_expl,
            "training_score": round(train_score, 4),
            "evaluation_score": round(eval_score, 4),
            "overfitting_gap": round(score_gap, 4),
            "cv_std": selected_cv_std_canonical,
        }

        cross_validation = {
            "fold_scores": [round(float(s), 4) for s in selected_cv_scores],
            "average_score": selected_cv_mean_canonical,
            "cross_validation_mean": selected_cv_mean_canonical,
            "standard_deviation": selected_cv_std_canonical,
            "cv_min": selected_cv_min_canonical,
            "cv_max": selected_cv_max_canonical,
            "cv_range": selected_cv_range_canonical,
            "scoring_metric": scoring_metric,
            "num_folds": cv_folds,
        }

        result_payload = {
            "selected_model": selected_model,
            "selected_model_name": selected_model_name,
            "model_selection": safe_primitive(model_selection),
            "model_comparison": safe_primitive(comparison_list),
            "model_stability": safe_primitive(model_stability),
            "cross_validation": safe_primitive(cross_validation),
            "fitted_models": fitted_models,
            "selected_meta": safe_primitive(selected_meta),
        }

        return result_payload, evidence_list
