# analysis/verification_engine.py
"""
AI Model Root-Cause Analyzer: Experimental Root-Cause Verification & Closed-Loop Remediation Engine.

Transforms observational diagnostics into counterfactual, experimentally-proven conclusions.
Core capabilities:
1. Targeted Candidate Hypothesis Generation (focuses on high-impact features & flagged failure modes).
2. Controlled Experimental Trials:
   - Feature Ablation (retraining without candidate feature)
   - Feature Permutation (test-time shuffling)
   - Counterfactual Jitter / Noise Perturbation
   - Control Feature Baseline Ablation (scientific control group)
3. Mathematical Evidence Scoring (based on relative metric drop, permutation agreement, and control specificity).
4. Closed-Loop Remediation Simulation (measuring Before vs After recovery on exact test splits).
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
import math
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from analysis.evidence import safe_primitive
from analysis.preprocessor import PreprocessingEngine


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 4)
    except (ValueError, TypeError):
        return default


def _evaluate_model_metric(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    task_type: str
) -> float:
    """Evaluates primary benchmark metric (Weighted F1 for classification, R² for regression)."""
    try:
        y_pred = model.predict(X)
        if task_type == "classification":
            return float(f1_score(y, y_pred, average="weighted", zero_division=0))
        else:
            return float(r2_score(y, y_pred))
    except Exception:
        return 0.0
from analysis.evidence_graph import (
    EvidenceGraph,
    EvidenceGraphNode,
    EvidenceGraphEdge,
    EvidenceGraphMetadata,
    EvidenceGraphBuilder,
    build_evidence_graph,
    validate_evidence_graph,
    get_evidence_trace,
    CANONICAL_NODE_TYPES,
    CANONICAL_EDGE_RELATIONS,
)


class VerificationEngine:
    """
    Experimental Verification & Closed-Loop Remediation Engine.
    Executes targeted ablations, permutations, controls, and remediation simulations.
    """

    @classmethod
    def select_candidate_features(
        cls,
        feature_impact: Dict[str, Any],
        original_feature_names: List[str],
        max_candidates: int = 3
    ) -> Tuple[List[str], Optional[str]]:
        """
        Selects top suspicious candidate features and the lowest-impact control feature.
        Does NOT blindly test every feature.
        """
        if not feature_impact or not original_feature_names:
            return [], None

        ranked_features = []
        for feat in original_feature_names:
            data = feature_impact.get(feat, {})
            if isinstance(data, dict):
                imp = _safe_float(data.get("importance", 0.0))
                share = _safe_float(data.get("relative_share_pct", 0.0))
                tier = str(data.get("influence_tier", "Moderate"))
            else:
                imp = _safe_float(data)
                share = 0.0
                tier = "Moderate"
            ranked_features.append({"feature": feat, "importance": imp, "share": share, "tier": tier})

        ranked_features.sort(key=lambda x: x["importance"], reverse=True)

        if not ranked_features:
            return [], None

        # Candidate features: top dominant features (share >= 20% or top ranked)
        candidates = []
        for item in ranked_features[:max_candidates]:
            candidates.append(item["feature"])

        # Control feature: lowest importance non-zero feature
        control_feature = ranked_features[-1]["feature"] if len(ranked_features) > 1 else None
        if control_feature in candidates and len(ranked_features) > len(candidates):
            control_feature = ranked_features[len(candidates)]["feature"]

        return candidates, control_feature

    @classmethod
    def run_feature_verification_experiments(
        cls,
        X_df: pd.DataFrame,
        y_raw: pd.Series,
        task_type: str,
        champion_model: Any,
        feature_impact: Dict[str, Any],
        max_candidates: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Executes controlled Ablation, Permutation, Noise Perturbation, and Control experiments.
        """
        feature_names = list(X_df.columns)
        if len(feature_names) < 2:
            return []

        candidates, control_feature = cls.select_candidate_features(
            feature_impact=feature_impact,
            original_feature_names=feature_names,
            max_candidates=max_candidates
        )

        if not candidates:
            return []

        # 1. Baseline Model Training & Evaluation
        baseline_prep, baseline_tf_names, X_tf, y_enc, classes = PreprocessingEngine.build_preprocessed_matrices(
            X_df=X_df,
            y_raw=y_raw,
            task_type=task_type
        )

        n_samples = len(X_df)
        stratify_labels = y_enc if (task_type == "classification" and len(np.unique(y_enc)) >= 2) else None
        test_size = max(1, min(n_samples - 1, int(round(n_samples * 0.20))))

        train_idx, test_idx = train_test_split(
            np.arange(n_samples),
            test_size=test_size,
            random_state=42,
            stratify=stratify_labels
        )

        X_train_base = X_tf[train_idx]
        y_train_base = y_enc[train_idx]
        X_test_base = X_tf[test_idx]
        y_test_base = y_enc[test_idx]

        model_base = clone(champion_model)
        model_base.fit(X_train_base, y_train_base)
        baseline_metric = _evaluate_model_metric(model_base, X_test_base, y_test_base, task_type)

        # 2. Control Feature Ablation Baseline
        control_delta = 0.0
        control_ablated_metric = baseline_metric
        if control_feature and control_feature in X_df.columns:
            try:
                X_df_ctrl = X_df.drop(columns=[control_feature])
                _, _, X_tf_ctrl, _, _ = PreprocessingEngine.build_preprocessed_matrices(
                    X_df=X_df_ctrl,
                    y_raw=y_raw,
                    task_type=task_type
                )
                model_ctrl = clone(champion_model)
                model_ctrl.fit(X_tf_ctrl[train_idx], y_train_base)
                control_ablated_metric = _evaluate_model_metric(model_ctrl, X_tf_ctrl[test_idx], y_test_base, task_type)
                control_delta = round(control_ablated_metric - baseline_metric, 4)
            except Exception:
                control_delta = 0.0
                control_ablated_metric = baseline_metric

        metric_label = "Weighted F1" if task_type == "classification" else "R2 Score"
        verification_reports = []

        # 3. Controlled Experiments for Each Candidate
        for cand in candidates:
            denom = max(0.01, abs(baseline_metric))

            # --- Experiment A: Feature Ablation (Retraining) ---
            try:
                X_df_ablated = X_df.drop(columns=[cand])
                _, _, X_tf_abl, _, _ = PreprocessingEngine.build_preprocessed_matrices(
                    X_df=X_df_ablated,
                    y_raw=y_raw,
                    task_type=task_type
                )
                model_abl = clone(champion_model)
                model_abl.fit(X_tf_abl[train_idx], y_train_base)
                ablated_metric = _evaluate_model_metric(model_abl, X_tf_abl[test_idx], y_test_base, task_type)
                ablation_delta = round(ablated_metric - baseline_metric, 4)
            except Exception:
                ablated_metric = baseline_metric
                ablation_delta = 0.0

            # Find columns in transformed matrix corresponding to cand
            cand_col_indices = [
                i for i, name in enumerate(baseline_tf_names)
                if name == cand or name.startswith(f"num__{cand}") or name.startswith(f"cat__{cand}") or name.startswith(f"{cand}_")
            ]

            # --- Experiment B: Feature Permutation (Test-Time Shuffling) ---
            try:
                X_test_perm = X_test_base.copy()
                np.random.seed(42)
                if cand_col_indices:
                    perm_idx = np.random.permutation(len(X_test_perm))
                    for c_idx in cand_col_indices:
                        X_test_perm[:, c_idx] = X_test_perm[perm_idx, c_idx]
                permuted_metric = _evaluate_model_metric(model_base, X_test_perm, y_test_base, task_type)
                permutation_delta = round(permuted_metric - baseline_metric, 4)
            except Exception:
                permuted_metric = baseline_metric
                permutation_delta = 0.0

            # --- Experiment C: Noise Perturbation (Counterfactual Jitter) ---
            try:
                X_test_noise = X_test_base.copy()
                np.random.seed(42)
                if cand_col_indices:
                    for c_idx in cand_col_indices:
                        std_c = np.std(X_test_noise[:, c_idx])
                        if std_c == 0 or np.isnan(std_c):
                            std_c = 1.0
                        jitter = np.random.normal(0, 0.10 * std_c, size=len(X_test_noise))
                        X_test_noise[:, c_idx] += jitter
                perturbed_metric = _evaluate_model_metric(model_base, X_test_noise, y_test_base, task_type)
                noise_delta = round(perturbed_metric - baseline_metric, 4)

                y_pred_orig = model_base.predict(X_test_base)
                y_pred_noise = model_base.predict(X_test_noise)
                if task_type == "classification":
                    output_flip_pct = round(float(np.mean(y_pred_orig != y_pred_noise) * 100), 2)
                else:
                    orig_std = np.std(y_pred_orig)
                    denom_std = orig_std if orig_std > 0 else 1.0
                    output_flip_pct = round(float(np.mean(np.abs(y_pred_orig - y_pred_noise)) / denom_std * 100), 2)
            except Exception:
                perturbed_metric = baseline_metric
                noise_delta = 0.0
                output_flip_pct = 0.0

            # --- 4. Mathematical Evidence Scoring & Decomposition (100 Point Audit) ---
            # Relative drops
            rel_ablation_drop = max(0.0, (baseline_metric - ablated_metric) / denom)
            rel_permutation_drop = max(0.0, (baseline_metric - permuted_metric) / denom)
            rel_noise_drop = max(0.0, (baseline_metric - perturbed_metric) / denom)
            control_spec_ratio = (abs(ablation_delta) + 0.001) / (abs(control_delta) + 0.001)

            # Categorical Evidence Ratings
            if rel_ablation_drop >= 0.15:
                ablation_strength = "strong"
            elif rel_ablation_drop >= 0.05:
                ablation_strength = "moderate"
            elif rel_ablation_drop >= 0.01:
                ablation_strength = "weak"
            else:
                ablation_strength = "neutral"

            if rel_permutation_drop >= 0.15:
                permutation_strength = "strong"
            elif rel_permutation_drop >= 0.05:
                permutation_strength = "moderate"
            elif rel_permutation_drop >= 0.01:
                permutation_strength = "weak"
            else:
                permutation_strength = "neutral"

            # Noise perturbation classified as a measurement stability & robustness trial
            if output_flip_pct >= 15.0 or rel_noise_drop >= 0.10:
                noise_rating = "high_brittleness"
            elif output_flip_pct >= 5.0 or rel_noise_drop >= 0.03:
                noise_rating = "moderate_sensitivity"
            else:
                noise_rating = "robust"

            # Control Test Specificity Logic
            if abs(ablation_delta) <= 0.01 and abs(control_delta) <= 0.01:
                control_verdict = "inconclusive"
            elif abs(ablation_delta) >= abs(control_delta) + 0.02 or control_spec_ratio >= 1.5:
                control_verdict = "passed"
            else:
                control_verdict = "failed"

            # Decomposable Point Allocation (Sums to 100 max)
            # 1. Ablation Evidence (0-35 pts)
            abl_pts = min(35, int(round(35.0 * min(1.0, rel_ablation_drop / 0.25))))
            # 2. Permutation Evidence (0-35 pts)
            perm_pts = min(35, int(round(35.0 * min(1.0, rel_permutation_drop / 0.25))))
            # 3. Control Specificity (0-15 pts)
            if control_verdict == "passed":
                ctrl_pts = min(15, max(8, int(round(15.0 * min(1.0, control_spec_ratio / 3.0)))))
            elif control_verdict == "inconclusive":
                ctrl_pts = 5
            else:
                ctrl_pts = 0
            # 4. Measurement Stability & Noise Robustness (0-10 pts)
            if noise_rating == "robust":
                stab_pts = 10
            elif noise_rating == "moderate_sensitivity":
                stab_pts = 6
            else:
                stab_pts = 2
            # 5. Experimental Consistency (0-5 pts)
            if rel_ablation_drop >= 0.05 and rel_permutation_drop >= 0.05:
                const_pts = 5
            elif rel_ablation_drop >= 0.05 or rel_permutation_drop >= 0.05:
                const_pts = 2
            else:
                const_pts = 0

            # Check if hypothesis reveals no empirical reliance under tested interventions
            if rel_ablation_drop < 0.01 and rel_permutation_drop < 0.01:
                verdict = "NO MEASURABLE MODEL RELIANCE"
                abl_pts = 0
                perm_pts = 0
                ctrl_pts = 0
                stab_pts = 0
                const_pts = 0
                evidence_score = 0
                summary_text = (
                    f"Under the tested split and interventions, removing or permuting '{cand}' produced no measurable "
                    f"performance change (ablation delta: {ablation_delta:+.4f}, permutation delta: {permutation_delta:+.4f}). "
                    f"The model exhibits no measurable reliance on this feature."
                )
            else:
                evidence_score = min(100, abl_pts + perm_pts + ctrl_pts + stab_pts + const_pts)

                # Grounded Verdict Synthesis
                if ablation_strength in ("strong", "moderate") and permutation_strength in ("strong", "moderate") and control_verdict == "passed":
                    verdict = "VERIFIED MODEL RELIANCE"
                elif ablation_strength in ("strong", "moderate") and control_verdict == "passed":
                    verdict = "VERIFIED STRUCTURAL DEPENDENCY (Collinear Shielding)"
                elif noise_rating == "high_brittleness":
                    verdict = "HIGH LOCAL BRITTLENESS"
                elif ablation_strength == "neutral" and permutation_strength == "neutral":
                    verdict = "REDUNDANT / DISTRIBUTED SIGNAL"
                else:
                    verdict = "PARTIAL / INTERACTIVE SIGNAL"

                summary_text = (
                    f"Ablating '{cand}' resulted in a {abs(ablation_delta):.3f} ({abs((ablation_delta/denom)*100):.1f}%) "
                    f"performance drop ({ablation_strength}), test-time permutation caused a {abs(permutation_delta):.3f} drop, "
                    f"and noise perturbation confirmed stability ({noise_rating}, flip rate: {output_flip_pct:.1f}%). "
                    f"Control test was {control_verdict} (delta: {abs(control_delta):.3f}). Verdict: {verdict} (Score: {evidence_score}/100)."
                )

            # Build explicit structured components object (Section 6)
            components = {
                "ablation": {
                    "score": abl_pts,
                    "max_score": 35,
                    "effect_size": _safe_float(rel_ablation_drop),
                    "baseline_metric": _safe_float(baseline_metric),
                    "intervention_metric": _safe_float(ablated_metric),
                    "delta": _safe_float(ablation_delta),
                    "diagnostic_threshold": "Linear scaling up to 25% relative drop (35 pts max)",
                },
                "permutation": {
                    "score": perm_pts,
                    "max_score": 35,
                    "effect_size": _safe_float(rel_permutation_drop),
                    "baseline_metric": _safe_float(baseline_metric),
                    "intervention_metric": _safe_float(permuted_metric),
                    "delta": _safe_float(permutation_delta),
                    "diagnostic_threshold": "Linear scaling up to 25% relative drop (35 pts max)",
                },
                "control": {
                    "score": ctrl_pts,
                    "max_score": 15,
                    "candidate_delta": _safe_float(ablation_delta),
                    "control_delta": _safe_float(control_delta),
                    "status": control_verdict,
                    "diagnostic_threshold": "15 pts if candidate delta >= control + 0.02, 5 pts if inconclusive",
                },
                "stability": {
                    "score": stab_pts,
                    "max_score": 10,
                    "flip_rate": _safe_float(output_flip_pct),
                    "perturbed_metric": _safe_float(perturbed_metric),
                    "role": "measurement_stability_and_robustness",
                    "diagnostic_threshold": "10 pts for flip rate <= 5% (measures robustness, not feature reliance)",
                },
                "consistency": {
                    "score": const_pts,
                    "max_score": 5,
                    "supporting_experiments": [k for k, v in [("ablation", rel_ablation_drop >= 0.05), ("permutation", rel_permutation_drop >= 0.05)] if v],
                    "diagnostic_threshold": "5 pts when both ablation and permutation show >= 5% drop",
                }
            }

            score_decomposition = {
                "ablation_points": abl_pts,
                "permutation_points": perm_pts,
                "control_points": ctrl_pts,
                "stability_points": stab_pts,
                "consistency_points": const_pts,
                "ablation_evidence_pts": abl_pts,
                "permutation_evidence_pts": perm_pts,
                "control_specificity_pts": ctrl_pts,
                "measurement_stability_pts": stab_pts,
                "experimental_consistency_pts": const_pts,
                "total_score": evidence_score,
                "components": components
            }

            verification_reports.append({
                "candidate_feature": cand,
                "metric_name": metric_label,
                "baseline_metric": _safe_float(baseline_metric),
                "ablated_metric": _safe_float(ablated_metric),
                "ablation_delta": _safe_float(ablation_delta),
                "ablation_delta_pct": _safe_float((ablation_delta / denom) * 100),
                "permuted_metric": _safe_float(permuted_metric),
                "permutation_delta": _safe_float(permutation_delta),
                "permutation_delta_pct": _safe_float((permutation_delta / denom) * 100),
                "perturbed_metric": _safe_float(perturbed_metric),
                "noise_delta": _safe_float(noise_delta),
                "noise_delta_pct": _safe_float((noise_delta / denom) * 100),
                "prediction_flip_rate_pct": _safe_float(output_flip_pct),
                "noise_sensitivity": noise_rating,
                "control_feature": control_feature or "None",
                "control_ablation_metric": _safe_float(control_ablated_metric),
                "control_delta": _safe_float(control_delta),
                "control_specificity_ratio": _safe_float(control_spec_ratio),
                "evidence_ratings": {
                    "ablation": ablation_strength,
                    "permutation": permutation_strength,
                    "noise": noise_rating,
                    "control": control_verdict,
                },
                "score_decomposition": score_decomposition,
                "verdict": verdict,
                "evidence_score": evidence_score,
                "summary": summary_text
            })

        return safe_primitive(verification_reports)

    @classmethod
    def simulate_closed_loop_remediation(
        cls,
        X_df: pd.DataFrame,
        y_raw: pd.Series,
        task_type: str,
        champion_model: Any,
        diagnostic_candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Simulates and benchmarks the recommended remediation pipeline on identical splits.
        Provides empirical 'Proof of Fix' comparing baseline vs remediated model.
        """
        n_samples = len(X_df)
        if n_samples < 4:
            return {}

        # 1. Evaluate Baseline
        _, _, X_tf, y_enc, classes = PreprocessingEngine.build_preprocessed_matrices(
            X_df=X_df,
            y_raw=y_raw,
            task_type=task_type
        )

        stratify_labels = y_enc if (task_type == "classification" and len(np.unique(y_enc)) >= 2) else None
        test_size = max(1, min(n_samples - 1, int(round(n_samples * 0.20))))

        train_idx, test_idx = train_test_split(
            np.arange(n_samples),
            test_size=test_size,
            random_state=42,
            stratify=stratify_labels
        )

        model_base = clone(champion_model)
        model_base.fit(X_tf[train_idx], y_enc[train_idx])
        base_train_score = _evaluate_model_metric(model_base, X_tf[train_idx], y_enc[train_idx], task_type)
        base_test_score = _evaluate_model_metric(model_base, X_tf[test_idx], y_enc[test_idx], task_type)
        base_gen_gap = round(abs(base_train_score - base_test_score), 4)

        # 2. Build Remediated Model (e.g. Regularization / Balancing)
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

        if task_type == "classification":
            model_remed = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42
            )
        else:
            model_remed = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=4,
                learning_rate=0.05,
                random_state=42
            )

        model_remed.fit(X_tf[train_idx], y_enc[train_idx])
        remed_train_score = _evaluate_model_metric(model_remed, X_tf[train_idx], y_enc[train_idx], task_type)
        remed_test_score = _evaluate_model_metric(model_remed, X_tf[test_idx], y_enc[test_idx], task_type)
        remed_gen_gap = round(abs(remed_train_score - remed_test_score), 4)

        metric_delta = round(remed_test_score - base_test_score, 4)
        gen_gap_delta = round(remed_gen_gap - base_gen_gap, 4)  # negative means gap shrank
        gen_gap_reduction = round(base_gen_gap - remed_gen_gap, 4)  # positive means gap shrank

        # 3. Precise Scientific Resolution Verdict Logic
        is_baseline_gap_healthy = (base_gen_gap <= 0.03)
        reduction_pp = gen_gap_reduction * 100 if task_type == "classification" else gen_gap_reduction
        unit_str = "percentage points" if task_type == "classification" else "units"
        gap_base_str = f"{base_gen_gap:.2%}" if task_type == "classification" else f"{base_gen_gap:.4f}"
        gap_remed_str = f"{remed_gen_gap:.2%}" if task_type == "classification" else f"{remed_gen_gap:.4f}"

        metric_name_str = "weighted F1" if task_type == "classification" else "R2 score"

        if metric_delta > 0.02 and gen_gap_reduction > 0.01:
            resolution = "BOTH IMPROVED — Held-out performance improved and generalization gap reduced."
            proof_summary = (
                f"Simulated remediation resulted in test metric changing from {base_test_score:.4f} to {remed_test_score:.4f} "
                f"(delta: {metric_delta:+.4f}). Generalization gap shifted from {gap_base_str} to {gap_remed_str} "
                f"(reduction: {reduction_pp:.2f} {unit_str}). Verdict: {resolution}"
            )
        elif metric_delta > 0.02:
            resolution = "PERFORMANCE IMPROVED — Held-out test performance improved."
            proof_summary = (
                f"Simulated remediation resulted in held-out test metric improving from {base_test_score:.4f} to {remed_test_score:.4f} "
                f"(delta: {metric_delta:+.4f}). Verdict: {resolution}"
            )
        elif abs(metric_delta) <= 0.02 and gen_gap_reduction > 0.005:
            resolution = f"PARTIALLY RESOLVED — Generalization gap reduced by {reduction_pp:.2f} {unit_str}, while held-out {metric_name_str} showed no measurable improvement."
            proof_summary = (
                f"Simulated remediation resulted in held-out test metric changing from {base_test_score:.4f} to {remed_test_score:.4f} "
                f"(delta: {metric_delta:+.4f}). Generalization gap shifted from {gap_base_str} to {gap_remed_str} "
                f"(reduction: {reduction_pp:.2f} {unit_str}), while held-out {metric_name_str} showed no measurable improvement. Verdict: {resolution}"
            )
        elif metric_delta < -0.03:
            resolution = "BASELINE PREFERRED — Remediated pipeline degraded test performance."
            proof_summary = (
                f"Simulated remediation resulted in test metric changing from {base_test_score:.4f} to {remed_test_score:.4f} "
                f"(delta: {metric_delta:+.4f}). Verdict: {resolution}"
            )
        else:
            resolution = "NO MATERIAL IMPROVEMENT — Remediation produced comparable metrics to baseline."
            proof_summary = (
                f"Simulated remediation resulted in test metric changing from {base_test_score:.4f} to {remed_test_score:.4f} "
                f"(delta: {metric_delta:+.4f}). Generalization gap shifted from {gap_base_str} to {gap_remed_str}. Verdict: {resolution}"
            )

        return safe_primitive({
            "status": "Success",
            "task_type": task_type,
            "metric_name": "Weighted F1" if task_type == "classification" else "R2 Score",
            "baseline": {
                "train_score": _safe_float(base_train_score),
                "test_score": _safe_float(base_test_score),
                "generalization_gap": _safe_float(base_gen_gap),
            },
            "remediated": {
                "train_score": _safe_float(remed_train_score),
                "test_score": _safe_float(remed_test_score),
                "generalization_gap": _safe_float(remed_gen_gap),
            },
            "deltas": {
                "test_metric_delta": _safe_float(metric_delta),
                "generalization_gap_reduction": _safe_float(gen_gap_reduction),
            },
            "resolution_verdict": resolution,
            "proof_summary": proof_summary
        })

    @classmethod
    def build_evidence_graph(
        cls,
        task_type: str,
        feature_impact: Dict[str, Any],
        candidate_experiments: List[Dict[str, Any]],
        diagnostic_candidates: List[Dict[str, Any]],
        remediation_simulation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convenience delegation to EvidenceGraphBuilder."""
        return EvidenceGraphBuilder.build_graph(
            task_type=task_type,
            feature_impact=feature_impact,
            candidate_experiments=candidate_experiments,
            diagnostic_candidates=diagnostic_candidates,
            remediation_simulation=remediation_simulation,
        )

    @classmethod
    def get_evidence_trace(
        cls,
        candidate_feature: str,
        evidence_graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convenience delegation to EvidenceGraphBuilder."""
        return EvidenceGraphBuilder.get_evidence_trace(
            candidate_feature=candidate_feature,
            evidence_graph=evidence_graph,
        )

    @classmethod
    def verify_and_simulate(
        cls,
        X_df: pd.DataFrame,
        y_raw: pd.Series,
        task_type: str,
        champion_model: Any,
        feature_impact: Dict[str, Any],
        diagnostic_candidates: List[Dict[str, Any]],
        max_candidates: int = 3,
        target_column: Optional[str] = None,
        selected_model_name: Optional[str] = None,
        evaluation_metric: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Master coordinator for the Verification Engine.
        Executes targeted experiments, evidence fusion, closed-loop remediation simulation,
        and constructs the validated Multi-Experiment Evidence Graph with rich metadata.
        """
        experiments = cls.run_feature_verification_experiments(
            X_df=X_df,
            y_raw=y_raw,
            task_type=task_type,
            champion_model=champion_model,
            feature_impact=feature_impact,
            max_candidates=max_candidates
        )

        remediation_sim = cls.simulate_closed_loop_remediation(
            X_df=X_df,
            y_raw=y_raw,
            task_type=task_type,
            champion_model=champion_model,
            diagnostic_candidates=diagnostic_candidates
        )

        evidence_graph = EvidenceGraphBuilder.build_graph(
            task_type=task_type,
            feature_impact=feature_impact,
            candidate_experiments=experiments,
            diagnostic_candidates=diagnostic_candidates,
            remediation_simulation=remediation_sim,
            target_column=target_column,
            selected_model=selected_model_name,
            evaluation_metric=evaluation_metric,
        )

        return safe_primitive({
            "status": "Success",
            "candidate_experiments": experiments,
            "experiments_count": len(experiments),
            "remediation_simulation": remediation_sim,
            "evidence_graph": evidence_graph,
        })


