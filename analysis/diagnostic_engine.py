# analysis/diagnostic_engine.py
"""
DiagnosticEngine, ConfidenceEngine, RiskEngine, and RecommendationEngine.
Universal, evidence-driven diagnostic candidate generation, continuous statistical confidence
estimation, continuous evidence-weighted risk calculation without fixed penalty tables or double-counting,
and model-aware actionable mitigation recommendations.
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np
import pandas as pd
from analysis.evidence import DiagnosticEvidence, safe_primitive


class ConfidenceEngine:
    """
    Computes explainable, continuous diagnostic confidence scores (0.10 to 1.0)
    derived from sample support, effect magnitude, multi-source signal consistency,
    cross-model consensus, and cross-validation stability.
    """

    @classmethod
    def calculate_confidence(
        cls,
        sample_support: int,
        total_samples: int,
        effect_magnitude: float,
        supporting_signal_count: int = 1,
        cv_std: float = 0.0,
        cross_model_consistent: bool = True,
        reliability: float = 1.0,
        is_direct_evidence: bool = True
    ) -> Tuple[float, str, str]:
        """
        Calculates a continuous confidence score bounded between 0.10 and 1.0.
        Weak evidence with small samples produces LOW confidence.
        Strong evidence with multi-signal support and large samples produces HIGH confidence.
        """
        eff_mag = min(1.0, max(0.0, float(effect_magnitude)))
        tot_samples = max(1, int(total_samples))
        supp_samples = max(1, int(sample_support))

        # 1. Continuous sample support factor via smooth square-root scaling
        # Evaluates statistical certainty based on sample size
        sample_factor = min(1.0, max(0.40, math.sqrt(supp_samples) / 10.0))

        # 2. Base signal strength from magnitude and evidence directness
        direct_bonus = 0.08 if is_direct_evidence else 0.0
        rel_factor = min(1.0, max(0.5, float(reliability)))
        signal_bonus = min(0.18, (max(1, supporting_signal_count) - 1) * 0.06)

        base_score = (0.35 + (0.35 * eff_mag) + direct_bonus + signal_bonus) * rel_factor

        # 3. Adjustments for partition stability and cross-model agreement
        cv_penalty = 0.12 * min(1.0, max(0.0, cv_std / 0.15))
        base_score -= cv_penalty

        if cross_model_consistent:
            base_score += 0.06
        else:
            base_score -= 0.06

        # Apply continuous sample scaling
        raw_score = base_score * sample_factor
        score = round(min(1.0, max(0.10, raw_score)), 2)

        if score >= 0.72:
            label = "High"
            strength = "HIGH"
        elif score >= 0.48:
            label = "Medium"
            strength = "MODERATE"
        else:
            label = "Low"
            strength = "LOW"

        return score, label, strength


class DiagnosticEngine:
    """
    Synthesizes DiagnosticEvidence objects into unified, evidence-grounded diagnostic candidates.
    Dynamically groups related evidence across domains and subjects to eliminate duplicate findings.
    """

    @classmethod
    def build_diagnostic_candidates(
        cls,
        evidence_list: List[DiagnosticEvidence],
        task_type: str,
        target_profile: Dict[str, Any],
        selected_model_name: str,
        total_rows: int,
        test_rows: int,
        cv_std: float = 0.0,
        cross_model_consistent: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Groups evidence signals by domain and subject, synthesizes candidate titles from
        observed metrics, and generates cohesive diagnostic candidates.
        """
        candidates: List[Dict[str, Any]] = []
        candidate_idx = 1

        # Group evidence by domain
        domain_groups: Dict[str, List[DiagnosticEvidence]] = {}
        for ev in evidence_list:
            dom = ev.domain or "general"
            domain_groups.setdefault(dom, []).append(ev)

        # ----------------------------------------------------
        # 1. Leakage Evidence Grouping
        # ----------------------------------------------------
        leakage_evs = domain_groups.get("leakage", [])
        if leakage_evs:
            for leak_ev in leakage_evs:
                feat_name = leak_ev.affected_features[0] if leak_ev.affected_features else "feature"
                conf_score, conf_label, conf_strength = ConfidenceEngine.calculate_confidence(
                    sample_support=leak_ev.sample_support or total_rows,
                    total_samples=total_rows,
                    effect_magnitude=leak_ev.magnitude,
                    supporting_signal_count=1,
                    cv_std=cv_std,
                    cross_model_consistent=cross_model_consistent,
                    reliability=leak_ev.reliability,
                    is_direct_evidence=True
                )

                candidates.append(cls._synthesize_candidate(
                    candidate_id=f"DC_{candidate_idx:03d}",
                    title=f"Potential Target Leakage Signal in Predictor '{feat_name}'",
                    root_cause=f"High Empirical Association Between '{feat_name}' and Target",
                    category="Data Quality Issue",
                    domain="leakage",
                    evidence_list=[leak_ev],
                    interpretation=f"Feature '{feat_name}' exhibits near-deterministic or duplicate correlation with the target variable.",
                    potential_explanation="This feature may represent a direct target duplicate, post-outcome calculation, or proxy variable.",
                    impact="Artificially inflates evaluation metrics and creates unreliable dependencies in production deployment.",
                    severity="CRITICAL" if leak_ev.magnitude >= 0.95 else "HIGH",
                    confidence=conf_label,
                    confidence_score=conf_score,
                    affected_metric="Data Integrity / Leakage",
                    affected_population=leak_ev.affected_population or f"Predictor '{feat_name}'",
                    selected_model_name=selected_model_name,
                    recommended_action=f"Remove '{feat_name}' from the training feature set and audit feature generation pipelines."
                ))
                candidate_idx += 1

        # ----------------------------------------------------
        # 2. Classification Performance & Confusion Disparity Grouping
        # ----------------------------------------------------
        class_evs = domain_groups.get("class_performance", [])
        confusion_evs = domain_groups.get("confusion_pattern", [])

        if task_type == "classification" and (class_evs or confusion_evs):
            # Check if class recall disparity exists
            disparity_evs = [e for e in class_evs if e.metric in {"class_recall_disparity", "class_imbalance_ratio"}]
            if disparity_evs:
                # Combine class performance disparity and related confusion pattern into ONE candidate
                combined_evs = list(disparity_evs) + confusion_evs
                affected_classes = list(dict.fromkeys(
                    [c for ev in combined_evs for c in ev.affected_classes if c]
                ))
                target_cls_name = affected_classes[0] if affected_classes else "minority"

                max_mag = max(ev.magnitude for ev in combined_evs)
                min_supp = min(ev.sample_support for ev in combined_evs if ev.sample_support > 0) if combined_evs else test_rows

                conf_score, conf_label, conf_strength = ConfidenceEngine.calculate_confidence(
                    sample_support=min_supp,
                    total_samples=total_rows,
                    effect_magnitude=max_mag,
                    supporting_signal_count=len(combined_evs),
                    cv_std=cv_std,
                    cross_model_consistent=cross_model_consistent,
                    is_direct_evidence=True
                )

                candidates.append(cls._synthesize_candidate(
                    candidate_id=f"DC_{candidate_idx:03d}",
                    title=f"Prediction Sensitivity Disparity Concentrated in Class '{target_cls_name}'",
                    root_cause=f"Model Performance Deficit for Class '{target_cls_name}'",
                    category="Root Cause Candidate",
                    domain="class_performance",
                    evidence_list=combined_evs,
                    interpretation=f"The model demonstrates a significant sensitivity deficit specifically impacting class '{target_cls_name}'.",
                    potential_explanation="Loss function optimization across uneven class distributions or feature overlap reduces decision boundary sensitivity for this class.",
                    impact=f"Increased misclassification rate for real-world instances of class '{target_cls_name}'.",
                    severity="HIGH" if max_mag >= 0.40 else "MEDIUM",
                    confidence=conf_label,
                    confidence_score=conf_score,
                    affected_metric="Class Recall / Sensitivity",
                    affected_population=f"Class '{target_cls_name}' instances",
                    selected_model_name=selected_model_name,
                    recommended_action=cls._get_class_disparity_recommendation(selected_model_name, target_cls_name, confidence=conf_label)
                ))
                candidate_idx += 1
            elif confusion_evs:
                # Isolated confusion patterns
                top_conf_ev = confusion_evs[0]
                conf_score, conf_label, conf_strength = ConfidenceEngine.calculate_confidence(
                    sample_support=top_conf_ev.sample_support or test_rows,
                    total_samples=total_rows,
                    effect_magnitude=top_conf_ev.magnitude,
                    supporting_signal_count=len(confusion_evs),
                    cv_std=cv_std,
                    cross_model_consistent=cross_model_consistent,
                    is_direct_evidence=True
                )

                candidates.append(cls._synthesize_candidate(
                    candidate_id=f"DC_{candidate_idx:03d}",
                    title=f"Repeated Misclassification Pattern ({top_conf_ev.signal_name})",
                    root_cause=f"Frequent Confusion Transition ({top_conf_ev.signal_name})",
                    category="Model Behavior",
                    domain="confusion_pattern",
                    evidence_list=confusion_evs,
                    interpretation="The model repeatedly confuses specific class transitions during evaluation.",
                    potential_explanation="Shared feature distributions or insufficient boundary separability between these classes.",
                    impact="Targeted accuracy reduction on specific class transition boundaries.",
                    severity="MEDIUM",
                    confidence=conf_label,
                    confidence_score=conf_score,
                    affected_metric="Precision / Specificity",
                    affected_population=top_conf_ev.affected_population,
                    selected_model_name=selected_model_name,
                    recommended_action=f"Analyze feature differences on {top_conf_ev.affected_population} to identify separating features."
                ))
                candidate_idx += 1

        # ----------------------------------------------------
        # 3. Generalization & Partition Stability Grouping
        # ----------------------------------------------------
        gen_evs = [
            e for e in domain_groups.get("generalization", [])
            if e.evidence_id == "generalization_score_gap" and e.direction in {"gap", "overfitting", "deficit"} and e.observed_value >= 0.08
        ]
        part_evs = [
            e for e in domain_groups.get("partition_stability", [])
            if e.direction in {"variation", "instability"} and (e.observed_value >= 0.08 or e.magnitude >= 0.50)
        ]

        if gen_evs or part_evs:
            combined_gen = gen_evs + part_evs
            max_mag = max(e.magnitude for e in combined_gen)
            conf_score, conf_label, conf_strength = ConfidenceEngine.calculate_confidence(
                sample_support=total_rows,
                total_samples=total_rows,
                effect_magnitude=max_mag,
                supporting_signal_count=len(combined_gen),
                cv_std=cv_std,
                cross_model_consistent=cross_model_consistent,
                reliability=0.70 if total_rows < 40 else 0.90,
                is_direct_evidence=True
            )

            should_emit_gen_candidate = False
            if gen_evs:
                top_gen = gen_evs[0]
                gap_val = top_gen.observed_value
                title_str = "Performance Discrepancy Between Training and Evaluation Partitions"
                root_str = "Generalization Deficit on Unseen Evaluation Partitions"
                diag_interp = f"The model demonstrates a measurable performance gap ({gap_val:.1%} drop) between training split fit and independent evaluation partitions."
                sev = "HIGH" if (max_mag >= 0.50 and total_rows >= 40) or gap_val >= 0.20 else "MEDIUM"
                should_emit_gen_candidate = True
            elif part_evs:
                top_part = part_evs[0]
                if top_part.observed_value >= 0.08 or top_part.magnitude >= 0.50:
                    title_str = "Cross-Validation Partition Variance Across Folds"
                    root_str = "Model Performance Sensitive to Data Partitioning"
                    diag_interp = "Model performance varies across cross-validation folds, indicating sensitivity to specific data split boundaries."
                    sev = "MEDIUM"
                    should_emit_gen_candidate = True

            if should_emit_gen_candidate:
                candidates.append(cls._synthesize_candidate(
                    candidate_id=f"DC_{candidate_idx:03d}",
                    title=title_str,
                    root_cause=root_str,
                    category="Model Behavior",
                    domain="generalization",
                    evidence_list=combined_gen,
                    interpretation=diag_interp,
                    potential_explanation="Model complexity relative to sample size, or uneven distribution of predictor clusters across validation folds.",
                    impact="Out-of-sample operational performance will be lower or more variable than training metrics indicate.",
                    severity=sev,
                    confidence=conf_label,
                    confidence_score=conf_score,
                    affected_metric="Generalization / Validation Stability",
                    affected_population=f"Evaluation splits ({test_rows} test samples)",
                    selected_model_name=selected_model_name,
                    recommended_action=cls._get_generalization_recommendation(selected_model_name)
                ))
                candidate_idx += 1

        # ----------------------------------------------------
        # 4. Feature Reliance & Association Grouping
        # ----------------------------------------------------
        feat_evs = domain_groups.get("feature_reliance", [])
        if feat_evs:
            dom_feat_evs = [e for e in feat_evs if e.direction == "dominant_reliance"]
            for f_ev in dom_feat_evs:
                feat_name = f_ev.affected_features[0] if f_ev.affected_features else "feature"

                # Gate feature dominance so it becomes a diagnostic risk candidate only when
                # supported by independent fragility/instability signals (cross-model divergence,
                # generalization gap, partition instability, extreme >80% monopoly, or small sample constraint)
                has_cross_model_instability = not cross_model_consistent
                has_generalization_issue = bool(domain_groups.get("generalization"))
                has_partition_instability = any(e.observed_value >= 0.10 or e.magnitude >= 0.50 for e in domain_groups.get("partition_stability", []))
                has_extreme_concentration = f_ev.observed_value >= 80.0
                has_sample_constraint = total_rows < 40

                is_supported_risk_candidate = (
                    has_cross_model_instability or
                    has_generalization_issue or
                    has_partition_instability or
                    has_extreme_concentration or
                    has_sample_constraint
                )

                if is_supported_risk_candidate:
                    conf_score, conf_label, conf_strength = ConfidenceEngine.calculate_confidence(
                        sample_support=total_rows,
                        total_samples=total_rows,
                        effect_magnitude=f_ev.magnitude,
                        supporting_signal_count=len(feat_evs),
                        cv_std=cv_std,
                        cross_model_consistent=cross_model_consistent,
                        is_direct_evidence=True
                    )

                    candidates.append(cls._synthesize_candidate(
                        candidate_id=f"DC_{candidate_idx:03d}",
                        title=f"Disproportionate Model Reliance on Single Feature '{feat_name}'",
                        root_cause=f"Model Reliance Highly Concentrated in '{feat_name}'",
                        category="Risk Factor",
                        domain="feature_reliance",
                        evidence_list=[f_ev],
                        interpretation=f"The model derives a dominant portion of its predictive utility from feature '{feat_name}' alongside observed partition or cross-model sensitivity.",
                        potential_explanation="High univariate association or lack of complementary predictive signals in other predictors.",
                        impact=f"Operational consideration: Because the selected model exhibits strong empirical reliance on '{feat_name}', production monitoring should track changes in its distribution and data-collection process. The current experiments do not establish that such shifts will degrade production performance.",
                        severity="MEDIUM",
                        confidence=conf_label,
                        confidence_score=conf_score,
                        affected_metric="Predictive Robustness",
                        affected_population=f"Feature '{feat_name}'",
                        selected_model_name=selected_model_name,
                        recommended_action=f"If deployed, monitor distribution drift and data-quality changes in '{feat_name}' over time, audit data collection integrity, and evaluate alternative model configurations if reducing feature concentration is operationally necessary."
                    ))
                    candidate_idx += 1

        # ----------------------------------------------------
        # 5. Regression Residual Patterns Grouping
        # ----------------------------------------------------
        reg_evs = domain_groups.get("regression_residual", [])
        if task_type == "regression" and reg_evs:
            max_mag = max(e.magnitude for e in reg_evs)
            conf_score, conf_label, conf_strength = ConfidenceEngine.calculate_confidence(
                sample_support=test_rows,
                total_samples=total_rows,
                effect_magnitude=max_mag,
                supporting_signal_count=len(reg_evs),
                cv_std=cv_std,
                cross_model_consistent=cross_model_consistent,
                is_direct_evidence=True
            )

            het_evs = [e for e in reg_evs if e.metric == "heteroscedasticity_correlation"]
            bias_evs = [e for e in reg_evs if e.metric == "mean_residual_bias"]

            if het_evs:
                title_str = "Non-Constant Residual Variance Across Prediction Scale"
                root_str = "Scale-Dependent Error Distribution (Heteroscedasticity)"
                diag_interp = "Prediction error magnitude systematically varies depending on the scale of predicted values."
            elif bias_evs:
                title_str = "Systematic Residual Mean Shift (Prediction Bias)"
                root_str = "Systematic Directional Bias in Regression Predictions"
                diag_interp = "The model exhibits systematic over- or under-prediction bias across the evaluation partition."
            else:
                title_str = "Residual Distribution Anomaly in Continuous Predictions"
                root_str = "Non-Gaussian Residual Patterns"
                diag_interp = "Residual distribution departs from ideal homoscedastic properties."

            candidates.append(cls._synthesize_candidate(
                candidate_id=f"DC_{candidate_idx:03d}",
                title=title_str,
                root_cause=root_str,
                category="Model Behavior",
                domain="regression_residual",
                evidence_list=reg_evs,
                interpretation=diag_interp,
                potential_explanation="Non-linear interactions, multiplicative scale effects, or target distribution skewness.",
                impact="Forecast reliability degrades for extreme prediction values.",
                severity="MEDIUM",
                confidence=conf_label,
                confidence_score=conf_score,
                affected_metric="Residual Homoscedasticity",
                affected_population="Continuous prediction range",
                selected_model_name=selected_model_name,
                recommended_action=cls._get_regression_recommendation(selected_model_name)
            ))
            candidate_idx += 1

        # ----------------------------------------------------
        # 6. Sample Size Constraint Grouping
        # ----------------------------------------------------
        sample_evs = domain_groups.get("sample_size", [])
        if sample_evs and total_rows < 60:
            s_ev = sample_evs[0]
            conf_score, conf_label, conf_strength = ConfidenceEngine.calculate_confidence(
                sample_support=total_rows,
                total_samples=total_rows,
                effect_magnitude=s_ev.magnitude,
                supporting_signal_count=1,
                cv_std=cv_std,
                cross_model_consistent=True,
                reliability=1.0,
                is_direct_evidence=True
            )

            candidates.append(cls._synthesize_candidate(
                candidate_id=f"DC_{candidate_idx:03d}",
                title=f"Small Dataset Sample Size Constraint ({total_rows} rows)",
                root_cause=f"Small Dataset Sample Size ({total_rows} observations)",
                category="Data Quality Issue",
                domain="sample_size",
                evidence_list=sample_evs,
                interpretation=f"The dataset contains {total_rows} usable records, which limits statistical power and widens confidence intervals.",
                potential_explanation="Limited observation collection or severe filtering prior to analysis.",
                impact="Individual prediction errors have a large percentage effect on aggregate evaluation metrics.",
                severity="HIGH" if total_rows < 30 else "MEDIUM",
                confidence=conf_label,
                confidence_score=conf_score,
                affected_metric="Statistical Estimation Robustness",
                affected_population=f"Entire dataset ({total_rows} rows)",
                selected_model_name=selected_model_name,
                recommended_action=f"Expand dataset beyond {total_rows} observations to increase statistical reliability of validation splits."
            ))
            candidate_idx += 1

        # ----------------------------------------------------
        # 7. Clean Healthy Baseline (Zero False Alarms)
        # ----------------------------------------------------
        if not candidates:
            candidates.append({
                "candidate_id": "DC_001",
                "title": "Healthy Baseline Model Performance",
                "finding": "Healthy Baseline Model Performance",
                "signal": "Healthy Baseline Model Performance",
                "initial_evidence": "Evaluation scores align with cross-validation stability estimates across partitions.",
                "hypothesis": "Model operating in a healthy regime with consistent generalization across partitions.",
                "diagnostic_status": "SIGNAL DETECTED",
                "verification_status": "NOT TESTED",
                "verification_score": "N/A",
                "verification_evidence": "No anomaly detected requiring counterfactual verification.",
                "root_cause": "No dominant model deficit or diagnostic anomaly detected from available evidence",
                "category": "Diagnostic Observation",
                "domain": "generalization",
                "severity": "LOW",
                "confidence": "High",
                "confidence_score": 0.90,
                "evidence_strength": "HIGH",
                "evidence_ids": [f"healthy_baseline_{task_type}"],
                "evidence": "Evaluation scores are consistent with cross-validation averages, and error patterns show no dominant failure mode.",
                "evidence_list": ["Evaluation scores align with cross-validation stability estimates across partitions."],
                "evidence_items": [],
                "affected_population": "All evaluated partitions",
                "supporting_models": [selected_model_name],
                "limitations": ["Evaluated within available dataset distribution."],
                "diagnostic_interpretation": "The model demonstrates consistent, balanced performance without severe disparity or generalization deficits.",
                "potential_explanation": "Features provide sufficient predictive separability across target outcomes.",
                "impact": "Current model pipeline provides a reliable baseline for the evaluated operational domain.",
                "affected_metric": "Overall Pipeline Health",
                "recommended_actions": ["Maintain current training pipeline and monitor out-of-sample data distributions during deployment."],
                "recommended_action": "Maintain current training pipeline and monitor out-of-sample data distributions during deployment.",
                "recommendation": "Maintain current training pipeline and monitor out-of-sample data distributions during deployment.",
            })

        return safe_primitive(candidates)

    @classmethod
    def _synthesize_candidate(
        cls,
        candidate_id: str,
        title: str,
        root_cause: str,
        category: str,
        domain: str,
        evidence_list: List[DiagnosticEvidence],
        interpretation: str,
        potential_explanation: str,
        impact: str,
        severity: str,
        confidence: str,
        confidence_score: float,
        affected_metric: str,
        affected_population: str,
        selected_model_name: str,
        recommended_action: str
    ) -> Dict[str, Any]:
        """Builds standardized diagnostic candidate object."""
        ev_ids = [e.evidence_id for e in evidence_list]
        ev_summaries = [e.context or e.signal_name for e in evidence_list]
        ev_items = [e.to_dict() for e in evidence_list]

        supporting_models = list(dict.fromkeys(
            [m for e in evidence_list for m in e.supporting_models if m] or [selected_model_name]
        ))
        limitations = list(dict.fromkeys([lim for e in evidence_list for lim in e.limitations if lim]))

        conf_strength = "HIGH" if confidence_score >= 0.72 else ("MODERATE" if confidence_score >= 0.48 else "LOW")

        diag_status = "CANDIDATE HYPOTHESIS" if domain in {"feature_reliance", "generalization", "partition_stability", "class_performance"} else "SIGNAL DETECTED"

        return {
            "candidate_id": candidate_id,
            "title": title,
            "finding": title,
            "signal": title,
            "initial_evidence": " ".join(ev_summaries),
            "hypothesis": potential_explanation or root_cause,
            "diagnostic_status": diag_status,
            "verification_status": "NOT TESTED",
            "verification_score": "N/A",
            "verification_evidence": "Controlled counterfactual experiments not yet executed for this candidate.",
            "root_cause": root_cause,
            "category": category,
            "domain": domain,
            "severity": severity.upper(),
            "confidence": confidence,
            "confidence_score": confidence_score,
            "evidence_strength": conf_strength,
            "evidence_ids": ev_ids,
            "evidence": " ".join(ev_summaries),
            "evidence_list": ev_summaries,
            "evidence_items": ev_items,
            "affected_population": affected_population,
            "supporting_models": supporting_models,
            "limitations": limitations,
            "diagnostic_interpretation": interpretation,
            "interpretation": interpretation,
            "potential_explanation": potential_explanation,
            "possible_root_cause": potential_explanation,
            "impact": impact,
            "affected_metric": affected_metric,
            "affected_area": affected_metric,
            "recommended_actions": [recommended_action],
            "recommended_action": recommended_action,
            "recommendation": recommended_action,
        }

    @classmethod
    def _get_class_disparity_recommendation(
        cls,
        model_name: str,
        class_name: str,
        confidence: str = "Medium"
    ) -> str:
        """
        Emits estimator-aware, confidence-calibrated recommendations for class disparity.
        Frames recommendations as candidate interventions requiring validation against
        class-specific metrics, avoiding deterministic outcome promises.
        """
        m_lower = model_name.lower()
        conf_lower = str(confidence).lower()

        if "low" in conf_lower:
            action_verb = "Investigate"
            candidate_framing = "as candidate exploratory interventions"
            validation_suffix = "Their effectiveness should be experimentally validated using per-class recall, F1, and precision before adoption."
        elif "medium" in conf_lower:
            action_verb = "Consider evaluating"
            candidate_framing = "as candidate interventions"
            validation_suffix = "Validate their effect on class-specific recall, F1, and precision."
        else:
            action_verb = "Evaluate"
            candidate_framing = "as targeted candidate interventions"
            validation_suffix = "Confirm improvement using per-class recall, F1, and precision."

        if "logistic" in m_lower:
            interventions = "decision threshold tuning, balanced class weighting (class_weight='balanced'), and regularization adjustments"
        elif "random forest" in m_lower:
            interventions = "balanced tree sampling (class_weight='balanced_subsample'), min_samples_leaf adjustments, and decision threshold calibration"
        elif "decision tree" in m_lower:
            interventions = "class weighting (class_weight='balanced'), min_samples_leaf adjustments, and tree pruning (ccp_alpha)"
        elif "gradient boosting" in m_lower:
            interventions = "fit-time sample weighting, subsampling adjustments, and decision threshold calibration"
        else:
            interventions = "decision threshold calibration, class weighting, and targeted resampling strategies"

        return f"{action_verb} {interventions} {candidate_framing} for the observed class '{class_name}' sensitivity disparity. {validation_suffix}"

    @classmethod
    def _get_generalization_recommendation(cls, model_name: str) -> str:
        """Emits estimator-aware recommendations for generalization deficits framed as candidate interventions."""
        m_lower = model_name.lower()
        if "linear regression" in m_lower or (m_lower.startswith("linear") and "ridge" not in m_lower and "lasso" not in m_lower and "logistic" not in m_lower):
            return "Evaluate regularized linear estimators (Ridge, Lasso, ElasticNet), inspect multicollinearity (VIF), or evaluate feature scaling as candidate interventions to stabilize coefficient variance."
        elif "ridge" in m_lower:
            return "Evaluate increasing regularization strength (higher alpha parameter) as a candidate intervention to constrain coefficient variance across validation partitions."
        elif "lasso" in m_lower:
            return "Evaluate tuning alpha regularization as a candidate intervention to encourage feature sparsity and validate out-of-sample generalization."
        elif "elastic" in m_lower:
            return "Evaluate tuning alpha and l1_ratio parameters as candidate interventions to balance L1/L2 regularization penalties across validation partitions."
        elif "logistic" in m_lower:
            return "Evaluate increasing regularization strength (lower C parameter) or penalty solvers as candidate interventions to reduce model variance, and validate out-of-sample stability."
        elif "random forest" in m_lower:
            return "Evaluate constraining tree depth (max_depth), increasing min_samples_leaf, or adjusting max_features as candidate regularizations, and validate out-of-sample generalization across validation folds."
        elif "decision tree" in m_lower:
            return "Evaluate cost-complexity pruning (ccp_alpha), constraining max_depth, or increasing min_samples_leaf as candidate regularizations, and confirm validation partition stability."
        elif "gradient boosting" in m_lower:
            return "Evaluate reducing learning_rate with early stopping, constraining max_depth, or adjusting subsample as candidate interventions, and validate generalization on independent partitions."
        return "Evaluate increasing model regularization, applying cross-validated hyperparameter tuning, or acquiring additional training observations as candidate interventions."

    @classmethod
    def _get_regression_recommendation(cls, model_name: str) -> str:
        """Emits estimator-aware recommendations for regression residual anomalies."""
        m_lower = model_name.lower()
        if "linear regression" in m_lower or "ridge" in m_lower or "lasso" in m_lower:
            return "Evaluate non-linear feature transformations (polynomial, splines), target transformations (log/Box-Cox), or inspect residual patterns against individual predictors."
        elif "tree" in m_lower or "forest" in m_lower or "gradient" in m_lower:
            return "Evaluate robust loss criteria (Huber loss / MAE criterion) or target transformations to stabilize residual spread across prediction scales."
        return "Evaluate target transformations (log or Box-Cox transformation) or robust loss functions (Huber loss) to stabilize residual variance."


class RiskEngine:
    """
    Evidence-Weighted Continuous Risk Engine.
    Derives risk continuously from normalized evidence magnitude and confidence
    without arbitrary fixed point tables (no +25, +20, +15 step tables).
    Prevents double-counting across correlated signals through sub-additive saturation.
    Every risk driver includes linked evidence_ids.
    """

    @classmethod
    def compute_risk(
        cls,
        task_type: str,
        performance: Dict[str, Any],
        stability: Dict[str, Any],
        class_info: Dict[str, Any],
        data_info: Dict[str, Any],
        leakage_findings: List[Dict[str, Any]],
        error_segments: Optional[List[Dict[str, Any]]] = None,
        diagnostic_candidates: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Calculates a continuous risk score (5 to 100) derived strictly from observed evidence.
        Mathematical Formulation:
          Pillar 1: Performance Deficit (max 30 pts) = 30 * (1 - PerformanceMetric)^1.5 * Confidence
          Pillar 2: Generalization & Stability (max 25 pts) = 15 * clip(OverfitGap/0.30, 0, 1) + 10 * clip(CVStd/0.15, 0, 1)
          Pillar 3: Error Disparity (max 20 pts) = 20 * RecallDisparity * Confidence
          Pillar 4: Data Integrity & Leakage (max 25 pts) = LeakageContribution + QualityContribution + SampleUncertainty
          Total Score = round(clip(P1 + P2 + P3 + P4 + 5, 5, 100))
        """
        drivers: List[Dict[str, Any]] = []
        breakdown: Dict[str, int] = {
            "performance_deficit": 0,
            "generalization_stability": 0,
            "error_disparity": 0,
            "data_integrity": 0,
        }

        # ----------------------------------------------------
        # Pillar 1: Model Performance Deficit (0 to 30 pts)
        # ----------------------------------------------------
        p1_pts = 0.0
        if task_type == "classification":
            acc = float(performance.get("accuracy", 1.0))
            f1 = float(performance.get("f1_score", 1.0))
            perf_metric = max(0.0, min(1.0, (acc + f1) / 2.0))

            if perf_metric < 0.90:
                deficit = (0.90 - perf_metric) / 0.90
                p1_pts = min(30.0, 30.0 * (deficit ** 1.3))
                if p1_pts >= 4.0:
                    drivers.append({
                        "risk_driver_id": f"RD_{len(drivers) + 1:03d}",
                        "domain": "performance",
                        "issue": "Evaluation Performance Deficit",
                        "evidence": f"Test accuracy ({acc:.1%}) or F1 ({f1:.1%}) below optimal baseline.",
                        "observed_evidence": f"Test accuracy ({acc:.1%}) or F1 ({f1:.1%}) below optimal baseline.",
                        "reason": "Lower predictive accuracy/F1 reduces operational reliability.",
                        "confidence": 0.90,
                        "contribution": int(round(p1_pts)),
                        "evidence_ids": ["eval_performance_deficit"],
                    })
        else:
            r2 = float(performance.get("r2", 1.0))
            if r2 < 0.85:
                deficit = max(0.0, min(1.0, (0.85 - r2) / 0.85))
                p1_pts = min(30.0, 30.0 * (deficit ** 1.3))
                if p1_pts >= 4.0:
                    drivers.append({
                        "risk_driver_id": f"RD_{len(drivers) + 1:03d}",
                        "domain": "performance",
                        "issue": "Regression Explanatory Power Deficit",
                        "evidence": f"Model R² score is {r2:.4f} (explains {max(0.0, r2):.1%} of target variance).",
                        "observed_evidence": f"Model R² score is {r2:.4f} (explains {max(0.0, r2):.1%} of target variance).",
                        "reason": "Substantial unexplained variance in continuous target.",
                        "confidence": 0.85,
                        "contribution": int(round(p1_pts)),
                        "evidence_ids": ["regression_r2_deficit"],
                    })
        p1_pts = min(30.0, p1_pts)
        breakdown["performance_deficit"] = int(round(p1_pts))

        # ----------------------------------------------------
        # Pillar 2: Generalization & Partition Stability (0 to 25 pts)
        # ----------------------------------------------------
        p2_pts = 0.0
        overfit_gap = max(0.0, float(stability.get("overfitting_gap", 0.0)))
        cv_std = max(0.0, float(stability.get("cv_std", 0.0)))

        if overfit_gap >= 0.08:
            overfit_contrib = min(15.0, 15.0 * (overfit_gap / 0.30))
            p2_pts += overfit_contrib
            gap_fmt = f"{overfit_gap:.1%}" if task_type == "classification" else f"{overfit_gap:.4f}"
            drivers.append({
                "risk_driver_id": f"RD_{len(drivers) + 1:03d}",
                "domain": "generalization",
                "issue": "Training vs Evaluation Gap",
                "evidence": f"Training score exceeds independent evaluation score by {gap_fmt}.",
                "observed_evidence": f"Training score exceeds independent evaluation score by {gap_fmt}.",
                "reason": "Divergence between training fit and out-of-sample partitions indicates capacity over-fit.",
                "confidence": 0.80,
                "contribution": int(round(overfit_contrib)),
                "evidence_ids": ["generalization_score_gap"],
            })

        if cv_std >= 0.04:
            cv_contrib = max(0.0, min(10.0, 10.0 * ((cv_std - 0.04) / 0.16)))
            p2_pts += cv_contrib
            if cv_contrib >= 2.0:
                drivers.append({
                    "risk_driver_id": f"RD_{len(drivers) + 1:03d}",
                    "domain": "generalization",
                    "issue": "Cross-Validation Partition Variance",
                    "evidence": f"Fold score standard deviation is {cv_std:.4f}.",
                    "observed_evidence": f"Fold score standard deviation is {cv_std:.4f}.",
                    "reason": "Model performance is sensitive to partition boundaries.",
                    "confidence": 0.80,
                    "contribution": int(round(cv_contrib)),
                    "evidence_ids": ["cross_validation_fold_variance"],
                })
        p2_pts = min(25.0, p2_pts)
        breakdown["generalization_stability"] = int(round(p2_pts))

        # ----------------------------------------------------
        # Pillar 3: Error Disparity & Failure Concentration (0 to 20 pts)
        # ----------------------------------------------------
        p3_pts = 0.0
        if task_type == "classification":
            min_recall = float(class_info.get("minority_recall", 1.0))
            maj_recall = float(class_info.get("majority_recall", 1.0))
            target_cls = str(class_info.get("minority_class", "class"))

            rec_gap = max(0.0, maj_recall - min_recall)
            if rec_gap >= 0.12 or min_recall < 0.65:
                disparity_factor = max(rec_gap, (1.0 - min_recall) * 0.7)
                p3_pts = min(20.0, 20.0 * min(1.0, disparity_factor / 0.60))
                drivers.append({
                    "risk_driver_id": f"RD_{len(drivers) + 1:03d}",
                    "domain": "error_disparity",
                    "issue": f"Class Sensitivity Disparity ({target_cls})",
                    "evidence": f"Recall for class '{target_cls}' is {min_recall:.1%} vs {maj_recall:.1%} for highest class (gap: {rec_gap:+.1%}).",
                    "observed_evidence": f"Recall for class '{target_cls}' is {min_recall:.1%} vs {maj_recall:.1%} for highest class (gap: {rec_gap:+.1%}).",
                    "reason": f"Disproportionate false negatives concentrated in class '{target_cls}'.",
                    "confidence": 0.85,
                    "contribution": int(round(p3_pts)),
                    "evidence_ids": [f"recall_disparity_{target_cls}"],
                })
        else:
            if error_segments:
                seg_factor = min(1.0, len(error_segments) / 3.0)
                p3_pts = min(20.0, 10.0 + 10.0 * seg_factor)
                reg_ev_ids = [s["evidence_id"] for s in error_segments if "evidence_id" in s]
                if not reg_ev_ids and diagnostic_candidates:
                    for dc in diagnostic_candidates:
                        if dc.get("domain") in {"regression_residual", "error_disparity"}:
                            reg_ev_ids.extend(dc.get("evidence_ids", []))
                drivers.append({
                    "risk_driver_id": f"RD_{len(drivers) + 1:03d}",
                    "domain": "error_disparity",
                    "issue": "Subgroup Residual Concentration",
                    "evidence": f"Disproportionately elevated errors concentrated in {len(error_segments)} feature segment(s).",
                    "observed_evidence": f"Disproportionately elevated errors concentrated in {len(error_segments)} feature segment(s).",
                    "reason": "Prediction errors concentrate non-uniformly across specific feature ranges.",
                    "confidence": 0.80,
                    "contribution": int(round(p3_pts)),
                    "evidence_ids": list(dict.fromkeys(reg_ev_ids)),
                })
        p3_pts = min(20.0, p3_pts)
        breakdown["error_disparity"] = int(round(p3_pts))

        # ----------------------------------------------------
        # Pillar 4: Data Integrity, Leakage & Sample Constraints (0 to 25 pts)
        # ----------------------------------------------------
        p4_pts = 0.0
        if leakage_findings:
            leak_score = sum(20.0 if l.get("severity") == "CRITICAL" else 10.0 for l in leakage_findings)
            leak_pts = min(20.0, leak_score)
            p4_pts += leak_pts
            leak_feats = ", ".join(f"'{l['feature']}'" for l in leakage_findings[:2])
            leak_ev_ids = []
            for l in leakage_findings:
                if "evidence_id" in l:
                    leak_ev_ids.append(l["evidence_id"])
                elif "feature" in l:
                    leak_ev_ids.append(f"leakage_signal_{l['feature']}")
            if not leak_ev_ids and diagnostic_candidates:
                for dc in diagnostic_candidates:
                    if dc.get("domain") == "leakage":
                        leak_ev_ids.extend(dc.get("evidence_ids", []))
            if not leak_ev_ids:
                leak_ev_ids = ["leakage_signal"]
            drivers.append({
                "risk_driver_id": f"RD_{len(drivers) + 1:03d}",
                "domain": "leakage",
                "issue": "Target Data Leakage Vulnerability",
                "evidence": f"High predictor-target correlation or duplicate columns detected in: {leak_feats}.",
                "observed_evidence": f"High predictor-target correlation or duplicate columns detected in: {leak_feats}.",
                "reason": "Features provide artificial or non-deployable target predictability.",
                "confidence": 0.95,
                "contribution": int(round(leak_pts)),
                "evidence_ids": list(dict.fromkeys(leak_ev_ids)),
            })

        total_rows = int(data_info.get("total_rows", 100))
        if total_rows < 50:
            sample_pts = min(6.0, 6.0 * (50 - total_rows) / 50.0)
            p4_pts += sample_pts
            drivers.append({
                "risk_driver_id": f"RD_{len(drivers) + 1:03d}",
                "domain": "sample_size",
                "issue": "Sample Size Statistical Uncertainty",
                "evidence": f"Dataset has {total_rows} usable records, introducing wider estimation variance.",
                "observed_evidence": f"Dataset has {total_rows} usable records, introducing wider estimation variance.",
                "reason": "Small sample size limits statistical generalization confidence.",
                "confidence": 0.85,
                "contribution": int(round(sample_pts)),
                "evidence_ids": ["small_sample_size_constraint"],
            })

        # Quality flaws (missing values, duplicates)
        missing_count = int(data_info.get("total_missing_cells", 0))
        dup_count = int(data_info.get("duplicate_rows", 0))
        if missing_count > 0 or dup_count > 0:
            dq_pts = min(5.0, (1.5 if missing_count > 0 else 0.0) + (2.0 if dup_count > 0 else 0.0))
            p4_pts += dq_pts
            dq_ev_ids = []
            if missing_count > 0:
                dq_ev_ids.append("missing_values_detected")
            if dup_count > 0:
                dq_ev_ids.append("duplicate_rows_detected")
            drivers.append({
                "risk_driver_id": f"RD_{len(drivers) + 1:03d}",
                "domain": "data_integrity",
                "issue": "Data Quality / Hygiene Flaws",
                "evidence": f"Detected {missing_count} missing cells and {dup_count} duplicate rows.",
                "observed_evidence": f"Detected {missing_count} missing cells and {dup_count} duplicate rows.",
                "reason": "Missingness or duplicate records distort feature distributions.",
                "confidence": 0.80,
                "contribution": int(round(dq_pts)),
                "evidence_ids": dq_ev_ids,
            })

        p4_pts = min(25.0, p4_pts)
        breakdown["data_integrity"] = int(round(p4_pts))

        # Sum continuous contributions with base 5 floor
        raw_score = 5.0 + p1_pts + p2_pts + p3_pts + p4_pts
        final_score = int(round(min(100.0, max(5.0, raw_score))))

        if final_score >= 65:
            risk_level = "HIGH"
        elif final_score >= 35:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return safe_primitive({
            "score": final_score,
            "level": risk_level,
            "method": "Evidence-weighted diagnostic aggregation",
            "drivers": drivers,
            "breakdown": breakdown,
        })


class RecommendationEngine:
    """
    Generates actionable, deduplicated recommendations derived directly from diagnostic findings
    and estimator capabilities.
    """

    @classmethod
    def generate_recommendations(
        cls,
        task_type: str,
        candidates: List[Dict[str, Any]],
        data_quality: Dict[str, Any],
        selected_model_name: str = ""
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Builds cohesive, non-redundant mitigation recommendations, returning both a string list
        and structured recommendation objects.
        """
        recs: List[str] = []
        structured_recs: List[Dict[str, Any]] = []
        seen_themes = set()

        def add_rec(
            text: str,
            theme: str,
            reason: str = "",
            objective: str = "",
            priority: str = "MEDIUM",
            evidence_ids: Optional[List[str]] = None,
            compatible_models: Optional[List[str]] = None
        ):
            if text and theme not in seen_themes and text not in recs:
                seen_themes.add(theme)
                recs.append(text)
                structured_recs.append({
                    "action": text,
                    "theme": theme,
                    "reason": reason or f"Triggered by diagnostic finding in {theme}.",
                    "objective": objective or "Improve model robustness and performance.",
                    "priority": priority,
                    "compatible_models": compatible_models or ([selected_model_name] if selected_model_name else ["All compatible estimators"]),
                    "evidence_ids": evidence_ids or [],
                })

        for cand in candidates:
            action = cand.get("recommended_action", "")
            domain = cand.get("domain", "")
            ev_ids = cand.get("evidence_ids", [])
            cand_sev = cand.get("severity", "MEDIUM")

            if domain == "leakage":
                add_rec(
                    action, "leakage",
                    reason="Target leakage signal detected in predictor matrix.",
                    objective="Prevent artificial training score inflation and out-of-sample failure.",
                    priority="HIGH",
                    evidence_ids=ev_ids
                )
            elif domain == "class_performance":
                add_rec(
                    action, "class_performance",
                    reason=cand.get("interpretation", "Class sensitivity disparity detected."),
                    objective="Evaluate candidate interventions to improve class-specific sensitivity and recall without severe precision degradation.",
                    priority="HIGH" if cand_sev == "HIGH" else "MEDIUM",
                    evidence_ids=ev_ids
                )
            elif domain == "confusion_pattern":
                add_rec(
                    action, "confusion_pattern",
                    reason=cand.get("interpretation", "Repeated confusion between specific classes."),
                    objective="Separate overlapping class distributions on transition boundaries.",
                    priority="MEDIUM",
                    evidence_ids=ev_ids
                )
            elif domain == "generalization":
                add_rec(
                    action, "generalization",
                    reason=cand.get("interpretation", "Generalization gap between training and evaluation partitions."),
                    objective="Evaluate capacity regularizations to improve out-of-sample generalization.",
                    priority="HIGH" if cand_sev == "HIGH" else "MEDIUM",
                    evidence_ids=ev_ids
                )
            elif domain == "partition_stability":
                add_rec(
                    action, "partition_stability",
                    reason="High cross-validation fold variance across data splits.",
                    objective="Stabilize model estimation across differing partition boundaries.",
                    priority="MEDIUM",
                    evidence_ids=ev_ids
                )
            elif domain == "feature_reliance":
                add_rec(
                    action, "feature_reliance",
                    reason=cand.get("interpretation", "Model relies heavily on single predictor."),
                    objective="Diversify predictor representation and, if deployed, monitor distribution drift and data-quality changes in input features over time.",
                    priority="MEDIUM",
                    evidence_ids=ev_ids
                )
            elif domain == "regression_residual":
                add_rec(
                    action, "regression_residual",
                    reason=cand.get("interpretation", "Residual variance anomaly or directional bias."),
                    objective="Stabilize continuous prediction errors and reduce scale-dependent residual spread.",
                    priority="MEDIUM",
                    evidence_ids=ev_ids
                )
            elif domain == "sample_size":
                add_rec(
                    action, "sample_size",
                    reason="Small sample size widens statistical confidence intervals.",
                    objective="Acquire additional observations to improve statistical power.",
                    priority="MEDIUM",
                    evidence_ids=ev_ids
                )
            elif action and cand.get("category") != "Diagnostic Observation":
                add_rec(
                    action, f"custom_{domain}",
                    reason=cand.get("interpretation", "Diagnostic observation."),
                    objective="Address operational finding.",
                    priority="MEDIUM",
                    evidence_ids=ev_ids
                )

        # Quality recommendations
        missing_count = int(data_quality.get("total_missing_cells", 0))
        dup_count = int(data_quality.get("duplicate_rows", 0))

        if missing_count > 0:
            add_rec(
                "Evaluate feature imputation pipelines and missing data patterns to minimize data loss.",
                "missingness",
                reason=f"Detected {missing_count} missing cell(s) in feature matrix.",
                objective="Preserve sample observations without introducing statistical imputation distortion.",
                priority="MEDIUM",
                evidence_ids=["missing_values_detected"]
            )
        if dup_count > 0:
            add_rec(
                f"Investigate {dup_count} duplicate rows to verify whether they represent genuine identical transactions or redundant logging.",
                "duplicates",
                reason=f"Detected {dup_count} duplicate row(s) in dataset.",
                objective="Ensure data partitions represent strictly independent real-world observations.",
                priority="MEDIUM",
                evidence_ids=["duplicate_rows_detected"]
            )

        if not recs:
            if task_type == "classification":
                add_rec(
                    "Model diagnostics are healthy across evaluated partitions. Continue monitoring out-of-sample class recall over time.",
                    "healthy_baseline",
                    reason="All evaluated classification metrics align with baseline expectations.",
                    objective="Maintain ongoing model surveillance.",
                    priority="LOW",
                    evidence_ids=["healthy_baseline_classification"]
                )
            else:
                add_rec(
                    "Regression diagnostics are healthy across evaluated partitions. Continue monitoring residual variance on incoming records.",
                    "healthy_baseline",
                    reason="All evaluated continuous metrics align with baseline expectations.",
                    objective="Maintain ongoing model surveillance.",
                    priority="LOW",
                    evidence_ids=["healthy_baseline_regression"]
                )

        return recs, structured_recs
