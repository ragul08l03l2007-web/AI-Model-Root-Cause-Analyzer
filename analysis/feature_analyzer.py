# analysis/feature_analyzer.py
"""
FeatureAnalyzer: Feature predictive importance, empirical non-causal relationships,
adaptive feature binning, and cross-model stability assessment.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from analysis.evidence import DiagnosticEvidence, safe_primitive


class FeatureAnalyzer:
    """
    Extracts feature importance, calculates non-causal relationship summaries,
    performs adaptive binning across arbitrary classes, and evaluates cross-model feature stability.
    """

    @classmethod
    def extract_raw_importance(
        cls,
        model: Any,
        transformed_feature_names: List[str],
        original_feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Extracts importance scores and maps one-hot encoded dummies back to original feature names.
        """
        raw_scores = None
        if hasattr(model, "feature_importances_"):
            raw_scores = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            if coef.ndim == 1:
                raw_scores = np.abs(coef)
            else:
                raw_scores = np.mean(np.abs(coef), axis=0)

        if raw_scores is None or len(raw_scores) != len(transformed_feature_names):
            raw_scores = np.ones(len(transformed_feature_names))

        # Aggregate dummy features back to original feature names
        # Sort candidate feature names by length descending to match the most specific feature first
        sorted_candidates = sorted(original_feature_names, key=len, reverse=True)
        aggregated: Dict[str, float] = {col: 0.0 for col in original_feature_names}
        for name, score in zip(transformed_feature_names, raw_scores):
            matched = False
            for orig in sorted_candidates:
                if (
                    name == orig
                    or name == f"num__{orig}"
                    or name == f"cat__{orig}"
                    or name.startswith(f"cat__{orig}_")
                    or name.startswith(f"cat__{orig}__")
                    or name.startswith(f"{orig}_")
                    or name.startswith(f"num__{orig}_")
                ):
                    aggregated[orig] += float(score)
                    matched = True
                    break
            if not matched:
                aggregated[name] = aggregated.get(name, 0.0) + float(score)

        return aggregated

    @classmethod
    def analyze_feature_impact(
        cls,
        model: Any,
        transformed_feature_names: List[str],
        original_feature_names: List[str],
        X_df: pd.DataFrame,
        y_raw: pd.Series,
        task_type: str,
        target_profile: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[DiagnosticEvidence]]:
        """
        Calculates relative share percentages, influence tiers, non-causal predictive association,
        adaptive binned relationships, and emits DiagnosticEvidence.
        """
        raw_imp = cls.extract_raw_importance(model, transformed_feature_names, original_feature_names)
        total_imp = sum(raw_imp.values())
        if total_imp <= 0:
            total_imp = 1.0

        sorted_features = sorted(raw_imp.items(), key=lambda x: x[1], reverse=True)
        evidence_list: List[DiagnosticEvidence] = []
        n_samples = len(X_df)

        feature_impact: Dict[str, Any] = {}
        for rank, (feat, score) in enumerate(sorted_features, 1):
            share_pct = round((score / total_imp) * 100, 2)
            if share_pct >= 40.0:
                tier = "Very Strong Model Influence"
            elif share_pct >= 20.0:
                tier = "Strong Model Influence"
            elif share_pct >= 8.0:
                tier = "Moderate Model Influence"
            else:
                tier = "Low Model Influence"

            direction = ""
            rel_summary = ""
            if feat in X_df.columns:
                series = X_df[feat]
                if pd.api.types.is_numeric_dtype(series):
                    num_series = pd.to_numeric(series, errors="coerce")
                    if task_type == "regression":
                        num_y = pd.to_numeric(y_raw, errors="coerce")
                        valid = num_series.notna() & num_y.notna()
                        if valid.sum() >= 5:
                            corr = float(num_series[valid].corr(num_y[valid]))
                            is_linear_model = hasattr(model, "coef_") and not hasattr(model, "estimators_")
                            if is_linear_model:
                                if corr >= 0.50:
                                    direction = "Higher values strongly associated with higher target values"
                                    rel_summary = f"Strong positive linear correlation (r = {corr:+.2f}) with target."
                                elif corr >= 0.25:
                                    direction = "Higher values positively correlated with higher target values"
                                    rel_summary = f"Positive linear correlation (r = {corr:+.2f}) with target."
                                elif corr <= -0.50:
                                    direction = "Higher values strongly associated with lower target values"
                                    rel_summary = f"Strong negative linear correlation (r = {corr:+.2f}) with target."
                                elif corr <= -0.25:
                                    direction = "Higher values negatively correlated with target values"
                                    rel_summary = f"Negative linear correlation (r = {corr:+.2f}) with target."
                                elif abs(corr) >= 0.10:
                                    direction = "Moderate linear correlation across numeric feature range"
                                    rel_summary = f"Moderate linear correlation (r = {corr:+.2f})."
                                else:
                                    direction = "Non-linear or mixed relationship observed across the feature range"
                                    rel_summary = f"Weak linear correlation (r = {corr:+.2f})."
                            else:
                                if abs(corr) >= 0.35:
                                    tendency = "Positive" if corr > 0 else "Negative"
                                    direction = f"{tendency} empirical tendency observed across feature values"
                                    rel_summary = f"{tendency} empirical correlation (r = {corr:+.2f}) with target."
                                elif abs(corr) >= 0.15:
                                    direction = "Moderate empirical association across feature range"
                                    rel_summary = f"Moderate empirical correlation (r = {corr:+.2f})."
                                else:
                                    direction = "Non-linear or mixed relationship observed across the feature range"
                                    rel_summary = f"Weak linear correlation (r = {corr:+.2f})."
                    else:
                        # Multiclass / Binary generic group means
                        try:
                            means = num_series.groupby(y_raw.astype(str), observed=False).mean()
                            if len(means) >= 2:
                                highest_class = str(means.idxmax())
                                lowest_class = str(means.idxmin())
                                direction = f"Highest average values observed in class '{highest_class}', lowest in class '{lowest_class}'"
                                rel_summary = f"Mean varies from {means.min():.2f} (class '{lowest_class}') to {means.max():.2f} (class '{highest_class}')."
                        except Exception:
                            direction = "Differentiates classification boundaries across observation groups"
                            rel_summary = "Numerical variation informs model classification partitions."
                else:
                    direction = "Categorical distribution shift across predicted target subsets"
                    rel_summary = "Categorical variable informs partition boundaries."

            feature_impact[feat] = {
                "importance": round(float(score), 4),
                "relative_share_pct": share_pct,
                "influence_tier": tier,
                "rank": rank,
                "direction": direction,
                "relationship_summary": rel_summary,
                "interpretation_note": "Feature importance reflects predictive association within the trained model and does not establish real-world causation.",
            }

        # Check for single dominant predictor reliance
        if sorted_features:
            top_f, top_score = sorted_features[0]
            top_share = (top_score / total_imp) * 100
            second_share = (sorted_features[1][1] / total_imp) * 100 if len(sorted_features) > 1 else 0.0

            if top_share >= 48.0 and n_samples >= 15:
                evidence_list.append(DiagnosticEvidence(
                    evidence_id=f"dominant_feature_reliance_{top_f}",
                    signal_name=f"High Model Reliance on Single Feature '{top_f}' ({top_share:.1f}% Share)",
                    domain="feature_reliance",
                    metric="feature_importance_share",
                    observed_value=round(top_share, 2),
                    baseline_value=round(100.0 / max(1, len(sorted_features)), 2),
                    magnitude=min(1.0, top_share / 100.0),
                    direction="dominant_reliance",
                    sample_support=n_samples,
                    affected_population=f"Feature '{top_f}' (accounts for {top_share:.1f}% of predictive weight)",
                    affected_features=[top_f],
                    signal_type="behavioral",
                    strength="HIGH" if top_share >= 60.0 else "MODERATE",
                    reliability=0.90,
                    model_scope="selected_model",
                    context=f"Feature '{top_f}' provides {top_share:.1f}% of total model importance (next highest feature provides {second_share:.1f}%)."
                ))

        # Adaptive Binned Relationships (Top 4 Features)
        feature_relationships: List[Dict[str, Any]] = []
        for feat, _ in sorted_features[:4]:
            if feat not in X_df.columns:
                continue
            series = X_df[feat]

            if pd.api.types.is_numeric_dtype(series):
                num_series = pd.to_numeric(series, errors="coerce")
                valid = num_series.notna() & y_raw.notna()
                if valid.sum() >= 6:
                    u_vals = num_series[valid].nunique()
                    n_bins = min(4, max(2, u_vals // 3)) if u_vals >= 3 else 2
                    try:
                        qcuts = pd.qcut(num_series[valid], q=n_bins, duplicates="drop")
                        bins = []
                        for bin_label, grp_idx in num_series[valid].groupby(qcuts, observed=False).groups.items():
                            grp_y = y_raw.loc[grp_idx]
                            b_data: Dict[str, Any] = {
                                "bin": str(bin_label),
                                "sample_count": len(grp_y),
                            }
                            if task_type == "regression":
                                num_grp_y = pd.to_numeric(grp_y, errors="coerce").dropna()
                                b_data["target_mean"] = round(float(num_grp_y.mean()), 4) if not num_grp_y.empty else 0.0
                                b_data["target_median"] = round(float(num_grp_y.median()), 4) if not num_grp_y.empty else 0.0
                            else:
                                v_counts = grp_y.value_counts()
                                dom_cls = str(v_counts.index[0]) if not v_counts.empty else "N/A"
                                dom_pct = round((v_counts.iloc[0] / len(grp_y)) * 100, 1) if not v_counts.empty else 0.0
                                b_data["dominant_class"] = dom_cls
                                b_data["dominant_class_pct"] = dom_pct
                                # Multi-class frequency map for completeness
                                b_data["class_counts"] = {str(k): int(v) for k, v in v_counts.items()}
                            bins.append(b_data)

                        feature_relationships.append({
                            "feature": str(feat),
                            "feature_type": "numeric",
                            "bins": bins,
                        })
                    except Exception:
                        pass
            else:
                # Categorical relationships across top categories
                top_cats = series.value_counts().head(5).index
                categories = []
                total_cat_samples = len(series.dropna())
                covered_samples = 0
                for cat_val in top_cats:
                    mask = (series.astype(str) == str(cat_val)) & y_raw.notna()
                    grp_y = y_raw[mask]
                    covered_samples += len(grp_y)
                    c_data: Dict[str, Any] = {
                        "category": str(cat_val),
                        "sample_count": len(grp_y),
                    }
                    if task_type == "regression":
                        num_grp_y = pd.to_numeric(grp_y, errors="coerce").dropna()
                        c_data["target_mean"] = round(float(num_grp_y.mean()), 4) if not num_grp_y.empty else 0.0
                        c_data["target_median"] = round(float(num_grp_y.median()), 4) if not num_grp_y.empty else 0.0
                    else:
                        v_counts = grp_y.value_counts()
                        dom_cls = str(v_counts.index[0]) if not v_counts.empty else "N/A"
                        dom_pct = round((v_counts.iloc[0] / len(grp_y)) * 100, 1) if not v_counts.empty else 0.0
                        c_data["dominant_class"] = dom_cls
                        c_data["dominant_class_pct"] = dom_pct
                        c_data["class_counts"] = {str(k): int(v) for k, v in v_counts.items()}
                    categories.append(c_data)

                coverage_pct = round((covered_samples / total_cat_samples) * 100, 1) if total_cat_samples > 0 else 100.0
                feature_relationships.append({
                    "feature": str(feat),
                    "feature_type": "categorical",
                    "categories": categories,
                    "coverage_pct": coverage_pct,
                })

        return safe_primitive(feature_impact), safe_primitive(feature_relationships), evidence_list

    @classmethod
    def evaluate_cross_model_consistency(
        cls,
        fitted_models: Dict[str, Any],
        transformed_feature_names: List[str],
        original_feature_names: List[str],
        model_predictions: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[Dict[str, Any], List[DiagnosticEvidence]]:
        """
        Evaluates feature ranking stability and predictive consensus across all fitted model architectures.
        Tracks per-feature model support counts, coverage ratios, rank consistency, and importance variation.
        """
        evidence_list: List[DiagnosticEvidence] = []
        if not fitted_models:
            return {
                "consistency_score": 0.5,
                "status": "Single Model Evaluated",
                "top_consistent_features": [],
                "feature_stability": {},
                "explanation": "Only one model was available for cross-model consistency assessment.",
            }, evidence_list

        model_names = list(fitted_models.keys())
        total_models = len(model_names)
        n_feats = max(1, len(original_feature_names))

        # Extract normalized importances for each model
        model_importances: Dict[str, Dict[str, float]] = {}
        for name, mdl in fitted_models.items():
            try:
                imp = cls.extract_raw_importance(mdl, transformed_feature_names, original_feature_names)
                s = sum(imp.values()) if sum(imp.values()) > 0 else 1.0
                model_importances[name] = {f: (v / s) for f, v in imp.items()}
            except Exception:
                pass

        if not model_importances:
            return {
                "consistency_score": 0.5,
                "status": "Single Model Evaluated",
                "top_consistent_features": [],
                "feature_stability": {},
                "explanation": "No fitted model importances available for consistency assessment.",
            }, evidence_list

        # Calculate per-feature statistics across models
        feature_stability: Dict[str, Dict[str, Any]] = {}
        top_features_per_model: Dict[str, List[str]] = {}
        for m_name, imp_map in model_importances.items():
            sorted_f = sorted(imp_map.items(), key=lambda x: x[1], reverse=True)
            top_features_per_model[m_name] = [f for f, _ in sorted_f[:3]]

        for feat in original_feature_names:
            shares = [model_importances[m].get(feat, 0.0) for m in model_importances]
            ranks = []
            support_count = 0
            for m in model_importances:
                ranked = [f for f, _ in sorted(model_importances[m].items(), key=lambda x: x[1], reverse=True)]
                if feat in ranked:
                    r = ranked.index(feat) + 1
                    ranks.append(r)
                    if r <= max(2, n_feats // 3):
                        support_count += 1

            cov = round(support_count / total_models, 4) if total_models > 0 else 0.0
            mean_imp = round(float(np.mean(shares)), 4)
            std_imp = round(float(np.std(shares)), 4)
            rank_rng = [min(ranks), max(ranks)] if ranks else [0, 0]
            rank_consistency = round(max(0.0, 1.0 - (rank_rng[1] - rank_rng[0]) / n_feats), 4) if ranks else 0.0

            if cov >= 0.75 and std_imp <= 0.15:
                stab_label = "HIGH"
            elif cov >= 0.50:
                stab_label = "MODERATE"
            else:
                stab_label = "LOW"

            feature_stability[feat] = {
                "feature": feat,
                "supporting_models": support_count,
                "total_models": total_models,
                "coverage": cov,
                "mean_normalized_importance": mean_imp,
                "mean_importance_share": round(mean_imp * 100, 2),
                "std_importance_share": round(std_imp * 100, 2),
                "rank_consistency": rank_consistency,
                "rank_range": rank_rng,
                "stability": stab_label,
            }

        # Measure prediction-level agreement if predictions provided
        prediction_agreement = None
        if model_predictions and len(model_predictions) >= 2:
            try:
                p_names = list(model_predictions.keys())
                agreements = []
                for i in range(len(p_names)):
                    for j in range(i + 1, len(p_names)):
                        p1 = model_predictions[p_names[i]]
                        p2 = model_predictions[p_names[j]]
                        if len(p1) == len(p2) and len(p1) > 0:
                            if np.issubdtype(p1.dtype, np.number) and np.issubdtype(p2.dtype, np.number):
                                if np.std(p1) > 0 and np.std(p2) > 0:
                                    corr = float(np.corrcoef(p1, p2)[0, 1])
                                    agreements.append(max(0.0, corr))
                                else:
                                    agreements.append(1.0 if np.allclose(p1, p2) else 0.5)
                            else:
                                acc = float(np.mean(p1 == p2))
                                agreements.append(acc)
                if agreements:
                    prediction_agreement = round(float(np.mean(agreements)), 4)
            except Exception:
                prediction_agreement = None

        # Check intersection of top features across models
        all_top_2 = [set(feats[:2]) for feats in top_features_per_model.values() if len(feats) >= 2]
        common = set.intersection(*all_top_2) if all_top_2 else set()
        sorted_common = sorted(list(common))

        high_cov_feats = [f for f, data in feature_stability.items() if data["coverage"] >= 0.75]

        if sorted_common and len(high_cov_feats) >= 1:
            status_str = f"Strong cross-model alignment: multiple distinct model architectures independently prioritize {', '.join(sorted_common)}."
            expl_str = f"Across {total_models} evaluated model candidates, {', '.join(sorted_common)} consistently rank in the top predictive features."
            score = 0.88

            evidence_list.append(DiagnosticEvidence(
                evidence_id="cross_model_feature_consensus",
                signal_name=f"Cross-Model Feature Alignment on {', '.join(sorted_common)}",
                domain="feature_reliance",
                metric="cross_model_consensus_score",
                observed_value=score,
                baseline_value=0.50,
                magnitude=0.85,
                direction="consensus",
                sample_support=total_models,
                affected_population=f"All {total_models} evaluated model architectures",
                affected_features=sorted_common,
                signal_type="structural",
                strength="HIGH",
                reliability=0.95,
                model_scope="cross_model",
                supporting_models=model_names,
                context=expl_str
            ))
        elif high_cov_feats:
            status_str = f"Moderate cross-model alignment: models partially converge on top predictor(s) ({', '.join(high_cov_feats[:2])})."
            expl_str = f"Candidate model architectures share core predictors ({', '.join(high_cov_feats[:2])}) but differ in secondary feature rank ordering."
            score = 0.65
        else:
            status_str = "Divergent feature prioritization: evaluated model architectures distribute weight across different predictor combinations."
            expl_str = "Candidate model architectures distribute predictive weight across differing feature sets."
            score = 0.45

        result_payload = {
            "consistency_score": score,
            "status": status_str,
            "top_consistent_features": sorted_common,
            "feature_stability": feature_stability,
            "top_features_per_model": top_features_per_model,
            "prediction_agreement": prediction_agreement,
            "explanation": expl_str,
        }
        return safe_primitive(result_payload), evidence_list
