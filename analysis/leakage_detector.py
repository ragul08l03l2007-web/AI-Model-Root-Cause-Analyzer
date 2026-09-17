# analysis/leakage_detector.py
"""
LeakageDetector: Discovers potential target leakage, identical feature columns,
and suspicious empirical target proxies. Emits DiagnosticEvidence objects.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from analysis.evidence import DiagnosticEvidence, safe_primitive


class LeakageDetector:
    """
    Scans dataset and model predictors for signs of data leakage and target proxies.
    """

    @classmethod
    def detect_leakage(
        cls,
        df: pd.DataFrame,
        target_column: str,
        task_type: str
    ) -> Tuple[List[Dict[str, Any]], List[DiagnosticEvidence]]:
        findings = []
        evidence_list: List[DiagnosticEvidence] = []
        if target_column not in df.columns:
            return findings, evidence_list

        target_series = df[target_column]
        total_rows = len(df)

        for col in df.columns:
            if col == target_column:
                continue

            series = df[col]
            feat_name = str(col)

            # 1. Exact identical column
            if series.equals(target_series):
                msg = f"Column '{feat_name}' is 100% identical to target column '{target_column}'."
                findings.append({
                    "feature": feat_name,
                    "reason": "Direct Duplicate of Target",
                    "evidence": msg,
                    "severity": "CRITICAL",
                    "evidence_id": f"leakage_duplicate_{feat_name}",
                })
                evidence_list.append(DiagnosticEvidence(
                    evidence_id=f"leakage_duplicate_{feat_name}",
                    signal_name=f"Direct Target Duplicate Feature '{feat_name}'",
                    domain="leakage",
                    metric="exact_match_ratio",
                    observed_value=1.0,
                    baseline_value=0.0,
                    magnitude=1.0,
                    direction="identical",
                    sample_support=total_rows,
                    affected_population=f"All {total_rows} rows",
                    affected_features=[feat_name],
                    signal_type="leakage",
                    strength="HIGH",
                    reliability=1.0,
                    model_scope="dataset",
                    context=msg
                ))
                continue

            # 2. Extreme correlation
            if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target_series):
                valid_mask = series.notna() & target_series.notna()
                if valid_mask.sum() >= 8:
                    try:
                        corr = abs(float(series[valid_mask].corr(target_series[valid_mask])))
                        if corr >= 0.98:
                            msg = f"Pearson correlation between '{feat_name}' and target is {corr:.4f} (>= 0.98)."
                            findings.append({
                                "feature": feat_name,
                                "reason": "Suspiciously Perfect Correlation with Target",
                                "evidence": msg,
                                "severity": "CRITICAL",
                                "evidence_id": f"leakage_correlation_{feat_name}",
                            })
                            evidence_list.append(DiagnosticEvidence(
                                evidence_id=f"leakage_correlation_{feat_name}",
                                signal_name=f"Near-Deterministic Target Association in '{feat_name}' (r = {corr:.4f})",
                                domain="leakage",
                                metric="pearson_correlation",
                                observed_value=round(corr, 4),
                                baseline_value=0.0,
                                magnitude=min(1.0, corr),
                                direction="near_deterministic",
                                sample_support=int(valid_mask.sum()),
                                affected_population=f"Feature '{feat_name}'",
                                affected_features=[feat_name],
                                signal_type="leakage",
                                strength="HIGH",
                                reliability=0.98,
                                model_scope="dataset",
                                context=msg
                            ))
                            continue
                        elif corr >= 0.90:
                            msg = f"Pearson correlation between '{feat_name}' and target is {corr:.4f} (>= 0.90)."
                            findings.append({
                                "feature": feat_name,
                                "reason": "Very Strong Correlation with Target (Possible Post-Outcome Variable)",
                                "evidence": msg,
                                "severity": "HIGH",
                                "evidence_id": f"leakage_correlation_{feat_name}",
                            })
                            evidence_list.append(DiagnosticEvidence(
                                evidence_id=f"leakage_correlation_{feat_name}",
                                signal_name=f"High Target Correlation in '{feat_name}' (r = {corr:.4f})",
                                domain="leakage",
                                metric="pearson_correlation",
                                observed_value=round(corr, 4),
                                baseline_value=0.0,
                                magnitude=min(1.0, corr),
                                direction="high_correlation",
                                sample_support=int(valid_mask.sum()),
                                affected_population=f"Feature '{feat_name}'",
                                affected_features=[feat_name],
                                signal_type="leakage",
                                strength="HIGH",
                                reliability=0.95,
                                model_scope="dataset",
                                context=msg
                            ))
                    except Exception:
                        pass

            # 3. Categorical exact overlap
            if not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_numeric_dtype(target_series):
                if series.astype(str).equals(target_series.astype(str)):
                    msg = f"Column '{feat_name}' matches target column '{target_column}' exactly in all rows."
                    findings.append({
                        "feature": feat_name,
                        "reason": "Direct Categorical Copy of Target",
                        "evidence": msg,
                        "severity": "CRITICAL",
                        "evidence_id": f"leakage_categorical_copy_{feat_name}",
                    })
                    evidence_list.append(DiagnosticEvidence(
                        evidence_id=f"leakage_categorical_copy_{feat_name}",
                        signal_name=f"Categorical Target Copy Feature '{feat_name}'",
                        domain="leakage",
                        metric="categorical_identity_ratio",
                        observed_value=1.0,
                        baseline_value=0.0,
                        magnitude=1.0,
                        direction="identical",
                        sample_support=total_rows,
                        affected_population=f"All {total_rows} rows",
                        affected_features=[feat_name],
                        signal_type="leakage",
                        strength="HIGH",
                        reliability=1.0,
                        model_scope="dataset",
                        context=msg
                    ))

        return safe_primitive(findings), evidence_list
