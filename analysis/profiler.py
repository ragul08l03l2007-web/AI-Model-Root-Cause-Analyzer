# analysis/profiler.py
"""
DataProfiler: Universal dataset profiling, type detection, and statistical quality analysis.
Dynamically audits any tabular dataset without domain-specific assumptions.
"""

import math
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from analysis.evidence import safe_primitive


class DataProfiler:
    """
    Performs comprehensive, dataset-driven profiling of arbitrary tabular datasets.
    """

    ID_KEYWORDS = {
        "id", "uuid", "guid", "key", "pk", "fk", "code", "hash", "token",
        "ssn", "identifier", "cust_id", "user_id", "record_id", "row_id"
    }

    @classmethod
    def profile_dataset(cls, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return cls._empty_profile()

        total_rows = len(df)
        total_columns = len(df.columns)
        column_names = [str(c) for c in df.columns]
        column_types = {str(col): str(df[col].dtype) for col in df.columns}

        # 1. Missingness & Uniqueness
        missing_values = {}
        missing_percentages = {}
        unique_counts = {}
        unique_ratios = {}

        for column in df.columns:
            count = int(df[column].isna().sum())
            missing_values[str(column)] = count
            missing_percentages[str(column)] = round((count / total_rows) * 100, 2) if total_rows > 0 else 0.0

            nunique = int(df[column].nunique(dropna=True))
            unique_counts[str(column)] = nunique
            unique_ratios[str(column)] = round((nunique / total_rows) * 100, 2) if total_rows > 0 else 0.0

        total_missing_cells = int(sum(missing_values.values()))
        total_cells = total_rows * total_columns
        missing_cell_percentage = round((total_missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

        duplicate_rows = int(df.duplicated().sum())
        duplicate_percentage = round((duplicate_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0

        # 2. Structural Column Discovery
        numeric_features = []
        categorical_features = []
        boolean_features = []
        datetime_features = []
        text_features = []
        constant_features = []
        near_constant_features = []
        identifier_candidates = []
        high_cardinality_features = []

        for column in df.columns:
            series = df[column]
            col_str = str(column).lower().strip()
            nunique = series.nunique(dropna=True)
            nunique_all = series.nunique(dropna=False)
            dtype_str = str(series.dtype).lower()

            # Constant check
            if nunique_all <= 1:
                constant_features.append(str(column))
                continue

            # Near-constant check (>= 98% non-nulls have same value)
            if nunique >= 1 and total_rows >= 20:
                top_freq = int(series.value_counts(dropna=True).iloc[0])
                if (top_freq / total_rows) >= 0.98:
                    near_constant_features.append(str(column))

            # Datetime detection
            is_datetime = False
            if "datetime" in dtype_str or "date" in dtype_str or "timestamp" in dtype_str:
                is_datetime = True
            elif dtype_str == "object" or dtype_str.startswith("string"):
                sample = series.dropna().head(20).astype(str)
                if not sample.empty and sample.str.len().mean() <= 35:
                    try:
                        pd.to_datetime(sample, format="mixed", errors="raise")
                        is_datetime = True
                    except Exception:
                        is_datetime = False

            if is_datetime:
                datetime_features.append(str(column))
                continue

            # Boolean check
            if dtype_str in {"bool", "boolean"}:
                boolean_features.append(str(column))
                numeric_features.append(str(column))
                continue

            # Numeric check
            if pd.api.types.is_numeric_dtype(series):
                vals = set(series.dropna().unique())
                if vals.issubset({0, 1, 0.0, 1.0}) and len(vals) == 2:
                    boolean_features.append(str(column))

                # Identifier check for sequential integers or ID keywords
                is_id_name = any(kw in col_str for kw in cls.ID_KEYWORDS)
                if nunique == total_rows and total_rows >= 10:
                    if is_id_name or (series.min() in {0, 1} and series.max() in {total_rows - 1, total_rows}):
                        identifier_candidates.append(str(column))

                numeric_features.append(str(column))
            else:
                # Non-numeric
                non_null = series.dropna().astype(str)
                is_id_name = any(kw in col_str for kw in cls.ID_KEYWORDS)
                if (nunique == total_rows or (total_rows >= 20 and nunique / total_rows >= 0.95)) and is_id_name:
                    identifier_candidates.append(str(column))
                elif total_rows >= 20 and nunique == total_rows and non_null.str.len().mean() < 40:
                    identifier_candidates.append(str(column))

                # Text vs Categorical detection
                avg_len = non_null.str.len().mean() if not non_null.empty else 0
                avg_words = non_null.str.split().str.len().mean() if not non_null.empty else 0

                if avg_len > 60 or avg_words > 8:
                    text_features.append(str(column))
                else:
                    categorical_features.append(str(column))
                    if nunique > 20 or (total_rows >= 50 and (nunique / total_rows) > 0.40):
                        high_cardinality_features.append(str(column))

        # 3. Numeric Statistics & IQR Outlier Analysis
        outliers = {}
        outlier_details = {}
        numeric_statistics = {}

        for column in numeric_features:
            if column in constant_features or column in boolean_features:
                outliers[str(column)] = 0
                continue

            clean_series = pd.to_numeric(df[column], errors="coerce").dropna().astype(float)
            if len(clean_series) < 4 or clean_series.nunique() <= 2:
                outliers[str(column)] = 0
                continue

            q1 = float(clean_series.quantile(0.25))
            q3 = float(clean_series.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)

            outlier_mask = (clean_series < lower_bound) | (clean_series > upper_bound)
            outlier_count = int(outlier_mask.sum())
            outliers[str(column)] = outlier_count

            outlier_details[str(column)] = {
                "count": outlier_count,
                "percentage": round((outlier_count / len(clean_series)) * 100, 2),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
            }

            std_val = float(clean_series.std()) if len(clean_series) > 1 else 0.0
            var_val = float(clean_series.var()) if len(clean_series) > 1 else 0.0
            skew_val = float(clean_series.skew()) if std_val > 0 else 0.0

            numeric_statistics[str(column)] = {
                "min": round(float(clean_series.min()), 4),
                "q25": round(q1, 4),
                "median": round(float(clean_series.median()), 4),
                "mean": round(float(clean_series.mean()), 4),
                "q75": round(q3, 4),
                "max": round(float(clean_series.max()), 4),
                "std": round(std_val, 4),
                "variance": round(var_val, 4),
                "skewness": round(skew_val, 4),
            }

        # 4. Class / Category Distributions
        class_distribution = {}
        for column in df.columns:
            if str(column) in constant_features:
                continue
            nunique = int(df[column].nunique(dropna=True))
            if 2 <= nunique <= 15:
                counts = df[column].value_counts(dropna=True).to_dict()
                class_distribution[str(column)] = {
                    str(safe_primitive(k)): int(v) for k, v in counts.items()
                }

        # 5. Multicollinearity
        high_correlations = []
        if len(numeric_features) >= 2:
            try:
                num_df = df[numeric_features].apply(pd.to_numeric, errors="coerce")
                corr_matrix = num_df.corr(method="pearson").abs()
                cols = list(corr_matrix.columns)
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        val = corr_matrix.iloc[i, j]
                        if not pd.isna(val) and val >= 0.85:
                            high_correlations.append({
                                "feature_a": str(cols[i]),
                                "feature_b": str(cols[j]),
                                "correlation": round(float(val), 4),
                            })
            except Exception:
                high_correlations = []

        # 6. Quality Score & Findings
        structured_findings = []
        warnings = []

        cols_with_missing = {k: v for k, v in missing_values.items() if v > 0}
        if cols_with_missing:
            pct = missing_cell_percentage
            cols_missing_str = ", ".join(
                f"'{c}' ({missing_percentages[c]}% missing)" for c in list(cols_with_missing.keys())[:4]
            )
            if len(cols_with_missing) > 4:
                cols_missing_str += f" and {len(cols_with_missing) - 4} more"

            sev = "HIGH" if pct > 10.0 else ("MEDIUM" if pct > 2.0 else "LOW")
            f_item = {
                "finding": "Missing Values in Feature Matrix",
                "evidence": f"{total_missing_cells:,} missing cell(s) ({pct}% of total cells) across: {cols_missing_str}.",
                "severity": sev,
                "impact": "Requires automated imputation or row dropping, which can introduce statistical distortion."
            }
            structured_findings.append(f_item)
            warnings.append(f_item["evidence"])

        if duplicate_rows > 0:
            sev = "HIGH" if duplicate_percentage > 10.0 else "MEDIUM"
            f_item = {
                "finding": "Duplicate Rows Present",
                "evidence": f"{duplicate_rows:,} duplicate row(s) ({duplicate_percentage}%) detected in the dataset.",
                "severity": sev,
                "impact": "Duplicate records can cause data leakage between training and testing partitions."
            }
            structured_findings.append(f_item)
            warnings.append(f_item["evidence"])

        if constant_features:
            f_item = {
                "finding": "Zero-Variance Constant Features",
                "evidence": f"Feature(s) containing only one unique value: {', '.join(constant_features)}.",
                "severity": "MEDIUM",
                "impact": "Constant columns provide zero predictive variance and should be excluded from modeling."
            }
            structured_findings.append(f_item)
            warnings.append(f_item["evidence"])

        if near_constant_features:
            f_item = {
                "finding": "Near-Constant Low-Variance Features",
                "evidence": f"Feature(s) where >=98% of values are identical: {', '.join(near_constant_features)}.",
                "severity": "LOW",
                "impact": "May contribute minimal predictive signal or cause sparse encoding issues."
            }
            structured_findings.append(f_item)
            warnings.append(f_item["evidence"])

        cols_with_outliers = [c for c, count in outliers.items() if count > 0]
        if cols_with_outliers:
            top_outliers = sorted(
                [(c, outliers[c], outlier_details[c]["percentage"]) for c in cols_with_outliers],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            outlier_str = ", ".join(f"'{c}' ({cnt} outliers, {pct}%)" for c, cnt, pct in top_outliers)
            f_item = {
                "finding": "Potential Statistical Outliers (IQR Rule)",
                "evidence": f"Potential numerical outliers detected via IQR rule [Q1 - 1.5*IQR, Q3 + 1.5*IQR] in {len(cols_with_outliers)} column(s): {outlier_str}.",
                "severity": "MEDIUM",
                "impact": "Unusual observations may skew linear model coefficients or inflate squared error metrics; verify if these represent valid extreme values."
            }
            structured_findings.append(f_item)
            warnings.append(f_item["evidence"])

        if high_cardinality_features:
            f_item = {
                "finding": "High-Cardinality Categorical Features",
                "evidence": f"Categorical feature(s) with high distinct category counts: {', '.join(high_cardinality_features)}.",
                "severity": "MEDIUM",
                "impact": "Expands one-hot encoding dimensions and can cause sparse-data overfitting."
            }
            structured_findings.append(f_item)
            warnings.append(f_item["evidence"])

        if identifier_candidates:
            f_item = {
                "finding": "Potential Non-Predictive Identifier Columns",
                "evidence": f"Identifier-like pattern or 100% uniqueness detected in: {', '.join(identifier_candidates)}.",
                "severity": "MEDIUM",
                "impact": "Identifiers must be reviewed to prevent models from memorizing specific record IDs."
            }
            structured_findings.append(f_item)
            warnings.append(f_item["evidence"])

        if high_correlations:
            corr_pairs_str = ", ".join(
                f"'{pair['feature_a']}' & '{pair['feature_b']}' (r={pair['correlation']:.2f})"
                for pair in high_correlations[:3]
            )
            f_item = {
                "finding": "Multicollinearity Between Features",
                "evidence": f"Strong correlation (r >= 0.85) detected between: {corr_pairs_str}.",
                "severity": "MEDIUM",
                "impact": "Redundant feature information makes individual linear coefficients unstable."
            }
            structured_findings.append(f_item)
            warnings.append(f_item["evidence"])

        if not warnings:
            warnings.append("No major data-quality problems detected in the dataset.")

        # Quality scoring (0-100)
        quality_score = 100
        deductions = []

        if total_missing_cells > 0:
            missing_rate = total_missing_cells / total_cells
            pts = min(35, int(missing_rate * 100 * 2))
            quality_score -= pts
            deductions.append(f"{total_missing_cells:,} missing cell(s) ({missing_cell_percentage}%)")

        if duplicate_rows > 0:
            dup_rate = duplicate_rows / total_rows
            pts = min(25, int(dup_rate * 100 * 2))
            quality_score -= pts
            deductions.append(f"{duplicate_rows:,} duplicate row(s) ({duplicate_percentage}%)")

        if constant_features:
            pts = min(15, len(constant_features) * 5)
            quality_score -= pts
            deductions.append(f"{len(constant_features)} constant feature(s)")

        if cols_with_outliers:
            pts = min(15, len(cols_with_outliers) * 3)
            quality_score -= pts
            deductions.append(f"potential outliers in {len(cols_with_outliers)} numeric column(s)")

        if high_cardinality_features:
            pts = min(10, len(high_cardinality_features) * 3)
            quality_score -= pts
            deductions.append(f"{len(high_cardinality_features)} high-cardinality feature(s)")

        quality_score = max(10, min(100, quality_score))
        overall_quality = "GOOD" if quality_score >= 80 else ("MODERATE" if quality_score >= 50 else "POOR")

        if not deductions:
            quality_reason = "Zero missing values and zero duplicate rows detected; all features show healthy distribution variance."
        else:
            quality_reason = f"{'; '.join(deductions[:3])} detected; no critical structural corruption found."

        duplicate_handling_note = (
            f"{duplicate_rows} duplicate row(s) ({duplicate_percentage}%) detected. "
            "Duplicates are retained in feature matrices during exploration; ensure duplicate entries represent independent real-world observations."
            if duplicate_rows > 0
            else "Zero duplicate rows detected in the dataset."
        )

        return safe_primitive({
            "total_rows": total_rows,
            "total_columns": total_columns,
            "column_names": column_names,
            "column_types": column_types,
            "missing_values": missing_values,
            "missing_percentages": missing_percentages,
            "total_missing_cells": total_missing_cells,
            "missing_cell_percentage": missing_cell_percentage,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": duplicate_percentage,
            "duplicate_handling_note": duplicate_handling_note,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "boolean_features": boolean_features,
            "datetime_features": datetime_features,
            "text_features": text_features,
            "constant_features": constant_features,
            "near_constant_features": near_constant_features,
            "identifier_candidates": identifier_candidates,
            "high_cardinality_features": high_cardinality_features,
            "unique_counts": unique_counts,
            "unique_ratios": unique_ratios,
            "outliers": outliers,
            "outlier_details": outlier_details,
            "numeric_statistics": numeric_statistics,
            "class_distribution": class_distribution,
            "high_correlations": high_correlations,
            "structured_findings": structured_findings,
            "warnings": warnings,
            "quality_score": quality_score,
            "overall_quality": overall_quality,
            "quality_reason": quality_reason,
        })

    @classmethod
    def _empty_profile(cls) -> Dict[str, Any]:
        return {
            "total_rows": 0,
            "total_columns": 0,
            "column_names": [],
            "column_types": {},
            "missing_values": {},
            "missing_percentages": {},
            "total_missing_cells": 0,
            "missing_cell_percentage": 0.0,
            "duplicate_rows": 0,
            "duplicate_percentage": 0.0,
            "numeric_features": [],
            "categorical_features": [],
            "boolean_features": [],
            "datetime_features": [],
            "text_features": [],
            "constant_features": [],
            "near_constant_features": [],
            "identifier_candidates": [],
            "high_cardinality_features": [],
            "unique_counts": {},
            "unique_ratios": {},
            "outliers": {},
            "outlier_details": {},
            "numeric_statistics": {},
            "class_distribution": {},
            "high_correlations": [],
            "structured_findings": [],
            "warnings": ["Dataset is empty or invalid."],
            "quality_score": 0,
            "overall_quality": "POOR",
            "quality_reason": "Dataset is empty.",
            "duplicate_handling_note": "No rows present.",
        }
