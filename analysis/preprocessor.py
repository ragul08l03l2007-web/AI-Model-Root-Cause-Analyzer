# analysis/preprocessor.py
"""
PreprocessingEngine: Dynamic tabular data preprocessing, feature exclusion rationales,
datetime decomposition, and adaptive encoding pipeline.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from analysis.evidence import safe_primitive


def _make_one_hot_encoder():
    """Construct OneHotEncoder with cross-version compatibility for sparse matrices."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class PreprocessingEngine:
    """
    Handles feature discovery, transparent exclusions, datetime feature engineering,
    and adaptive sklearn ColumnTransformer construction.
    """

    ID_KEYWORDS = {
        "id", "uuid", "guid", "key", "pk", "fk", "code", "hash", "token",
        "ssn", "identifier", "cust_id", "user_id", "record_id", "row_id"
    }

    @classmethod
    def detect_identifiers(cls, df: pd.DataFrame, target_column: str) -> List[str]:
        id_columns = []
        total_rows = len(df)
        for col in df.columns:
            if col == target_column:
                continue
            series = df[col]
            col_str = str(col).lower().strip()
            nunique = int(series.nunique(dropna=True))
            dtype_str = str(series.dtype).lower()

            is_id_name = any(kw in col_str for kw in cls.ID_KEYWORDS)
            is_numeric = pd.api.types.is_numeric_dtype(series)

            if is_id_name and (nunique >= 3 or total_rows < 5):
                id_columns.append(str(col))
            elif nunique == total_rows and total_rows >= 10:
                if is_id_name or not is_numeric or (is_numeric and series.min() in {0, 1} and series.max() in {total_rows - 1, total_rows}):
                    id_columns.append(str(col))
            elif not is_numeric and total_rows >= 20 and (nunique / total_rows) >= 0.95 and is_id_name:
                id_columns.append(str(col))

        return list(dict.fromkeys(id_columns))

    @classmethod
    def detect_datetime_features(cls, df: pd.DataFrame, target_column: str) -> List[str]:
        dt_columns = []
        for col in df.columns:
            if col == target_column:
                continue
            series = df[col]
            dtype_str = str(series.dtype).lower()
            if "datetime" in dtype_str or "date" in dtype_str or "timestamp" in dtype_str:
                dt_columns.append(str(col))
            elif dtype_str == "object" or dtype_str.startswith("string"):
                sample = series.dropna().head(20).astype(str)
                if not sample.empty and sample.str.len().mean() <= 35:
                    try:
                        pd.to_datetime(sample, format="mixed", errors="raise")
                        dt_columns.append(str(col))
                    except Exception:
                        pass
        return dt_columns

    @classmethod
    def detect_text_features(cls, df: pd.DataFrame, target_column: str) -> List[str]:
        text_columns = []
        for col in df.columns:
            if col == target_column:
                continue
            series = df[col]
            if not pd.api.types.is_numeric_dtype(series):
                non_null = series.dropna().astype(str)
                if not non_null.empty:
                    avg_char_len = non_null.str.len().mean()
                    avg_words = non_null.str.split().str.len().mean()
                    if avg_char_len > 60 or avg_words > 8:
                        text_columns.append(str(col))
        return text_columns

    @classmethod
    def prepare_features(
        cls,
        df: pd.DataFrame,
        target_column: str
    ) -> Tuple[pd.DataFrame, List[str], List[str], List[str], Dict[str, str]]:
        """
        Filters features, records transparent exclusions, expands datetime features,
        and segregates numeric and categorical columns.
        """
        total_rows = len(df)
        constant_columns = [
            str(col) for col in df.columns
            if col != target_column and df[col].nunique(dropna=False) <= 1
        ]
        detected_identifiers = cls.detect_identifiers(df, target_column)
        detected_text = cls.detect_text_features(df, target_column)
        detected_datetime = cls.detect_datetime_features(df, target_column)

        excluded_features: Dict[str, str] = {}
        candidate_features = [str(c) for c in df.columns if c != target_column]

        for col in constant_columns:
            excluded_features[col] = "Constant column containing only 1 distinct value (zero predictive variance)."

        for col in detected_text:
            excluded_features[col] = "Unstructured high-entropy text column unsuitable for standard tabular models."

        remaining_after_specials = [
            c for c in candidate_features
            if c not in excluded_features and c not in detected_identifiers
        ]
        if remaining_after_specials:
            for col in detected_identifiers:
                excluded_features[col] = "Detected as a unique non-predictive identifier column that may cause row memorization."

        usable_features = [c for c in candidate_features if c not in excluded_features]
        if not usable_features:
            usable_features = [c for c in candidate_features if c not in constant_columns]
            excluded_features.clear()

        if not usable_features:
            raise ValueError("No usable feature columns remain after data inspection.")

        X_df = df[usable_features].copy()

        # Datetime expansion
        for dt_col in detected_datetime:
            if dt_col in X_df.columns:
                try:
                    parsed_dt = pd.to_datetime(X_df[dt_col], format="mixed", errors="coerce")
                    X_df[f"{dt_col}_year"] = parsed_dt.dt.year.fillna(2000)
                    X_df[f"{dt_col}_month"] = parsed_dt.dt.month.fillna(1)
                    X_df[f"{dt_col}_day"] = parsed_dt.dt.day.fillna(1)
                    X_df[f"{dt_col}_dayofweek"] = parsed_dt.dt.dayofweek.fillna(0)
                    X_df = X_df.drop(columns=[dt_col])
                except Exception:
                    pass

        feature_names = [str(c) for c in X_df.columns]
        numeric_columns = [str(c) for c in X_df.select_dtypes(include=["number", "bool"]).columns]
        categorical_columns = [c for c in feature_names if c not in numeric_columns]

        return X_df, feature_names, numeric_columns, categorical_columns, excluded_features

    @classmethod
    def build_column_transformer(
        cls,
        numeric_columns: List[str],
        categorical_columns: List[str]
    ) -> ColumnTransformer:
        """
        Builds a robust sklearn ColumnTransformer with appropriate imputation and encoding.
        """
        transformers = []

        if numeric_columns:
            from sklearn.pipeline import Pipeline
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ])
            transformers.append(("num", num_pipe, numeric_columns))

        if categorical_columns:
            from sklearn.pipeline import Pipeline
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_one_hot_encoder()),
            ])
            transformers.append(("cat", cat_pipe, categorical_columns))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    @classmethod
    def build_preprocessed_matrices(
        cls,
        X_df: pd.DataFrame,
        y_raw: pd.Series,
        task_type: str = "classification"
    ) -> Tuple[ColumnTransformer, List[str], np.ndarray, np.ndarray, List[str]]:
        """
        Builds, fits, and returns preprocessed feature matrices and target arrays.
        Returns: (preprocessor, transformed_feature_names, X_transformed, y_encoded, classes)
        """
        from sklearn.preprocessing import LabelEncoder
        from scipy import sparse

        feature_names = [str(c) for c in X_df.columns]
        numeric_columns = [str(c) for c in X_df.select_dtypes(include=["number", "bool"]).columns]
        categorical_columns = [c for c in feature_names if c not in numeric_columns]

        preprocessor = cls.build_column_transformer(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns
        )

        X_transformed = preprocessor.fit_transform(X_df)
        if sparse.issparse(X_transformed):
            X_transformed = X_transformed.toarray()
        else:
            X_transformed = np.asarray(X_transformed)

        # Get feature names
        try:
            transformed_feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            transformed_feature_names = []
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

        # Target encoding
        if task_type == "classification":
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_raw.astype(str))
            classes = [str(c) for c in label_encoder.classes_]
        else:
            y_encoded = np.asarray(pd.to_numeric(y_raw, errors="coerce").fillna(0.0), dtype=float)
            classes = []

        return preprocessor, transformed_feature_names, X_transformed, y_encoded, classes

