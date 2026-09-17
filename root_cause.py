import pandas as pd


def analyze_root_cause(df, target_column):
    """
    Analyze dataset problems that could affect
    an AI/ML model's performance.
    """

    report = {}

    # Basic dataset information
    report["rows"] = len(df)
    report["columns"] = len(df.columns)

    # Missing values
    missing = df.isnull().sum()
    report["missing_values"] = int(missing.sum())

    # Duplicate rows
    report["duplicate_rows"] = int(df.duplicated().sum())

    # Root causes
    root_causes = []

    # Target column analysis
    if target_column in df.columns:

        report["target_column"] = target_column
        report["target_missing"] = int(df[target_column].isnull().sum())
        report["target_unique_values"] = int(df[target_column].nunique())

        distribution = (
            df[target_column]
            .value_counts(dropna=False)
            .to_dict()
        )

        report["target_distribution"] = distribution

        # Check for class imbalance
        if len(distribution) >= 2:

            total = sum(distribution.values())
            largest_class = max(distribution.values())
            smallest_class = min(distribution.values())

            imbalance_ratio = smallest_class / largest_class

            report["imbalance_ratio"] = round(imbalance_ratio, 2)

            if imbalance_ratio < 0.5:
                root_causes.append(
                    "Class imbalance detected in target column"
                )

    else:
        report["target_column"] = "Not found"

        root_causes.append(
            "Target column was not found in the dataset"
        )

    # Check missing values
    if report["missing_values"] > 0:
        root_causes.append(
            "Missing values detected in the dataset"
        )

    # Check duplicate rows
    if report["duplicate_rows"] > 0:
        root_causes.append(
            "Duplicate rows detected in the dataset"
        )

    # Final diagnosis
    if root_causes:
        report["root_causes"] = root_causes
        report["status"] = "Warning"
    else:
        report["root_causes"] = ["No obvious data-related root causes detected"]
        report["status"] = "Healthy"

    return report