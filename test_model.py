import pandas as pd
import random

from analysis.model_analyzer import analyze_model
from analysis.data_quality import analyze_data_quality
from report_generator import generate_report


# ============================================================
# CREATE SYNTHETIC DATASET
# ============================================================

random.seed(42)

records = []

for _ in range(200):

    age = random.randint(21, 60)

    base_salary = 18000 + (age - 21) * 3500

    salary = base_salary + random.randint(
        -12000,
        18000
    )

    salary = max(
        18000,
        salary
    )

    score = (
        (age - 40) * 0.08
        + (salary - 100000) / 100000
    )

    probability_of_fail = (
        1 / (1 + pow(2.71828, -score))
    )

    result = (
        "Fail"
        if random.random() < probability_of_fail
        else "Pass"
    )

    records.append({
        "age": age,
        "salary": salary,
        "result": result
    })


df = pd.DataFrame(records)


# ============================================================
# DATA QUALITY ANALYSIS
# ============================================================

data_quality = analyze_data_quality(df)


# ============================================================
# MODEL ANALYSIS
# ============================================================

result = analyze_model(
    df,
    "result"
)


# ============================================================
# EXTRACT NESTED RESULTS
# ============================================================

performance = result[
    "model_performance"
]

cross_validation = result[
    "cross_validation"
]

error_analysis = result[
    "error_analysis"
]


# ============================================================
# HEADER
# ============================================================

print()

print("=" * 50)
print("       AI MODEL ROOT-CAUSE ANALYZER")
print("=" * 50)


# ============================================================
# DATASET INFORMATION
# ============================================================

print()

print("DATASET INFORMATION")
print("-" * 50)

print(
    f"Total Rows     : "
    f"{len(df)}"
)

print(
    f"Total Features : "
    f"{len(df.columns) - 1}"
)

print(
    "Dataset Type   : "
    "Synthetic realistic test data"
)


# ============================================================
# DATA QUALITY
# ============================================================

print()

print("DATA QUALITY ANALYSIS")
print("-" * 50)

print(
    f"Overall Quality : "
    f"{data_quality['overall_quality']}"
)

total_missing = sum(
    data_quality["missing_values"].values()
)

print(
    f"Missing Values  : "
    f"{total_missing}"
)

print(
    f"Duplicate Rows  : "
    f"{data_quality['duplicate_rows']}"
)

print(
    f"Numeric Features: "
    f"{len(data_quality['numeric_features'])}"
)

print()

print("Potential Outliers:")

outliers = data_quality["outliers"]

if outliers:

    for feature, count in outliers.items():

        print(
            f"- {feature:<10}: "
            f"{count}"
        )

else:

    print("- None")


print()

print("Data Quality Warnings:")

for warning in data_quality["warnings"]:

    print(
        f"- {warning}"
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

print()

print("MODEL PERFORMANCE")
print("-" * 50)

print(
    f"Training Rows : "
    f"{performance['training_rows']}"
)

print(
    f"Testing Rows  : "
    f"{performance['testing_rows']}"
)

print(
    f"Accuracy      : "
    f"{performance['accuracy']:.0%}"
)

print(
    f"Precision     : "
    f"{performance['precision']:.0%}"
)

print(
    f"Recall        : "
    f"{performance['recall']:.0%}"
)

print(
    f"F1 Score      : "
    f"{performance['f1_score']:.0%}"
)


# ============================================================
# CROSS VALIDATION
# ============================================================

print()

print("CROSS-VALIDATION")
print("-" * 50)

fold_scores = cross_validation[
    "fold_scores"
]

if fold_scores:

    print(
        "Fold Scores   : "
        + ", ".join(
            f"{score}%"
            for score in fold_scores
        )
    )

print(
    f"Average Score : "
    f"{cross_validation['average_score']:.0%}"
)


# ============================================================
# OVERALL RISK
# ============================================================

print()

print("OVERALL RISK")
print("-" * 50)

print(
    result["overall_risk"]
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()

print("CLASS DISTRIBUTION")
print("-" * 50)

for label, count in result[
    "class_distribution"
].items():

    print(
        f"{label:<10}: "
        f"{count}"
    )


# ============================================================
# WARNINGS
# ============================================================

print()

print("WARNINGS")
print("-" * 50)

warnings = result.get(
    "warnings",
    []
)

if warnings:

    for warning in warnings:

        print(warning)

else:

    print(
        "No major warnings detected."
    )


# ============================================================
# PRIORITY ISSUES
# ============================================================

print()

print("PRIORITY ISSUES")
print("-" * 50)

priority_issues = result.get(
    "priority_issues",
    []
)

if priority_issues:

    for issue in priority_issues:

        print(
            f"[{issue['priority']}] "
            f"{issue['issue']}"
        )

        print(
            f"  Reason: "
            f"{issue['reason']}"
        )

else:

    print(
        "No major priority issues detected."
    )


# ============================================================
# FEATURE IMPACT
# ============================================================

print()

print("FEATURE IMPACT")
print("-" * 50)

feature_impact = result.get(
    "feature_impact",
    {}
)

for feature, information in feature_impact.items():

    print(
        f"{feature:<10}: "
        f"{information['importance']:.4f} "
        f"-> {information['direction']}"
    )


# ============================================================
# ROOT CAUSES
# ============================================================

print()

print("ROOT CAUSES")
print("-" * 50)

root_causes = result.get(
    "root_causes",
    []
)

if root_causes:

    for index, cause in enumerate(
        root_causes,
        start=1
    ):

        print(
            f"{index}. {cause}"
        )

else:

    print(
        "No major root causes identified."
    )


# ============================================================
# ERROR ANALYSIS
# ============================================================

print()

print("ERROR ANALYSIS")
print("-" * 50)

print(
    f"True Negatives  : "
    f"{error_analysis['true_negatives']}"
)

print(
    f"False Positives : "
    f"{error_analysis['false_positives']}"
)

print(
    f"False Negatives : "
    f"{error_analysis['false_negatives']}"
)

print(
    f"True Positives   : "
    f"{error_analysis['true_positives']}"
)

print()

print(
    f"Main Error: "
    f"{error_analysis['main_error']}"
)


# ============================================================
# INDIVIDUAL PREDICTION ERRORS
# ============================================================

print()

print("INDIVIDUAL PREDICTION ERRORS")
print("-" * 50)

errors = result.get(
    "prediction_errors",
    []
)

if errors:

    for index, error in enumerate(
        errors,
        start=1
    ):

        print(
            f"Error #{index}"
        )

        print(
            f"Dataset Row : "
            f"{error['dataset_row']}"
        )

        print(
            f"Actual      : "
            f"{error['actual']}"
        )

        print(
            f"Predicted   : "
            f"{error['predicted']}"
        )

        print(
            f"Error Type  : "
            f"{error['error_type']}"
        )

        print(
            f"Confidence  : "
            f"{error['confidence_percentage']:.2f}%"
        )

        print(
            f"Confidence Level : "
            f"{error['confidence_level']}"
        )

        print()

        print("Feature values:")

        for feature, value in error[
            "feature_values"
        ].items():

            print(
                f"- {feature:<10}: "
                f"{value}"
            )

        print()

        print(
            "Top contributing features:"
        )

        for feature in error[
            "top_contributing_features"
        ]:

            print(
                f"- {feature}"
            )

        print()

        print("-" * 50)

else:

    print(
        "No prediction errors detected."
    )


# ============================================================
# HIGH-CONFIDENCE ERRORS
# ============================================================

print()

print("HIGH-CONFIDENCE ERRORS")
print("-" * 50)

high_confidence_errors = result.get(
    "high_confidence_errors",
    []
)

if high_confidence_errors:

    for error in high_confidence_errors:

        print(error)

else:

    print(
        "No high-confidence errors detected."
    )


# ============================================================
# SUSPICIOUS RECORDS
# ============================================================

print()

print(
    "SUSPICIOUS / POTENTIALLY "
    "MISLABELED RECORDS"
)

print("-" * 50)

suspicious_records = result.get(
    "suspicious_records",
    []
)

print(
    f"Suspicious Records : "
    f"{len(suspicious_records)}"
)

for record in suspicious_records:

    print(record)


# ============================================================
# RECOMMENDATIONS
# ============================================================

print()

print("RECOMMENDATIONS")
print("-" * 50)

recommendations = result.get(
    "recommendations",
    []
)

if recommendations:

    for index, recommendation in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"{index}. {recommendation}"
        )

else:

    print(
        "No additional recommendations."
    )


# ============================================================
# PROFESSIONAL DIAGNOSTIC REPORT
# ============================================================

generate_report(
    data_quality,
    result
)