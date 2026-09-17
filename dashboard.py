from http.server import BaseHTTPRequestHandler, HTTPServer
import csv
import html
import io
import json
import random
from urllib.parse import urlparse

import pandas as pd

from analysis.model_analyzer import analyze_model
from analysis.data_quality import analyze_data_quality


# ============================================================
# DEMO DATASET
# Uses the same deterministic dataset as test_model.py.
# ============================================================

random.seed(42)

records = []

for _ in range(200):
    age = random.randint(21, 60)

    base_salary = 18000 + (age - 21) * 3500

    salary = base_salary + random.randint(-12000, 18000)
    salary = max(18000, salary)

    score = (
        (age - 40) * 0.08
        + (salary - 100000) / 100000
    )

    probability_of_fail = 1 / (1 + pow(2.71828, -score))

    target = "Fail" if random.random() < probability_of_fail else "Pass"

    records.append(
        {
            "age": age,
            "salary": salary,
            "result": target,
        }
    )


df = pd.DataFrame(records)


# ============================================================
# ANALYSIS
# ============================================================

data_quality = analyze_data_quality(df)
result = analyze_model(df, "result")

performance = result.get("model_performance", {})
cross_validation = result.get("cross_validation", {})
error_analysis = result.get("error_analysis", {})


# ============================================================
# HELPERS
# ============================================================

def safe(value):
    return html.escape(str(value))


def number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def as_percent(value):
    value = number(value)
    return value * 100 if 0 <= value <= 1 else value


def confidence_value(record):
    if "confidence" in record:
        return as_percent(record.get("confidence", 0))

    if "confidence_percentage" in record:
        return number(
            record.get(
                "confidence_percentage",
                0
            )
        )

    return 0.0


def record_row(record):
    if "row" in record:
        return record.get("row")

    if "dataset_row" in record:
        return record.get("dataset_row")

    return "N/A"


def error_type(record):
    if record.get("error_type"):
        return record["error_type"]

    actual = str(
        record.get(
            "actual",
            ""
        )
    )

    predicted = str(
        record.get(
            "predicted",
            ""
        )
    )

    if actual and predicted:
        return f"Misclassification ({actual} → {predicted})"

    return "Prediction Error"


def total_missing(values):

    if isinstance(values, dict):

        total = 0

        for value in values.values():

            try:
                total += int(value)

            except (
                TypeError,
                ValueError
            ):
                pass

        return total

    try:
        return int(values)

    except (
        TypeError,
        ValueError
    ):
        return 0


def total_outliers(values):

    if isinstance(values, dict):

        total = 0

        for value in values.values():

            try:
                total += int(value)

            except (
                TypeError,
                ValueError
            ):
                pass

        return total

    try:
        return int(values)

    except (
        TypeError,
        ValueError
    ):
        return 0


def csv_response(
    rows,
    filename,
    fieldnames=None
):

    buffer = io.StringIO()

    if fieldnames is None:

        fieldnames = []

        for row in rows:

            if isinstance(
                row,
                dict
            ):

                for key in row.keys():

                    if key not in fieldnames:
                        fieldnames.append(key)

    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )

    writer.writeheader()

    for row in rows:
        writer.writerow(row)

    return (
        "text/csv; charset=utf-8",
        filename,
        buffer.getvalue().encode(
            "utf-8-sig"
        ),
    )


def json_response(
    payload,
    filename
):

    content = json.dumps(
        payload,
        indent=2,
        default=str,
    ).encode(
        "utf-8"
    )

    return (
        "application/json; charset=utf-8",
        filename,
        content,
    )


# ============================================================
# NORMALIZED DATA FOR EXPORTS
# ============================================================

prediction_errors = result.get(
    "prediction_errors",
    []
)

if not isinstance(
    prediction_errors,
    list
):
    prediction_errors = []


high_confidence_errors = result.get(
    "high_confidence_errors",
    []
)

if not isinstance(
    high_confidence_errors,
    list
):
    high_confidence_errors = []


suspicious_records = result.get(
    "suspicious_records",
    []
)

if not isinstance(
    suspicious_records,
    list
):
    suspicious_records = []


feature_impact = result.get(
    "feature_impact",
    {}
)

if not isinstance(
    feature_impact,
    dict
):
    feature_impact = {}


recommendations = result.get(
    "recommendations",
    []
)

if not isinstance(
    recommendations,
    list
):
    recommendations = []


# ============================================================
# VALUES FOR UI
# ============================================================

accuracy = as_percent(
    performance.get(
        "accuracy",
        0
    )
)

precision = as_percent(
    performance.get(
        "precision",
        0
    )
)

recall = as_percent(
    performance.get(
        "recall",
        0
    )
)

f1_score = as_percent(
    performance.get(
        "f1_score",
        0
    )
)

training_rows = int(
    number(
        performance.get(
            "training_rows",
            0
        )
    )
)

testing_rows = int(
    number(
        performance.get(
            "testing_rows",
            0
        )
    )
)


dataset_size = int(
    number(
        data_quality.get(
            "total_rows",
            len(df)
        )
    )
)

feature_count = max(
    0,
    int(
        number(
            data_quality.get(
                "total_columns",
                len(df.columns)
            )
        )
    ) - 1
)

missing_values = total_missing(
    data_quality.get(
        "missing_values",
        {}
    )
)

duplicate_rows = int(
    number(
        data_quality.get(
            "duplicate_rows",
            0
        )
    )
)

outlier_count = total_outliers(
    data_quality.get(
        "outliers",
        {}
    )
)

data_quality_status = data_quality.get(
    "overall_quality",
    "UNKNOWN"
)


raw_fold_scores = cross_validation.get(
    "fold_scores",
    []
)

fold_scores = (
    raw_fold_scores
    if isinstance(
        raw_fold_scores,
        list
    )
    else []
)

cv_scores = [
    as_percent(score)
    for score in fold_scores
]

if cv_scores:

    cv_average = (
        sum(cv_scores)
        / len(cv_scores)
    )

else:

    cv_average = as_percent(
        cross_validation.get(
            "average_score",
            0
        )
    )


overall_risk = str(
    result.get(
        "overall_risk",
        "UNKNOWN"
    )
)

risk_score = int(
    number(
        result.get(
            "risk_score",
            0
        )
    )
)


class_distribution = result.get(
    "class_distribution",
    {}
)

if not isinstance(
    class_distribution,
    dict
):
    class_distribution = {}


if (
    set(
        class_distribution.keys()
    )
    == {"result"}
    and isinstance(
        class_distribution.get("result"),
        dict
    )
):

    class_distribution = (
        class_distribution[
            "result"
        ]
    )


true_negatives = int(
    number(
        error_analysis.get(
            "true_negatives",
            0
        )
    )
)

false_positives = int(
    number(
        error_analysis.get(
            "false_positives",
            0
        )
    )
)

false_negatives = int(
    number(
        error_analysis.get(
            "false_negatives",
            0
        )
    )
)

true_positives = int(
    number(
        error_analysis.get(
            "true_positives",
            0
        )
    )
)

main_error = result.get(
    "main_error",
    error_analysis.get(
        "main_error",
        "No major error detected.",
    ),
)


root_causes = result.get(
    "root_causes",
    []
)

if not isinstance(
    root_causes,
    list
):
    root_causes = []


priority_issues = result.get(
    "priority_issues",
    []
)

if not isinstance(
    priority_issues,
    list
):
    priority_issues = []


warnings = result.get(
    "warnings",
    []
)

if not isinstance(
    warnings,
    list
):
    warnings = []


# ============================================================
# PRIORITY ISSUES HTML
# ============================================================

priority_html = ""

for index, issue in enumerate(
    priority_issues,
    start=1
):

    if isinstance(
        issue,
        dict
    ):

        priority = issue.get(
            "priority",
            "INFO"
        )

        title = issue.get(
            "issue",
            "Unspecified issue"
        )

        reason = issue.get(
            "reason",
            "No explanation provided."
        )

    else:

        priority = "INFO"
        title = str(issue)
        reason = ""

    badge_class = (
        str(priority)
        .lower()
        .replace(
            " ",
            "-"
        )
    )

    priority_html += f"""
    <div class="issue-card">

        <div class="issue-number">
            {index}
        </div>

        <div class="issue-content">

            <div class="issue-heading">

                <span class="badge {safe(badge_class)}">
                    {safe(priority)}
                </span>

                <span class="issue-title">
                    {safe(title)}
                </span>

            </div>

            <div class="issue-reason">
                {safe(reason)}
            </div>

        </div>

    </div>
    """


if not priority_html:

    priority_html = """
    <div class="empty">
        No priority issues were reported.
    </div>
    """


# ============================================================
# FEATURE IMPACT HTML
# ============================================================

feature_rows = []

for (
    feature_name,
    feature_data
) in feature_impact.items():

    if isinstance(
        feature_data,
        dict
    ):

        importance = number(
            feature_data.get(
                "importance",
                0
            )
        )

        direction = feature_data.get(
            "direction",
            ""
        )

    else:

        importance = number(
            feature_data,
            0
        )

        direction = ""

    feature_rows.append(
        (
            str(feature_name),
            importance,
            str(direction)
        )
    )


feature_rows.sort(
    key=lambda item: item[1],
    reverse=True
)


feature_html = ""

for (
    feature_name,
    importance,
    direction
) in feature_rows:

    feature_html += f"""
    <div class="feature-row">

        <div class="feature-name">
            {safe(feature_name)}
        </div>

        <div class="feature-value">
            {importance:.4f}
        </div>

        <div class="feature-direction">
            {safe(direction)}
        </div>

    </div>
    """


if not feature_html:

    feature_html = """
    <div class="empty">
        No feature-impact information was reported.
    </div>
    """


# ============================================================
# ROOT CAUSES HTML
# ============================================================

root_cause_html = ""

for index, cause in enumerate(
    root_causes,
    start=1
):

    root_cause_html += f"""
    <div class="numbered-row">

        <div class="row-number">
            {index}
        </div>

        <div class="row-text">
            {safe(cause)}
        </div>

    </div>
    """


if not root_cause_html:

    root_cause_html = """
    <div class="empty">
        No root causes were reported.
    </div>
    """


# ============================================================
# PREDICTION ERRORS HTML
# ============================================================

prediction_html = ""

for error in prediction_errors:

    if not isinstance(
        error,
        dict
    ):
        continue

    row = record_row(error)

    actual = error.get(
        "actual",
        "N/A"
    )

    predicted = error.get(
        "predicted",
        "N/A"
    )

    confidence = confidence_value(
        error
    )

    kind = error_type(
        error
    )

    prediction_html += f"""
    <div class="prediction-row">

        <div>
            Row {safe(row)}
        </div>

        <div class="prediction-result">

            <span>
                {safe(actual)}
            </span>

            <span class="arrow">
                →
            </span>

            <span>
                {safe(predicted)}
            </span>

        </div>

        <div class="error-kind">
            {safe(kind)}
        </div>

        <div class="confidence">
            {confidence:.2f}%
        </div>

    </div>
    """


if not prediction_html:

    prediction_html = """
    <div class="empty">
        No individual prediction errors were reported.
    </div>
    """


# ============================================================
# HIGH-CONFIDENCE ERRORS HTML
# ============================================================

high_confidence_html = ""

for error in high_confidence_errors:

    if not isinstance(
        error,
        dict
    ):
        continue

    row = record_row(
        error
    )

    actual = error.get(
        "actual",
        "N/A"
    )

    predicted = error.get(
        "predicted",
        "N/A"
    )

    confidence = confidence_value(
        error
    )

    kind = error_type(
        error
    )

    high_confidence_html += f"""
    <div class="prediction-row highlight-orange">

        <div>
            Row {safe(row)}
        </div>

        <div class="prediction-result">

            <span>
                {safe(actual)}
            </span>

            <span class="arrow">
                →
            </span>

            <span>
                {safe(predicted)}
            </span>

        </div>

        <div class="error-kind">
            {safe(kind)}
        </div>

        <div class="confidence">
            {confidence:.2f}%
        </div>

    </div>
    """


if not high_confidence_html:

    high_confidence_html = """
    <div class="empty">
        No high-confidence errors detected.
    </div>
    """


# ============================================================
# SUSPICIOUS RECORDS HTML
# ============================================================

suspicious_html = ""

for record in suspicious_records:

    if not isinstance(
        record,
        dict
    ):
        continue

    row = record_row(
        record
    )

    actual = record.get(
        "actual",
        "N/A"
    )

    predicted = record.get(
        "predicted",
        "N/A"
    )

    confidence = confidence_value(
        record
    )

    suspicious_html += f"""
    <div class="prediction-row highlight-red">

        <div>
            Row {safe(row)}
        </div>

        <div class="prediction-result">

            <span>
                {safe(actual)}
            </span>

            <span class="arrow">
                →
            </span>

            <span>
                {safe(predicted)}
            </span>

        </div>

        <div class="suspicious-label">
            Suspicious
        </div>

        <div class="confidence">
            {confidence:.2f}%
        </div>

    </div>
    """


if not suspicious_html:

    suspicious_html = """
    <div class="empty">
        No suspicious records detected.
    </div>
    """


# ============================================================
# RECOMMENDATIONS HTML
# ============================================================

recommendations_html = ""

for index, recommendation in enumerate(
    recommendations,
    start=1
):

    recommendations_html += f"""
    <div class="numbered-row">

        <div class="row-number">
            {index}
        </div>

        <div class="row-text">
            {safe(recommendation)}
        </div>

    </div>
    """


if not recommendations_html:

    recommendations_html = """
    <div class="empty">
        No recommendations were reported.
    </div>
    """


# ============================================================
# WARNINGS HTML
# ============================================================

warnings_html = ""

for warning in warnings:

    warnings_html += f"""
    <div class="warning-row">
        {safe(warning)}
    </div>
    """


if not warnings_html:

    warnings_html = """
    <div class="empty">
        No model warnings were reported.
    </div>
    """


# ============================================================
# CV HTML
# ============================================================

cv_cards_html = ""

for index, score in enumerate(
    cv_scores,
    start=1
):

    cv_cards_html += f"""
    <div class="cv-card">

        <div class="cv-label">
            Fold {index}
        </div>

        <div class="cv-score">
            {score:.0f}%
        </div>

    </div>
    """


cv_cards_html += f"""
<div class="cv-card average-card">

    <div class="cv-label">
        Average
    </div>

    <div class="cv-score">
        {cv_average:.1f}%
    </div>

</div>
"""


# ============================================================
# CLASS DISTRIBUTION HTML
# ============================================================

class_cards_html = ""

for (
    class_name,
    count
) in class_distribution.items():

    class_cards_html += f"""
    <div class="card metric-card">

        <div class="card-label">
            {safe(class_name)}
        </div>

        <div class="metric-value">
            {int(number(count))}
        </div>

    </div>
    """


# ============================================================
# CSV EXPORT DATA
# ============================================================

prediction_csv_rows = []

for error in prediction_errors:

    if not isinstance(
        error,
        dict
    ):
        continue

    prediction_csv_rows.append(
        {
            "row":
                record_row(error),

            "actual":
                error.get(
                    "actual",
                    ""
                ),

            "predicted":
                error.get(
                    "predicted",
                    ""
                ),

            "error_type":
                error_type(error),

            "confidence_percent":
                f"{confidence_value(error):.2f}",
        }
    )


suspicious_csv_rows = []

for record in suspicious_records:

    if not isinstance(
        record,
        dict
    ):
        continue

    suspicious_csv_rows.append(
        {
            "row":
                record_row(record),

            "actual":
                record.get(
                    "actual",
                    ""
                ),

            "predicted":
                record.get(
                    "predicted",
                    ""
                ),

            "confidence_percent":
                f"{confidence_value(record):.2f}",
        }
    )


feature_csv_rows = []

for (
    feature_name,
    importance,
    direction
) in feature_rows:

    feature_csv_rows.append(
        {
            "feature":
                feature_name,

            "importance":
                f"{importance:.4f}",

            "direction":
                direction,
        }
    )


recommendation_csv_rows = [

    {
        "number":
            index,

        "recommendation":
            recommendation,
    }

    for index, recommendation
    in enumerate(
        recommendations,
        start=1
    )

]


summary_csv_rows = [

    {
        "metric":
            "accuracy",

        "value":
            f"{accuracy:.2f}%"
    },

    {
        "metric":
            "precision",

        "value":
            f"{precision:.2f}%"
    },

    {
        "metric":
            "recall",

        "value":
            f"{recall:.2f}%"
    },

    {
        "metric":
            "f1_score",

        "value":
            f"{f1_score:.2f}%"
    },

    {
        "metric":
            "training_rows",

        "value":
            training_rows
    },

    {
        "metric":
            "testing_rows",

        "value":
            testing_rows
    },

    {
        "metric":
            "cv_average",

        "value":
            f"{cv_average:.2f}%"
    },

    {
        "metric":
            "overall_risk",

        "value":
            overall_risk
    },

    {
        "metric":
            "risk_score",

        "value":
            risk_score
    },

    {
        "metric":
            "dataset_size",

        "value":
            dataset_size
    },

    {
        "metric":
            "feature_count",

        "value":
            feature_count
    },

    {
        "metric":
            "missing_values",

        "value":
            missing_values
    },

    {
        "metric":
            "duplicate_rows",

        "value":
            duplicate_rows
    },

    {
        "metric":
            "outliers",

        "value":
            outlier_count
    },

    {
        "metric":
            "true_negatives",

        "value":
            true_negatives
    },

    {
        "metric":
            "false_positives",

        "value":
            false_positives
    },

    {
        "metric":
            "false_negatives",

        "value":
            false_negatives
    },

    {
        "metric":
            "true_positives",

        "value":
            true_positives
    },

]


FULL_ANALYSIS_EXPORT = {

    "dataset": {

        "rows":
            dataset_size,

        "features":
            feature_count,

        "columns":
            list(df.columns),

    },

    "data_quality":
        data_quality,

    "result":
        result,

}


# ============================================================
# DASHBOARD HTML
# ============================================================

DASHBOARD_HTML = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<meta
    http-equiv="Cache-Control"
    content="no-cache, no-store, must-revalidate"
>

<meta
    http-equiv="Pragma"
    content="no-cache"
>

<meta
    http-equiv="Expires"
    content="0"
>

<title>
    AI Model Root-Cause Analyzer
</title>


<style>

* {{
    box-sizing:
        border-box;
}}


body {{

    margin:
        0;

    background:
        #f4f7fb;

    color:
        #172033;

    font-family:
        Arial,
        Helvetica,
        sans-serif;
}}


.header {{

    background:
        linear-gradient(
            135deg,
            #111827,
            #1e3a5f
        );

    color:
        #fff;

    padding:
        30px 20px 26px;
}}


.header-inner {{

    width:
        min(
            1180px,
            92%
        );

    margin:
        auto;

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    gap:
        20px;
}}


.brand h1 {{

    margin:
        0;

    font-size:
        29px;
}}


.brand p {{

    margin:
        8px 0 0;

    color:
        #cbd5e1;
}}


.risk-pill {{

    padding:
        10px 16px;

    border-radius:
        999px;

    background:
        #dcfce7;

    color:
        #166534;

    font-weight:
        800;

    white-space:
        nowrap;
}}


.container {{

    width:
        min(
            1180px,
            92%
        );

    margin:
        26px auto 50px;
}}


.toolbar {{

    display:
        flex;

    flex-wrap:
        wrap;

    gap:
        10px;

    margin-bottom:
        26px;
}}


.btn {{

    display:
        inline-flex;

    align-items:
        center;

    justify-content:
        center;

    padding:
        10px 14px;

    border-radius:
        9px;

    text-decoration:
        none;

    background:
        #fff;

    color:
        #1e293b;

    border:
        1px solid #dbe3ee;

    font-weight:
        700;

    box-shadow:
        0 2px 7px
        rgba(
            15,
            23,
            42,
            .05
        );
}}


.btn.primary {{

    background:
        #1d4ed8;

    color:
        #fff;

    border-color:
        #1d4ed8;
}}


.btn:hover {{

    opacity:
        .9;
}}


.section-title {{

    margin:
        30px 0 13px;

    font-size:
        21px;

    font-weight:
        800;
}}


.grid {{

    display:
        grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                190px,
                1fr
            )
        );

    gap:
        15px;
}}


.card,
.summary-card,
.risk-card {{

    background:
        #fff;

    border:
        1px solid #e6ebf2;

    border-radius:
        12px;

    padding:
        19px;

    box-shadow:
        0 3px 12px
        rgba(
            15,
            23,
            42,
            .05
        );
}}


.metric-card {{

    min-height:
        115px;
}}


.card-label,
.cv-label {{

    color:
        #64748b;

    font-size:
        13px;

    font-weight:
        700;
}}


.metric-value {{

    margin-top:
        7px;

    font-size:
        29px;

    font-weight:
        800;
}}


.metric-subtext {{

    margin-top:
        4px;

    color:
        #64748b;

    font-size:
        13px;
}}


.good {{

    color:
        #15803d;
}}


.performance-card {{

    position:
        relative;

    overflow:
        hidden;
}}


.performance-bar {{

    height:
        6px;

    margin-top:
        12px;

    background:
        #e7edf5;

    border-radius:
        999px;

    overflow:
        hidden;
}}


.performance-fill {{

    height:
        100%;

    background:
        #2563eb;

    border-radius:
        999px;
}}


.risk-card {{

    border-left:
        5px solid #16a34a;
}}


.risk-value {{

    font-size:
        30px;

    font-weight:
        800;
}}


.risk-score {{

    margin-top:
        8px;

    color:
        #64748b;
}}


.cv-grid {{

    display:
        grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                125px,
                1fr
            )
        );

    gap:
        10px;
}}


.cv-card {{

    background:
        #fff;

    padding:
        16px;

    border-radius:
        10px;

    border:
        1px solid #e6ebf2;
}}


.cv-score {{

    margin-top:
        6px;

    font-size:
        25px;

    font-weight:
        800;
}}


.average-card {{

    border:
        2px solid #1e3a8a;
}}


.issue-card {{

    display:
        flex;

    gap:
        13px;

    background:
        #fff;

    padding:
        16px;

    margin-bottom:
        10px;

    border-radius:
        10px;

    border:
        1px solid #e6ebf2;
}}


.issue-number,
.row-number {{

    width:
        31px;

    min-width:
        31px;

    height:
        31px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        50%;

    background:
        #eef2f7;

    font-weight:
        800;
}}


.issue-content {{

    flex:
        1;
}}


.issue-heading {{

    display:
        flex;

    flex-wrap:
        wrap;

    align-items:
        center;

    gap:
        9px;
}}


.issue-title {{

    font-weight:
        800;
}}


.issue-reason {{

    margin-top:
        7px;

    color:
        #64748b;

    line-height:
        1.5;
}}


.badge {{

    padding:
        4px 8px;

    border-radius:
        999px;

    font-size:
        11px;

    font-weight:
        800;
}}


.badge.high {{

    background:
        #fee2e2;

    color:
        #991b1b;
}}


.badge.medium {{

    background:
        #fef3c7;

    color:
        #92400e;
}}


.badge.low {{

    background:
        #dcfce7;

    color:
        #166534;
}}


.badge.info {{

    background:
        #e0f2fe;

    color:
        #075985;
}}


.confusion-grid {{

    display:
        grid;

    grid-template-columns:
        repeat(
            4,
            1fr
        );

    gap:
        10px;
}}


.confusion-cell {{

    background:
        #f8fafc;

    border-radius:
        10px;

    padding:
        18px 12px;

    text-align:
        center;
}}


.confusion-value {{

    font-size:
        29px;

    font-weight:
        800;
}}


.confusion-label {{

    margin-top:
        5px;

    color:
        #64748b;

    font-size:
        12px;

    font-weight:
        700;
}}


.definition-list {{

    margin-top:
        19px;

    line-height:
        1.6;

    color:
        #64748b;
}}


.main-error {{

    margin-top:
        18px;

    padding-top:
        16px;

    border-top:
        1px solid #e5e7eb;

    line-height:
        1.5;
}}


.feature-header,
.feature-row {{

    display:
        grid;

    grid-template-columns:
        minmax(
            190px,
            1fr
        )
        150px
        190px;

    gap:
        14px;

    align-items:
        center;
}}


.feature-header {{

    padding-bottom:
        10px;

    color:
        #64748b;

    font-weight:
        800;

    border-bottom:
        2px solid #e5e7eb;
}}


.feature-row {{

    padding:
        13px 0;

    border-bottom:
        1px solid #e5e7eb;
}}


.feature-row:last-child {{

    border-bottom:
        0;
}}


.feature-name {{

    font-weight:
        700;
}}


.feature-value {{

    font-weight:
        800;
}}


.feature-direction {{

    color:
        #475569;
}}


.numbered-row {{

    display:
        flex;

    gap:
        12px;

    align-items:
        flex-start;

    padding:
        12px 0;

    border-bottom:
        1px solid #e5e7eb;

    line-height:
        1.5;
}}


.numbered-row:last-child {{

    border-bottom:
        0;
}}


.row-text {{

    flex:
        1;
}}


.prediction-row {{

    display:
        grid;

    grid-template-columns:
        100px
        170px
        1fr
        100px;

    gap:
        12px;

    align-items:
        center;

    padding:
        12px 0;

    border-bottom:
        1px solid #e5e7eb;
}}


.prediction-row:last-child {{

    border-bottom:
        0;
}}


.prediction-result {{

    display:
        flex;

    gap:
        7px;

    font-weight:
        700;
}}


.arrow {{

    color:
        #64748b;
}}


.error-kind {{

    color:
        #b91c1c;

    font-weight:
        700;
}}


.confidence {{

    text-align:
        right;

    font-weight:
        800;
}}


.highlight-orange {{

    background:
        #fff7ed;

    padding-left:
        10px;

    padding-right:
        10px;

    border-radius:
        8px;
}}


.highlight-red {{

    background:
        #fef2f2;

    padding-left:
        10px;

    padding-right:
        10px;

    border-radius:
        8px;
}}


.suspicious-label {{

    color:
        #991b1b;

    font-weight:
        800;
}}


.warning-row {{

    padding:
        12px;

    background:
        #fff7ed;

    border-left:
        4px solid #f59e0b;

    border-radius:
        7px;

    color:
        #78350f;

    margin-bottom:
        8px;
}}


.empty {{

    padding:
        15px;

    background:
        #f8fafc;

    border-radius:
        8px;

    color:
        #64748b;
}}


.footer {{

    text-align:
        center;

    color:
        #64748b;

    padding:
        25px 0 5px;
}}


@media (
    max-width: 760px
) {{

    .header-inner {{

        flex-direction:
            column;

        align-items:
            flex-start;
    }}


    .confusion-grid {{

        grid-template-columns:
            repeat(
                2,
                1fr
            );
    }}


    .feature-header {{

        display:
            none;
    }}


    .feature-row {{

        grid-template-columns:
            1fr;

        gap:
            5px;
    }}


    .prediction-row {{

        grid-template-columns:
            1fr;

        gap:
            6px;
    }}


    .confidence {{

        text-align:
            left;
    }}

}}

</style>

</head>


<body>


<header class="header">

    <div class="header-inner">

        <div class="brand">

            <h1>
                AI Model Root-Cause Analyzer
            </h1>

            <p>
                Automated Machine Learning Diagnostic Dashboard
            </p>

        </div>

        <div class="risk-pill">

            Risk:
            {safe(overall_risk)}

        </div>

    </div>

</header>


<main class="container">


<div class="toolbar">

    <a
        class="btn primary"
        href="/download/dataset.csv"
    >
        Download Dataset CSV
    </a>


    <a
        class="btn"
        href="/download/prediction-errors.csv"
    >
        Prediction Errors CSV
    </a>


    <a
        class="btn"
        href="/download/suspicious-records.csv"
    >
        Suspicious Records CSV
    </a>


    <a
        class="btn"
        href="/download/feature-impact.csv"
    >
        Feature Impact CSV
    </a>


    <a
        class="btn"
        href="/download/recommendations.csv"
    >
        Recommendations CSV
    </a>


    <a
        class="btn"
        href="/download/summary.csv"
    >
        Summary CSV
    </a>


    <a
        class="btn"
        href="/download/analysis.json"
    >
        Full Analysis JSON
    </a>

</div>


<div class="section-title">
    Model Performance
</div>


<div class="grid">


    <div class="card performance-card">

        <div class="card-label">
            Accuracy
        </div>

        <div class="metric-value">
            {accuracy:.0f}%
        </div>

        <div class="performance-bar">

            <div
                class="performance-fill"
                style="width:{min(accuracy, 100):.0f}%"
            ></div>

        </div>

    </div>


    <div class="card performance-card">

        <div class="card-label">
            Precision
        </div>

        <div class="metric-value">
            {precision:.0f}%
        </div>

        <div class="performance-bar">

            <div
                class="performance-fill"
                style="width:{min(precision, 100):.0f}%"
            ></div>

        </div>

    </div>


    <div class="card performance-card">

        <div class="card-label">
            Recall
        </div>

        <div class="metric-value">
            {recall:.0f}%
        </div>

        <div class="performance-bar">

            <div
                class="performance-fill"
                style="width:{min(recall, 100):.0f}%"
            ></div>

        </div>

    </div>


    <div class="card performance-card">

        <div class="card-label">
            F1 Score
        </div>

        <div class="metric-value">
            {f1_score:.0f}%
        </div>

        <div class="performance-bar">

            <div
                class="performance-fill"
                style="width:{min(f1_score, 100):.0f}%"
            ></div>

        </div>

    </div>


</div>


<div class="section-title">
    Dataset Health
</div>


<div class="grid">


    <div class="card metric-card">

        <div class="card-label">
            Dataset Size
        </div>

        <div class="metric-value">
            {dataset_size}
        </div>

        <div class="metric-subtext">
            rows
        </div>

    </div>


    <div class="card metric-card">

        <div class="card-label">
            Features
        </div>

        <div class="metric-value">
            {feature_count}
        </div>

        <div class="metric-subtext">
            input features
        </div>

    </div>


    <div class="card metric-card">

        <div class="card-label">
            Data Quality
        </div>

        <div class="metric-value">
            {safe(data_quality_status)}
        </div>

    </div>


    <div class="card metric-card">

        <div class="card-label">
            Missing Values
        </div>

        <div class="metric-value good">
            {missing_values}
        </div>

    </div>


    <div class="card metric-card">

        <div class="card-label">
            Duplicate Rows
        </div>

        <div class="metric-value good">
            {duplicate_rows}
        </div>

    </div>


    <div class="card metric-card">

        <div class="card-label">
            Outliers
        </div>

        <div class="metric-value good">
            {outlier_count}
        </div>

    </div>


</div>


<div class="section-title">
    Dataset Split
</div>


<div class="grid">


    <div class="card metric-card">

        <div class="card-label">
            Training Rows
        </div>

        <div class="metric-value">
            {training_rows}
        </div>

    </div>


    <div class="card metric-card">

        <div class="card-label">
            Testing Rows
        </div>

        <div class="metric-value">
            {testing_rows}
        </div>

    </div>


</div>


<div class="section-title">
    Cross Validation
</div>


<div class="cv-grid">

    {cv_cards_html}

</div>


<div class="section-title">
    Overall Risk
</div>


<div class="risk-card">

    <div class="risk-value">
        {safe(overall_risk)}
    </div>

    <div class="risk-score">

        Risk Score:
        <strong>
            {risk_score}/100
        </strong>

    </div>

</div>


<div class="section-title">
    Class Distribution
</div>


<div class="grid">

    {class_cards_html}

</div>


<div class="section-title">
    Priority Issues
</div>


{priority_html}


<div class="section-title">
    Error Analysis
</div>


<div class="summary-card">


    <div class="confusion-grid">


        <div class="confusion-cell">

            <div class="confusion-value">
                {true_negatives}
            </div>

            <div class="confusion-label">
                True Negative
            </div>

        </div>


        <div class="confusion-cell">

            <div class="confusion-value">
                {false_positives}
            </div>

            <div class="confusion-label">
                False Positive
            </div>

        </div>


        <div class="confusion-cell">

            <div class="confusion-value">
                {false_negatives}
            </div>

            <div class="confusion-label">
                False Negative
            </div>

        </div>


        <div class="confusion-cell">

            <div class="confusion-value">
                {true_positives}
            </div>

            <div class="confusion-label">
                True Positive
            </div>

        </div>


    </div>


    <div class="definition-list">

        <strong>
            True Negative:
        </strong>

        Actual Pass correctly predicted as Pass.

        <br><br>


        <strong>
            False Positive:
        </strong>

        Actual Pass incorrectly predicted as Fail.

        <br><br>


        <strong>
            False Negative:
        </strong>

        Actual Fail incorrectly predicted as Pass.

        <br><br>


        <strong>
            True Positive:
        </strong>

        Actual Fail correctly predicted as Fail.

    </div>


    <div class="main-error">

        <strong>
            Main error pattern:
        </strong>

        {safe(main_error)}

    </div>


</div>


<div class="section-title">
    Feature Impact
</div>


<div class="summary-card">


    <div class="feature-header">

        <div>
            Feature
        </div>

        <div>
            Importance
        </div>

        <div>
            Direction
        </div>

    </div>


    {feature_html}


</div>


<div class="section-title">
    Root Causes
</div>


<div class="summary-card">

    {root_cause_html}

</div>


<div class="section-title">
    Individual Prediction Errors
</div>


<div class="summary-card">

    {prediction_html}

</div>


<div class="section-title">
    High-Confidence Errors
</div>


<div class="summary-card">

    {high_confidence_html}

</div>


<div class="section-title">
    Suspicious Records
</div>


<div class="summary-card">

    {suspicious_html}

</div>


<div class="section-title">
    Recommendations
</div>


<div class="summary-card">

    {recommendations_html}

</div>


<div class="section-title">
    Warnings
</div>


<div class="summary-card">

    {warnings_html}

</div>


<div class="footer">

    AI Model Root-Cause Analyzer |
    Local Diagnostic Dashboard

</div>


</main>


</body>

</html>
"""


# ============================================================
# HTTP SERVER
# ============================================================

class DashboardHandler(
    BaseHTTPRequestHandler
):

    def _send_download(
        self,
        content_type,
        filename,
        content
    ):

        self.send_response(200)

        self.send_header(
            "Content-Type",
            content_type
        )

        self.send_header(
            "Content-Disposition",
            f'attachment; filename="{filename}"'
        )

        self.send_header(
            "Content-Length",
            str(len(content))
        )

        self.send_header(
            "Cache-Control",
            "no-store"
        )

        self.end_headers()

        self.wfile.write(
            content
        )


    def do_GET(self):

        path = urlparse(
            self.path
        ).path


        if path == "/":

            content = (
                DASHBOARD_HTML
                .encode("utf-8")
            )

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8"
            )

            self.send_header(
                "Cache-Control",
                "no-store, no-cache, must-revalidate, max-age=0"
            )

            self.send_header(
                "Content-Length",
                str(len(content))
            )

            self.end_headers()

            self.wfile.write(
                content
            )

            return


        if path == "/download/dataset.csv":

            rows = df.to_dict(
                orient="records"
            )

            (
                content_type,
                filename,
                content
            ) = csv_response(
                rows,
                "ai_model_demo_dataset.csv",
                list(df.columns),
            )

            self._send_download(
                content_type,
                filename,
                content
            )

            return


        if path == "/download/prediction-errors.csv":

            (
                content_type,
                filename,
                content
            ) = csv_response(
                prediction_csv_rows,
                "prediction_errors.csv",
                [
                    "row",
                    "actual",
                    "predicted",
                    "error_type",
                    "confidence_percent",
                ],
            )

            self._send_download(
                content_type,
                filename,
                content
            )

            return


        if path == "/download/suspicious-records.csv":

            (
                content_type,
                filename,
                content
            ) = csv_response(
                suspicious_csv_rows,
                "suspicious_records.csv",
                [
                    "row",
                    "actual",
                    "predicted",
                    "confidence_percent",
                ],
            )

            self._send_download(
                content_type,
                filename,
                content
            )

            return


        if path == "/download/feature-impact.csv":

            (
                content_type,
                filename,
                content
            ) = csv_response(
                feature_csv_rows,
                "feature_impact.csv",
                [
                    "feature",
                    "importance",
                    "direction",
                ],
            )

            self._send_download(
                content_type,
                filename,
                content
            )

            return


        if path == "/download/recommendations.csv":

            (
                content_type,
                filename,
                content
            ) = csv_response(
                recommendation_csv_rows,
                "recommendations.csv",
                [
                    "number",
                    "recommendation",
                ],
            )

            self._send_download(
                content_type,
                filename,
                content
            )

            return


        if path == "/download/summary.csv":

            (
                content_type,
                filename,
                content
            ) = csv_response(
                summary_csv_rows,
                "analysis_summary.csv",
                [
                    "metric",
                    "value",
                ],
            )

            self._send_download(
                content_type,
                filename,
                content
            )

            return


        if path == "/download/analysis.json":

            (
                content_type,
                filename,
                content
            ) = json_response(
                FULL_ANALYSIS_EXPORT,
                "full_analysis.json",
            )

            self._send_download(
                content_type,
                filename,
                content
            )

            return


        self.send_response(404)

        self.send_header(
            "Content-Type",
            "text/plain; charset=utf-8"
        )

        self.end_headers()

        self.wfile.write(
            b"Not found"
        )


    def log_message(
        self,
        format,
        *args
    ):

        return


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    server = HTTPServer(
        (
            "localhost",
            8000
        ),
        DashboardHandler
    )


    print(
        "=" * 60
    )

    print(
        "       AI MODEL ROOT-CAUSE ANALYZER"
    )

    print(
        "=" * 60
    )

    print()

    print(
        "Dashboard running at:"
    )

    print(
        "http://localhost:8000"
    )

    print()

    print(
        "CSV and JSON downloads are enabled."
    )

    print(
        "Press CTRL+C to stop the server."
    )

    print()

    server.serve_forever()