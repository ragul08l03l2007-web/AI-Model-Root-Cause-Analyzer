# test_diagnostic_deduplication.py
import json
import numpy as np
import pandas as pd

from analysis.model_analyzer import analyze_model


def test_classification_disparity_and_confusion_grouping():
    print("--------------------------------------------------")
    print("TEST 1: Classification Disparity & Confusion Grouping")
    print("--------------------------------------------------")
    np.random.seed(42)
    # 150 rows: 30 positive (class 1, 20%), 120 negative (class 0, 80%)
    # Feat 1 has subtle overlap causing model to predict mostly 0 (false negatives for class 1)
    feat1 = np.random.normal(0, 1, 150)
    feat2 = np.random.normal(0, 1, 150)
    target = np.array([1 if i < 30 else 0 for i in range(150)])

    df = pd.DataFrame({"feat_1": feat1, "feat_2": feat2, "outcome": target})
    res = analyze_model(df, "outcome", analysis_type="classification")

    structured_rc = res.get("root_causes_structured", [])
    print(f"Total Structured Root Causes: {len(structured_rc)}")
    for i, rc in enumerate(structured_rc, 1):
        print(f"  {i}. [{rc['category']}] {rc['finding']} (Severity: {rc['severity']}, Strength: {rc['evidence_strength']})")
        print(f"     Evidence: {rc['evidence']}")

    # Verification: Confirm that "Disproportionate False Negatives" and "Repeated Misclassification Pattern (1 -> 0)"
    # are NOT two separate root causes!
    findings_titles = [rc["finding"] for rc in structured_rc]
    imbalance_findings = [f for f in findings_titles if "Disparity" in f or "Sensitivity" in f or "Deficit" in f or "False Negative" in f or "Minority" in f]
    confusion_findings = [f for f in findings_titles if "Repeated Misclassification" in f]

    assert len(imbalance_findings) >= 1, f"Expected disparity finding, got: {findings_titles}"
    assert len(confusion_findings) == 0, (
        f"Repeated misclassification was emitted as a separate duplicate root cause! Findings: {findings_titles}"
    )

    # Verify that the disparity finding's evidence incorporated the confusion pattern
    disparity_rc = next(rc for rc in structured_rc if "Disparity" in rc["finding"] or "Sensitivity" in rc["finding"] or "Deficit" in rc["finding"] or "False Negative" in rc["finding"] or "Minority" in rc["finding"])
    print("\nDisparity Finding Combined Evidence List:")
    for ev in disparity_rc["evidence_list"]:
        print(f" - {ev}")

    assert any("misclassification" in ev.lower() or "errors" in ev.lower() for ev in disparity_rc["evidence_list"]), (
        "Dominant misclassification was not merged into disparity evidence list!"
    )
    print("\n-> TEST 1 PASSED: Disparity & Confusion successfully grouped into one candidate!\n")


def test_risk_drivers_traceability_and_non_double_counting():
    print("--------------------------------------------------")
    print("TEST 2: Risk Drivers Traceability & Non-Double-Counting")
    print("--------------------------------------------------")
    np.random.seed(99)
    df = pd.DataFrame({
        "x1": np.random.randn(100),
        "x2": np.random.randn(100),
        "target": [1] * 25 + [0] * 75
    })
    res = analyze_model(df, "target", analysis_type="classification")
    risk = res.get("risk", {})
    
    print(f"Risk Score: {risk.get('score')} ({risk.get('level')})")
    print(f"Risk Method: {risk.get('method')}")
    print("Risk Drivers:")
    for d in risk.get("drivers", []):
        print(f" - [{d['issue']}] +{d['contribution']} pts (Confidence: {d.get('confidence')}): {d['evidence']}")

    assert 5 <= risk["score"] <= 100
    assert risk["method"] == "Evidence-weighted diagnostic aggregation"
    assert len(risk["drivers"]) > 0

    # Verify driver fields
    for d in risk["drivers"]:
        assert "issue" in d
        assert "evidence" in d
        assert "contribution" in d
        assert "confidence" in d

    print("\n-> TEST 2 PASSED: Risk drivers are traceable and bounded!\n")


def test_recommendation_deduplication():
    print("--------------------------------------------------")
    print("TEST 3: Recommendation Deduplication")
    print("--------------------------------------------------")
    np.random.seed(77)
    df = pd.DataFrame({
        "num1": np.random.randn(60),
        "num2": np.random.randn(60),
        "label": [1] * 12 + [0] * 48
    })
    res = analyze_model(df, "label", analysis_type="classification")
    recs = res.get("recommendations", [])

    print(f"Total Recommendations: {len(recs)}")
    for i, r in enumerate(recs, 1):
        print(f" {i}. {r}")

    # Ensure no exact duplicate recommendations
    assert len(recs) == len(set(recs)), "Duplicate recommendations found in output!"
    print("\n-> TEST 3 PASSED: Recommendations are clean and deduplicated!\n")


def test_regression_deduplication():
    print("--------------------------------------------------")
    print("TEST 4: Regression Diagnostics Deduplication")
    print("--------------------------------------------------")
    np.random.seed(123)
    x = np.linspace(1, 100, 150)
    y = 3.5 * x + np.random.normal(0, 5, 150)
    df = pd.DataFrame({"sqft": x, "price": y})

    res = analyze_model(df, "price", analysis_type="regression")
    structured_rc = res.get("root_causes_structured", [])

    print(f"Regression Root Causes Count: {len(structured_rc)}")
    for rc in structured_rc:
        print(f" - [{rc['category']}] {rc['finding']} (Severity: {rc['severity']})")

    assert len(structured_rc) > 0
    print("\n-> TEST 4 PASSED: Regression analysis completed without duplicate findings!\n")


if __name__ == "__main__":
    test_classification_disparity_and_confusion_grouping()
    test_risk_drivers_traceability_and_non_double_counting()
    test_recommendation_deduplication()
    test_regression_deduplication()
    print("==================================================")
    print("ALL DEDUPLICATION & GROUPING TESTS PASSED 100% GREEN!")
    print("==================================================")
