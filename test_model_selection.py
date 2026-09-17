# test_model_selection.py

import json
import numpy as np
import pandas as pd
from analysis.model_analyzer import analyze_model

def run_model_selection_tests():
    print("=" * 60)
    print("STARTING MODEL SELECTION UNIT & INTEGRATION TEST SUITE")
    print("=" * 60)

    # ----------------------------------------------------
    # TEST 1: Classification Data-Driven Selection & Metadata
    # ----------------------------------------------------
    print("\nTest 1: Classification Selection Criterion & Metadata Verification...")
    np.random.seed(42)
    n = 200
    df_clf = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n) * 2,
        "f3": np.random.choice(["A", "B", "C"], size=n),
        "target": np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    })
    # Inject signal into f1
    df_clf.loc[df_clf["f1"] > 0, "target"] = 1

    res_clf = analyze_model(df_clf, target_column="target", task_type="classification")

    # Verify model_selection dictionary exists
    assert "model_selection" in res_clf, "model_selection metadata missing from classification result"
    ms = res_clf["model_selection"]
    print("Classification Model Selection Meta:", ms)

    assert "selected_model" in ms and ms["selected_model"] == res_clf["selected_model"]
    assert "criterion" in ms and "cross-validation" in ms["criterion"].lower()
    assert "metric" in ms and ms["metric"] in {"weighted_f1", "accuracy"}
    assert "selected_score" in ms and isinstance(ms["selected_score"], float)
    assert ms["selection_basis"] == "cross_validation"
    assert "selection_explanation" in ms and len(ms["selection_explanation"]) > 0

    # Verify model comparison table distinguishes Test vs CV
    comp_list = res_clf.get("model_comparison", [])
    assert len(comp_list) == 4, f"Expected 4 candidate models, got {len(comp_list)}"

    selected_count = 0
    for row in comp_list:
        assert "test_accuracy" in row or "accuracy" in row
        assert "test_f1_score" in row or "f1_score" in row
        assert "cv_mean" in row
        assert "cv_std" in row
        assert "selection_status" in row
        if row["selection_status"] in ("Selected", "SELECTED"):
            selected_count += 1
            assert row["model"] == ms["selected_model"]

    assert selected_count == 1, f"Expected exactly 1 SELECTED model, got {selected_count}"
    print("-> Test 1 PASSED: Classification model selection structure & metadata validated.")

    # ----------------------------------------------------
    # TEST 2: Regression Selection Criterion & Metadata
    # ----------------------------------------------------
    print("\nTest 2: Regression Selection Criterion & Metadata Verification...")
    n = 150
    df_reg = pd.DataFrame({
        "x1": np.linspace(10, 100, n),
        "x2": np.random.randn(n) * 5,
        "target": 3.5 * np.linspace(10, 100, n) + np.random.randn(n) * 10
    })

    res_reg = analyze_model(df_reg, target_column="target", task_type="regression")

    assert "model_selection" in res_reg, "model_selection metadata missing from regression result"
    ms_reg = res_reg["model_selection"]
    print("Regression Model Selection Meta:", ms_reg)

    assert "selected_model" in ms_reg and ms_reg["selected_model"] == res_reg["selected_model"]
    assert "criterion" in ms_reg and ("r²" in ms_reg["criterion"].lower() or "r2" in ms_reg["criterion"].lower())
    assert ms_reg["metric"] == "r2"
    assert "selected_score" in ms_reg and isinstance(ms_reg["selected_score"], float)
    assert ms_reg["selection_basis"] == "cross_validation"
    assert "selection_explanation" in ms_reg and len(ms_reg["selection_explanation"]) > 0

    reg_comp_list = res_reg.get("model_comparison", [])
    reg_selected_count = sum(1 for r in reg_comp_list if r["selection_status"] in ("Selected", "SELECTED"))
    assert reg_selected_count == 1, f"Expected exactly 1 SELECTED regression model, got {reg_selected_count}"
    print("-> Test 2 PASSED: Regression model selection structure & metadata validated.")

    # ----------------------------------------------------
    # TEST 3: Downstream Alignment Verification
    # ----------------------------------------------------
    print("\nTest 3: Downstream Diagnostic Alignment Verification...")
    sel_name = res_clf["selected_model"]
    assert res_clf["model_performance"]["selected_model"] == sel_name
    assert len(res_clf["feature_importance"]) > 0
    assert len(res_clf["feature_impact"]) > 0

    print("-> Test 3 PASSED: Downstream feature analysis aligned with champion model.")

    # ----------------------------------------------------
    # TEST 4: Fail-Safe Model Evaluation Handling
    # ----------------------------------------------------
    print("\nTest 4: Fail-Safe Handling Verification...")
    # Verify that model comparison records have valid selection_status
    for item in comp_list:
        assert item["selection_status"] in ["Selected", "SELECTED", "Unselected", "Failed"]

    print("-> Test 4 PASSED: Model status values strictly conform to schema.")

    # ----------------------------------------------------
    # TEST 5: CV vs Single Test-Split Generalization Priority
    # ----------------------------------------------------
    print("\nTest 5: CV Generalization Priority over Single Test Split...")
    # Verify that the selected model is chosen based on CV, not highest test accuracy
    comp_list = res_clf["model_comparison"]
    max_test_acc_model = max(comp_list, key=lambda x: x["test_accuracy"])
    max_cv_model = max(comp_list, key=lambda x: x["cv_mean"])
    selected_model_row = next(r for r in comp_list if r["selection_status"] in ("Selected", "SELECTED"))

    print(f"  Model with highest Test Acc: {max_test_acc_model['model']} ({max_test_acc_model['test_accuracy']:.1%})")
    print(f"  Model with highest CV Mean : {max_cv_model['model']} ({max_cv_model['cv_mean']:.1%})")
    print(f"  Actually Selected Model    : {selected_model_row['model']} (Status: {selected_model_row['selection_status']})")

    assert selected_model_row["model"] == max_cv_model["model"], "Selected model must match highest CV model, not single test split"
    print("-> Test 5 PASSED: Generalization CV criterion strictly prioritized over single test accuracy.")

    print("\n" + "=" * 60)
    print("ALL MODEL SELECTION TESTS COMPLETED SUCCESSFULLY (100% GREEN)!")
    print("=" * 60)

if __name__ == "__main__":
    run_model_selection_tests()

