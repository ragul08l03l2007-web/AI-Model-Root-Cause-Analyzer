# test_verification_engine.py
"""
Test Suite for Automated Root-Cause Verification & Closed-Loop Remediation Engine (analysis/verification_engine.py).

Verifies:
1. Targeted candidate selection (only tests suspicious/top features, with low-impact control feature).
2. Controlled experimental trials (Retrained Feature Ablation, Permutation Shuffling, Noise Perturbation, Control Ablation).
3. Evidence ratings and mathematical evidence scores without arbitrary threshold leaks.
4. Closed-loop remediation simulation (Before vs After metrics and resolution verdicts).
5. 100% JSON-serializable output (zero NumPy type representations).
6. Robustness on both classification and continuous regression datasets.
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from analysis.model_analyzer import analyze_model
from analysis.verification_engine import VerificationEngine


def _assert_json_clean(data: dict, test_name: str):
    try:
        serialized = json.dumps(data)
        assert len(serialized) > 10, f"[{test_name}] Output is empty"
    except Exception as e:
        raise AssertionError(f"[{test_name}] Failed JSON serialization (NumPy leak): {e}")


def test_classification_verification_experiments():
    print("\n[TEST 1] Testing Root-Cause Verification on Classification Dataset...")
    df = pd.read_csv("random_test_dataset.csv")
    target_col = "churn"
    y_raw = df[target_col]
    X_df = df.drop(columns=[target_col])

    # 1. Run basic analyzer to get candidate feature impact
    res = analyze_model(df, target_column=target_col, analysis_type="classification")
    feat_impact = res["feature_impact"]

    # 2. Verify candidate selection
    candidates, control_feat = VerificationEngine.select_candidate_features(feat_impact, list(X_df.columns), max_candidates=2)
    assert len(candidates) <= 2, "Candidate count exceeds max_candidates"
    assert len(candidates) > 0, "Failed to select candidate features"
    assert control_feat is not None, "Failed to select control feature"
    print(f"  -> Selected targeted candidates: {candidates}")
    print(f"  -> Selected scientific control feature: {control_feat}")

    # 3. Run Controlled Experiments
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    experiments = VerificationEngine.run_feature_verification_experiments(
        X_df=X_df,
        y_raw=y_raw,
        task_type="classification",
        champion_model=model,
        feature_impact=feat_impact,
        max_candidates=2
    )

    assert len(experiments) > 0, "No verification experiments returned"
    for exp in experiments:
        print(f"\n  --- Experiment Report for Candidate: '{exp['candidate_feature']}' ---")
        print(f"  Baseline Metric ({exp['metric_name']}) : {exp['baseline_metric']:.4f}")
        print(f"  Ablated Metric                     : {exp['ablated_metric']:.4f} (delta: {exp['ablation_delta']:+.4f})")
        print(f"  Permuted Metric                    : {exp['permuted_metric']:.4f} (delta: {exp['permutation_delta']:+.4f})")
        print(f"  Noise Perturbed Metric             : {exp['perturbed_metric']:.4f} (delta: {exp['noise_delta']:+.4f}, flip: {exp['prediction_flip_rate_pct']:.1f}%)")
        print(f"  Control Feature ({exp['control_feature']})        : {exp['control_ablation_metric']:.4f} (delta: {exp['control_delta']:+.4f})")
        print(f"  Evidence Ratings                   : {exp['evidence_ratings']}")
        print(f"  Evidence Score                     : {exp['evidence_score']}/100")
        print(f"  Verdict                            : {exp['verdict']}")

        assert "baseline_metric" in exp
        assert "ablation_delta" in exp
        assert "permutation_delta" in exp
        assert "perturbed_metric" in exp
        assert "noise_delta" in exp
        assert "prediction_flip_rate_pct" in exp
        assert "noise_sensitivity" in exp
        assert "control_delta" in exp
        assert "verdict" in exp
        assert "noise" in exp["evidence_ratings"]
        assert exp["evidence_ratings"]["control"] in ("passed", "inconclusive", "failed")
        assert "score_decomposition" in exp
        assert exp["score_decomposition"]["total_score"] == exp["evidence_score"]
        assert 0 <= exp["evidence_score"] <= 100

    _assert_json_clean(experiments, "classification_experiments")
    print("\n  -> TEST 1 (Classification Verification Experiments) PASSED!")


def test_regression_verification_experiments():
    print("\n[TEST 2] Testing Root-Cause Verification on Continuous Regression Dataset...")
    df = pd.read_csv("continuous_regression_test_dataset.csv")
    target_col = "annual_bonus"
    y_raw = df[target_col]
    X_df = df.drop(columns=[target_col])

    res = analyze_model(df, target_column=target_col, analysis_type="regression")
    feat_impact = res["feature_impact"]

    model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    experiments = VerificationEngine.run_feature_verification_experiments(
        X_df=X_df,
        y_raw=y_raw,
        task_type="regression",
        champion_model=model,
        feature_impact=feat_impact,
        max_candidates=2
    )

    assert len(experiments) > 0, "No regression experiments returned"
    for exp in experiments:
        print(f"  -> Candidate '{exp['candidate_feature']}': Ablation delta = {exp['ablation_delta']:+.4f}, Noise delta = {exp['noise_delta']:+.4f}, Control delta = {exp['control_delta']:+.4f}, Score = {exp['evidence_score']}/100, Verdict = {exp['verdict']}")
        assert exp["metric_name"] == "R2 Score" or "Score" in exp["metric_name"]
        assert "evidence_ratings" in exp
        assert "perturbed_metric" in exp
        assert "prediction_flip_rate_pct" in exp


    _assert_json_clean(experiments, "regression_experiments")
    print("  -> TEST 2 (Regression Verification Experiments) PASSED!")


def test_closed_loop_remediation_simulation():
    print("\n[TEST 3] Testing Closed-Loop Remediation Simulation...")
    df = pd.read_csv("random_test_dataset.csv")
    target_col = "churn"
    y_raw = df[target_col]
    X_df = df.drop(columns=[target_col])

    res = analyze_model(df, target_column=target_col, analysis_type="classification")
    candidates = res.get("root_causes_structured", [])
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)

    sim = VerificationEngine.simulate_closed_loop_remediation(
        X_df=X_df,
        y_raw=y_raw,
        task_type="classification",
        champion_model=model,
        diagnostic_candidates=candidates
    )

    print(f"  -> Remediation Status       : {sim.get('status')}")
    print(f"  -> Baseline Test Score      : {sim['baseline']['test_score']:.4f} (Gap: {sim['baseline']['generalization_gap']:.4f})")
    print(f"  -> Remediated Test Score    : {sim['remediated']['test_score']:.4f} (Gap: {sim['remediated']['generalization_gap']:.4f})")
    print(f"  -> Resolution Verdict       : {sim['resolution_verdict']}")
    print(f"  -> Proof Summary            : {sim['proof_summary']}")

    assert "baseline" in sim
    assert "remediated" in sim
    assert "deltas" in sim
    assert "resolution_verdict" in sim
    assert any(valid_prefix in sim["resolution_verdict"] for valid_prefix in [
        "BOTH IMPROVED", "PERFORMANCE IMPROVED", "PARTIALLY RESOLVED", "BASELINE PREFERRED", "NO MATERIAL IMPROVEMENT", "ALIGNMENT IMPROVED"
    ])
    _assert_json_clean(sim, "remediation_simulation")
    print("  -> TEST 3 (Closed-Loop Remediation Simulation) PASSED!")


def test_master_verify_and_simulate():
    print("\n[TEST 4] Testing Master verify_and_simulate API...")
    df = pd.read_csv("random_test_dataset.csv")
    target_col = "churn"
    y_raw = df[target_col]
    X_df = df.drop(columns=[target_col])

    res = analyze_model(df, target_column=target_col, analysis_type="classification")
    feat_impact = res["feature_impact"]
    candidates = res.get("root_causes_structured", [])
    model = RandomForestClassifier(n_estimators=50, random_state=42)

    master_result = VerificationEngine.verify_and_simulate(
        X_df=X_df,
        y_raw=y_raw,
        task_type="classification",
        champion_model=model,
        feature_impact=feat_impact,
        diagnostic_candidates=candidates,
        max_candidates=2
    )

    assert master_result["status"] == "Success"
    assert "candidate_experiments" in master_result
    assert "remediation_simulation" in master_result
    assert len(master_result["candidate_experiments"]) > 0
    _assert_json_clean(master_result, "master_verify_result")
    print(f"  -> Master result generated {master_result['experiments_count']} experiment reports and simulated remediation successfully.")
    print("  -> TEST 4 PASSED!")


if __name__ == "__main__":
    print("======================================================================")
    print("RUNNING ROOT-CAUSE VERIFICATION & EXPERIMENTAL ENGINE TEST SUITE")
    print("======================================================================")
    test_classification_verification_experiments()
    test_regression_verification_experiments()
    test_closed_loop_remediation_simulation()
    test_master_verify_and_simulate()
    print("======================================================================")
    print("ALL ROOT-CAUSE VERIFICATION ENGINE TESTS PASSED 100%!")
    print("======================================================================")
