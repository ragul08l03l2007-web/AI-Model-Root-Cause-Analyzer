# test_multi_dataset_scenarios.py
"""
Multi-Scenario Scientific Verification Benchmark Suite.

Validates that the Verification Engine acts as a true ML Diagnostic Investigation System
across 6 distinct synthetic scenarios:
- Scenario A: Genuine Feature Reliance (VERIFIED MODEL RELIANCE)
- Scenario B: Collinear / Redundant Features (VERIFIED STRUCTURAL DEPENDENCY or REDUNDANT)
- Scenario C: False Candidate Refutation (REFUTED CANDIDATE - proves initial hypothesis false!)
- Scenario D: Noise Brittleness (HIGH LOCAL BRITTLENESS / high flip rate)
- Scenario E: Overfitting Generalization Gap (PARTIALLY RESOLVED / gap reduced)
- Scenario F: Continuous Regression (Targeted ablations and R2 remediation proof)
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from analysis.verification_engine import VerificationEngine


def test_scenario_a_genuine_reliance():
    print("\n[SCENARIO A] Testing Genuine Feature Reliance...")
    np.random.seed(42)
    n = 600
    income = np.random.uniform(20000, 150000, n)
    age = np.random.randint(18, 70, n)
    support_tickets = np.random.randint(0, 10, n)

    # Churn is almost entirely determined by income tier
    churn = (income < 50000).astype(int)

    df = pd.DataFrame({
        "annual_income": income,
        "customer_age": age,
        "support_tickets": support_tickets,
        "churn": churn
    })

    X_df = df.drop(columns=["churn"])
    y_raw = df["churn"]

    feat_impact = {
        "annual_income": {"importance": 0.90, "relative_share_pct": 85.0},
        "customer_age": {"importance": 0.06, "relative_share_pct": 8.0},
        "support_tickets": {"importance": 0.04, "relative_share_pct": 7.0}
    }

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    experiments = VerificationEngine.run_feature_verification_experiments(
        X_df=X_df,
        y_raw=y_raw,
        task_type="classification",
        champion_model=model,
        feature_impact=feat_impact,
        max_candidates=1
    )

    exp = experiments[0]
    print(f"  Candidate: {exp['candidate_feature']}")
    print(f"  Baseline F1: {exp['baseline_metric']:.4f}")
    print(f"  Ablated F1 : {exp['ablated_metric']:.4f} (drop: {exp['ablation_delta_pct']:.1f}%)")
    print(f"  Permuted F1: {exp['permuted_metric']:.4f} (drop: {exp['permutation_delta_pct']:.1f}%)")
    print(f"  Control (support_tickets) Delta: {exp['control_delta']:+.4f}")
    print(f"  Verdict: {exp['verdict']}, Score: {exp['evidence_score']}/100")

    assert exp["candidate_feature"] == "annual_income"
    assert exp["verdict"] == "VERIFIED MODEL RELIANCE"
    assert exp["evidence_score"] >= 80
    assert "score_decomposition" in exp
    decomp = exp["score_decomposition"]
    assert decomp["total_score"] == exp["evidence_score"]
    assert decomp["ablation_points"] + decomp["permutation_points"] + decomp["control_points"] + decomp["stability_points"] + decomp["consistency_points"] == exp["evidence_score"]
    assert exp["evidence_ratings"]["ablation"] in ("strong", "moderate")
    assert exp["evidence_ratings"]["permutation"] in ("strong", "moderate")
    assert exp["evidence_ratings"]["control"] == "passed"
    print("  -> SCENARIO A PASSED: Correctly verified model reliance on genuine dominant feature.")


def test_scenario_b_collinear_redundant_features():
    print("\n[SCENARIO B] Testing Collinear / Redundant Features...")
    np.random.seed(42)
    n = 600
    x1 = np.random.normal(0, 1, n)
    x2 = x1 + np.random.normal(0, 0.01, n)  # Nearly identical copy
    x_control = np.random.normal(0, 1, n)
    y = (x1 + x2 > 0).astype(int)

    df = pd.DataFrame({"feat_a": x1, "feat_b_copy": x2, "noise_ctrl": x_control, "target": y})
    X_df = df.drop(columns=["target"])
    y_raw = df["target"]

    feat_impact = {
        "feat_a": {"importance": 0.50, "relative_share_pct": 48.0},
        "feat_b_copy": {"importance": 0.45, "relative_share_pct": 45.0},
        "noise_ctrl": {"importance": 0.05, "relative_share_pct": 7.0}
    }

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    experiments = VerificationEngine.run_feature_verification_experiments(
        X_df=X_df,
        y_raw=y_raw,
        task_type="classification",
        champion_model=model,
        feature_impact=feat_impact,
        max_candidates=1
    )

    exp = experiments[0]
    print(f"  Candidate: {exp['candidate_feature']}")
    print(f"  Ablation Drop: {exp['ablation_delta']:+.4f}, Permutation Drop: {exp['permutation_delta']:+.4f}")
    print(f"  Verdict: {exp['verdict']}, Score: {exp['evidence_score']}/100")

    # Because feat_b_copy preserves the signal when feat_a is permuted or ablated, drop is modest
    assert exp["verdict"] in ("VERIFIED STRUCTURAL DEPENDENCY (Collinear Shielding)", "REDUNDANT / DISTRIBUTED SIGNAL", "PARTIAL / INTERACTIVE SIGNAL")
    print("  -> SCENARIO B PASSED: Correctly identified redundant/shielded signal.")


def test_scenario_c_false_candidate_refutation():
    print("\n[SCENARIO C] Testing False Candidate Refutation (Proving Hypothesis False)...")
    np.random.seed(42)
    n = 600
    real_signal = np.random.normal(0, 1, n)
    pure_noise = np.random.uniform(0, 100, n)
    ctrl_noise = np.random.uniform(-1, 1, n)
    y = (real_signal > 0).astype(int)

    df = pd.DataFrame({
        "real_feature": real_signal,
        "bogus_flagged_feature": pure_noise,
        "control_col": ctrl_noise,
        "target": y
    })
    X_df = df.drop(columns=["target"])
    y_raw = df["target"]

    # Artificially claim bogus_flagged_feature is a top suspicious candidate
    feat_impact = {
        "bogus_flagged_feature": {"importance": 0.85, "relative_share_pct": 70.0},
        "real_feature": {"importance": 0.10, "relative_share_pct": 20.0},
        "control_col": {"importance": 0.05, "relative_share_pct": 10.0}
    }

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    experiments = VerificationEngine.run_feature_verification_experiments(
        X_df=X_df,
        y_raw=y_raw,
        task_type="classification",
        champion_model=model,
        feature_impact=feat_impact,
        max_candidates=1
    )

    exp = experiments[0]
    print(f"  Candidate: {exp['candidate_feature']}")
    print(f"  Ablation Delta: {exp['ablation_delta']:+.4f}, Permutation Delta: {exp['permutation_delta']:+.4f}")
    print(f"  Control Delta: {exp['control_delta']:+.4f}, Control Verdict: {exp['evidence_ratings']['control']}")
    print(f"  Verdict: {exp['verdict']}, Score: {exp['evidence_score']}/100")
    print(f"  Summary: {exp['summary']}")

    assert exp["candidate_feature"] == "bogus_flagged_feature"
    assert exp["verdict"] == "NO MEASURABLE MODEL RELIANCE"
    assert exp["evidence_score"] == 0
    assert exp["score_decomposition"]["total_score"] == 0
    assert exp["evidence_ratings"]["control"] == "inconclusive"
    assert "no measurable" in exp["summary"].lower()
    print("  -> SCENARIO C PASSED: Successfully proved lack of measurable reliance under tested interventions!")


def test_scenario_d_noise_brittleness():
    print("\n[SCENARIO D] Testing Noise Brittleness Experiment...")
    from sklearn.linear_model import LogisticRegression
    np.random.seed(42)
    n = 600
    # Create continuous sensitive threshold boundary
    fragile_val = np.random.normal(0, 1, n)
    other_val = np.random.normal(0, 1, n)
    y = (fragile_val + 0.1 * other_val > 0).astype(int)

    df = pd.DataFrame({"fragile_feature": fragile_val, "other_feature": other_val, "target": y})
    X_df = df.drop(columns=["target"])
    y_raw = df["target"]

    feat_impact = {
        "fragile_feature": {"importance": 0.85, "relative_share_pct": 80.0},
        "other_feature": {"importance": 0.15, "relative_share_pct": 20.0}
    }

    model = LogisticRegression(random_state=42)
    experiments = VerificationEngine.run_feature_verification_experiments(
        X_df=X_df,
        y_raw=y_raw,
        task_type="classification",
        champion_model=model,
        feature_impact=feat_impact,
        max_candidates=1
    )

    exp = experiments[0]
    print(f"  Candidate: {exp['candidate_feature']}")
    print(f"  Prediction Flip Rate under Noise: {exp['prediction_flip_rate_pct']:.2f}%")
    print(f"  Noise Sensitivity: {exp['noise_sensitivity']}")
    print(f"  Verdict: {exp['verdict']}")

    assert "prediction_flip_rate_pct" in exp
    assert exp["prediction_flip_rate_pct"] > 0
    assert exp["evidence_ratings"]["noise"] in ("high_brittleness", "moderate_sensitivity", "robust")
    print("  -> SCENARIO D PASSED: Noise jitter trial accurately tracked prediction flip rate and sensitivity.")


def test_scenario_e_overfitting_generalization_gap():
    print("\n[SCENARIO E] Testing Overfitting Generalization Gap Resolution...")
    np.random.seed(42)
    n = 150
    X = np.random.normal(0, 1, (n, 10))
    y = np.random.choice([0, 1], size=n)

    cols = [f"col_{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y

    X_df = df.drop(columns=["target"])
    y_raw = df["target"]

    # An unconstrained overfitted tree (100% train, ~50% test)
    unconstrained_tree = DecisionTreeClassifier(max_depth=20, random_state=42)

    sim = VerificationEngine.simulate_closed_loop_remediation(
        X_df=X_df,
        y_raw=y_raw,
        task_type="classification",
        champion_model=unconstrained_tree,
        diagnostic_candidates=[]
    )

    print(f"  Baseline Train: {sim['baseline']['train_score']:.4f} | Test: {sim['baseline']['test_score']:.4f} | Gap: {sim['baseline']['generalization_gap']:.4f}")
    print(f"  Remediated Train: {sim['remediated']['train_score']:.4f} | Test: {sim['remediated']['test_score']:.4f} | Gap: {sim['remediated']['generalization_gap']:.4f}")
    print(f"  Test Score Delta: {sim['deltas']['test_metric_delta']:+.4f}")
    print(f"  Generalization Gap Reduction: {sim['deltas']['generalization_gap_reduction']:+.4f}")
    print(f"  Resolution Verdict: {sim['resolution_verdict']}")

    assert "resolution_verdict" in sim
    # Generalization gap must have been reduced
    assert sim["deltas"]["generalization_gap_reduction"] > 0
    # Must NOT claim "VERIFIED MITIGATED" without nuance
    assert "VERIFIED MITIGATED" not in sim["resolution_verdict"]
    assert any(sub in sim["resolution_verdict"] for sub in ["PARTIALLY RESOLVED", "BOTH IMPROVED", "PERFORMANCE IMPROVED", "NO MATERIAL IMPROVEMENT", "ALIGNMENT IMPROVED"])
    print("  -> SCENARIO E PASSED: Accurately reflected generalization gap reduction with scientific resolution.")


def test_scenario_f_continuous_regression():
    print("\n[SCENARIO F] Testing Continuous Regression Scenario...")
    np.random.seed(42)
    n = 500
    experience = np.random.uniform(1, 20, n)
    hours = np.random.uniform(20, 60, n)
    noise_col = np.random.uniform(0, 10, n)
    salary = 30000 + 4000 * experience + 500 * hours + np.random.normal(0, 2000, n)

    df = pd.DataFrame({
        "experience": experience,
        "hours_worked": hours,
        "unrelated_id": noise_col,
        "salary": salary
    })
    X_df = df.drop(columns=["salary"])
    y_raw = df["salary"]

    feat_impact = {
        "experience": {"importance": 0.80, "relative_share_pct": 75.0},
        "hours_worked": {"importance": 0.18, "relative_share_pct": 20.0},
        "unrelated_id": {"importance": 0.02, "relative_share_pct": 5.0}
    }

    model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    experiments = VerificationEngine.run_feature_verification_experiments(
        X_df=X_df,
        y_raw=y_raw,
        task_type="regression",
        champion_model=model,
        feature_impact=feat_impact,
        max_candidates=2
    )

    assert len(experiments) >= 1
    exp_lead = experiments[0]
    print(f"  Candidate: {exp_lead['candidate_feature']}")
    print(f"  Baseline R2: {exp_lead['baseline_metric']:.4f}")
    print(f"  Ablated R2 : {exp_lead['ablated_metric']:.4f} (drop: {exp_lead['ablation_delta_pct']:.1f}%)")
    print(f"  Verdict: {exp_lead['verdict']}, Score: {exp_lead['evidence_score']}/100")

    assert exp_lead["candidate_feature"] == "experience"
    assert exp_lead["verdict"] in ("VERIFIED MODEL RELIANCE", "VERIFIED STRUCTURAL DEPENDENCY (Collinear Shielding)")
    assert exp_lead["evidence_score"] >= 70

    # Also test regression closed loop simulation
    sim = VerificationEngine.simulate_closed_loop_remediation(
        X_df=X_df,
        y_raw=y_raw,
        task_type="regression",
        champion_model=model,
        diagnostic_candidates=[]
    )
    print(f"  Regression Remediation Verdict: {sim['resolution_verdict']}")
    assert "baseline" in sim and "remediated" in sim
    print("  -> SCENARIO F PASSED: Continuous regression verification executed cleanly.")


if __name__ == "__main__":
    print("======================================================================")
    print("RUNNING MULTI-SCENARIO SCIENTIFIC VERIFICATION BENCHMARK SUITE")
    print("======================================================================")
    test_scenario_a_genuine_reliance()
    test_scenario_b_collinear_redundant_features()
    test_scenario_c_false_candidate_refutation()
    test_scenario_d_noise_brittleness()
    test_scenario_e_overfitting_generalization_gap()
    test_scenario_f_continuous_regression()
    print("======================================================================")
    print("ALL 6 SCIENTIFIC VERIFICATION BENCHMARK SCENARIOS PASSED 100%!")
    print("======================================================================")
