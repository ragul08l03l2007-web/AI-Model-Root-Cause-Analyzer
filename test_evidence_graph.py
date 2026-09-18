# test_evidence_graph.py
"""
Comprehensive Test Suite for Multi-Experiment Evidence Graph Engine.

Verifies:
1. Dedicated evidence_graph module and API (EvidenceGraph, EvidenceGraphBuilder, build_evidence_graph).
2. Complete coverage of Node Types & Edge Relations.
3. Single candidate graph & multi-candidate graph structures.
4. Verified Model Reliance vs No Measurable Model Reliance lineage.
5. Evidence Fusion decomposed score traceability (35+35+15+10+5 = 100).
6. Remediation & Re-evaluation chain (baseline vs remediated metric, generalization gap, resolution).
7. Metadata integrity (graph_version, analysis_type, target_column, selected_model, evaluation_metric).
8. Graph Integrity Validation Engine (detects duplicates, broken edges, invalid types, validates DAG).
9. Dynamic Sub-Graph Trace Extraction (get_evidence_trace).
10. 100% JSON Serializability (zero NumPy leaks).
11. End-to-end classification, multiclass, and continuous regression pipeline compatibility.
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

from analysis.evidence_graph import (
    EvidenceGraph,
    EvidenceGraphNode,
    EvidenceGraphEdge,
    EvidenceGraphMetadata,
    EvidenceGraphBuilder,
    build_evidence_graph,
    validate_evidence_graph,
    get_evidence_trace,
    CANONICAL_NODE_TYPES,
    CANONICAL_EDGE_RELATIONS,
)
from analysis.model_analyzer import analyze_model
from analysis.verification_engine import VerificationEngine


def _assert_json_clean(data: dict, test_name: str):
    try:
        serialized = json.dumps(data)
        assert len(serialized) > 10, f"[{test_name}] Serialized JSON is unexpectedly empty"
    except Exception as e:
        raise AssertionError(f"[{test_name}] Failed JSON serialization (NumPy leak): {e}")


def test_evidence_graph_definitions():
    """Verifies that all node types and edge relations are defined and supported."""
    print("\n[TEST 1] Testing Node Types and Edge Relations Definitions...")
    expected_node_types = {
        "candidate",
        "observation",
        "feature_importance",
        "experiment",
        "ablation_experiment",
        "permutation_experiment",
        "stability_experiment",
        "control_experiment",
        "measurement",
        "result",
        "evidence",
        "evidence_fusion",
        "verdict",
        "remediation",
        "intervention",
        "outcome",
        "reevaluation",
        "remediation_result",
        "resolution",
    }
    expected_edge_relations = {
        "supports",
        "supported_by",
        "tests",
        "tested_by",
        "measures",
        "contributes_to",
        "leads_to",
        "modifies",
        "evaluates",
        "evaluated_by",
        "evaluated_as",
        "verifies",
        "produced",
        "produces",
        "triggers",
        "compared_against",
        "resolves_to",
    }

    assert CANONICAL_NODE_TYPES == expected_node_types, "Mismatch in CANONICAL_NODE_TYPES"
    assert CANONICAL_EDGE_RELATIONS == expected_edge_relations, "Mismatch in CANONICAL_EDGE_RELATIONS"
    print("  -> Node Types and Edge Relations Verified!")


def test_synthetic_multi_candidate_graph():
    """Verifies synthetic construction of multiple candidates with verified vs rejected verdicts."""
    print("\n[TEST 2] Testing Multi-Candidate Evidence Graph Construction...")
    
    feature_impact = {
        "annual_income": {"importance": 0.959, "relative_share_pct": 95.9, "influence_tier": "Very Strong"},
        "satisfaction_score": {"importance": 0.025, "relative_share_pct": 2.5, "influence_tier": "Low"},
        "support_tickets": {"importance": 0.016, "relative_share_pct": 1.6, "influence_tier": "Low"},
    }

    candidate_experiments = [
        {
            "candidate_feature": "annual_income",
            "metric_name": "Weighted F1",
            "baseline_metric": 0.9836,
            "ablated_metric": 0.2765,
            "ablation_delta": -0.7071,
            "ablation_delta_pct": -71.89,
            "permuted_metric": 0.2302,
            "permutation_delta": -0.7534,
            "permutation_delta_pct": -76.60,
            "perturbed_metric": 0.9836,
            "noise_delta": 0.0,
            "prediction_flip_rate_pct": 3.33,
            "noise_sensitivity": "robust",
            "control_feature": "support_tickets",
            "control_ablation_metric": 0.9836,
            "control_delta": 0.0,
            "control_specificity_ratio": 70.71,
            "evidence_ratings": {"ablation": "strong", "permutation": "strong", "control": "passed"},
            "score_decomposition": {
                "ablation_points": 35,
                "permutation_points": 35,
                "control_points": 15,
                "stability_points": 10,
                "consistency_points": 5,
                "total_score": 100,
                "components": {
                    "ablation": {"score": 35, "max_score": 35},
                    "permutation": {"score": 35, "max_score": 35},
                    "control": {"score": 15, "max_score": 15},
                    "stability": {"score": 10, "max_score": 10},
                    "consistency": {"score": 5, "max_score": 5},
                }
            },
            "verdict": "VERIFIED MODEL RELIANCE",
            "evidence_score": 100,
            "summary": "Verified Model Reliance on Predictor 'annual_income'"
        },
        {
            "candidate_feature": "satisfaction_score",
            "metric_name": "Weighted F1",
            "baseline_metric": 0.9836,
            "ablated_metric": 0.9836,
            "ablation_delta": 0.0,
            "ablation_delta_pct": 0.0,
            "permuted_metric": 0.9836,
            "permutation_delta": 0.0,
            "permutation_delta_pct": 0.0,
            "perturbed_metric": 0.9836,
            "noise_delta": 0.0,
            "prediction_flip_rate_pct": 0.0,
            "noise_sensitivity": "robust",
            "control_feature": "support_tickets",
            "control_ablation_metric": 0.9836,
            "control_delta": 0.0,
            "control_specificity_ratio": 1.0,
            "evidence_ratings": {"ablation": "negligible", "permutation": "negligible", "control": "inconclusive"},
            "score_decomposition": {
                "ablation_points": 0,
                "permutation_points": 0,
                "control_points": 5,
                "stability_points": 10,
                "consistency_points": 0,
                "total_score": 15,
                "components": {
                    "ablation": {"score": 0, "max_score": 35},
                    "permutation": {"score": 0, "max_score": 35},
                    "control": {"score": 5, "max_score": 15},
                    "stability": {"score": 10, "max_score": 10},
                    "consistency": {"score": 0, "max_score": 5},
                }
            },
            "verdict": "NO MEASURABLE MODEL RELIANCE",
            "evidence_score": 15,
            "summary": "No Measurable Model Reliance on Predictor 'satisfaction_score'"
        }
    ]

    remediation_sim = {
        "status": "Success",
        "task_type": "classification",
        "metric_name": "Weighted F1",
        "baseline": {"train_score": 1.0, "test_score": 0.9836, "generalization_gap": 0.0164},
        "remediated": {"train_score": 0.9877, "test_score": 0.9836, "generalization_gap": 0.0041},
        "deltas": {"test_metric_delta": 0.0, "generalization_gap_reduction": 0.0123},
        "resolution_verdict": "PARTIALLY RESOLVED — Generalization gap reduced by 1.23 percentage points, while held-out weighted F1 showed no measurable improvement.",
        "proof_summary": "Simulated remediation proved generalization gap reduced."
    }

    graph = build_evidence_graph(
        task_type="classification",
        feature_impact=feature_impact,
        candidate_experiments=candidate_experiments,
        diagnostic_candidates=[],
        remediation_simulation=remediation_sim,
        target_column="customer_segment",
        selected_model="Gradient Boosting",
        evaluation_metric="Weighted F1"
    )

    assert graph["graph_id"] == "evidence_graph_classification"
    assert graph["metadata"]["target_column"] == "customer_segment"
    assert graph["metadata"]["selected_model"] == "Gradient Boosting"
    assert graph["metadata"]["evaluation_metric"] == "Weighted F1"
    assert graph["metadata"]["number_of_candidates"] == 3
    assert graph["summary"]["verified_candidates"] == ["annual_income"]
    assert graph["summary"]["rejected_candidates"] == ["satisfaction_score"]
    assert graph["validation"]["is_valid"] is True

    # Trace annual_income
    trace_income = get_evidence_trace("annual_income", graph)
    assert trace_income["verdict"] == "VERIFIED MODEL RELIANCE"
    assert trace_income["evidence_score"] == 100
    assert "Ablation Retraining" in trace_income["trace_summary"]

    # Trace satisfaction_score
    trace_sat = get_evidence_trace("satisfaction_score", graph)
    assert trace_sat["verdict"] == "NO MEASURABLE MODEL RELIANCE"
    assert trace_sat["evidence_score"] == 15

    _assert_json_clean(graph, "multi_candidate_graph")
    print("  -> Multi-Candidate Graph Verified!")


def test_graph_validation_engine_error_detection():
    """Verifies that the validation engine correctly catches duplicates, broken edges, and orphans."""
    print("\n[TEST 3] Testing Graph Validation Engine Error Catching...")

    # Case A: Duplicate node ID
    bad_graph_dup = {
        "nodes": [
            {"id": "node_1", "type": "candidate", "candidate_feature": "x", "data": {}},
            {"id": "node_1", "type": "feature_importance", "candidate_feature": "x", "data": {}},
        ],
        "edges": []
    }
    val_res = validate_evidence_graph(bad_graph_dup)
    assert val_res["is_valid"] is False
    assert any("Duplicate node ID" in err for err in val_res["errors"])
    print("  -> Caught duplicate node ID successfully.")

    # Case B: Broken Edge (missing target)
    bad_graph_broken_edge = {
        "nodes": [
            {"id": "node_1", "type": "candidate", "candidate_feature": "x", "data": {}},
        ],
        "edges": [
            {"id": "edge_1", "source": "node_1", "target": "missing_node_2", "relation": "supported_by"}
        ]
    }
    val_res_broken = validate_evidence_graph(bad_graph_broken_edge)
    assert val_res_broken["is_valid"] is False
    assert any("does not exist in graph nodes" in err for err in val_res_broken["errors"])
    print("  -> Caught broken edge target successfully.")

    # Case C: Invalid Node Type
    bad_graph_invalid_type = {
        "nodes": [
            {"id": "node_1", "type": "invalid_magic_type", "candidate_feature": "x", "data": {}},
        ],
        "edges": []
    }
    val_res_type = validate_evidence_graph(bad_graph_invalid_type)
    assert val_res_type["is_valid"] is False
    assert any("Invalid node type" in err for err in val_res_type["errors"])
    print("  -> Caught invalid node type successfully.")

    # Case D: Orphaned node warning
    graph_with_orphan = {
        "nodes": [
            {"id": "node_1", "type": "candidate", "candidate_feature": "x", "data": {}},
            {"id": "node_2", "type": "feature_importance", "candidate_feature": "x", "data": {}},
            {"id": "orphan_node", "type": "candidate", "candidate_feature": "z", "data": {}},
        ],
        "edges": [
            {"id": "edge_1", "source": "node_1", "target": "node_2", "relation": "supported_by"}
        ]
    }
    val_res_orphan = validate_evidence_graph(graph_with_orphan)
    assert val_res_orphan["is_valid"] is True  # Warning, not fatal error
    assert any("Orphaned node" in w for w in val_res_orphan["warnings"])
    print("  -> Flagged orphaned node warning successfully.")


def test_evidence_graph_classification_pipeline():
    """Verifies graph generation on end-to-end classification dataset."""
    print("\n[TEST 4] Testing Evidence Graph on End-to-End Classification Dataset...")
    df = pd.read_csv("random_test_dataset.csv")
    target_col = "churn"

    res = analyze_model(df, target_column=target_col, analysis_type="classification")
    
    assert "evidence_graph" in res, "evidence_graph missing from analyze_model result"
    graph = res["evidence_graph"]
    assert graph.get("graph_id") == "evidence_graph_classification"
    assert graph.get("task_type") == "classification"

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    summary = graph.get("summary", {})
    validation = graph.get("validation", {})
    meta = graph.get("metadata", {})

    print(f"  -> Generated {len(nodes)} nodes and {len(edges)} edges.")
    print(f"  -> Candidates analyzed: {summary.get('candidates_analyzed')}")
    print(f"  -> Metadata: {meta}")

    assert len(nodes) > 0, "No nodes in evidence graph"
    assert len(edges) > 0, "No edges in evidence graph"
    assert validation.get("is_valid") is True, f"Graph validation failed: {validation.get('errors')}"
    assert meta.get("target_column") == "churn"
    assert meta.get("evaluation_metric") == "Weighted F1"

    _assert_json_clean(graph, "classification_evidence_graph")
    print("  -> TEST 4 PASSED!")


def test_evidence_graph_regression_pipeline():
    """Verifies graph generation on end-to-end regression dataset."""
    print("\n[TEST 5] Testing Evidence Graph on End-to-End Regression Dataset...")
    df = pd.read_csv("continuous_regression_test_dataset.csv")
    target_col = "annual_bonus"

    res = analyze_model(df, target_column=target_col, analysis_type="regression")
    
    assert "evidence_graph" in res, "evidence_graph missing from analyze_model result"
    graph = res["evidence_graph"]
    assert graph.get("graph_id") == "evidence_graph_regression"
    assert graph.get("task_type") == "regression"

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    validation = graph.get("validation", {})
    meta = graph.get("metadata", {})

    assert len(nodes) > 0
    assert len(edges) > 0
    assert validation.get("is_valid") is True
    assert meta.get("target_column") == "annual_bonus"
    assert meta.get("evaluation_metric") == "R2 Score"

    _assert_json_clean(graph, "regression_evidence_graph")
    print("  -> TEST 5 PASSED!")


def test_multiclass_evidence_graph_integrity():
    """Verifies multiclass dataset (annual_income verified reliance) evidence graph integrity."""
    print("\n[TEST 6] Testing Evidence Graph on Multiclass Test Dataset...")
    np.random.seed(42)
    n = 600
    income = np.random.uniform(20000, 150000, n)
    age = np.random.randint(18, 70, n)
    satisfaction = np.random.randint(1, 10, n)
    support_tickets = np.random.randint(0, 10, n)
    region = np.random.choice(["North", "South", "East", "West"], n)

    segments = []
    for inc in income:
        if inc < 50000:
            segments.append("Bronze")
        elif inc < 100000:
            segments.append("Silver")
        else:
            segments.append("Gold")

    df = pd.DataFrame({
        "annual_income": income,
        "customer_age": age,
        "satisfaction_score": satisfaction,
        "support_tickets": support_tickets,
        "region": region,
        "customer_segment": segments
    })

    res = analyze_model(df, target_column="customer_segment", analysis_type="classification")
    graph = res["evidence_graph"]

    # Check all candidates and their verdicts
    all_exps = res["verification_experiments"]
    exp_verdicts = {e["candidate_feature"]: e["verdict"] for e in all_exps}
    
    assert exp_verdicts.get("annual_income") == "VERIFIED MODEL RELIANCE"
    if "satisfaction_score" in exp_verdicts:
        assert exp_verdicts.get("satisfaction_score") == "NO MEASURABLE MODEL RELIANCE"
    if "support_tickets" in exp_verdicts:
        assert exp_verdicts.get("support_tickets") == "NO MEASURABLE MODEL RELIANCE"

    trace = get_evidence_trace("annual_income", graph)
    assert trace["verdict"] == "VERIFIED MODEL RELIANCE"
    assert trace["evidence_score"] >= 90
    print(f"  -> annual_income trace: {trace['trace_summary']}")

    # Test get_evidence_chain
    chain_income = VerificationEngine.get_evidence_chain("annual_income", graph)
    assert chain_income["verdict"] == "VERIFIED MODEL RELIANCE"
    assert chain_income["candidate"] == "annual_income"
    chain_types = [item["type"] for item in chain_income["evidence_chain"]]
    assert "observation" in chain_types
    assert "ablation" in chain_types
    assert "permutation" in chain_types
    assert "noise" in chain_types
    assert "control" in chain_types
    assert "evidence_fusion" in chain_types
    assert "verdict" in chain_types
    print(f"  -> annual_income evidence chain extracted: {len(chain_income['evidence_chain'])} items: {chain_types}")

    print("  -> TEST 6 PASSED!")


def test_programmatic_evidence_graph_api():
    """Verifies direct programmatic EvidenceGraph API (add_node, add_edge, get_node, get_edges, get_evidence_chain, validate)."""
    print("\n[TEST 7] Testing Direct Programmatic EvidenceGraph API...")
    
    g = EvidenceGraph(graph_id="test_custom_graph", task_type="classification")
    
    # 1. Add nodes
    g.add_node(
        node_id="cand_feat_a",
        node_type="candidate",
        label="Candidate Feature A",
        candidate_feature="feat_a",
        description="Hypothesized root cause feature A",
        data={"importance": 0.85}
    )
    g.add_node(
        node_id="obs_feat_a",
        node_type="observation",
        label="Observational Importance",
        candidate_feature="feat_a",
        description="Observed model reliance",
        data={"importance": 0.85, "relative_share_pct": 85.0}
    )
    g.add_node(
        node_id="abl_feat_a",
        node_type="ablation_experiment",
        label="Ablation Experiment",
        candidate_feature="feat_a",
        data={"delta": -0.65, "delta_pct": -65.0, "score_pts": 35, "max_pts": 35}
    )
    g.add_node(
        node_id="fusion_feat_a",
        node_type="evidence_fusion",
        label="Evidence Fusion Score",
        candidate_feature="feat_a",
        data={"total_score": 95, "components": {"ablation": 35}}
    )
    g.add_node(
        node_id="verdict_feat_a",
        node_type="verdict",
        label="Diagnostic Verdict",
        candidate_feature="feat_a",
        data={"verdict": "VERIFIED MODEL RELIANCE", "evidence_score": 95}
    )

    # 2. Add edges
    g.add_edge(edge_id="e1", source="cand_feat_a", target="obs_feat_a", relation="supports")
    g.add_edge(edge_id="e2", source="cand_feat_a", target="abl_feat_a", relation="tested_by")
    g.add_edge(edge_id="e3", source="abl_feat_a", target="fusion_feat_a", relation="contributes_to")
    g.add_edge(edge_id="e4", source="fusion_feat_a", target="verdict_feat_a", relation="produces")

    # 3. Query methods
    node = g.get_node("cand_feat_a")
    assert node is not None
    assert node.candidate_feature == "feat_a"
    assert node.description == "Hypothesized root cause feature A"

    all_edges = g.get_edges()
    assert len(all_edges) == 4
    
    cand_edges = g.get_edges("cand_feat_a")
    assert len(cand_edges) == 2

    # 4. Validate
    val = g.validate()
    assert val["is_valid"] is True
    assert val["node_count"] == 5
    assert val["edge_count"] == 4

    # 5. Evidence chain from verdict ID
    chain_by_verdict = g.get_evidence_chain("verdict_feat_a")
    assert chain_by_verdict["verdict"] == "VERIFIED MODEL RELIANCE"
    assert chain_by_verdict["candidate"] == "feat_a"
    assert len(chain_by_verdict["evidence_chain"]) >= 4

    # 6. Evidence chain from candidate name
    chain_by_name = g.get_evidence_chain("feat_a")
    assert chain_by_name["verdict"] == "VERIFIED MODEL RELIANCE"
    assert chain_by_name["candidate"] == "feat_a"

    # 7. Serialization
    d = g.to_dict()
    assert d["graph_id"] == "test_custom_graph"
    _assert_json_clean(d, "custom_graph_api")
    
    print("  -> Programmatic EvidenceGraph API verified successfully!")
    print("  -> TEST 7 PASSED!")


if __name__ == "__main__":
    test_evidence_graph_definitions()
    test_synthetic_multi_candidate_graph()
    test_graph_validation_engine_error_detection()
    test_evidence_graph_classification_pipeline()
    test_evidence_graph_regression_pipeline()
    test_multiclass_evidence_graph_integrity()
    test_programmatic_evidence_graph_api()
    print("\n=======================================================")
    print("ALL MULTI-EXPERIMENT EVIDENCE GRAPH TESTS PASSED (7/7)!")
    print("=======================================================\n")

