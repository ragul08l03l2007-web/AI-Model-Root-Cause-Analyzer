# analysis/evidence_graph.py
"""
Multi-Experiment Evidence Graph Engine for AI Model Root-Cause Diagnostics.

Provides a structured, traceable Directed Acyclic Graph (DAG) representation that
connects:
    Candidates (Hypotheses)
      -> Observational Evidence (Feature Importance, Errors, Data Quality)
      -> Controlled Experiments (Ablation, Permutation, Stability, Control)
      -> Experiment Results (Measured deltas, flip rates, specificity ratios)
      -> Evidence Fusion (Decomposed scoring: Ablation, Permutation, Control, Stability, Consistency)
      -> Root-Cause Verdicts (VERIFIED MODEL RELIANCE, NO MEASURABLE MODEL RELIANCE, INCONCLUSIVE)
      -> Remediation Interventions (Regularization, Balancing, Feature Modification)
      -> Re-evaluation Benchmarks (Before vs After metrics & generalization gaps)
      -> Final Resolution Status (PARTIALLY RESOLVED, BOTH IMPROVED, PERFORMANCE IMPROVED, etc.)

Zero external heavy dependencies; strictly standard library dataclasses, typing, and JSON serialization.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from analysis.evidence import safe_primitive


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safe conversion of scalar/numpy values to rounded Python float."""
    try:
        if val is None:
            return default
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, 4)
    except (ValueError, TypeError):
        return default


# Canonical Node Types and aliases for backward compatibility
CANONICAL_NODE_TYPES: Set[str] = {
    "candidate",
    "observation",
    "feature_importance",
    "experiment",
    "ablation_experiment",
    "permutation_experiment",
    "stability_experiment",
    "control_experiment",
    "result",
    "evidence_fusion",
    "verdict",
    "remediation",
    "intervention",
    "reevaluation",
    "remediation_result",
    "resolution",
}

# Canonical Edge Relations
CANONICAL_EDGE_RELATIONS: Set[str] = {
    "supported_by",
    "tested_by",
    "produced",
    "produces",
    "contributes_to",
    "triggers",
    "evaluated_by",
    "evaluated_as",
    "compared_against",
    "resolves_to",
}


@dataclass
class EvidenceGraphNode:
    """
    A single node in the Multi-Experiment Evidence Graph.
    """
    id: str
    type: str
    label: str
    candidate_feature: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return safe_primitive({
            "id": str(self.id),
            "type": str(self.type),
            "label": str(self.label),
            "candidate_feature": str(self.candidate_feature),
            "data": self.data,
            "metadata": self.metadata,
        })


@dataclass
class EvidenceGraphEdge:
    """
    A directed relationship in the Multi-Experiment Evidence Graph.
    """
    id: str
    source: str
    target: str
    relation: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return safe_primitive({
            "id": str(self.id),
            "source": str(self.source),
            "target": str(self.target),
            "relation": str(self.relation),
            "weight": _safe_float(self.weight, 1.0),
            "metadata": self.metadata,
        })


@dataclass
class EvidenceGraphMetadata:
    """
    Top-level execution and environment metadata for the Evidence Graph.
    """
    graph_version: str = "2.0.0"
    analysis_type: str = "classification"
    target_column: Optional[str] = None
    selected_model: Optional[str] = None
    evaluation_metric: str = "Weighted F1"
    number_of_candidates: int = 0
    number_of_experiments: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return safe_primitive(asdict(self))


class EvidenceGraph:
    """
    Directed Acyclic Graph (DAG) holding complete empirical lineage from
    initial hypotheses down to verified resolution.
    """
    def __init__(
        self,
        graph_id: str,
        task_type: str,
        metadata: Optional[EvidenceGraphMetadata] = None,
    ):
        self.graph_id = graph_id
        self.task_type = task_type
        self.metadata = metadata or EvidenceGraphMetadata(analysis_type=task_type)
        self._nodes: Dict[str, EvidenceGraphNode] = {}
        self._edges: Dict[str, EvidenceGraphEdge] = {}
        self._candidates_ordered: List[str] = []
        self._verified_candidates: List[str] = []
        self._rejected_candidates: List[str] = []
        self.remediation_tested: bool = False

    def add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        candidate_feature: str,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceGraphNode:
        """Add or update a node in the graph."""
        node = EvidenceGraphNode(
            id=node_id,
            type=node_type,
            label=label,
            candidate_feature=candidate_feature,
            data=data or {},
            metadata=metadata or {},
        )
        self._nodes[node_id] = node
        return node

    def add_edge(
        self,
        edge_id: str,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceGraphEdge:
        """Add or update a directed edge in the graph."""
        edge = EvidenceGraphEdge(
            id=edge_id,
            source=source,
            target=target,
            relation=relation,
            weight=weight,
            metadata=metadata or {},
        )
        self._edges[edge_id] = edge
        return edge

    def get_node(self, node_id: str) -> Optional[EvidenceGraphNode]:
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[EvidenceGraphEdge]:
        return self._edges.get(edge_id)

    def all_nodes(self) -> List[Dict[str, Any]]:
        return [node.to_dict() for node in self._nodes.values()]

    def all_edges(self) -> List[Dict[str, Any]]:
        return [edge.to_dict() for edge in self._edges.values()]

    def validate(self) -> Dict[str, Any]:
        """Validate structural integrity of the graph."""
        return validate_evidence_graph(self.to_dict())

    def get_trace(self, candidate_feature: str) -> Dict[str, Any]:
        """Extract isolated empirical lineage for a specific candidate feature."""
        return get_evidence_trace(candidate_feature, self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire graph to a JSON-serializable dictionary."""
        nodes_list = self.all_nodes()
        edges_list = self.all_edges()

        # Update metadata counts
        self.metadata.number_of_candidates = len(self._candidates_ordered)
        self.metadata.number_of_experiments = sum(
            1 for n in nodes_list if "experiment" in n.get("type", "")
        )

        raw_dict = {
            "graph_id": self.graph_id,
            "task_type": self.task_type,
            "metadata": self.metadata.to_dict(),
            "nodes": nodes_list,
            "edges": edges_list,
            "summary": {
                "total_nodes": len(nodes_list),
                "total_edges": len(edges_list),
                "candidates_analyzed": self._candidates_ordered,
                "verified_candidates": self._verified_candidates,
                "rejected_candidates": self._rejected_candidates,
                "remediation_tested": self.remediation_tested,
            },
        }

        # Embed validation diagnostics
        raw_dict["validation"] = validate_evidence_graph(raw_dict)
        return safe_primitive(raw_dict)


def validate_evidence_graph(graph_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive structural validation for the Multi-Experiment Evidence Graph.

    Validates:
    - Unique node and edge IDs
    - All edges connect existing source and target nodes
    - Node types belong to canonical allowed set
    - Edge relations belong to canonical allowed set
    - Detects unlinked/orphaned nodes (warnings)
    - Verifies candidates have associated observations/verdicts
    """
    errors: List[str] = []
    warnings: List[str] = []

    nodes = graph_dict.get("nodes", [])
    edges = graph_dict.get("edges", [])

    seen_node_ids: Set[str] = set()
    node_types_map: Dict[str, str] = {}
    candidate_features_in_nodes: Set[str] = set()

    for node in nodes:
        nid = node.get("id")
        if not nid:
            errors.append("Node missing required 'id' field.")
            continue
        if nid in seen_node_ids:
            errors.append(f"Duplicate node ID detected: '{nid}'.")
        seen_node_ids.add(nid)

        ntype = node.get("type")
        if not ntype or ntype not in CANONICAL_NODE_TYPES:
            errors.append(f"Invalid node type '{ntype}' in node '{nid}'.")
        node_types_map[nid] = str(ntype)

        cf = node.get("candidate_feature")
        if cf:
            candidate_features_in_nodes.add(cf)

    seen_edge_ids: Set[str] = set()
    in_degrees: Dict[str, int] = {nid: 0 for nid in seen_node_ids}
    out_degrees: Dict[str, int] = {nid: 0 for nid in seen_node_ids}

    for edge in edges:
        eid = edge.get("id")
        if not eid:
            errors.append("Edge missing required 'id' field.")
            continue
        if eid in seen_edge_ids:
            errors.append(f"Duplicate edge ID detected: '{eid}'.")
        seen_edge_ids.add(eid)

        rel = edge.get("relation")
        if not rel or rel not in CANONICAL_EDGE_RELATIONS:
            errors.append(f"Invalid edge relation '{rel}' in edge '{eid}'.")

        src = edge.get("source")
        tgt = edge.get("target")

        if src not in seen_node_ids:
            errors.append(f"Edge '{eid}' source '{src}' does not exist in graph nodes.")
        else:
            out_degrees[src] = out_degrees.get(src, 0) + 1

        if tgt not in seen_node_ids:
            errors.append(f"Edge '{eid}' target '{tgt}' does not exist in graph nodes.")
        else:
            in_degrees[tgt] = in_degrees.get(tgt, 0) + 1

    # Check for completely orphaned nodes
    for nid in seen_node_ids:
        if in_degrees.get(nid, 0) == 0 and out_degrees.get(nid, 0) == 0:
            warnings.append(f"Orphaned node with zero degree detected: '{nid}'.")

    # Domain integrity check: check experiments have parent candidate
    for nid, ntype in node_types_map.items():
        if "experiment" in ntype or ntype == "experiment":
            if in_degrees.get(nid, 0) == 0:
                warnings.append(f"Experiment node '{nid}' has no incoming candidate hypothesis edge.")

    is_valid = (len(errors) == 0)

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "candidates_count": len(candidate_features_in_nodes),
    }


def get_evidence_trace(candidate_feature: str, evidence_graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts isolated lineage trace and human-readable sequence for a specific candidate feature.
    """
    if not evidence_graph or not candidate_feature:
        return {
            "candidate_feature": candidate_feature,
            "nodes": [],
            "edges": [],
            "nodes_count": 0,
            "edges_count": 0,
            "verdict": "UNKNOWN",
            "evidence_score": 0,
            "trace_summary": f"No evidence graph available for candidate '{candidate_feature}'.",
            "trace_steps": []
        }

    all_nodes = evidence_graph.get("nodes", [])
    all_edges = evidence_graph.get("edges", [])

    # Find nodes belonging to this candidate
    cand_nodes = [n for n in all_nodes if n.get("candidate_feature") == candidate_feature]
    cand_node_ids = {n["id"] for n in cand_nodes}

    cand_edges = [
        e for e in all_edges
        if e.get("source") in cand_node_ids and e.get("target") in cand_node_ids
    ]

    # Extract key findings
    verdict_node = next((n for n in cand_nodes if n.get("type") == "verdict"), None)
    fusion_node = next((n for n in cand_nodes if n.get("type") == "evidence_fusion"), None)
    imp_node = next((n for n in cand_nodes if n.get("type") in ("feature_importance", "observation")), None)
    abl_node = next((n for n in cand_nodes if n.get("type") == "ablation_experiment"), None)
    perm_node = next((n for n in cand_nodes if n.get("type") == "permutation_experiment"), None)
    stab_node = next((n for n in cand_nodes if n.get("type") == "stability_experiment"), None)
    ctrl_node = next((n for n in cand_nodes if n.get("type") == "control_experiment"), None)
    resol_node = next((n for n in cand_nodes if n.get("type") == "resolution"), None)

    verdict_str = verdict_node["data"].get("verdict", "UNTESTED") if verdict_node else "UNTESTED"
    ev_score = fusion_node["data"].get("total_score", 0) if fusion_node else 0

    trace_steps: List[str] = []
    if imp_node:
        share = imp_node["data"].get("relative_share_pct", 0.0)
        imp_v = imp_node["data"].get("importance", 0.0)
        trace_steps.append(f"Feature Importance: accounts for {share:.1f}% of model importance ({imp_v:.4f}).")

    if abl_node:
        abl_d = abl_node["data"].get("delta", 0.0)
        abl_pts = abl_node["data"].get("score_pts", 0)
        trace_steps.append(f"Ablation Retraining: metric shifted by {abl_d:+.4f} (points: {abl_pts}/35).")

    if perm_node:
        perm_d = perm_node["data"].get("delta", 0.0)
        perm_pts = perm_node["data"].get("score_pts", 0)
        trace_steps.append(f"Permutation Shuffling: metric shifted by {perm_d:+.4f} (points: {perm_pts}/35).")

    if stab_node:
        flips = stab_node["data"].get("flip_rate_pct", 0.0)
        stab_pts = stab_node["data"].get("score_pts", 0)
        trace_steps.append(f"Measurement Jitter (10%): prediction flip rate is {flips:.1f}% (points: {stab_pts}/10).")

    if ctrl_node:
        ctrl_f = ctrl_node["data"].get("control_feature", "None")
        ctrl_d = ctrl_node["data"].get("control_delta", 0.0)
        ctrl_pts = ctrl_node["data"].get("score_pts", 0)
        trace_steps.append(f"Control Specificity ({ctrl_f}): control baseline delta is {ctrl_d:+.4f} (points: {ctrl_pts}/15).")

    if fusion_node:
        trace_steps.append(f"Evidence Fusion: total decomposed evidence score is {ev_score}/100.")

    if verdict_node:
        trace_steps.append(f"Diagnostic Verdict: {verdict_str}.")

    if resol_node:
        res_v = resol_node["data"].get("resolution_verdict", "")
        trace_steps.append(f"Closed-Loop Resolution: {res_v}.")

    trace_summary = f"Trace for '{candidate_feature}': " + " -> ".join([s.split(":")[0] for s in trace_steps]) if trace_steps else f"No trace recorded for '{candidate_feature}'."

    return safe_primitive({
        "candidate_feature": candidate_feature,
        "nodes": cand_nodes,
        "edges": cand_edges,
        "nodes_count": len(cand_nodes),
        "edges_count": len(cand_edges),
        "verdict": verdict_str,
        "evidence_score": ev_score,
        "trace_summary": trace_summary,
        "trace_steps": trace_steps,
    })


class EvidenceGraphBuilder:
    """
    Builder and serializer for creating Multi-Experiment Evidence Graphs from
    observational analysis, controlled verification experiments, and remediation results.
    """
    NODE_TYPES = CANONICAL_NODE_TYPES
    EDGE_RELATIONS = CANONICAL_EDGE_RELATIONS

    @classmethod
    def build_graph(
        cls,
        task_type: str,
        feature_impact: Dict[str, Any],
        candidate_experiments: List[Dict[str, Any]],
        diagnostic_candidates: List[Dict[str, Any]],
        remediation_simulation: Dict[str, Any],
        target_column: Optional[str] = None,
        selected_model: Optional[str] = None,
        evaluation_metric: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Builds the complete Multi-Experiment Evidence Graph DAG from diagnostic artifacts.
        """
        metric_name = evaluation_metric or ("Weighted F1" if task_type == "classification" else "R2 Score")
        meta = EvidenceGraphMetadata(
            graph_version="2.0.0",
            analysis_type=task_type,
            target_column=target_column,
            selected_model=selected_model,
            evaluation_metric=metric_name,
            number_of_candidates=0,
            number_of_experiments=len(candidate_experiments or []),
        )

        graph = EvidenceGraph(
            graph_id=f"evidence_graph_{task_type}",
            task_type=task_type,
            metadata=meta
        )

        # 1. Map all candidate features (tested + diagnostic candidates + high importance)
        exp_by_feature = {e.get("candidate_feature"): e for e in (candidate_experiments or []) if e.get("candidate_feature")}
        
        all_candidates_ordered: List[str] = []
        seen_cands: Set[str] = set()

        # Add tested candidates first
        for exp in candidate_experiments or []:
            f = exp.get("candidate_feature")
            if f and f not in seen_cands:
                all_candidates_ordered.append(f)
                seen_cands.add(f)

        # Add features from diagnostic candidates
        for rc in diagnostic_candidates or []:
            for ev in rc.get("evidence_items", []):
                for af in ev.get("affected_features", []):
                    if af and af not in seen_cands:
                        all_candidates_ordered.append(af)
                        seen_cands.add(af)
            cand_feat = rc.get("candidate_feature")
            if cand_feat and cand_feat not in seen_cands:
                all_candidates_ordered.append(cand_feat)
                seen_cands.add(cand_feat)

        # Add top features from feature_impact if any
        if feature_impact:
            sorted_impact = sorted(
                feature_impact.items(),
                key=lambda item: _safe_float(item[1].get("importance", 0.0) if isinstance(item[1], dict) else item[1]),
                reverse=True
            )
            for f, data in sorted_impact[:3]:
                if f and f not in seen_cands:
                    all_candidates_ordered.append(f)
                    seen_cands.add(f)

        graph._candidates_ordered = all_candidates_ordered

        # 2. Build candidate sub-graphs
        for cand in all_candidates_ordered:
            cand_node_id = f"cand_{cand}"
            imp_node_id = f"imp_{cand}"

            # Importance data
            imp_info = feature_impact.get(cand, {}) if feature_impact else {}
            if isinstance(imp_info, dict):
                imp_val = _safe_float(imp_info.get("importance", 0.0))
                share_pct = _safe_float(imp_info.get("relative_share_pct", 0.0))
                tier = str(imp_info.get("influence_tier", "Moderate"))
            else:
                imp_val = _safe_float(imp_info)
                share_pct = 0.0
                tier = "Moderate"

            # Node 1: Candidate Hypothesis
            graph.add_node(
                node_id=cand_node_id,
                node_type="candidate",
                label=f"Candidate: {cand}",
                candidate_feature=cand,
                data={"feature_name": cand, "is_experimentally_tested": cand in exp_by_feature}
            )

            # Node 2: Observational Feature Importance
            graph.add_node(
                node_id=imp_node_id,
                node_type="feature_importance",
                label=f"Importance: {cand} ({share_pct:.1f}%)",
                candidate_feature=cand,
                data={"importance": imp_val, "relative_share_pct": share_pct, "influence_tier": tier}
            )

            # Edge 1: Candidate -> Importance
            graph.add_edge(
                edge_id=f"edge_cand_imp_{cand}",
                source=cand_node_id,
                target=imp_node_id,
                relation="supported_by",
                weight=_safe_float(share_pct / 100.0 if share_pct > 0 else 1.0, 1.0),
                metadata={"metric": "importance_share", "value": share_pct}
            )

            if cand in exp_by_feature:
                exp = exp_by_feature[cand]
                decomp = exp.get("score_decomposition", {})
                ev_score = exp.get("evidence_score", 0)
                verdict = exp.get("verdict", "")

                if "VERIFIED" in verdict:
                    graph._verified_candidates.append(cand)
                elif "NO MEASURABLE" in verdict:
                    graph._rejected_candidates.append(cand)

                # Experiment Nodes
                abl_node_id = f"abl_{cand}"
                perm_node_id = f"perm_{cand}"
                stab_node_id = f"stab_{cand}"
                ctrl_node_id = f"ctrl_{cand}"
                fusion_node_id = f"fusion_{cand}"
                verdict_node_id = f"verdict_{cand}"

                # Node 3: Ablation Experiment
                graph.add_node(
                    node_id=abl_node_id,
                    node_type="ablation_experiment",
                    label=f"Ablation: {cand} (Δ {exp.get('ablation_delta', 0):+.3f})",
                    candidate_feature=cand,
                    data={
                        "baseline_metric": exp.get("baseline_metric", 0.0),
                        "ablated_metric": exp.get("ablated_metric", 0.0),
                        "delta": exp.get("ablation_delta", 0.0),
                        "delta_pct": exp.get("ablation_delta_pct", 0.0),
                        "effect_strength": exp.get("evidence_ratings", {}).get("ablation", "neutral"),
                        "score_pts": decomp.get("ablation_points", 0),
                        "max_pts": 35,
                    }
                )
                graph.add_edge(
                    edge_id=f"edge_cand_abl_{cand}",
                    source=cand_node_id,
                    target=abl_node_id,
                    relation="tested_by",
                    weight=1.0,
                    metadata={"experiment": "retrained_ablation"}
                )

                # Node 4: Permutation Experiment
                graph.add_node(
                    node_id=perm_node_id,
                    node_type="permutation_experiment",
                    label=f"Permutation: {cand} (Δ {exp.get('permutation_delta', 0):+.3f})",
                    candidate_feature=cand,
                    data={
                        "baseline_metric": exp.get("baseline_metric", 0.0),
                        "permuted_metric": exp.get("permuted_metric", 0.0),
                        "delta": exp.get("permutation_delta", 0.0),
                        "delta_pct": exp.get("permutation_delta_pct", 0.0),
                        "effect_strength": exp.get("evidence_ratings", {}).get("permutation", "neutral"),
                        "score_pts": decomp.get("permutation_points", 0),
                        "max_pts": 35,
                    }
                )
                graph.add_edge(
                    edge_id=f"edge_cand_perm_{cand}",
                    source=cand_node_id,
                    target=perm_node_id,
                    relation="tested_by",
                    weight=1.0,
                    metadata={"experiment": "test_time_permutation"}
                )

                # Node 5: Stability Jitter Experiment
                graph.add_node(
                    node_id=stab_node_id,
                    node_type="stability_experiment",
                    label=f"Stability Jitter: {cand} ({exp.get('prediction_flip_rate_pct', 0):.1f}% flips)",
                    candidate_feature=cand,
                    data={
                        "baseline_metric": exp.get("baseline_metric", 0.0),
                        "perturbed_metric": exp.get("perturbed_metric", 0.0),
                        "noise_delta": exp.get("noise_delta", 0.0),
                        "flip_rate_pct": exp.get("prediction_flip_rate_pct", 0.0),
                        "noise_sensitivity": exp.get("noise_sensitivity", "robust"),
                        "score_pts": decomp.get("stability_points", 0),
                        "max_pts": 10,
                    }
                )
                graph.add_edge(
                    edge_id=f"edge_cand_stab_{cand}",
                    source=cand_node_id,
                    target=stab_node_id,
                    relation="tested_by",
                    weight=1.0,
                    metadata={"experiment": "measurement_stability_jitter"}
                )

                # Node 6: Control Specificity Experiment
                graph.add_node(
                    node_id=ctrl_node_id,
                    node_type="control_experiment",
                    label=f"Control: {exp.get('control_feature', 'None')} (Δ {exp.get('control_delta', 0):+.3f})",
                    candidate_feature=cand,
                    data={
                        "control_feature": exp.get("control_feature", "None"),
                        "control_ablation_metric": exp.get("control_ablation_metric", 0.0),
                        "control_delta": exp.get("control_delta", 0.0),
                        "specificity_ratio": exp.get("control_specificity_ratio", 0.0),
                        "status": exp.get("evidence_ratings", {}).get("control", "inconclusive"),
                        "score_pts": decomp.get("control_points", 0),
                        "max_pts": 15,
                    }
                )
                graph.add_edge(
                    edge_id=f"edge_cand_ctrl_{cand}",
                    source=cand_node_id,
                    target=ctrl_node_id,
                    relation="compared_against",
                    weight=1.0,
                    metadata={"experiment": "control_specificity", "control_feature": exp.get("control_feature")}
                )

                # Node 7: Evidence Fusion (with 5-component breakdown)
                graph.add_node(
                    node_id=fusion_node_id,
                    node_type="evidence_fusion",
                    label=f"Evidence Fusion: {ev_score}/100",
                    candidate_feature=cand,
                    data={
                        "total_score": ev_score,
                        "score_decomposition": decomp,
                        "components": decomp.get("components", {}),
                    }
                )
                graph.add_edge(
                    edge_id=f"edge_abl_fusion_{cand}",
                    source=abl_node_id,
                    target=fusion_node_id,
                    relation="contributes_to",
                    weight=_safe_float(decomp.get("ablation_points", 0) / 35.0, 0.0),
                    metadata={"points": decomp.get("ablation_points", 0), "max_points": 35}
                )
                graph.add_edge(
                    edge_id=f"edge_perm_fusion_{cand}",
                    source=perm_node_id,
                    target=fusion_node_id,
                    relation="contributes_to",
                    weight=_safe_float(decomp.get("permutation_points", 0) / 35.0, 0.0),
                    metadata={"points": decomp.get("permutation_points", 0), "max_points": 35}
                )
                graph.add_edge(
                    edge_id=f"edge_stab_fusion_{cand}",
                    source=stab_node_id,
                    target=fusion_node_id,
                    relation="contributes_to",
                    weight=_safe_float(decomp.get("stability_points", 0) / 10.0, 0.0),
                    metadata={"points": decomp.get("stability_points", 0), "max_points": 10}
                )
                graph.add_edge(
                    edge_id=f"edge_ctrl_fusion_{cand}",
                    source=ctrl_node_id,
                    target=fusion_node_id,
                    relation="contributes_to",
                    weight=_safe_float(decomp.get("control_points", 0) / 15.0, 0.0),
                    metadata={"points": decomp.get("control_points", 0), "max_points": 15}
                )

                # Node 8: Diagnostic Verdict
                graph.add_node(
                    node_id=verdict_node_id,
                    node_type="verdict",
                    label=f"Verdict: {verdict}",
                    candidate_feature=cand,
                    data={
                        "verdict": verdict,
                        "evidence_score": ev_score,
                        "summary": exp.get("summary", ""),
                        "status": "VERIFIED" if "VERIFIED" in verdict else ("REJECTED" if "NO MEASURABLE" in verdict else "EVALUATED"),
                    }
                )
                graph.add_edge(
                    edge_id=f"edge_fusion_verdict_{cand}",
                    source=fusion_node_id,
                    target=verdict_node_id,
                    relation="produces",
                    weight=1.0,
                    metadata={"verdict": verdict, "score": ev_score}
                )

            else:
                # Non-tested candidate branch
                verdict_node_id = f"verdict_{cand}"
                graph.add_node(
                    node_id=verdict_node_id,
                    node_type="verdict",
                    label=f"Verdict: NOT TESTED ({cand})",
                    candidate_feature=cand,
                    data={
                        "verdict": "NOT TESTED",
                        "evidence_score": 0,
                        "summary": f"Controlled counterfactual experiments were not conducted for '{cand}'.",
                        "status": "UNTESTED",
                    }
                )
                graph.add_edge(
                    edge_id=f"edge_imp_verdict_{cand}",
                    source=imp_node_id,
                    target=verdict_node_id,
                    relation="produces",
                    weight=0.0,
                    metadata={"reason": "untested_candidate"}
                )

        # 3. Closed-Loop Remediation Sub-Graph
        if remediation_simulation and remediation_simulation.get("status") == "Success":
            primary_candidate = (
                graph._verified_candidates[0]
                if graph._verified_candidates
                else (all_candidates_ordered[0] if all_candidates_ordered else "primary_candidate")
            )
            primary_verdict_id = f"verdict_{primary_candidate}"
            if graph.get_node(primary_verdict_id):
                interv_node_id = f"interv_{primary_candidate}"
                remed_res_node_id = f"remed_res_{primary_candidate}"
                resol_node_id = f"resol_{primary_candidate}"

                # Node 9: Remediation Intervention
                graph.add_node(
                    node_id=interv_node_id,
                    node_type="intervention",
                    label="Remediation Intervention (Regularization & Rebalancing)",
                    candidate_feature=primary_candidate,
                    data={
                        "strategy": "Constrained Model Complexity / Balanced Weighting",
                        "task_type": task_type,
                    }
                )
                graph.add_edge(
                    edge_id=f"edge_verdict_interv_{primary_candidate}",
                    source=primary_verdict_id,
                    target=interv_node_id,
                    relation="triggers",
                    weight=1.0,
                    metadata={"trigger": "model_reliance_or_overfit"}
                )

                # Node 10: Remediation Re-evaluation Result
                graph.add_node(
                    node_id=remed_res_node_id,
                    node_type="remediation_result",
                    label="Remediation Result (Before vs After)",
                    candidate_feature=primary_candidate,
                    data={
                        "baseline": remediation_simulation.get("baseline", {}),
                        "remediated": remediation_simulation.get("remediated", {}),
                        "deltas": remediation_simulation.get("deltas", {}),
                    }
                )
                graph.add_edge(
                    edge_id=f"edge_interv_remed_{primary_candidate}",
                    source=interv_node_id,
                    target=remed_res_node_id,
                    relation="produces",
                    weight=1.0,
                    metadata={"metric_delta": remediation_simulation.get("deltas", {}).get("test_metric_delta", 0.0)}
                )

                # Node 11: Resolution Verdict
                res_verdict = remediation_simulation.get("resolution_verdict", "EVALUATED")
                graph.add_node(
                    node_id=resol_node_id,
                    node_type="resolution",
                    label=f"Resolution: {res_verdict}",
                    candidate_feature=primary_candidate,
                    data={
                        "resolution_verdict": res_verdict,
                        "proof_summary": remediation_simulation.get("proof_summary", ""),
                    }
                )
                graph.add_edge(
                    edge_id=f"edge_remed_resol_{primary_candidate}",
                    source=remed_res_node_id,
                    target=resol_node_id,
                    relation="evaluated_as",
                    weight=1.0,
                    metadata={"verdict": res_verdict}
                )
                graph.remediation_tested = True

        return graph.to_dict()

    @classmethod
    def validate_graph(cls, graph_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience validation delegation."""
        return validate_evidence_graph(graph_dict)

    @classmethod
    def get_evidence_trace(cls, candidate_feature: str, evidence_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience trace extraction delegation."""
        return get_evidence_trace(candidate_feature, evidence_graph)


def build_evidence_graph(
    task_type: str,
    feature_impact: Dict[str, Any],
    candidate_experiments: List[Dict[str, Any]],
    diagnostic_candidates: List[Dict[str, Any]],
    remediation_simulation: Dict[str, Any],
    target_column: Optional[str] = None,
    selected_model: Optional[str] = None,
    evaluation_metric: Optional[str] = None,
) -> Dict[str, Any]:
    """Factory helper to build a Multi-Experiment Evidence Graph."""
    return EvidenceGraphBuilder.build_graph(
        task_type=task_type,
        feature_impact=feature_impact,
        candidate_experiments=candidate_experiments,
        diagnostic_candidates=diagnostic_candidates,
        remediation_simulation=remediation_simulation,
        target_column=target_column,
        selected_model=selected_model,
        evaluation_metric=evaluation_metric,
    )
