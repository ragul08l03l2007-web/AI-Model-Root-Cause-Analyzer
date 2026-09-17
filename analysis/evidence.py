# analysis/evidence.py
"""
DiagnosticEvidence model and EvidenceCollection bus.
Encapsulates individual empirical signals discovered during dataset profiling,
target analysis, preprocessing, model training, cross-validation, and error analysis.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import math
import numpy as np
import pandas as pd


def safe_primitive(val: Any) -> Any:
    """Convert any NumPy / pandas / scalar type into a JSON-serializable Python native type."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return [safe_primitive(v) for v in val]
    if isinstance(val, dict):
        return {str(k): safe_primitive(v) for k, v in val.items()}
    if isinstance(val, np.ndarray):
        return [safe_primitive(v) for v in val.tolist()]
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), 4)
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    return str(val)


@dataclass
class DiagnosticEvidence:
    """
    Formal canonical representation of a single empirical diagnostic signal.
    Distinguishes raw signals from model behaviors, dataset properties, and root causes.
    """
    evidence_id: str
    signal_name: str
    domain: str                                 # e.g. class_performance, generalization, leakage, data_quality, feature_reliance, regression_residual, partition_stability, sample_size
    metric: str                                 # e.g. recall_gap, overfit_gap, cv_std, missing_ratio, correlation
    observed_value: Any
    baseline_value: Optional[Any] = None
    magnitude: float = 0.0                      # Normalized effect magnitude (0.0 to 1.0)
    direction: str = ""                         # e.g. "positive", "negative", "higher", "lower", "disparity"
    sample_support: int = 0                     # Number of observations providing signal
    affected_population: str = ""               # Human-readable population description
    affected_features: List[str] = field(default_factory=list)
    affected_classes: List[str] = field(default_factory=list)
    signal_type: str = "direct"                 # direct, behavioral, structural, leakage, data_quality
    strength: str = "MODERATE"                  # HIGH, MODERATE, LOW
    reliability: float = 1.0                    # 0.0 to 1.0 based on statistical test stability
    model_scope: str = "selected_model"         # selected_model, cross_model, dataset
    supporting_models: List[str] = field(default_factory=list)
    stability: float = 1.0                      # 0.0 to 1.0
    limitations: List[str] = field(default_factory=list)
    context: str = ""
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to clean JSON-serializable dictionary with backward-compatible helper keys."""
        d = asdict(self)
        ctx = self.context or self.summary or f"{self.metric}: {self.observed_value}"
        d["context"] = ctx
        d["signal"] = self.signal_name or self.summary or f"{self.metric} = {self.observed_value}"
        d["value"] = self.observed_value
        d["strength"] = self.strength
        return safe_primitive(d)


class EvidenceBus:
    """
    Collection manager and registry for DiagnosticEvidence objects during analysis.
    """
    def __init__(self):
        self._evidence: List[DiagnosticEvidence] = []

    def emit(self, evidence: DiagnosticEvidence) -> None:
        """Register a new diagnostic evidence signal."""
        self._evidence.append(evidence)

    def add(self, **kwargs) -> DiagnosticEvidence:
        """Convenience factory to create and register evidence in one step."""
        ev = DiagnosticEvidence(**kwargs)
        self._evidence.append(ev)
        return ev

    def all(self) -> List[DiagnosticEvidence]:
        """Return all registered evidence records."""
        return list(self._evidence)

    def by_domain(self, domain: str) -> List[DiagnosticEvidence]:
        """Filter evidence records by analytical domain."""
        return [e for e in self._evidence if e.domain == domain]

    def by_feature(self, feature_name: str) -> List[DiagnosticEvidence]:
        """Filter evidence records affecting a specific feature."""
        return [e for e in self._evidence if feature_name in e.affected_features]

    def by_class(self, class_name: str) -> List[DiagnosticEvidence]:
        """Filter evidence records affecting a specific target class."""
        return [e for e in self._evidence if class_name in e.affected_classes]

    def to_list_of_dicts(self) -> List[Dict[str, Any]]:
        """Serialize all evidence to dictionary representations."""
        return [e.to_dict() for e in self._evidence]
