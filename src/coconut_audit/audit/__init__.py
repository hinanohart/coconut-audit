"""Audit primitives: Steering / Shortcut / Drift scorers, Verdict aggregator."""

from coconut_audit.audit.drift import DriftScorer, latent_drift_score
from coconut_audit.audit.shortcut import ShortcutDetector
from coconut_audit.audit.steering import SteeringProbe
from coconut_audit.audit.verdict import (
    AuditVerdict,
    Verdict,
    aggregate_verdict,
    label_for_metric,
)

__all__ = [
    "AuditVerdict",
    "DriftScorer",
    "ShortcutDetector",
    "SteeringProbe",
    "Verdict",
    "aggregate_verdict",
    "label_for_metric",
    "latent_drift_score",
]
