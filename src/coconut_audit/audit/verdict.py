"""Verdict aggregator: combines per-probe metrics into a final AuditVerdict."""

from __future__ import annotations

from dataclasses import dataclass

from coconut_audit.core.types import AuditVerdict

__all__ = ["AuditVerdict", "Verdict", "aggregate_verdict"]


@dataclass(slots=True)
class Verdict:
    """Per-probe verdict slice: a label plus the metric that produced it."""

    label: AuditVerdict
    metric_name: str
    metric_value: float
    threshold: float


def aggregate_verdict(verdicts: list[Verdict]) -> AuditVerdict:
    """Aggregate per-probe verdicts into a final one.

    Policy (v0.1.0):
      - any FAIL → FAIL
      - any WARN (and no FAIL) → WARN
      - all PASS → PASS
      - empty input → NO_VERDICT
    """
    if not verdicts:
        return AuditVerdict.NO_VERDICT
    labels = {v.label for v in verdicts}
    if AuditVerdict.FAIL in labels:
        return AuditVerdict.FAIL
    if AuditVerdict.WARN in labels:
        return AuditVerdict.WARN
    if labels == {AuditVerdict.PASS}:
        return AuditVerdict.PASS
    return AuditVerdict.NO_VERDICT
