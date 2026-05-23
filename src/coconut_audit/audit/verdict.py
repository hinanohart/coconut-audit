"""Verdict aggregator: combines per-probe metrics into a final AuditVerdict."""

from __future__ import annotations

import math
from dataclasses import dataclass

from coconut_audit.core.types import AuditVerdict

__all__ = ["AuditVerdict", "Verdict", "aggregate_verdict", "label_for_metric"]


@dataclass(slots=True)
class Verdict:
    """Per-probe verdict slice: a label plus the metric that produced it."""

    label: AuditVerdict
    metric_name: str
    metric_value: float
    threshold: float


def label_for_metric(
    value: float, warn_at: float, fail_at: float, *, higher_is_worse: bool = True
) -> AuditVerdict:
    """Threshold a single metric into PASS / WARN / FAIL.

    With `higher_is_worse=True` (default), FAIL fires when value >= fail_at,
    WARN at value >= warn_at; otherwise the direction is flipped. Non-finite
    inputs (NaN / inf) return NO_VERDICT — a NaN metric must never silently
    classify as PASS in an audit tool.
    """
    if not math.isfinite(value):
        return AuditVerdict.NO_VERDICT
    if higher_is_worse:
        if warn_at > fail_at:
            raise ValueError(
                f"label_for_metric(higher_is_worse=True) requires warn_at <= fail_at; "
                f"got warn_at={warn_at}, fail_at={fail_at}"
            )
        if value >= fail_at:
            return AuditVerdict.FAIL
        if value >= warn_at:
            return AuditVerdict.WARN
        return AuditVerdict.PASS
    if warn_at < fail_at:
        raise ValueError(
            f"label_for_metric(higher_is_worse=False) requires warn_at >= fail_at; "
            f"got warn_at={warn_at}, fail_at={fail_at}"
        )
    if value <= fail_at:
        return AuditVerdict.FAIL
    if value <= warn_at:
        return AuditVerdict.WARN
    return AuditVerdict.PASS


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
