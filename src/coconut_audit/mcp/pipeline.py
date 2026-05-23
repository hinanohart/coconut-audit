"""High-level audit pipeline.

v0.1.0 ships a `demo_mode=True` path that drives the audit primitives with
synthetic activations so the end-to-end flow is testable without a model
download. The real-model path (HF inference → forward hook → SAE encode →
benchmark loop) lands in v0.1.1+.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import torch

from coconut_audit.audit import (
    DriftScorer,
    ShortcutDetector,
    SteeringProbe,
    Verdict,
    aggregate_verdict,
    label_for_metric,
)
from coconut_audit.core import AuditConfig, AuditReport
from coconut_audit.core.types import ProbeKind
from coconut_audit.reports.jsonl import LedgerWriter


def _synthetic_steering(seed: int) -> tuple[dict[str, float], Verdict]:
    g = torch.Generator().manual_seed(seed)
    clean = torch.randn(2, 6, 8, generator=g)
    perturbed = clean + 0.1 * torch.randn(2, 6, 8, generator=g)
    mask = torch.tensor([[0, 0, 0, 1, 1, 1]] * 2)
    metrics = SteeringProbe().run(clean, perturbed, mask)
    label = label_for_metric(metrics["ratio"], warn_at=2.0, fail_at=5.0)
    return metrics, Verdict(
        label=label,
        metric_name="steering_ratio",
        metric_value=metrics["ratio"],
        threshold=5.0,
    )


def _synthetic_shortcut(seed: int) -> tuple[dict[str, float], Verdict]:
    g = torch.Generator().manual_seed(seed)
    id_correct = (torch.rand(32, generator=g) > 0.2).int()
    ood_correct = (torch.rand(32, generator=g) > 0.7).int()
    metrics = ShortcutDetector().run(id_correct, ood_correct)
    label = label_for_metric(metrics["gap"], warn_at=0.2, fail_at=0.4)
    return metrics, Verdict(
        label=label,
        metric_name="shortcut_gap",
        metric_value=metrics["gap"],
        threshold=0.4,
    )


def _synthetic_drift(seed: int) -> tuple[dict[str, float], Verdict]:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(64, 16, generator=g)
    b = a + 0.5 * torch.randn(64, 16, generator=g)
    score = DriftScorer().score(a, b)
    metrics = {"wasserstein_1": score}
    label = label_for_metric(score, warn_at=0.3, fail_at=0.7)
    return metrics, Verdict(
        label=label,
        metric_name="wasserstein_1",
        metric_value=score,
        threshold=0.7,
    )


_PROBE_HANDLERS = {
    ProbeKind.STEERING: _synthetic_steering,
    ProbeKind.SHORTCUT: _synthetic_shortcut,
    ProbeKind.DRIFT: _synthetic_drift,
}


def run_audit_pipeline(
    config: AuditConfig,
    *,
    demo_mode: bool = True,
    ledger_path: Path | None = None,
) -> AuditReport:
    """Run an audit. Returns the `AuditReport`; optionally appends to a JSONL ledger.

    v0.1.0 only supports `demo_mode=True` (synthetic activations).
    """
    if not demo_mode:
        raise NotImplementedError(
            "Real-model audit pipeline lands in v0.1.1+. "
            "Pass demo_mode=True to run with synthetic activations."
        )
    handler = _PROBE_HANDLERS.get(config.probe_kind)
    if handler is None:
        raise ValueError(f"unsupported probe_kind: {config.probe_kind}")
    metrics, verdict = handler(config.seed)
    final = aggregate_verdict([verdict])

    report = AuditReport(
        audit_id=str(uuid.uuid4()),
        model_id=config.model_id,
        sae_id=config.sae_id,
        probe_kind=config.probe_kind,
        benchmark=config.benchmark,
        verdict=final,
        metrics=metrics,
        n_samples=config.n_samples,
        notes=[
            f"demo_mode=True (v0.1.0 synthetic pipeline; threshold metric={verdict.metric_name}).",
        ],
    )

    if ledger_path is not None:
        LedgerWriter(path=ledger_path).append(report)
    return report


def diff_reports(a: AuditReport, b: AuditReport) -> dict[str, object]:
    """Compare two `AuditReport`s; return a dict of per-metric deltas."""
    keys = sorted(set(a.metrics) | set(b.metrics))
    deltas: dict[str, float] = {}
    for k in keys:
        deltas[k] = float(b.metrics.get(k, 0.0) - a.metrics.get(k, 0.0))
    verdict_changed = a.verdict != b.verdict
    return {
        "a_audit_id": a.audit_id,
        "b_audit_id": b.audit_id,
        "a_verdict": a.verdict.value,
        "b_verdict": b.verdict.value,
        "verdict_changed": verdict_changed,
        "metric_deltas": deltas,
    }


def find_report_in_ledger(ledger_path: Path, audit_id: str) -> AuditReport:
    """Load the first `AuditReport` matching `audit_id` from a JSONL ledger.

    A corrupt line is surfaced as a typed `RuntimeError` (with the offending
    line number) instead of a raw `json.JSONDecodeError`, which would leak
    file offsets through the MCP error payload.
    """
    if not ledger_path.exists():
        raise FileNotFoundError(f"ledger not found: {ledger_path}")
    with ledger_path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"corrupt JSONL at {ledger_path}:{lineno} ({e.msg}); "
                    "likely a concurrent-write interleave from a pre-0.1.0.post1 "
                    "writer — see LedgerWriter locking notes."
                ) from e
            if obj.get("audit_id") == audit_id:
                return AuditReport.model_validate(obj)
    raise KeyError(f"audit_id {audit_id!r} not found in {ledger_path}")
