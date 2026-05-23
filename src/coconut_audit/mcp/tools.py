"""MCP tool surface: audit_run / audit_get / audit_diff.

These are plain Python functions; the stdio MCP server in `server.py` exposes
them under the standard MCP tool calling convention.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from coconut_audit.core import AuditConfig
from coconut_audit.core.types import ProbeKind
from coconut_audit.mcp.pipeline import (
    diff_reports,
    find_report_in_ledger,
    run_audit_pipeline,
)

_DEFAULT_LEDGER = Path(os.environ.get("COCONUT_AUDIT_LEDGER", "audit_reports/ledger.jsonl"))


def audit_run(
    model_id: str,
    sae_id: str,
    probe: str = "steering",
    benchmark: str | None = None,
    n_samples: int = 32,
    seed: int = 0,
    demo_mode: bool = True,
    ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a fresh audit. Returns the `AuditReport` as a dict."""
    config = AuditConfig(
        model_id=model_id,
        sae_id=sae_id,
        probe_kind=ProbeKind(probe.lower()),
        benchmark=benchmark,
        n_samples=n_samples,
        seed=seed,
    )
    ledger = Path(ledger_path) if ledger_path is not None else _DEFAULT_LEDGER
    report = run_audit_pipeline(config, demo_mode=demo_mode, ledger_path=ledger)
    return report.model_dump(mode="json")


def audit_get(audit_id: str, ledger_path: str | Path | None = None) -> dict[str, Any]:
    """Look up a stored `AuditReport` by `audit_id` in the JSONL ledger."""
    ledger = Path(ledger_path) if ledger_path is not None else _DEFAULT_LEDGER
    report = find_report_in_ledger(ledger, audit_id)
    return report.model_dump(mode="json")


def audit_diff(
    audit_id_a: str,
    audit_id_b: str,
    ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Diff two stored audit reports by their IDs."""
    ledger = Path(ledger_path) if ledger_path is not None else _DEFAULT_LEDGER
    a = find_report_in_ledger(ledger, audit_id_a)
    b = find_report_in_ledger(ledger, audit_id_b)
    return diff_reports(a, b)
