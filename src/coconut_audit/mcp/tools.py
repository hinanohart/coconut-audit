"""MCP tool surface: audit_run / audit_get / audit_diff."""

from __future__ import annotations

from typing import Any


def audit_run(
    model_id: str,
    probe: str = "steering",
    benchmark: str | None = None,
    sae_id: str | None = None,
) -> dict[str, Any]:  # pragma: no cover - Phase 4
    """Run a fresh audit. Returns an `AuditReport` dict."""
    raise NotImplementedError("audit_run lands in v0.1.0 Phase 4")


def audit_get(audit_id: str) -> dict[str, Any]:  # pragma: no cover - Phase 4
    """Retrieve a stored `AuditReport` by `audit_id` from the JSONL ledger."""
    raise NotImplementedError("audit_get lands in v0.1.0 Phase 4")


def audit_diff(audit_id_a: str, audit_id_b: str) -> dict[str, Any]:  # pragma: no cover - Phase 4
    """Diff two audit reports (e.g., pre/post-fine-tune)."""
    raise NotImplementedError("audit_diff lands in v0.1.0 Phase 4")
