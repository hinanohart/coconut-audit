"""Core data types: LatentTrace, AuditReport, AuditVerdict, ProbeKind."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProbeKind(StrEnum):
    """Which audit probe was applied."""

    STEERING = "steering"
    SHORTCUT = "shortcut"
    DRIFT = "drift"


class AuditVerdict(StrEnum):
    """Final verdict aggregated from one or more probes."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    NO_VERDICT = "NO_VERDICT"


class LatentTrace(BaseModel):
    """A single (token, layer, activation) latent observation, post-SAE projection.

    `feature_indices` and `feature_activations` are the sparse non-zero
    coordinates of the SAE feature vector at this site.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    layer: int
    hook_site: str
    token_index: int
    feature_indices: list[int] = Field(default_factory=list)
    feature_activations: list[float] = Field(default_factory=list)
    raw_norm: float | None = None


class AuditReport(BaseModel):
    """Top-level audit result. Emitted as JSON + HTML, appended to JSONL ledger."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "0.1.0"
    audit_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_id: str
    sae_id: str
    probe_kind: ProbeKind
    benchmark: str | None = None
    verdict: AuditVerdict
    metrics: dict[str, float] = Field(default_factory=dict)
    n_samples: int = 0
    traces: list[LatentTrace] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
