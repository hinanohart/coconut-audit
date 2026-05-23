"""Audit configuration loader (YAML or dict)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from coconut_audit.core.types import ProbeKind


class AuditConfig(BaseModel):
    """User-facing audit configuration."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    sae_id: str
    probe_kind: ProbeKind = ProbeKind.STEERING
    benchmark: str | None = None
    layers: list[int] = Field(default_factory=list)
    hook_site: str = "resid_post"
    n_samples: int = 32
    seed: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


def load_config(source: str | Path | dict[str, Any]) -> AuditConfig:
    """Load `AuditConfig` from a YAML path or a dict."""
    if isinstance(source, dict):
        return AuditConfig.model_validate(source)
    path = Path(source)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return AuditConfig.model_validate(data)
