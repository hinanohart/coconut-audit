"""Unit tests for coconut_audit.core."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from coconut_audit.core import (
    AuditConfig,
    AuditReport,
    AuditVerdict,
    LatentTrace,
    ProbeKind,
    load_config,
)


def test_latent_trace_is_frozen() -> None:
    t = LatentTrace(model_id="gpt2", layer=0, hook_site="x", token_index=0)
    with pytest.raises(ValidationError):
        t.layer = 1  # type: ignore[misc]


def test_audit_report_serializable_round_trip() -> None:
    r = AuditReport(
        audit_id="abc",
        model_id="gpt2",
        sae_id="jbloom/GPT2-Small-SAEs",
        probe_kind=ProbeKind.STEERING,
        verdict=AuditVerdict.PASS,
        metrics={"steering_sensitivity": 0.81},
        n_samples=4,
    )
    payload = r.model_dump(mode="json")
    rebuilt = AuditReport.model_validate(payload)
    assert rebuilt.verdict == AuditVerdict.PASS
    assert rebuilt.probe_kind == ProbeKind.STEERING
    assert rebuilt.metrics["steering_sensitivity"] == pytest.approx(0.81)


def test_audit_config_from_dict() -> None:
    cfg = load_config(
        {
            "model_id": "gpt2",
            "sae_id": "jbloom/GPT2-Small-SAEs",
            "probe_kind": "shortcut",
            "layers": [3, 5, 7],
            "n_samples": 16,
        }
    )
    assert isinstance(cfg, AuditConfig)
    assert cfg.probe_kind == ProbeKind.SHORTCUT
    assert cfg.layers == [3, 5, 7]
    assert cfg.n_samples == 16


def test_audit_config_from_yaml(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "model_id: gpt2\nsae_id: jbloom/GPT2-Small-SAEs\nprobe_kind: drift\nn_samples: 8\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.probe_kind == ProbeKind.DRIFT
    assert cfg.n_samples == 8


def test_audit_config_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        load_config({"model_id": "gpt2", "sae_id": "x", "unknown_field": 1})
