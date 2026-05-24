"""Unit tests for coconut_audit.mcp pipeline + tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coconut_audit.core import AuditConfig
from coconut_audit.core.types import AuditVerdict, ProbeKind
from coconut_audit.mcp import (
    audit_diff,
    audit_get,
    audit_run,
    diff_reports,
    find_report_in_ledger,
    run_audit_pipeline,
)
from coconut_audit.mcp.tools import _validate_ledger_path


def _cfg(probe: ProbeKind, *, seed: int = 0) -> AuditConfig:
    return AuditConfig(
        model_id="gpt2",
        sae_id="jbloom/GPT2-Small-SAEs",
        probe_kind=probe,
        n_samples=4,
        seed=seed,
    )


@pytest.mark.parametrize("probe", list(ProbeKind))
def test_run_audit_pipeline_demo_mode(probe: ProbeKind, tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    report = run_audit_pipeline(_cfg(probe), demo_mode=True, ledger_path=ledger)
    assert report.probe_kind == probe
    assert report.verdict in {v for v in AuditVerdict}
    assert report.metrics  # non-empty
    assert ledger.exists()
    lines = ledger.read_text().strip().splitlines()
    assert len(lines) == 1


def test_run_audit_pipeline_real_mode_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="v0.1.1"):
        run_audit_pipeline(_cfg(ProbeKind.STEERING), demo_mode=False)


def test_find_report_in_ledger_round_trip(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    r = run_audit_pipeline(_cfg(ProbeKind.SHORTCUT), demo_mode=True, ledger_path=ledger)
    fetched = find_report_in_ledger(ledger, r.audit_id)
    assert fetched.audit_id == r.audit_id
    assert fetched.verdict == r.verdict


def test_find_report_in_ledger_missing_id(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    run_audit_pipeline(_cfg(ProbeKind.DRIFT), demo_mode=True, ledger_path=ledger)
    with pytest.raises(KeyError, match="not found"):
        find_report_in_ledger(ledger, "no-such-id")


def test_find_report_in_ledger_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_report_in_ledger(tmp_path / "nope.jsonl", "x")


def test_diff_reports_metric_deltas(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    a = run_audit_pipeline(_cfg(ProbeKind.STEERING, seed=1), demo_mode=True, ledger_path=ledger)
    b = run_audit_pipeline(_cfg(ProbeKind.STEERING, seed=2), demo_mode=True, ledger_path=ledger)
    d = diff_reports(a, b)
    assert d["a_audit_id"] == a.audit_id
    assert d["b_audit_id"] == b.audit_id
    assert isinstance(d["metric_deltas"], dict)
    assert set(d["metric_deltas"]).issuperset({"latent_sensitivity", "cot_sensitivity", "ratio"})


def test_audit_run_tool_returns_dict_and_appends_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COCONUT_AUDIT_LEDGER_ROOT", str(tmp_path))
    ledger = tmp_path / "ledger.jsonl"
    out = audit_run(
        model_id="gpt2",
        sae_id="jbloom/GPT2-Small-SAEs",
        probe="steering",
        n_samples=4,
        seed=0,
        demo_mode=True,
        ledger_path=str(ledger),
    )
    assert isinstance(out, dict)
    assert out["model_id"] == "gpt2"
    assert ledger.exists()


def test_audit_get_and_diff_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COCONUT_AUDIT_LEDGER_ROOT", str(tmp_path))
    ledger = tmp_path / "ledger.jsonl"
    a = audit_run(
        model_id="gpt2",
        sae_id="x",
        probe="shortcut",
        seed=1,
        ledger_path=str(ledger),
    )
    b = audit_run(
        model_id="gpt2",
        sae_id="x",
        probe="shortcut",
        seed=2,
        ledger_path=str(ledger),
    )
    fetched = audit_get(a["audit_id"], ledger_path=str(ledger))
    assert fetched["audit_id"] == a["audit_id"]

    diffed = audit_diff(a["audit_id"], b["audit_id"], ledger_path=str(ledger))
    assert diffed["a_audit_id"] == a["audit_id"]
    assert diffed["b_audit_id"] == b["audit_id"]
    # both reports must JSON-serialize cleanly
    json.dumps(fetched)
    json.dumps(diffed)


def test_validate_ledger_path_accepts_in_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = _validate_ledger_path("audit_reports/ledger.jsonl")
    assert resolved == (tmp_path / "audit_reports" / "ledger.jsonl").resolve()
    assert resolved.is_relative_to(tmp_path.resolve())


def test_validate_ledger_path_default_is_in_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = _validate_ledger_path(None)
    assert resolved.is_relative_to(tmp_path.resolve())


def test_validate_ledger_path_rejects_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="out-of-bounds"):
        _validate_ledger_path("../../etc/passwd")


def test_validate_ledger_path_rejects_absolute_outside_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="out-of-bounds"):
        _validate_ledger_path("/etc/passwd")


def test_validate_ledger_path_honors_ledger_root_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ledgers"
    root.mkdir()
    monkeypatch.setenv("COCONUT_AUDIT_LEDGER_ROOT", str(root))
    resolved = _validate_ledger_path("sub/ledger.jsonl")
    assert resolved == (root / "sub" / "ledger.jsonl").resolve()
    with pytest.raises(ValueError, match="out-of-bounds"):
        _validate_ledger_path("../escape.jsonl")


def test_audit_run_tool_rejects_traversal_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="out-of-bounds"):
        audit_run(
            model_id="gpt2",
            sae_id="x",
            probe="steering",
            ledger_path="../../tmp/evil.jsonl",
        )


def test_audit_get_tool_rejects_absolute_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="out-of-bounds"):
        audit_get("any-id", ledger_path="/etc/passwd")


def test_audit_run_tool_in_base_ledger_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    out = audit_run(
        model_id="gpt2",
        sae_id="x",
        probe="steering",
        ledger_path="nested/ledger.jsonl",
    )
    ledger = tmp_path / "nested" / "ledger.jsonl"
    assert ledger.exists()
    fetched = audit_get(out["audit_id"], ledger_path="nested/ledger.jsonl")
    assert fetched["audit_id"] == out["audit_id"]
