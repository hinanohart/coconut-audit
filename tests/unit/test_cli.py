"""CLI smoke tests via Click's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from coconut_audit.cli.main import cli


def test_cli_run_demo_mode_emits_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COCONUT_AUDIT_LEDGER_ROOT", str(tmp_path))
    runner = CliRunner()
    ledger = tmp_path / "ledger.jsonl"
    out_dir = tmp_path / "reports"
    result = runner.invoke(
        cli,
        [
            "run",
            "--model",
            "gpt2",
            "--sae",
            "jbloom/GPT2-Small-SAEs",
            "--probe",
            "steering",
            "--n-samples",
            "4",
            "--out-dir",
            str(out_dir),
            "--ledger",
            str(ledger),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "audit_id" in result.output
    assert "verdict" in result.output
    assert ledger.exists()
    assert any(out_dir.glob("*.json"))
    assert any(out_dir.glob("*.html"))


def test_cli_real_mode_raises_helpful_error() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "--model", "gpt2", "--sae", "x", "--real"],
    )
    assert result.exit_code != 0
    assert "v0.1.1" in result.output


def test_cli_get_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COCONUT_AUDIT_LEDGER_ROOT", str(tmp_path))
    runner = CliRunner()
    ledger = tmp_path / "ledger.jsonl"
    out_dir = tmp_path / "reports"
    run = runner.invoke(
        cli,
        [
            "run",
            "--model",
            "gpt2",
            "--sae",
            "x",
            "--probe",
            "drift",
            "--out-dir",
            str(out_dir),
            "--ledger",
            str(ledger),
        ],
    )
    assert run.exit_code == 0
    audit_id = run.output.split("audit_id")[1].split("\n")[0].split(":")[1].strip()

    got = runner.invoke(cli, ["get", audit_id, "--ledger", str(ledger)])
    assert got.exit_code == 0
    payload = json.loads(got.output)
    assert payload["audit_id"] == audit_id


def test_cli_diff_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COCONUT_AUDIT_LEDGER_ROOT", str(tmp_path))
    runner = CliRunner()
    ledger = tmp_path / "ledger.jsonl"
    out_dir = tmp_path / "reports"
    ids: list[str] = []
    for seed in (1, 2):
        r = runner.invoke(
            cli,
            [
                "run",
                "--model",
                "gpt2",
                "--sae",
                "x",
                "--probe",
                "shortcut",
                "--seed",
                str(seed),
                "--out-dir",
                str(out_dir),
                "--ledger",
                str(ledger),
            ],
        )
        assert r.exit_code == 0
        ids.append(r.output.split("audit_id")[1].split("\n")[0].split(":")[1].strip())

    d = runner.invoke(cli, ["diff", ids[0], ids[1], "--ledger", str(ledger)])
    assert d.exit_code == 0
    payload = json.loads(d.output)
    assert payload["a_audit_id"] == ids[0]
    assert payload["b_audit_id"] == ids[1]
