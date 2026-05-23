"""Unit tests for coconut_audit.reports (XSS escape verified)."""

from __future__ import annotations

import json
from pathlib import Path

from coconut_audit.core.types import AuditReport, AuditVerdict, ProbeKind
from coconut_audit.reports import (
    LedgerWriter,
    dump_json_report,
    render_html_report,
)


def _report(**overrides: object) -> AuditReport:
    base = {
        "audit_id": "abc-123",
        "model_id": "gpt2",
        "sae_id": "jbloom/GPT2-Small-SAEs",
        "probe_kind": ProbeKind.STEERING,
        "verdict": AuditVerdict.PASS,
        "metrics": {"steering_ratio": 0.42},
        "n_samples": 4,
        "notes": ["demo_mode=True"],
    }
    base.update(overrides)
    return AuditReport(**base)  # type: ignore[arg-type]


def test_dump_json_report_round_trip(tmp_path: Path) -> None:
    r = _report()
    p = dump_json_report(r, tmp_path / "out" / "report.json")
    assert p.exists()
    obj = json.loads(p.read_text("utf-8"))
    assert obj["audit_id"] == "abc-123"
    assert obj["verdict"] == "PASS"


def test_ledger_writer_appends_jsonl(tmp_path: Path) -> None:
    led = tmp_path / "ledger.jsonl"
    w = LedgerWriter(path=led)
    w.append(_report(audit_id="id-1"))
    w.append(_report(audit_id="id-2"))
    lines = led.read_text("utf-8").strip().splitlines()
    assert [json.loads(line)["audit_id"] for line in lines] == ["id-1", "id-2"]


def test_html_report_escapes_xss_payload(tmp_path: Path) -> None:
    payload = "<script>alert('xss')</script>"
    r = _report(
        audit_id=payload,
        model_id=payload,
        sae_id=payload,
        notes=[payload, "another " + payload],
    )
    out = render_html_report(r, tmp_path / "report.html")
    html = out.read_text("utf-8")
    # raw payload must not appear
    assert "<script>" not in html
    assert "</script>" not in html
    # escaped variant must appear (each user-controlled field should escape it)
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html


def test_html_report_includes_verdict_class(tmp_path: Path) -> None:
    out = render_html_report(_report(verdict=AuditVerdict.FAIL), tmp_path / "r.html")
    html = out.read_text("utf-8")
    assert 'class="verdict FAIL"' in html


def test_html_report_metric_attribute_quote_escaping(tmp_path: Path) -> None:
    # quote=True in html.escape must turn " and ' into &quot; / &#x27;
    r = _report(metrics={'attr="injected"': 1.0})
    out = render_html_report(r, tmp_path / "r.html")
    html = out.read_text("utf-8")
    assert 'attr="injected"' not in html
    assert "attr=&quot;injected&quot;" in html


def test_html_report_escapes_benchmark_field(tmp_path: Path) -> None:
    """`benchmark` is optional but user-controlled — must be escaped too."""
    payload = "<script>alert('benchmark-xss')</script>"
    r = _report(benchmark=payload)
    out = render_html_report(r, tmp_path / "r.html")
    html = out.read_text("utf-8")
    assert "<script>" not in html
    assert "&lt;script&gt;alert(&#x27;benchmark-xss&#x27;)&lt;/script&gt;" in html
