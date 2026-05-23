"""HTML report renderer (XSS-safe via stdlib html.escape).

Every user-controlled field (model_id, sae_id, audit_id, benchmark, notes,
metric keys) is routed through `html.escape(..., quote=True)` before
interpolation. We deliberately avoid `jinja2` here so the escape policy is
visible and grep-auditable on every field — this inherits the
`recurrentlens 0.1.0.post1` XSS-hotfix lesson, but tightens it: a missing
escape on a single field becomes a code-review-visible diff rather than a
template-engine config bug.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from coconut_audit.core.types import AuditReport


def _e(value: object) -> str:
    """Escape any value for safe HTML interpolation (text or attribute)."""
    return escape(str(value), quote=True)


_CSS = (
    "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
    "max-width:920px;margin:2em auto;padding:0 1em}"
    ".verdict{font-size:1.5em;padding:0.5em 1em;border-radius:6px;display:inline-block}"
    ".PASS{background:#e6f4ea;color:#137333}"
    ".WARN{background:#fef7e0;color:#b06000}"
    ".FAIL{background:#fce8e6;color:#c5221f}"
    ".NO_VERDICT{background:#f1f3f4;color:#5f6368}"
    ".demo-banner{background:#fff3cd;color:#664d03;padding:0.8em 1em;"
    "border:2px solid #b08800;border-radius:6px;margin-bottom:1em;"
    "font-weight:600}"
    "table{border-collapse:collapse;margin-top:1em}"
    "th,td{border:1px solid #dadce0;padding:0.4em 0.8em;text-align:left}"
)


def _is_demo_report(report: AuditReport) -> bool:
    """A report is a demo-mode synthetic run if any note advertises it."""
    return any("demo_mode=True" in n for n in report.notes)


def render_html_report(report: AuditReport, path: str | Path) -> Path:
    """Render `report` to a self-contained HTML file at `path`.

    Every interpolated field is HTML-escaped via `html.escape(..., quote=True)`.
    """
    verdict_class = _e(report.verdict.value)

    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>coconut-audit report — {_e(report.audit_id)}</title>",
        f"<style>{_CSS}</style></head><body>",
    ]
    if _is_demo_report(report):
        parts.append(
            '<div class="demo-banner">'
            "⚠ DEMO MODE — verdict is computed from SYNTHETIC activations "
            "(<code>torch.randn</code> seed). "
            "v0.1.x does not exercise real-model inference; treat as advisory only."
            "</div>"
        )
    parts.extend(
        [
            "<h1>coconut-audit report</h1>",
            f"<p><strong>Audit ID:</strong> {_e(report.audit_id)}</p>",
            f"<p><strong>Model:</strong> {_e(report.model_id)}</p>",
            f"<p><strong>SAE:</strong> {_e(report.sae_id)}</p>",
            f"<p><strong>Probe:</strong> {_e(report.probe_kind.value)}</p>",
        ]
    )
    if report.benchmark:
        parts.append(f"<p><strong>Benchmark:</strong> {_e(report.benchmark)}</p>")
    parts.extend(
        [
            f"<p><strong>Created:</strong> {_e(report.created_at.isoformat())}</p>",
            f"<p><strong>n_samples:</strong> {_e(report.n_samples)}</p>",
            f'<p class="verdict {verdict_class}">Verdict: {verdict_class}</p>',
        ]
    )

    if report.metrics:
        parts.append("<h2>Metrics</h2><table><tr><th>Name</th><th>Value</th></tr>")
        for k, v in report.metrics.items():
            parts.append(f"<tr><td>{_e(k)}</td><td>{_e(f'{v:.6f}')}</td></tr>")
        parts.append("</table>")

    if report.notes:
        parts.append("<h2>Notes</h2><ul>")
        for n in report.notes:
            parts.append(f"<li>{_e(n)}</li>")
        parts.append("</ul>")

    parts.append("</body></html>")
    html_text = "".join(parts)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    return path
