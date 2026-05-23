"""coconut-audit CLI (click)."""

from __future__ import annotations

import json
from pathlib import Path

import click

from coconut_audit.core import AuditReport
from coconut_audit.mcp.tools import audit_diff, audit_get, audit_run
from coconut_audit.reports import dump_json_report, render_html_report


@click.group()
@click.version_option(package_name="coconut-audit")
def cli() -> None:
    """coconut-audit: audit harness for continuous-latent chain-of-thought."""


@cli.command()
@click.option("--model", "model_id", required=True, help="HuggingFace model ID.")
@click.option("--sae", "sae_id", required=True, help="HuggingFace SAE Hub ID.")
@click.option(
    "--probe",
    type=click.Choice(["steering", "shortcut", "drift"], case_sensitive=False),
    default="steering",
    show_default=True,
)
@click.option("--benchmark", default=None, help="Optional benchmark name (e.g. mmlu, hotpotqa).")
@click.option("--n-samples", default=32, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option(
    "--out-dir",
    default="audit_reports",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--ledger",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Override JSONL ledger path (default: $COCONUT_AUDIT_LEDGER or audit_reports/ledger.jsonl).",
)
@click.option(
    "--real",
    is_flag=True,
    help="Run against a real model (lands in v0.1.1+; currently raises NotImplementedError).",
)
def run(
    model_id: str,
    sae_id: str,
    probe: str,
    benchmark: str | None,
    n_samples: int,
    seed: int,
    out_dir: Path,
    ledger: Path | None,
    real: bool,
) -> None:
    """Run a fresh audit and emit JSON + HTML reports."""
    try:
        report_dict = audit_run(
            model_id=model_id,
            sae_id=sae_id,
            probe=probe,
            benchmark=benchmark,
            n_samples=n_samples,
            seed=seed,
            demo_mode=not real,
            ledger_path=str(ledger) if ledger is not None else None,
        )
    except NotImplementedError as e:
        raise click.ClickException(str(e)) from e

    report = AuditReport.model_validate(report_dict)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = dump_json_report(report, out_dir / f"{report.audit_id}.json")
    html_path = render_html_report(report, out_dir / f"{report.audit_id}.html")

    click.echo(f"audit_id : {report.audit_id}")
    click.echo(f"verdict  : {report.verdict.value}")
    click.echo(f"JSON     : {json_path}")
    click.echo(f"HTML     : {html_path}")


@cli.command()
@click.argument("audit_id")
@click.option(
    "--ledger",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
)
def get(audit_id: str, ledger: Path | None) -> None:
    """Look up a stored audit report by `audit_id` from the JSONL ledger."""
    try:
        result = audit_get(audit_id, ledger_path=str(ledger) if ledger is not None else None)
    except (FileNotFoundError, KeyError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(json.dumps(result, indent=2, ensure_ascii=False))


@cli.command()
@click.argument("audit_id_a")
@click.argument("audit_id_b")
@click.option(
    "--ledger",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
)
def diff(audit_id_a: str, audit_id_b: str, ledger: Path | None) -> None:
    """Diff two stored audit reports."""
    try:
        result = audit_diff(
            audit_id_a,
            audit_id_b,
            ledger_path=str(ledger) if ledger is not None else None,
        )
    except (FileNotFoundError, KeyError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    cli()
