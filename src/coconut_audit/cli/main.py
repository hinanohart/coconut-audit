"""coconut-audit CLI (click)."""

from __future__ import annotations

import click


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
)
@click.option("--benchmark", default=None, help="Optional benchmark name (e.g. mmlu, hotpotqa).")
@click.option("--out", "out_path", default="audit_reports/report.json", show_default=True)
def run(model_id: str, sae_id: str, probe: str, benchmark: str | None, out_path: str) -> None:
    """Run a fresh audit and emit JSON + HTML reports."""
    raise click.ClickException("`coconut-audit run` lands in v0.1.0 Phase 4-5.")


@cli.command()
@click.argument("audit_id")
def get(audit_id: str) -> None:
    """Look up a stored audit report by `audit_id` from the JSONL ledger."""
    raise click.ClickException("`coconut-audit get` lands in v0.1.0 Phase 4-5.")


@cli.command()
@click.argument("audit_id_a")
@click.argument("audit_id_b")
def diff(audit_id_a: str, audit_id_b: str) -> None:
    """Diff two audit reports."""
    raise click.ClickException("`coconut-audit diff` lands in v0.1.0 Phase 4-5.")


if __name__ == "__main__":  # pragma: no cover
    cli()
