"""Append-only JSONL ledger for audit reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from coconut_audit.core.types import AuditReport


@dataclass(slots=True)
class LedgerWriter:
    """Append `AuditReport`s to a JSONL file (one line per report)."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, report: AuditReport) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(report.model_dump(mode="json"), ensure_ascii=False))
            fh.write("\n")
