"""Append-only JSONL ledger for audit reports.

Two MCP clients (e.g. Claude Code + Cursor) can hit the default ledger
concurrently, and a single ``AuditReport`` JSON line can easily exceed
``PIPE_BUF`` (4 KiB) once ``traces`` is populated. We take an exclusive
``fcntl`` lock on POSIX so the line-write is atomic with respect to other
ledger writers, and ``fsync`` so a process kill mid-write does not leave a
torn line. The lock degrades to a no-op on non-POSIX platforms; that
matches our Linux-first deployment story and keeps Windows usable for dev.
"""

from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from coconut_audit.core.types import AuditReport

try:
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - Windows / non-POSIX
    fcntl = None  # type: ignore[assignment]
    _HAVE_FCNTL = False


@dataclass(slots=True)
class LedgerWriter:
    """Append `AuditReport`s to a JSONL file (one line per report)."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, report: AuditReport) -> None:
        line = json.dumps(report.model_dump(mode="json"), ensure_ascii=False) + "\n"
        with self.path.open("a", encoding="utf-8") as fh:
            if _HAVE_FCNTL:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.write(line)
                fh.flush()
                with suppress(OSError):
                    os.fsync(fh.fileno())
            finally:
                if _HAVE_FCNTL:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
