"""JSON report emitter."""

from __future__ import annotations

import json
from pathlib import Path

from coconut_audit.core.types import AuditReport


def dump_json_report(report: AuditReport, path: str | Path) -> Path:
    """Write `report` to `path` as pretty-printed JSON (UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path
