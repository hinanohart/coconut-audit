"""Shortcut detector: evaluate under biased / OOD prompt distributions.

Based on the methodology family in arxiv:2512.21711 (Shortcut experiments).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ShortcutDetector:
    """Compares in-distribution vs OOD-biased accuracy gaps."""

    bias_kind: str = "answer_letter_shuffle"

    def run(self) -> dict[str, float]:  # pragma: no cover - Phase 4
        raise NotImplementedError("ShortcutDetector.run lands in v0.1.0 Phase 4")
