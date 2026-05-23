"""Shortcut detector: in-distribution vs OOD-biased accuracy gap.

Methodology family from arxiv:2512.21711 (`Shortcut experiments`): if a model's
accuracy on a biased / OOD-perturbed variant of a benchmark collapses while its
in-distribution accuracy stays high, the apparent reasoning is shortcut-driven.

A large positive `gap` (id_acc - ood_acc) is the failure signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ShortcutDetector:
    """Score the accuracy gap between in-distribution and OOD-biased prompts."""

    bias_kind: str = "answer_letter_shuffle"

    def run(self, id_correct: Any, ood_correct: Any) -> dict[str, float]:
        """Score the ID-vs-OOD accuracy gap.

        Args:
          id_correct: shape [N] — bool/int per-example correctness on the ID set.
          ood_correct: shape [M] — bool/int per-example correctness on the OOD set.

        Returns dict with `id_acc`, `ood_acc`, `gap`.
        """
        if id_correct.ndim != 1 or ood_correct.ndim != 1:
            raise ValueError(
                f"expected 1D arrays, got id={tuple(id_correct.shape)} ood={tuple(ood_correct.shape)}"
            )
        id_acc = float(id_correct.float().mean())
        ood_acc = float(ood_correct.float().mean())
        return {"id_acc": id_acc, "ood_acc": ood_acc, "gap": id_acc - ood_acc}
