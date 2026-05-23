"""Latent drift scoring across SAE feature distributions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DriftScorer:
    """Computes drift between two latent-feature distributions."""

    method: str = "wasserstein"

    def score(self, a: object, b: object) -> float:  # pragma: no cover - Phase 4
        raise NotImplementedError("DriftScorer.score lands in v0.1.0 Phase 4")


def latent_drift_score(a: object, b: object) -> float:
    """Convenience entry-point (alias for `DriftScorer().score`)."""
    return DriftScorer().score(a, b)
