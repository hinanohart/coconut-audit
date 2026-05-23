"""Projects raw hidden activations through an SAE into sparse features."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FeatureProjector:
    """Wraps an SAE's `encode` for batched residual-stream projections."""

    sae: object

    def project(self, hidden: object) -> object:  # pragma: no cover - Phase 3
        raise NotImplementedError("FeatureProjector.project lands in v0.1.0 Phase 3")
