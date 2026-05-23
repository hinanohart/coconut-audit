"""Steering probe: perturb token subsets and measure reasoning-criticality.

Based on the methodology family in arxiv:2512.21711 (Steering experiments).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SteeringProbe:
    """Compares per-token-subset sensitivity (latent vs explicit CoT tokens)."""

    target_token_kinds: tuple[str, ...] = ("latent", "explicit_cot")

    def run(self) -> dict[str, float]:  # pragma: no cover - Phase 4
        raise NotImplementedError("SteeringProbe.run lands in v0.1.0 Phase 4")
