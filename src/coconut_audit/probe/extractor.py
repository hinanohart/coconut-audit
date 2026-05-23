"""Latent activation extractor (forward-hook-based)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LatentExtractor:
    """Captures hidden states at registered hook sites during a forward pass.

    Phase 2 implements the `torch.nn.Module.register_forward_hook` plumbing.
    """

    layer: int
    hook_site: str

    def extract(self) -> None:  # pragma: no cover - filled in Phase 2
        raise NotImplementedError("LatentExtractor.extract lands in v0.1.0 Phase 2")
