"""Pretrained SAE loader (HuggingFace Hub only).

Inherits the `recurrentlens 0.1.0.post1` hardening lesson: validate safetensors
metadata against tensor shapes before construction. Treat unknown SAEs as
untrusted code-adjacent artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PretrainedSAELoader:
    """Resolves an SAE Hub repo ID to local artifact paths and metadata."""

    sae_id: str
    revision: str | None = None
    cache_dir: Path | None = None

    def snapshot(self) -> Path:  # pragma: no cover - filled in Phase 3
        raise NotImplementedError("PretrainedSAELoader.snapshot lands in v0.1.0 Phase 3")


def load_pretrained_sae(
    sae_id: str,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> object:
    """Convenience: download + construct a pretrained SAE artifact.

    The returned object exposes `.encode(x) -> sparse_features` and
    `.decode(z) -> reconstruction`. Phase 3 wires the SAELens / sparsify
    / Goodfire backends.
    """
    _ = PretrainedSAELoader(sae_id=sae_id, revision=revision, cache_dir=cache_dir)
    raise NotImplementedError("load_pretrained_sae lands in v0.1.0 Phase 3")
