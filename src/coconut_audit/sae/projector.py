"""Projects raw hidden activations through an SAE into sparse features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from coconut_audit.sae.base import BaseSAE


@dataclass(slots=True)
class FeatureProjector:
    """Batch-friendly wrapper around `BaseSAE.encode`.

    Flattens the leading (batch × time) dims before projection so callers can
    pass shape [batch, seq, d_in] tensors directly.
    """

    sae: BaseSAE

    def project(self, hidden: Any) -> Any:
        if hidden.ndim < 2:
            raise ValueError(f"hidden must be at least 2D, got shape {tuple(hidden.shape)}")
        if hidden.shape[-1] != self.sae.d_in:
            raise ValueError(f"hidden last dim {hidden.shape[-1]} != sae.d_in {self.sae.d_in}")
        flat = hidden.reshape(-1, self.sae.d_in)
        z = self.sae.encode(flat)
        return z.reshape(*hidden.shape[:-1], self.sae.d_sae)
