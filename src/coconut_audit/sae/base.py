"""Base SAE protocol shared by all loader adapters."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseSAE(Protocol):
    """Minimal interface every loaded SAE must satisfy.

    Implementations may be torch.nn.Module subclasses or plain dataclasses;
    the audit pipeline only needs encode + decode and the two shape attrs.
    """

    d_in: int
    d_sae: int

    def encode(self, x: Any) -> Any:
        """Sparse feature activations: shape [..., d_sae]."""
        ...

    def decode(self, z: Any) -> Any:
        """Reconstruction: shape [..., d_in]."""
        ...
