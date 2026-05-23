"""Concrete linear-encoder + linear-decoder SAE (ReLU / JumpReLU activation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _relu(x: Any) -> Any:
    import torch

    return torch.relu(x)


def _jump_relu(x: Any, threshold: Any) -> Any:
    import torch

    return torch.where(x > threshold, x, torch.zeros_like(x))


@dataclass(slots=True)
class LinearSAE:
    """W_enc/W_dec linear SAE, used by Goodfire/SAELens/sparsify formats.

    Shape contract:
      W_enc: [d_in, d_sae]
      b_enc: [d_sae]
      W_dec: [d_sae, d_in]
      b_dec: [d_in]
      threshold: [d_sae] | scalar | None  (None => plain ReLU)
    """

    W_enc: Any
    b_enc: Any
    W_dec: Any
    b_dec: Any
    threshold: Any = None
    d_in: int = 0
    d_sae: int = 0
    activation: str = "relu"

    def __post_init__(self) -> None:
        if self.d_in == 0:
            self.d_in = int(self.W_enc.shape[0])
        if self.d_sae == 0:
            self.d_sae = int(self.W_enc.shape[1])
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        # recurrentlens 0.1.0.post1 lesson: validate before trusting Hub artifacts.
        if tuple(self.W_enc.shape) != (self.d_in, self.d_sae):
            raise ValueError(
                f"W_enc shape {tuple(self.W_enc.shape)} != ({self.d_in}, {self.d_sae})"
            )
        if tuple(self.W_dec.shape) != (self.d_sae, self.d_in):
            raise ValueError(
                f"W_dec shape {tuple(self.W_dec.shape)} != ({self.d_sae}, {self.d_in})"
            )
        if tuple(self.b_enc.shape) != (self.d_sae,):
            raise ValueError(f"b_enc shape {tuple(self.b_enc.shape)} != ({self.d_sae},)")
        if tuple(self.b_dec.shape) != (self.d_in,):
            raise ValueError(f"b_dec shape {tuple(self.b_dec.shape)} != ({self.d_in},)")
        if self.threshold is not None and tuple(self.threshold.shape) not in (
            (),
            (self.d_sae,),
        ):
            raise ValueError(
                f"threshold shape {tuple(self.threshold.shape)} not in [(), ({self.d_sae},)]"
            )

    def encode(self, x: Any) -> Any:
        pre = x @ self.W_enc + self.b_enc
        if self.activation == "jumprelu" and self.threshold is not None:
            return _jump_relu(pre, self.threshold)
        return _relu(pre)

    def decode(self, z: Any) -> Any:
        return z @ self.W_dec + self.b_dec
