"""Steering probe: per-token-kind perturbation sensitivity analysis.

Methodology family from arxiv:2512.21711 (`Steering experiments`): compare how
much a model's hidden state shifts in response to a perturbation applied to
*latent* tokens vs *explicit chain-of-thought* tokens. A ratio close to 1
indicates the latent tokens carry as much reasoning load as explicit ones;
a ratio close to 0 supports the "uninterpretable placeholder" hypothesis.

This primitive operates on pre-computed (clean, perturbed) activation pairs,
so it is fully testable without a model forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SteeringProbe:
    """Compare per-token-kind sensitivity from a (clean, perturbed) activation pair."""

    eps: float = 1e-8

    def run(
        self,
        clean_activations: Any,
        perturbed_activations: Any,
        token_kind_mask: Any,
    ) -> dict[str, float]:
        """Score per-kind sensitivity.

        Args:
          clean_activations: shape [N, T, D] — baseline hidden states.
          perturbed_activations: shape [N, T, D] — hidden states after perturb.
          token_kind_mask: shape [N, T] — int, 0=latent, 1=explicit_cot.

        Returns dict with `latent_sensitivity`, `cot_sensitivity`, `ratio`.
        `ratio` = latent / cot in [0, +inf); near 0 ⇒ latent tokens carry
        far less reasoning load than explicit CoT.
        """
        if clean_activations.shape != perturbed_activations.shape:
            raise ValueError(
                f"shape mismatch: clean {tuple(clean_activations.shape)} "
                f"vs perturbed {tuple(perturbed_activations.shape)}"
            )
        if token_kind_mask.shape != clean_activations.shape[:-1]:
            raise ValueError(
                f"token_kind_mask shape {tuple(token_kind_mask.shape)} "
                f"must match clean leading dims {tuple(clean_activations.shape[:-1])}"
            )

        latent_mask = token_kind_mask == 0
        cot_mask = token_kind_mask == 1
        if int(latent_mask.sum()) == 0:
            raise ValueError(
                "token_kind_mask has no latent tokens (mask==0); cannot compute "
                "latent sensitivity. Pass at least one latent token to audit."
            )
        if int(cot_mask.sum()) == 0:
            raise ValueError(
                "token_kind_mask has no explicit-CoT tokens (mask==1); cannot compute "
                "the comparison baseline."
            )
        delta = (perturbed_activations - clean_activations).norm(dim=-1)  # [N, T]
        latent_delta = float(delta[latent_mask].mean())
        cot_delta = float(delta[cot_mask].mean())
        ratio = latent_delta / max(cot_delta, self.eps)
        return {
            "latent_sensitivity": latent_delta,
            "cot_sensitivity": cot_delta,
            "ratio": ratio,
        }
