"""Latent drift scoring across SAE feature distributions.

`wasserstein` is a per-feature sorted-quantile (sliced Wasserstein-1)
approximation — O(N log N) per feature, no scipy dep, cheap enough to run
across thousands of SAE features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DriftScorer:
    """Drift between two latent-feature distributions over the same SAE.

    Args:
      method: `wasserstein` (sliced W-1, default) or `l2_mean` (||μ_a - μ_b||₂).
    """

    method: str = "wasserstein"

    def score(self, a: Any, b: Any) -> float:
        if a.shape[-1] != b.shape[-1]:
            raise ValueError(f"feature-dim mismatch: a {tuple(a.shape)} vs b {tuple(b.shape)}")
        if self.method == "wasserstein":
            return _sliced_wasserstein_1(a, b)
        if self.method == "l2_mean":
            return float((a.float().mean(0) - b.float().mean(0)).norm())
        raise ValueError(f"unknown drift method {self.method!r}")


def _sliced_wasserstein_1(a: Any, b: Any) -> float:
    """Per-feature sorted-quantile W-1, averaged across features.

    Both `a` and `b` are 2D [N, K] / [M, K]. If N != M the longer one is
    truncated to the shorter; this matches the standard 1D-Wasserstein
    on truncated empirical CDFs.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"expected 2D arrays, got a={tuple(a.shape)} b={tuple(b.shape)}")
    n = min(a.shape[0], b.shape[0])
    a_sorted = a[:n].float().sort(dim=0).values
    b_sorted = b[:n].float().sort(dim=0).values
    return float((a_sorted - b_sorted).abs().mean())


def latent_drift_score(a: Any, b: Any) -> float:
    """Convenience entrypoint (alias for `DriftScorer(method='wasserstein').score`)."""
    return DriftScorer().score(a, b)
