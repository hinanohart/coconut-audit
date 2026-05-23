"""Pretrained SAE loader (HuggingFace Hub only).

Inherits the `recurrentlens 0.1.0.post1` hardening lesson: validate tensor
shapes against declared `d_in` / `d_sae` before construction. Treat unknown
SAEs as untrusted code-adjacent artifacts.

v0.1.0 supports **safetensors only**; pickle-backed `.pt` artifacts are
refused with an actionable error (see ``SECURITY.md`` and Trail of Bits'
"pickles-in-pytorch" advisory). Goodfire / SAELens / EleutherAI sparsify
safetensors layouts are auto-detected via the ``_KEY_ALIASES`` table.

Known limitation: when ``d_in == d_sae`` (square 1× SAE) the weight
orientation cannot be inferred from shape alone; callers must declare it
explicitly via ``cfg["weight_layout"] in {"in_first", "out_first"}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coconut_audit.sae.linear import LinearSAE

_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "W_enc": ("W_enc", "encoder.weight", "sae.W_enc", "W_E"),
    "W_dec": ("W_dec", "decoder.weight", "sae.W_dec", "W_D"),
    "b_enc": ("b_enc", "encoder.bias", "sae.b_enc", "b_E"),
    "b_dec": ("b_dec", "decoder.bias", "sae.b_dec", "b_D"),
    "threshold": ("threshold", "jump_threshold", "log_threshold"),
}


def _pick(state_dict: dict[str, Any], canonical: str) -> Any:
    for k in _KEY_ALIASES[canonical]:
        if k in state_dict:
            return state_dict[k]
    if canonical == "threshold":
        return None
    raise KeyError(
        f"SAE state_dict missing required key {canonical!r}; "
        f"tried aliases {_KEY_ALIASES[canonical]}"
    )


def _orient(tensor: Any, expected_rows: int, expected_cols: int) -> Any:
    """Return tensor oriented so that `tensor.shape[0] == expected_rows`.

    Raises `ValueError` if the tensor is not 2D or if neither orientation
    matches (`d_in` × `d_sae`) — silent transposition of a malformed weight
    matrix would otherwise corrupt audit results downstream.
    """
    if tensor.ndim != 2:
        raise ValueError(
            f"SAE weight tensor must be 2D, got ndim={tensor.ndim} (shape={tuple(tensor.shape)})"
        )
    rows, cols = int(tensor.shape[0]), int(tensor.shape[1])
    if rows == expected_rows and cols == expected_cols:
        return tensor
    if rows == expected_cols and cols == expected_rows:
        return tensor.T
    raise ValueError(
        f"SAE weight tensor shape {tuple(tensor.shape)} does not match "
        f"expected ({expected_rows}, {expected_cols}) in either orientation"
    )


@dataclass(slots=True)
class PretrainedSAELoader:
    """Resolves an SAE Hub repo ID to local artifact paths + metadata."""

    sae_id: str
    revision: str | None = None
    cache_dir: Path | None = None

    def snapshot(self) -> Path:
        from huggingface_hub import snapshot_download

        local = snapshot_download(
            repo_id=self.sae_id,
            revision=self.revision,
            cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
        )
        return Path(local)


def _find_weights(root: Path) -> Path:
    """Locate the SAE weights file under `root`. Safetensors-only by policy.

    We deliberately do **not** accept pickle-backed `.pt` files. The SAE
    publisher should re-export as safetensors before publication; see
    https://huggingface.co/docs/safetensors for conversion snippets.
    """
    for pat in ("sae_weights.safetensors", "*.safetensors"):
        hits = sorted(root.glob(pat))
        if hits:
            return hits[0]
    pickle_hits = sorted(root.glob("*.pt"))
    if pickle_hits:
        raise FileNotFoundError(
            f"SAE artifact under {root} only ships pickle-backed .pt files "
            f"({[h.name for h in pickle_hits]}); coconut-audit refuses pickle loads. "
            f"Ask the publisher to re-export as safetensors, or convert locally."
        )
    raise FileNotFoundError(f"no .safetensors SAE weight file under {root}")


def _find_cfg(root: Path) -> dict[str, Any]:
    for name in ("cfg.json", "config.json", "sae_cfg.json"):
        path = root / name
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    return {}


def _load_state_dict(path: Path) -> dict[str, Any]:
    """Load a safetensors state_dict. Pickle-backed `.pt` is rejected upstream."""
    from safetensors.torch import load_file

    return load_file(str(path))


def build_linear_sae(state_dict: dict[str, Any], cfg: dict[str, Any]) -> LinearSAE:
    """Construct a `LinearSAE` from a state_dict + cfg dict."""
    d_in = int(cfg.get("d_in") or cfg.get("d_model") or 0)
    d_sae = int(cfg.get("d_sae") or cfg.get("d_features") or cfg.get("d_hidden") or 0)
    activation = str(cfg.get("activation_fn") or cfg.get("activation") or "relu").lower()

    W_enc = _pick(state_dict, "W_enc")
    W_dec = _pick(state_dict, "W_dec")
    b_enc = _pick(state_dict, "b_enc")
    b_dec = _pick(state_dict, "b_dec")
    threshold = _pick(state_dict, "threshold")

    if d_in == 0:
        d_in = int(b_dec.shape[0])
    if d_sae == 0:
        d_sae = int(b_enc.shape[0])

    if d_in == d_sae:
        layout = str(cfg.get("weight_layout") or "").lower()
        if layout not in {"in_first", "out_first"}:
            raise ValueError(
                f"Square SAE (d_in == d_sae == {d_in}) is shape-ambiguous; "
                "set cfg['weight_layout'] to 'in_first' (W_enc=[d_in, d_sae]) "
                "or 'out_first' (W_enc=[d_sae, d_in]) to disambiguate."
            )
        if layout == "out_first":
            W_enc = W_enc.T if W_enc.ndim == 2 else W_enc
            W_dec = W_dec.T if W_dec.ndim == 2 else W_dec
    else:
        W_enc = _orient(W_enc, d_in, d_sae)
        W_dec = _orient(W_dec, d_sae, d_in)

    return LinearSAE(
        W_enc=W_enc,
        b_enc=b_enc,
        W_dec=W_dec,
        b_dec=b_dec,
        threshold=threshold,
        d_in=d_in,
        d_sae=d_sae,
        activation=activation,
    )


def load_pretrained_sae(
    sae_id: str,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> LinearSAE:
    """Download a pretrained SAE from HuggingFace Hub and return a `LinearSAE`."""
    loader = PretrainedSAELoader(sae_id=sae_id, revision=revision, cache_dir=cache_dir)
    root = loader.snapshot()
    cfg = _find_cfg(root)
    weights_path = _find_weights(root)
    state_dict = _load_state_dict(weights_path)
    return build_linear_sae(state_dict, cfg)
