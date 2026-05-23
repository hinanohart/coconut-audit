"""Unit tests for coconut_audit.sae (synthetic state_dicts; no HF download)."""

from __future__ import annotations

import pytest
import torch

from coconut_audit.sae import (
    BaseSAE,
    FeatureProjector,
    LinearSAE,
    build_linear_sae,
)


def _make_linear_sae(d_in: int = 8, d_sae: int = 32, jump: bool = False) -> LinearSAE:
    g = torch.Generator().manual_seed(0xC0C0)
    return LinearSAE(
        W_enc=torch.randn(d_in, d_sae, generator=g),
        b_enc=torch.zeros(d_sae),
        W_dec=torch.randn(d_sae, d_in, generator=g),
        b_dec=torch.zeros(d_in),
        threshold=torch.zeros(d_sae) + 0.05 if jump else None,
        activation="jumprelu" if jump else "relu",
    )


def test_linear_sae_satisfies_base_protocol() -> None:
    sae = _make_linear_sae()
    assert isinstance(sae, BaseSAE)


def test_linear_sae_encode_decode_shapes() -> None:
    sae = _make_linear_sae(d_in=8, d_sae=32)
    x = torch.randn(4, 8)
    z = sae.encode(x)
    assert z.shape == (4, 32)
    assert (z >= 0).all()
    x_hat = sae.decode(z)
    assert x_hat.shape == (4, 8)


def test_linear_sae_jumprelu_thresholding() -> None:
    sae = _make_linear_sae(jump=True)
    x = torch.randn(2, 8)
    z = sae.encode(x)
    # All survivors must be strictly above the per-feature threshold.
    survivors = z[z > 0]
    assert (survivors > 0.05).all()


def test_linear_sae_validates_shape_mismatch() -> None:
    with pytest.raises(ValueError, match=r"W_enc shape"):
        LinearSAE(
            W_enc=torch.zeros(8, 31),  # wrong d_sae
            b_enc=torch.zeros(32),
            W_dec=torch.zeros(32, 8),
            b_dec=torch.zeros(8),
            d_in=8,
            d_sae=32,
        )


def test_feature_projector_handles_batch_seq() -> None:
    sae = _make_linear_sae(d_in=8, d_sae=32)
    proj = FeatureProjector(sae=sae)
    hidden = torch.randn(2, 5, 8)
    z = proj.project(hidden)
    assert z.shape == (2, 5, 32)


def test_feature_projector_rejects_wrong_d_in() -> None:
    sae = _make_linear_sae(d_in=8, d_sae=32)
    proj = FeatureProjector(sae=sae)
    with pytest.raises(ValueError, match="hidden last dim"):
        proj.project(torch.randn(2, 5, 7))


def test_build_linear_sae_from_canonical_state_dict() -> None:
    d_in, d_sae = 4, 16
    state = {
        "W_enc": torch.randn(d_in, d_sae),
        "b_enc": torch.zeros(d_sae),
        "W_dec": torch.randn(d_sae, d_in),
        "b_dec": torch.zeros(d_in),
    }
    cfg = {"d_in": d_in, "d_sae": d_sae, "activation_fn": "relu"}
    sae = build_linear_sae(state, cfg)
    assert sae.d_in == d_in
    assert sae.d_sae == d_sae
    z = sae.encode(torch.randn(3, d_in))
    assert z.shape == (3, d_sae)


def test_build_linear_sae_handles_aliased_keys_and_orientation() -> None:
    d_in, d_sae = 4, 16
    # `encoder.weight` is conventionally [d_sae, d_in] (PyTorch Linear); loader
    # must auto-orient to [d_in, d_sae].
    state = {
        "encoder.weight": torch.randn(d_sae, d_in),
        "encoder.bias": torch.zeros(d_sae),
        "decoder.weight": torch.randn(d_in, d_sae),
        "decoder.bias": torch.zeros(d_in),
    }
    sae = build_linear_sae(state, {})
    assert sae.W_enc.shape == (d_in, d_sae)
    assert sae.W_dec.shape == (d_sae, d_in)


def test_build_linear_sae_missing_key_errors_loudly() -> None:
    with pytest.raises(KeyError, match="W_enc"):
        build_linear_sae({}, {"d_in": 4, "d_sae": 16})


def test_find_weights_refuses_pickle_pt(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """SECURITY policy: pickle-backed .pt artifacts must be refused with a clear message."""
    from coconut_audit.sae.loader import _find_weights

    (tmp_path / "model.pt").write_bytes(b"\x80\x04N.")  # fake pickle bytes
    with pytest.raises(FileNotFoundError, match="pickle-backed"):
        _find_weights(tmp_path)


def test_find_weights_missing_returns_clear_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from coconut_audit.sae.loader import _find_weights

    with pytest.raises(FileNotFoundError, match="no .safetensors"):
        _find_weights(tmp_path)


def test_find_weights_prefers_canonical_filename(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from coconut_audit.sae.loader import _find_weights

    (tmp_path / "alt.safetensors").write_bytes(b"\x00")
    (tmp_path / "sae_weights.safetensors").write_bytes(b"\x00")
    picked = _find_weights(tmp_path)
    assert picked.name == "sae_weights.safetensors"


def test_build_linear_sae_rejects_square_without_weight_layout() -> None:
    """d_in == d_sae is shape-ambiguous; require explicit layout to avoid silent corruption."""
    d = 8
    state = {
        "W_enc": torch.randn(d, d),
        "b_enc": torch.zeros(d),
        "W_dec": torch.randn(d, d),
        "b_dec": torch.zeros(d),
    }
    with pytest.raises(ValueError, match="weight_layout"):
        build_linear_sae(state, {"d_in": d, "d_sae": d})


def test_build_linear_sae_square_with_explicit_layout_in_first() -> None:
    d = 8
    state = {
        "W_enc": torch.randn(d, d),
        "b_enc": torch.zeros(d),
        "W_dec": torch.randn(d, d),
        "b_dec": torch.zeros(d),
    }
    sae = build_linear_sae(state, {"d_in": d, "d_sae": d, "weight_layout": "in_first"})
    assert sae.W_enc.shape == (d, d)


def test_orient_rejects_non_2d_tensor() -> None:
    from coconut_audit.sae.loader import _orient

    with pytest.raises(ValueError, match="must be 2D"):
        _orient(torch.zeros(2, 3, 4), expected_rows=2, expected_cols=3)


def test_orient_rejects_mismatched_shape() -> None:
    """Neither orientation matches → ValueError instead of silent corruption."""
    from coconut_audit.sae.loader import _orient

    with pytest.raises(ValueError, match="does not match"):
        _orient(torch.zeros(5, 7), expected_rows=4, expected_cols=16)
