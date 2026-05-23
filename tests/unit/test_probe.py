"""Unit tests for coconut_audit.probe (mock-based; no HF download)."""

from __future__ import annotations

import types
from typing import Any

import pytest

from coconut_audit.probe import HookRegistry, hook_site_for
from coconut_audit.probe.extractor import LatentExtractor, _resolve_module


@pytest.mark.parametrize(
    ("model_id", "layer", "expected"),
    [
        ("gpt2", 3, "transformer.h.3"),
        ("gpt2-small", 0, "transformer.h.0"),
        ("EleutherAI/pythia-160m", 5, "gpt_neox.layers.5"),
        ("meta-llama/Llama-3.1-8B", 7, "model.layers.7"),
        ("Goodfire/DeepSeek-R1-SAE-l37", 37, "model.layers.37"),
        ("google/gemma-2-9b-it", 12, "model.layers.12"),
        ("totally-unknown/Mystery-Model", 2, "model.layers.2"),
    ],
)
def test_hook_site_for_known_families(model_id: str, layer: int, expected: str) -> None:
    assert hook_site_for(model_id, layer) == expected


def test_hook_registry_clear_removes_handles() -> None:
    reg = HookRegistry()
    removed: list[int] = []

    class _Handle:
        def __init__(self, i: int) -> None:
            self.i = i

        def remove(self) -> None:
            removed.append(self.i)

    reg.register(_Handle(1))
    reg.register(_Handle(2))
    assert len(reg.handles) == 2
    reg.clear()
    assert removed == [1, 2]
    assert reg.handles == []


def test_resolve_module_walks_dotted_path() -> None:
    leaf = object()
    layer = types.SimpleNamespace(ln_2=leaf)
    h = [None, layer]
    root = types.SimpleNamespace(transformer=types.SimpleNamespace(h=h))
    assert _resolve_module(root, "transformer.h.1.ln_2") is leaf


class _FakeModule:
    """Minimal stand-in for a torch.nn.Module that records forward-hook attach."""

    def __init__(self) -> None:
        self._hooks: list[Any] = []

    def register_forward_hook(self, fn: Any) -> Any:
        self._hooks.append(fn)
        handle = types.SimpleNamespace(remove=lambda: self._hooks.remove(fn))
        return handle


def test_latent_extractor_attach_and_detach_lifecycle() -> None:
    leaf = _FakeModule()
    layer = types.SimpleNamespace(input_layernorm=leaf)
    layers = [layer, layer]
    root = types.SimpleNamespace(model=types.SimpleNamespace(layers=layers))

    ex = LatentExtractor(
        model_id="meta-llama/Llama-3.1-8B",
        layer=0,
        hook_site="model.layers.0.input_layernorm",
    )
    ex.attach(root)
    assert len(leaf._hooks) == 1
    ex.detach()
    assert leaf._hooks == []


def test_hf_inference_client_blocked_by_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """COCONUT_AUDIT_SKIP_HF_DOWNLOAD=1 must hard-block HFInferenceClient.load."""
    from coconut_audit.probe.client import HFInferenceClient

    monkeypatch.setenv("COCONUT_AUDIT_SKIP_HF_DOWNLOAD", "1")
    with pytest.raises(RuntimeError, match="COCONUT_AUDIT_SKIP_HF_DOWNLOAD"):
        HFInferenceClient(model_id="gpt2").load()


def test_hf_inference_client_falls_back_to_legacy_torch_dtype_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """transformers <5 uses `torch_dtype=`; the client must retry on TypeError."""
    from coconut_audit.probe.client import HFInferenceClient

    monkeypatch.setenv("COCONUT_AUDIT_SKIP_HF_DOWNLOAD", "0")
    calls: list[dict[str, Any]] = []

    class _FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _model_id: str, **_kwargs: Any) -> Any:
            return object()

    class _FakeModel:
        def to(self, _device: str) -> Any:
            return self

        def eval(self) -> None:
            pass

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, _model_id: str, **kwargs: Any) -> Any:
            calls.append(kwargs)
            if "dtype" in kwargs:
                raise TypeError("unexpected keyword 'dtype'")
            return _FakeModel()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeTokenizer)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _FakeAutoModel)

    HFInferenceClient(model_id="gpt2", dtype="float32").load()
    assert len(calls) == 2
    assert "dtype" in calls[0]
    assert "torch_dtype" in calls[1]
    assert "dtype" not in calls[1]
