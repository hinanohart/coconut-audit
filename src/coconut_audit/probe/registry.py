"""Hook-site registry for known model families."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Per-family default forward-hook site (residual / pre-block path).
# Phase 2 ships the union of Coconut / Quiet-STaR / TTC reference architectures.
_FAMILY_HOOK_SITES: dict[str, str] = {
    "gpt2": "transformer.h.{layer}",
    "EleutherAI/pythia": "gpt_neox.layers.{layer}",
    "meta-llama/Llama": "model.layers.{layer}",
    "deepseek-ai/DeepSeek-R1": "model.layers.{layer}",
    "Goodfire/DeepSeek-R1": "model.layers.{layer}",
    "google/gemma": "model.layers.{layer}",
    "Qwen/Qwen": "model.layers.{layer}",
    "mistralai/Mistral": "model.layers.{layer}",
    "fnlp/Llama-Scope": "model.layers.{layer}",
}


def hook_site_for(model_id: str, layer: int) -> str:
    """Resolve a forward-hook site path for a known model family.

    Falls back to a generic Llama-style residual path if the family is unknown
    (every transformer wired through `transformers` exposes a `model.layers`
    sequence after v4.45, modulo GPT-2's `transformer.h`).
    """
    for prefix, template in _FAMILY_HOOK_SITES.items():
        if model_id.startswith(prefix):
            return template.format(layer=layer)
    return f"model.layers.{layer}"


@dataclass(slots=True)
class HookRegistry:
    """Tracks active forward hooks so they can be cleanly removed."""

    handles: list[Any] = field(default_factory=list)

    def register(self, handle: Any) -> None:
        self.handles.append(handle)

    def clear(self) -> None:
        for h in self.handles:
            remove = getattr(h, "remove", None)
            if callable(remove):
                remove()
        self.handles.clear()
