"""Hook-site registry for known model families."""

from __future__ import annotations

from dataclasses import dataclass, field

_FAMILY_HOOK_SITES: dict[str, str] = {
    # Coconut-paper families (Phase 2 expands)
    "gpt2": "transformer.h.{layer}.ln_2",
    "EleutherAI/pythia": "gpt_neox.layers.{layer}.input_layernorm",
    "meta-llama/Llama": "model.layers.{layer}.input_layernorm",
    "deepseek-ai/DeepSeek-R1": "model.layers.{layer}.input_layernorm",
    "google/gemma": "model.layers.{layer}.input_layernorm",
}


def hook_site_for(model_id: str, layer: int) -> str:
    """Resolve a forward-hook site path for a known model family.

    Falls back to a generic residual-post path if the family is unknown.
    """
    for prefix, template in _FAMILY_HOOK_SITES.items():
        if model_id.startswith(prefix):
            return template.format(layer=layer)
    return f"model.layers.{layer}.input_layernorm"


@dataclass(slots=True)
class HookRegistry:
    """Tracks active forward hooks so they can be cleanly removed."""

    handles: list[object] = field(default_factory=list)

    def register(self, handle: object) -> None:
        self.handles.append(handle)

    def clear(self) -> None:
        for h in self.handles:
            remove = getattr(h, "remove", None)
            if callable(remove):
                remove()
        self.handles.clear()
