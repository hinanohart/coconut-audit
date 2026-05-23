"""Latent activation extractor (forward-hook based, per-layer per-hook-site)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from coconut_audit.probe.registry import HookRegistry, hook_site_for


def _resolve_module(root: Any, dotted_path: str) -> Any:
    """Walk a dotted attribute path (e.g. 'transformer.h.5.ln_2') to a sub-module."""
    obj = root
    for part in dotted_path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


@dataclass(slots=True)
class LatentExtractor:
    """Captures hidden-state activations at the configured layer + hook site."""

    model_id: str
    layer: int
    hook_site: str | None = None
    captured: list[Any] = field(default_factory=list)
    _registry: HookRegistry = field(default_factory=HookRegistry, repr=False)

    def resolve_site(self) -> str:
        return self.hook_site or hook_site_for(self.model_id, self.layer)

    def attach(self, model: Any) -> None:
        """Register a forward hook on the resolved sub-module."""
        site = self.resolve_site()
        target = _resolve_module(model, site)

        def _hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            self.captured.append(tensor.detach())

        handle = target.register_forward_hook(_hook)
        self._registry.register(handle)

    def detach(self) -> None:
        self._registry.clear()

    def reset(self) -> None:
        self.captured.clear()

    def extract(self, model: Any, inputs: dict[str, Any]) -> list[Any]:
        """Run one forward pass and return the captured activations."""
        import torch

        self.reset()
        self.attach(model)
        try:
            with torch.inference_mode():
                model(**inputs)
        finally:
            self.detach()
        return list(self.captured)
