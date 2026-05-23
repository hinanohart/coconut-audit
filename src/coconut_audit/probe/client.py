"""HuggingFace inference client (lazy loader for Phase 2 full implementation)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HFInferenceClient:
    """Thin wrapper over `transformers.AutoModelForCausalLM`.

    v0.1.0 skeleton: actual model load happens in `Phase 2` (see CHANGELOG).
    """

    model_id: str
    device: str = "cpu"
    dtype: str = "auto"

    def load(self) -> None:  # pragma: no cover - filled in Phase 2
        raise NotImplementedError("HFInferenceClient.load lands in v0.1.0 Phase 2")
