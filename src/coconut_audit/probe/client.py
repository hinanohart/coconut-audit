"""HuggingFace inference client + activation-capturing forward pass.

v0.1.0 supports local CPU/GPU inference via `transformers.AutoModelForCausalLM`.
Remote `HF Inference API` lands in v0.2 (the API does not expose hidden-state
hooks, so we keep it as an optional non-audit codepath only).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class HFInferenceClient:
    """Wraps `transformers.AutoModelForCausalLM` + tokenizer with lazy loading."""

    model_id: str
    device: str = "cpu"
    dtype: str = "auto"
    trust_remote_code: bool = False
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    @property
    def model(self) -> Any:
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    def load(self) -> None:
        """Lazily load model + tokenizer. Honors `COCONUT_AUDIT_SKIP_HF_DOWNLOAD`.

        Cross-version transformers: tries `dtype=` first (transformers >=5.0,
        where `torch_dtype=` is deprecated) and falls back to `torch_dtype=`
        for transformers 4.x.
        """
        if os.environ.get("COCONUT_AUDIT_SKIP_HF_DOWNLOAD", "0") == "1":
            raise RuntimeError(
                "HFInferenceClient.load blocked by COCONUT_AUDIT_SKIP_HF_DOWNLOAD=1 "
                "(set in CI / offline mode). Unset to download the model."
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_dtype: Any = "auto" if self.dtype == "auto" else getattr(torch, self.dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=resolved_dtype,
                trust_remote_code=self.trust_remote_code,
            )
        except TypeError:
            # transformers <5.0 uses the legacy `torch_dtype=` keyword.
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=resolved_dtype,
                trust_remote_code=self.trust_remote_code,
            )
        self._model = model.to(self.device)
        self._model.eval()

    def encode(self, text: str) -> Any:
        """Tokenize `text` to a device-resident tensor dict."""
        toks = self.tokenizer(text, return_tensors="pt")
        return {k: v.to(self.device) for k, v in toks.items()}

    def forward(self, inputs: dict[str, Any]) -> Any:
        """Run a single forward pass under `torch.inference_mode()`."""
        import torch

        with torch.inference_mode():
            return self.model(**inputs, output_hidden_states=False, return_dict=True)
