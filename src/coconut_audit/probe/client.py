"""HuggingFace inference client + activation-capturing forward pass.

v0.1.0 supports local CPU/GPU inference via `transformers.AutoModelForCausalLM`.
The remote HF Inference API is intentionally **not** a v0.x target — the API
does not expose hidden-state hooks, so it cannot drive an SAE-grounded audit.
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

        Cross-version transformers: selects the dtype keyword via the installed
        `transformers.__version__` major (>=5 uses `dtype=`; 4.x uses
        `torch_dtype=`). `transformers` ≥5.x silently routes `torch_dtype=`
        through `**kwargs` with a deprecation warning rather than raising
        ``TypeError``, so a try/except cannot disambiguate the two versions.
        """
        if os.environ.get("COCONUT_AUDIT_SKIP_HF_DOWNLOAD", "0") == "1":
            raise RuntimeError(
                "HFInferenceClient.load blocked by COCONUT_AUDIT_SKIP_HF_DOWNLOAD=1 "
                "(set in CI / offline mode). Unset to download the model."
            )

        _ALLOWED_DTYPES = {
            "auto",
            "float32",
            "float16",
            "bfloat16",
            "float64",
            "int8",
        }
        if self.dtype not in _ALLOWED_DTYPES:
            raise ValueError(
                f"HFInferenceClient(dtype={self.dtype!r}) is not in the allowed "
                f"set {sorted(_ALLOWED_DTYPES)}; pass a torch dtype name."
            )

        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_dtype: Any = "auto" if self.dtype == "auto" else getattr(torch, self.dtype)

        try:
            _major = int(transformers.__version__.split(".", 1)[0])
        except (ValueError, AttributeError):
            _major = 4  # treat unknown version strings as legacy
        dtype_kw = "dtype" if _major >= 5 else "torch_dtype"

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **{dtype_kw: resolved_dtype},
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
