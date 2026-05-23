# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-23

### Added

- Initial scaffold of the `coconut-audit` framework.
- `core`: `LatentTrace`, `AuditReport`, `AuditVerdict`, `ProbeKind` types; `AuditConfig` loader (pydantic v2).
- `probe`: `HFInferenceClient`, `LatentExtractor`, `HookRegistry` for Coconut / Quiet-STaR / Test-Time-Compute families.
- `sae`: `PretrainedSAELoader` + `FeatureProjector` against HuggingFace Hub. License-clean primary targets: GPT-2 small (SAELens), Pythia 70M/160M (EleutherAI `sparsify`), DeepSeek-R1 (Goodfire).
- `audit`: `SteeringProbe`, `ShortcutDetector`, `DriftScorer`, `aggregate_verdict` (4-way verdict: `PASS` / `WARN` / `FAIL` / `NO_VERDICT`).
- `mcp`: stdio MCP server exposing `audit_run`, `audit_get`, `audit_diff` tools.
- `reports`: JSON + HTML (Jinja2 autoescape) + JSONL append-only ledger.
- `cli`: `coconut-audit run / get / diff` entrypoints (click).
- CI (ruff format/lint, no-secrets grep, pytest matrix on Python 3.11 + 3.12).
- Pre-commit hooks.
- Apache-2.0 license, SECURITY policy, CITATION.cff, CONTRIBUTING guide.

### Notes

- `transformers` 4.45 through 5.9+ are both supported. The `HFInferenceClient`
  tries the modern `dtype=` keyword first (transformers >=5.0) and falls back
  to the legacy `torch_dtype=` for 4.x. If you pin transformers >=5.0,
  remember its default `dtype` shifted to `"auto"` — pass an explicit dtype
  via `HFInferenceClient(dtype="float32")` for cross-version weight-hash
  determinism.

### Known limitations

- v0.1.0 does **not** reproduce the full numerics of [arxiv:2512.21711](https://arxiv.org/abs/2512.21711); see the v0.2 roadmap for the planned reproduction.
- v0.1.0 ships HF inference only. Self-trained SAEs and SSM backends arrive in v0.2.
- **Safetensors-only policy**: pickle-backed `.pt` SAE artifacts are refused.
  SAEs distributed only as `.pt` (e.g. `Goodfire/DeepSeek-R1-SAE-l37`) require
  either upstream re-export or a community conversion adapter — both targeted
  for v0.1.1.
- Goodfire Llama 3.x Instruct, Qwen-Scope, and Gemma Scope 2 adapters land in v0.1.1+.
