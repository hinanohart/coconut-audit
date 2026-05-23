# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0.post1] — 2026-05-24

Post-release hotfix driven by a 4-agent post-publication audit
(architect / omc-code-reviewer / verifier + integration critic). All changes
are documentation, defense-in-depth, and verdict-safety fixes; the public
API surface is unchanged.

### Fixed

- `probe/client.py`: replaced the `try/except TypeError` transformers dtype
  fallback (which never fires against real transformers because both
  `dtype=` and `torch_dtype=` are absorbed by `**kwargs`) with an explicit
  `transformers.__version__` major-branch dispatch. Added an allow-list of
  dtype names to catch typos at construction.
- `audit/verdict.py`: `label_for_metric(NaN, ...)` previously returned
  `PASS` silently; it now returns `NO_VERDICT`. Also validates the
  `warn_at` vs `fail_at` ordering instead of swallowing inverted thresholds.
- `core/types.py`: `AuditReport` is now `frozen=True` (matches `LatentTrace`).
  Post-construction mutation is rejected with `ValidationError`.
- `sae/loader.py`: docstring incorrectly claimed `v0.1.0 supports the
  Goodfire .pt-state-dict format directly`; the actual loader refuses
  pickle `.pt`. Docstring rewritten to match policy. `_orient` now rejects
  `ndim != 2` and refuses to silently disambiguate square (d_in == d_sae)
  matrices without an explicit `weight_layout` cfg hint.
- `mcp/pipeline.py`: corrupt JSONL lines in `find_report_in_ledger` now raise
  a typed `RuntimeError` with the offending line number instead of leaking
  raw `json.JSONDecodeError`.
- `reports/jsonl.py`: `LedgerWriter.append` now takes an exclusive `fcntl`
  lock + `fsync` so two MCP clients writing to the same default ledger
  cannot interleave bytes mid-line.
- `reports/html.py`: when `notes` includes `demo_mode=True`, the HTML report
  now prepends a prominent banner (`DEMO MODE — synthetic activations`) so
  a screenshot of the verdict pill cannot be cited out of context.
- `mcp/server.py`: the `audit_run` MCP tool description now explicitly
  warns that v0.1.0 verdicts come from synthetic activations.
- `CITATION.cff`: removed the false claim that v0.1.0 ships a DeepSeek-R1
  adapter (it does not; see Known limitations below).
- `README.md`: added a `Why not X` section so external readers can locate
  `coconut-audit`'s niche relative to Anthropic Circuit Tracer and Goodfire.

### Known limitations (carried forward to v0.1.1)

- The pipeline's `demo_mode=True` path drives the audit primitives with
  `torch.randn` activations and **does not call the SAE encoder or the
  forward hook**. The probe / SAE / projector primitives are wired and
  unit-tested in isolation but are not connected to the demo flow. The
  real-model path lands in v0.1.1+.
- MCP server lacks structured `logging` and per-exception-type
  classification (code-reviewer follow-up); errors are surfaced as a
  generic `{error, exception_type}` JSON payload.
- HF Hub revision pinning is by `revision` string only; SHA allow-listing
  of pretrained SAE artifacts is a v0.1.1 hardening target.

## [0.1.0] — 2026-05-23

### Added

- Initial scaffold of the `coconut-audit` framework.
- `core`: `LatentTrace`, `AuditReport`, `AuditVerdict`, `ProbeKind` types; `AuditConfig` loader (pydantic v2).
- `probe`: `HFInferenceClient`, `LatentExtractor`, `HookRegistry` for Coconut / Quiet-STaR / Test-Time-Compute families.
- `sae`: `PretrainedSAELoader` + `FeatureProjector` against HuggingFace Hub. Safetensors-only loader policy (pickle `.pt` is refused). License-clean primary targets (safetensors-shipping): GPT-2 small (SAELens), Pythia 70M/160M (EleutherAI `sparsify`), Llama Scope (fnlp), Gemma Scope (google). Goodfire DeepSeek-R1 is deferred to v0.1.1+ (upstream ships only `.pt`).
- `audit`: `SteeringProbe`, `ShortcutDetector`, `DriftScorer`, `aggregate_verdict` (4-way verdict: `PASS` / `WARN` / `FAIL` / `NO_VERDICT`).
- `mcp`: stdio MCP server exposing `audit_run`, `audit_get`, `audit_diff` tools.
- `reports`: JSON + HTML (stdlib `html.escape(..., quote=True)` per-field) + JSONL append-only ledger.
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
