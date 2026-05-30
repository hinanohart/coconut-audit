# coconut-audit

> Audit harness for continuous-latent chain-of-thought (Coconut / Quiet-STaR / Test-Time Compute) — SAE-grounded steering & shortcut probes with an MCP server.

[![CI](https://github.com/hinanohart/coconut-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/hinanohart/coconut-audit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)

**Status: v0.1.0 (alpha, framework + scaffold).** Pretrained SAE coverage and full reproduction of paper-grade metrics arrive in v0.1.1.

## Why

Recent work argues that continuous-latent reasoning (Coconut and successors) does **not** encode faithful step-by-step reasoning — the latent tokens behave as
"uninterpretable placeholders" that exploit dataset shortcuts ([arxiv:2512.21711](https://arxiv.org/abs/2512.21711)). There is no general-purpose OSS that:

1. Probes a deployed continuous-latent model's hidden activations,
2. Decomposes them through pretrained Sparse Autoencoders (SAEs) from HuggingFace Hub,
3. Runs **Steering** (token-subset perturbation) + **Shortcut** (OOD/biased prompt) experiments,
4. Emits an auditable verdict via the Model Context Protocol (MCP) so Claude / Cursor / Cline can call it.

`coconut-audit` ships exactly that pipeline, with HuggingFace inference as the only required runtime.

## Why not X

- **Anthropic's Circuit Tracer** is a Claude-internal tool — not an OSS framework you can point at `gpt2` / `pythia-160m` / `Llama-3.1-8B` from your terminal.
- **Goodfire Ember** is a closed-source platform targeting Llama-family models behind an API; `coconut-audit` is an MIT framework you can pin in CI, run offline, and extend to new families via a small `_FAMILY_HOOK_SITES` entry.
- **SAELens / sparsify** are excellent SAE training/inference libraries but stop short of an audit verdict; `coconut-audit` consumes their pretrained artifacts and adds the steering / shortcut / drift probes + MCP surface on top.

## Install

```bash
uv pip install coconut-audit                  # core (HF inference + SAE loader + MCP)
uv pip install "coconut-audit[saelens]"       # + SAELens GPT-2 / Gemma Scope loader
uv pip install "coconut-audit[sparsify]"      # + EleutherAI Pythia / Llama Scope loader
uv pip install "coconut-audit[datasets]"      # + MMLU / HotpotQA shortcut fixtures
```

## Quick start (CLI)

```bash
# 1. Run an audit (v0.1.0 ships demo_mode by default — synthetic activations
#    drive the probes so the end-to-end pipeline is exercisable without a
#    model download). Real-model inference lands in v0.1.1+.
coconut-audit run \
  --model gpt2 \
  --sae jbloom/GPT2-Small-SAEs \
  --probe shortcut \
  --out-dir audit_reports/

# 2. View the human-readable HTML report
open audit_reports/<audit_id>.html
```

> **Note**: v0.1.0 runs synthetic activations (`demo_mode=True`); pass `--real`
> to see the `NotImplementedError` placeholder for the v0.1.1+ real-model path.
> Verdicts from synthetic runs are advisory only — re-run on every release.

## Quick start (MCP — call from Claude / Cursor / Cline)

```jsonc
// ~/.config/claude/mcp.json
{
  "mcpServers": {
    "coconut-audit": {
      "command": "coconut-audit-mcp",
      "args": []
    }
  }
}
```

Then from the agent:

> Run `audit_run` with `model=gpt2`, `probe=steering`, `benchmark=mmlu_subset`.

## Supported pretrained SAE coverage (v0.1.0)

License-clean (Apache / MIT / CC-BY) primary targets shipping as **safetensors**:

| Model family | SAE source | Hub URL | License |
|---|---|---|---|
| GPT-2 small | SAELens `gpt2-small-res-jb` | [jbloom/GPT2-Small-SAEs](https://huggingface.co/jbloom/GPT2-Small-SAEs) | MIT |
| Pythia 70M / 160M | EleutherAI `sparsify` | [EleutherAI/sae-pythia-*](https://huggingface.co/EleutherAI) | MIT |
| Llama 3.1 8B | Llama Scope (fnlp) | [fnlp/Llama-Scope](https://huggingface.co/fnlp/Llama-Scope) | MIT |
| Gemma 2 | Gemma Scope | [google/gemma-scope](https://huggingface.co/google/gemma-scope) | CC-BY-4.0 |

**Safetensors-only policy**: `coconut-audit` refuses pickle-backed `.pt` artifacts
to keep the supply chain auditable. SAEs distributed only as `.pt`
(e.g. `Goodfire/DeepSeek-R1-SAE-l37`) become available once the publisher
re-exports as safetensors, or via a community-contributed conversion adapter in v0.1.1+.

Community-contributed extensions (Goodfire Instruct lines after safetensors export,
Qwen-Scope, Gemma Scope 2) land in v0.1.1+. PRs welcome.

## Portfolio map (related hinanohart OSS)

| Repo | Layer | Relation |
|---|---|---|
| [recurrentlens](https://github.com/hinanohart/recurrentlens) | Representation-side SAE framework (SSM-first) | Upstream SAE backend for v0.2 SSM models |
| [subjunctor](https://github.com/hinanohart/subjunctor) | Agent-side counterfactual gate (TS) | Conceptual sibling (alignment plumbing; MCP integration planned for subjunctor v0.4) |
| [exitkit](https://github.com/hinanohart/exitkit) | Nozick-style closest-continuer over snapshots | Conceptual sibling (alignment plumbing) |

## Design honesty

- **HF inference only.** v0.1.0 does **not** train new SAEs; we load pretrained artifacts. This is intentional — the audit harness should not introduce its own representation drift.
- **Best-effort audit, not certification.** Verdicts are advisory. Re-run on every model/SAE pin bump.
- **Coverage is partial.** A NO_VERDICT result on an unsupported model is a feature, not a bug.

## Citation

See [CITATION.cff](./CITATION.cff).

## License

[MIT](./LICENSE).
