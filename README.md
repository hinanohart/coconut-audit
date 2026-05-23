# coconut-audit

> Audit harness for continuous-latent chain-of-thought (Coconut / Quiet-STaR / Test-Time Compute) — SAE-grounded steering & shortcut probes with an MCP server.

[![CI](https://github.com/hinanohart/coconut-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/hinanohart/coconut-audit/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
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

## Install

```bash
uv pip install coconut-audit                  # core (HF inference + SAE loader + MCP)
uv pip install "coconut-audit[saelens]"       # + SAELens GPT-2 / Gemma Scope loader
uv pip install "coconut-audit[sparsify]"      # + EleutherAI Pythia / Llama Scope loader
uv pip install "coconut-audit[datasets]"      # + MMLU / HotpotQA shortcut fixtures
```

## Quick start (CLI)

```bash
# 1. Run an audit against a HF model with a pretrained SAE
coconut-audit run \
  --model Goodfire/DeepSeek-R1-SAE-l37 \
  --probe shortcut \
  --benchmark mmlu \
  --out audit_reports/deepseek_r1.json

# 2. View the human-readable HTML report
open audit_reports/deepseek_r1.html
```

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

License-clean (Apache / MIT / CC-BY) primary targets:

| Model family | SAE source | Hub URL | License |
|---|---|---|---|
| GPT-2 small | SAELens `gpt2-small-res-jb` | [jbloom/GPT2-Small-SAEs](https://huggingface.co/jbloom/GPT2-Small-SAEs) | MIT |
| Pythia 70M / 160M | EleutherAI `sparsify` | [EleutherAI/sae-pythia-*](https://huggingface.co/EleutherAI) | MIT |
| Llama 3.1 8B | EleutherAI / Llama Scope | [fnlp/Llama-Scope](https://huggingface.co/fnlp/Llama-Scope) | Apache-2.0 |
| DeepSeek-R1 | Goodfire | [Goodfire/DeepSeek-R1-SAE-l37](https://huggingface.co/Goodfire/DeepSeek-R1-SAE-l37) | MIT |
| Gemma 2 | Gemma Scope | [google/gemma-scope](https://huggingface.co/google/gemma-scope) | CC-BY-4.0 |

Community-contributed extensions (Goodfire Llama 3.x Instruct, Qwen-Scope, Gemma Scope 2) land in v0.1.1+. PRs welcome.

## Portfolio map (related hinanohart OSS)

| Repo | Layer | Relation |
|---|---|---|
| [recurrentlens](https://github.com/hinanohart/recurrentlens) | Representation-side SAE framework (SSM-first) | Upstream for v0.2 SSM backend |
| [subjunctor](https://github.com/hinanohart/subjunctor) | LLM agent counterfactual gate | Downstream MCP client |
| [exitkit](https://github.com/hinanohart/exitkit) | Nozick-style closest-continuer over snapshots | Conceptual sibling (alignment plumbing) |

## Design honesty

- **HF inference only.** v0.1.0 does **not** train new SAEs; we load pretrained artifacts. This is intentional — the audit harness should not introduce its own representation drift.
- **Best-effort audit, not certification.** Verdicts are advisory. Re-run on every model/SAE pin bump.
- **Coverage is partial.** A NO_VERDICT result on an unsupported model is a feature, not a bug.

## Citation

See [CITATION.cff](./CITATION.cff).

## License

[Apache-2.0](./LICENSE).
