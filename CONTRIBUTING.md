# Contributing

Thanks for your interest in `coconut-audit`. The library is in early alpha (v0.1.0); contributions are very welcome.

## Setup

```bash
git clone https://github.com/hinanohart/coconut-audit.git
cd coconut-audit
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,saelens,sparsify,datasets]"
pre-commit install
```

## Running tests

```bash
pytest -ra                                                # default (unit + smoke, no HF download)
pytest -ra -m "slow"                                      # slow tests (HF download)
pytest -ra -m "integration"                               # end-to-end audit pipeline
COCONUT_AUDIT_SKIP_HF_DOWNLOAD=1 pytest -ra               # offline CI mode
```

## Style

- `ruff format`
- `ruff check --fix`
- `pyright` (standard mode)

Pre-commit hooks enforce all three plus a `no-secrets` grep gate.

## High-leverage contributions for v0.1.0 → v0.1.1

1. **Pretrained SAE adapters**: add a `sae/adapters/<vendor>.py` entry for Goodfire Llama-3.x Instruct lines / Qwen-Scope / Gemma Scope 2. Each adapter needs a Hub repo ID, layer→hook-site map, and a license assertion.
2. **Steering / Shortcut benchmark fixtures**: contribute small (≤100-row) MMLU / HotpotQA / TruthfulQA subsets with known shortcut signatures, under `tests/fixtures/`.
3. **MCP tool surface**: extend `mcp/tools.py` with `audit_explain` (per-feature attribution) or `audit_compare` (model A vs B side-by-side).
4. **Honest-marketing docs**: any place the README overstates capability is a bug — PR welcomed.

## v0.2 roadmap

- SSM backend via `recurrentlens` (Mamba-2 / RWKV-7 hook adapter)
- `subjunctor` loose-coupling MCP bridge (counterfactual gate before audit)
- Self-train SAE pipeline (currently HF Hub only)
- Coconut shortcut metrics: full reproduction of arxiv:2512.21711 numerics

## Code of conduct

Be kind. Disagree about technical claims with citations.
