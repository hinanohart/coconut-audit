# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.0 | ✅ (current alpha) |

## Reporting a vulnerability

Please report security issues **privately** rather than opening a public GitHub
issue. Use **GitHub's "Report a vulnerability"** form on the [Security tab](https://github.com/hinanohart/coconut-audit/security/advisories/new),
or open a private security advisory.

For sensitive issues that require coordination with HuggingFace Hub (e.g.,
malicious SAE artifacts staged under a Hub repo), please also alert HuggingFace
via their standard responsible-disclosure channel.

## Scope

In scope:

- Code execution via SAE artifacts loaded from the Hub (`sae.load_pretrained_sae`).
- HTML injection / XSS in `reports.render_html_report` outputs.
- Path traversal in `reports.LedgerWriter` write paths.
- Secret leakage from environment variables, token files, or logs.
- MCP tool argument injection that can reach the host shell.
- Dependency CVEs that affect the documented public API.

Out of scope:

- Vulnerabilities in third-party packages (`torch`, `transformers`,
  `huggingface_hub`, `sae-lens`, `sparsify`, `mcp`) — please report upstream.
- Issues that require the attacker to already control the user's Python
  environment.
- The semantic accuracy of an audit verdict (`coconut-audit` is best-effort;
  re-run on every model/SAE pin bump).

## Hardening notes

- `sae.load_pretrained_sae` validates safetensors metadata against tensor shapes
  before construction. Treat unknown SAEs as untrusted code-adjacent artifacts.
- `reports.render_html_report` HTML-escapes every user-controlled field via
  `jinja2`'s autoescape. We inherit the lesson from `recurrentlens 0.1.0.post1`
  (an XSS vector in unescaped report fields). Do not disable autoescape.
- The MCP server is `stdio`-only by default. If you expose it over a network
  transport, gate it behind your own authentication layer.
