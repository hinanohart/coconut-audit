"""MCP stdio server entrypoint.

Exposes `audit_run`, `audit_get`, `audit_diff` over stdio. To register with
Claude Code / Cursor / Cline, add to your MCP config:

  {"command": "coconut-audit-mcp"}

The server is intentionally stateless beyond the JSONL ledger on disk
(`COCONUT_AUDIT_LEDGER` env var, default `audit_reports/ledger.jsonl`).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from coconut_audit.mcp.tools import audit_diff, audit_get, audit_run

# Relative, traversal-free ledger path: no leading "/" (absolute) and no ".."
# segment. Server-side `_validate_ledger_path` is the authoritative guard; this
# schema constraint just rejects obviously out-of-bounds inputs at the edge.
_LEDGER_PATH_SCHEMA: dict[str, Any] = {
    "type": ["string", "null"],
    "description": (
        "Optional ledger path, relative to the COCONUT_AUDIT_LEDGER_ROOT base "
        "(default: current working directory). Absolute paths and `..` traversal "
        "are rejected."
    ),
    "pattern": r"^(?!/)(?!.*(^|/)\.\.(/|$)).*$",
}

_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "audit_run": {
        "description": (
            "Run a fresh continuous-latent-CoT audit and append the report to the JSONL ledger. "
            "v0.1.0 runs SYNTHETIC activations (demo_mode=True by default); verdicts are "
            "advisory only and MUST NOT be cited as real-model evidence. Real-model "
            "inference (forward-hook + SAE encode + benchmark loop) lands in v0.1.1+."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {"type": "string"},
                "sae_id": {"type": "string"},
                "probe": {
                    "type": "string",
                    "enum": ["steering", "shortcut", "drift"],
                    "default": "steering",
                },
                "benchmark": {"type": ["string", "null"]},
                "n_samples": {"type": "integer", "default": 32},
                "seed": {"type": "integer", "default": 0},
                "demo_mode": {"type": "boolean", "default": True},
                "ledger_path": _LEDGER_PATH_SCHEMA,
            },
            "required": ["model_id", "sae_id"],
        },
    },
    "audit_get": {
        "description": "Look up a stored audit report by its `audit_id`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audit_id": {"type": "string"},
                "ledger_path": _LEDGER_PATH_SCHEMA,
            },
            "required": ["audit_id"],
        },
    },
    "audit_diff": {
        "description": "Diff two stored audit reports by their IDs (e.g. pre/post fine-tune).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "audit_id_a": {"type": "string"},
                "audit_id_b": {"type": "string"},
                "ledger_path": _LEDGER_PATH_SCHEMA,
            },
            "required": ["audit_id_a", "audit_id_b"],
        },
    },
}

_TOOL_HANDLERS = {
    "audit_run": audit_run,
    "audit_get": audit_get,
    "audit_diff": audit_diff,
}


async def _serve() -> None:
    try:
        import mcp.server.stdio
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as e:  # pragma: no cover - exercised at runtime only
        raise RuntimeError(
            "The `mcp` package is required to run the stdio server. "
            "Install with: uv pip install mcp"
        ) from e

    server = Server("coconut-audit")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(name=name, description=spec["description"], inputSchema=spec["inputSchema"])
            for name, spec in _TOOL_SCHEMAS.items()
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        handler = _TOOL_HANDLERS.get(name)
        if handler is None:
            return [TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]
        try:
            result = handler(**arguments)
        except Exception as e:  # surface as MCP tool error payload
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": str(e), "exception_type": type(e).__name__}),
                )
            ]
        return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]

    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:  # pragma: no cover - exercised at runtime only
    """Launch the MCP stdio server."""
    asyncio.run(_serve())


if __name__ == "__main__":  # pragma: no cover
    main()
