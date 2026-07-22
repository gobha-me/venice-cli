"""`venice mcp-serve` -- run an MCP server (stdio) exposing venice tools.

Direction A of the MCP epic (#16 / #14): venice is the *callee*. It speaks MCP over
stdio -- JSON-RPC frames on stdout -- exposing image/sfx/music/tts/upscale/bg-remove/
chat as MCP tools, so a host (Claude Code, or the #15 host) calls them instead of
shelling out to the CLI.

The `mcp` SDK is imported lazily (behind the `[mcp]` extra, Python >=3.10) so the
base stdlib-only CLI and `venice --help` keep working without it -- the same
discipline `chat` uses for the openai SDK. Once the server starts, stdout belongs to
the JSON-RPC transport, so this command's own diagnostics go to stderr only.
"""
from __future__ import annotations

import sys

from .. import auth
from .. import userconfig
from ..client import build_client_from_auth
from . import _mcp


def register(subparsers) -> None:
    p = subparsers.add_parser(
        "mcp-serve",
        help="Run an MCP server (stdio) exposing venice tools.",
        description=(
            "Speaks MCP over stdio: exposes venice image/sfx/music/tts/upscale/"
            "bg-remove/chat as MCP tools. Needs the [mcp] extra (Python >=3.10): "
            'pip install "venice-cli[mcp]". Attach it with, e.g., '
            "`claude mcp add venice -- venice mcp-serve`. Spend on paid tools is "
            "gated: costs over VENICE_MCP_MAX_SPEND (default $0.10) need confirm=true. "
            "Output files land in VENICE_MCP_OUTPUT_DIR (default: cwd)."
        ),
    )
    p.set_defaults(handler=_run)


def _run(args) -> int:
    mcp = _mcp.import_mcp("mcp-serve")  # lazy probe -> None + stderr hint if absent
    if mcp is None:
        return 2

    try:
        # Fail fast on auth *before* stdout is handed to the JSON-RPC transport.
        client = build_client_from_auth()
    except auth.AuthError as e:
        print(str(e), file=sys.stderr)
        return 2

    from ..mcp_server import serve  # lazy: only import FastMCP after the probe passes

    doc = userconfig.load_config()  # #58: honor defaults.<section>.* in exposed tools

    print("venice mcp-serve: starting stdio MCP server (Ctrl-C to stop)",
          file=sys.stderr)
    try:
        serve(client, doc=doc)
    except KeyboardInterrupt:
        return 130
    return 0
