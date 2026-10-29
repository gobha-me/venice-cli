"""Shared MCP argument precedence and text result encoding.

This optional SDK module is loaded only by the MCP server profiles.
"""
import json

from mcp import types


def _merged(defaults: dict, host: dict) -> dict:
    """Layer config `defaults` UNDER host-supplied args (#58): an explicit (non-None)
    host value wins; where the host omitted an arg (MCPServer fills the wrapper's None
    default) the config default applies; keys the wrapper never exposes (e.g. image
    safe_mode/hide_watermark) come purely from config. Same precedence as
    `_agent._make_paid`'s `{**defaults, **_clean(arguments)}`."""
    return {**defaults, **{k: v for k, v in host.items() if v is not None}}


def _vision_text_result(result: dict) -> types.CallToolResult:
    """Convert the existing delegated result envelope to one MCP text block."""
    return types.CallToolResult(
        content=[types.TextContent(text=json.dumps(result))],
        is_error=result.get("status") == "error",
    )
