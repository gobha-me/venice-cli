"""A tiny stdio MCP server used as a fixture by the #21 client integration test.

Run as ``python tests/_mcp_fake_server.py`` -- it speaks MCP over stdio and exposes
two tools: ``echo`` (annotated read-only) and ``write_note`` (side-effecting, no
read-only hint). The leading underscore keeps unittest's discovery from importing
this module (its ``import mcp`` would fail on Python 3.9), so it is only ever
launched as a subprocess by a test already gated on the `mcp` SDK being present.
"""
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

# Quiet the SDK's per-request INFO logging so it doesn't clutter test output.
server = FastMCP("fake", log_level="WARNING")


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def echo(text: str) -> str:
    """Return the text unchanged (read-only)."""
    return f"echo: {text}"


@server.tool()
def write_note(note: str) -> str:
    """Pretend to persist a note (side-effecting; no read-only hint)."""
    return f"wrote: {note}"


if __name__ == "__main__":
    server.run(transport="stdio")
