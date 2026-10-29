"""Record MCP tool schemas for every server profile without calling a tool.

Uses a placeholder client and explicit empty config; never loads credentials or
contacts Venice. Requires the optional MCP extra.
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from mcp import Client

from venice.client import VeniceClient
from venice.mcp_server import build_http_server, build_server


async def inventory():
    """Negotiate local MCP sessions and list schemas, with all I/O kept local."""
    lean = VeniceClient("test-fake-key")
    remote = dict(
        public_url="https://mcp.example.test/mcp", issuer_url="https://auth.example.test",
        jwks_url="https://auth.example.test/jwks", audience="audit", scopes=["venice:mcp"],
    )
    with tempfile.TemporaryDirectory(prefix="venice-mcp-surface-") as temp:
        servers = {
            "stdio": build_server(lean, doc={}, root=temp),
            "http_minimal": build_http_server(lean, doc={}, **remote),
            "http_media": build_http_server(lean, doc={}, media_dir=temp, **remote),
        }
        result = {}
        for profile, server in servers.items():
            async with Client(server) as session:
                listed = await session.list_tools()
                result[profile] = {
                    tool.name: tool.model_dump(mode="json", exclude_none=True)
                    for tool in listed.tools
                }
        return result


def main():
    """Write the profile inventory to an explicit output path."""
    output = Path(sys.argv[1])
    output.write_text(json.dumps(asyncio.run(inventory()), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
