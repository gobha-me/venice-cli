"""Tests for the `venice mcp-serve` subcommand and FastMCP wiring.

Two layers: the missing-`mcp`-extra path runs everywhere (it patches the SDK out);
the `build_server` wiring test is skipped unless the `mcp` SDK is importable (it is
absent on Python 3.9, where the extra's environment marker excludes it).
"""
import argparse
import importlib.util
import io
import sys
import unittest
from unittest import mock

_HAS_MCP = importlib.util.find_spec("mcp") is not None

EXPECTED_TOOLS = {
    "venice_image", "venice_tts", "venice_sfx", "venice_music",
    "venice_upscale", "venice_bg_remove", "venice_chat",
    "venice_video", "venice_image_edit",
}


class TestMissingExtra(unittest.TestCase):
    def test_missing_mcp_returns_2_with_hint(self):
        from venice.commands import mcp_serve

        err = io.StringIO()
        with mock.patch.dict(sys.modules, {"mcp": None}), \
             mock.patch.object(sys, "stderr", err):
            rc = mcp_serve._run(argparse.Namespace())
        self.assertEqual(rc, 2)
        self.assertIn('venice-cli[mcp]', err.getvalue())


@unittest.skipUnless(_HAS_MCP, "mcp SDK not installed (expected on Python 3.9)")
class TestServerWiring(unittest.TestCase):
    def test_build_server_exposes_exactly_nine_tools(self):
        from venice.mcp_server import build_server

        class FakeClient:
            api_key = "fake"
            base_url = "https://api.venice.ai/api/v1"

        server = build_server(FakeClient())
        names = {t.name for t in server._tool_manager.list_tools()}
        self.assertEqual(names, EXPECTED_TOOLS)


if __name__ == "__main__":
    unittest.main()
