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


@unittest.skipUnless(_HAS_MCP, "mcp SDK not installed (expected on Python 3.9)")
class TestConfigDefaultsWiring(unittest.TestCase):
    """#58: defaults.<section>.* are layered UNDER host args on mcp-serve, matching
    the chat/code agent path. The wrapper's underlying function is `Tool.fn`."""

    class _Client:
        api_key = "fake"
        base_url = "https://api.venice.ai/api/v1"

    def _spy(self):
        captured = {}

        # A real function (NOT a MagicMock): config_defaults_for introspects
        # inspect.signature(impl), so these named params are what let the matching
        # config keys be injected.
        def image_tool(client, prompt=None, *, hide_watermark=None, safe_mode=None,
                       steps=None, confirm=False, max_spend=None, output_dir=None,
                       **kw):
            captured.update(hide_watermark=hide_watermark, safe_mode=safe_mode,
                            steps=steps)
            captured.update(kw)
            return {"status": "ok"}

        return captured, image_tool

    def _invoke_image(self, doc, spy, **call_kwargs):
        """Build the server with `spy` patched in (so both build-time introspection
        and call-time delegation see it), then invoke the registered venice_image."""
        from venice.mcp_server import build_server
        from venice.commands import _mcp
        with mock.patch.object(_mcp, "image_tool", spy):
            server = build_server(self._Client(), doc=doc)
            fn = server._tool_manager.get_tool("venice_image").fn
            fn(prompt="p", **call_kwargs)

    def test_config_defaults_injected(self):
        captured, spy = self._spy()
        doc = {"defaults": {"image": {"hide_watermark": True, "safe_mode": False}}}
        self._invoke_image(doc, spy)
        self.assertIs(captured["hide_watermark"], True)   # from config
        self.assertIs(captured["safe_mode"], False)       # overrides impl default True

    def test_host_arg_overrides_config(self):
        captured, spy = self._spy()
        doc = {"defaults": {"image": {"steps": 40, "safe_mode": False}}}
        self._invoke_image(doc, spy, steps=5, safe_mode=True)
        self.assertEqual(captured["steps"], 5)            # explicit host arg wins
        self.assertIs(captured["safe_mode"], True)        # exposed flag, host wins

    def test_no_config_no_injection(self):
        captured, spy = self._spy()
        self._invoke_image({}, spy)
        self.assertIsNone(captured["hide_watermark"])     # impl default applies
        self.assertIsNone(captured["safe_mode"])
        self.assertIsNone(captured["steps"])


if __name__ == "__main__":
    unittest.main()
