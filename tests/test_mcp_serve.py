"""Tests for the `venice mcp-serve` subcommand and FastMCP wiring.

Two layers: the missing-`mcp`-extra path runs everywhere (it patches the SDK out);
the `build_server` wiring test is skipped unless the `mcp` SDK is importable (it is
absent on Python 3.9, where the extra's environment marker excludes it).
"""
import argparse
import importlib.util
import inspect
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

    # -- #57 Class B: the tri-stated booleans must survive the wrapper --------- #
    #
    # `_merged` layers config UNDER host args and drops only None. A wrapper param
    # declared `bool = False` therefore fills a concrete False when the host omits
    # the arg, and that False silently BEATS the config default -- while the CLI
    # path looks perfectly correct. These four are the guard for that: each new
    # _COMMAND_MAP boolean must reach its impl when the host says nothing.
    _CLASS_B_TOOLS = [
        ("music", "venice_music", "music_tool", "instrumental", dict(prompt="p")),
        ("video", "venice_video", "video_tool", "no_audio", dict(prompt="p")),
        ("upscale", "venice_upscale", "upscale_tool", "enhance",
         dict(input_path="in.png")),
        ("image_edit", "venice_image_edit", "image_edit_tool", "safe_mode",
         dict(prompt="p")),
    ]

    # -- #57 Class C1: the same trap, for the VALUED defaults ------------------ #
    #
    # Identical mechanism to Class B above, and identical blast radius: a wrapper
    # param left as `model: str = DEFAULT_IMAGE_MODEL` fills a concrete value
    # when the host omits the arg, and that value beats defaults.image.model --
    # silently, while the CLI path, the _COMMAND_MAP rows and the docs all look
    # correct. Ten params across six tools; each needs to reach its impl.
    _CLASS_C_TOOLS = [
        ("image", "venice_image", "image_tool", "model", "hidream",
         dict(prompt="p")),
        ("image", "venice_image", "image_tool", "format", "webp",
         dict(prompt="p")),
        ("image", "venice_image", "image_tool", "variants", 3, dict(prompt="p")),
        ("tts", "venice_tts", "tts_tool", "model", "tts-kokoro", dict(text="t")),
        ("tts", "venice_tts", "tts_tool", "format", "wav", dict(text="t")),
        ("sfx", "venice_sfx", "sfx_tool", "model", "mmaudio-v2-text-to-audio",
         dict(prompt="p")),
        ("sfx", "venice_sfx", "sfx_tool", "duration", 12, dict(prompt="p")),
        ("music", "venice_music", "music_tool", "model", "other-music",
         dict(prompt="p")),
        ("video", "venice_video", "video_tool", "duration", "10s",
         dict(prompt="p")),
        ("upscale", "venice_upscale", "upscale_tool", "scale", 3.0,
         dict(input_path="in.png")),
    ]

    def test_class_c_config_reaches_impl_when_host_omits_the_arg(self):
        for section, tool, impl, param, val, kwargs in self._CLASS_C_TOOLS:
            with self.subTest(tool=tool, param=param):
                doc = {"defaults": {section: {param: val}}}
                captured = self._invoke(section, tool, impl, param, kwargs, doc)
                self.assertEqual(captured.get(param), val,
                                 msg=f"{tool}.{param} did not receive the "
                                     f"config default")

    def test_class_c_host_arg_still_beats_config(self):
        for section, tool, impl, param, val, kwargs in self._CLASS_C_TOOLS:
            with self.subTest(tool=tool, param=param):
                doc = {"defaults": {section: {param: val}}}
                captured = self._invoke(
                    section, tool, impl, param, {**kwargs, param: val}, doc)
                self.assertEqual(captured.get(param), val)

    def test_class_c_wrapper_defaults_are_none(self):
        """The direct tripwire, needing no config doc at all: it fails the
        instant someone re-types a wrapper param back to a concrete literal,
        which is what made these ten unreachable from config in the first place.
        """
        import inspect as _inspect

        from venice.mcp_server import build_server

        server = build_server(self._Client(), doc=None)
        for _section, tool, _impl, param, _val, _kwargs in self._CLASS_C_TOOLS:
            with self.subTest(tool=tool, param=param):
                fn = server._tool_manager.get_tool(tool).fn
                default = _inspect.signature(fn).parameters[param].default
                self.assertIsNone(
                    default,
                    msg=f"{tool}.{param} default is {default!r}, not None -- it "
                        "will silently beat the config value")

    def _invoke(self, section, tool_name, impl_name, param, call_kwargs, doc):
        """Patch the named impl with a real function exposing `param`, build the
        server against `doc`, and invoke the registered tool."""
        from venice.mcp_server import build_server
        from venice.commands import _mcp

        captured = {}

        def spy(client, *a, **kw):
            captured.update(kw)
            return {"status": "ok"}

        spy.__signature__ = inspect.Signature([
            inspect.Parameter("client", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter(param, inspect.Parameter.KEYWORD_ONLY, default=None),
            inspect.Parameter("kw", inspect.Parameter.VAR_KEYWORD),
        ])

        with mock.patch.object(_mcp, impl_name, spy):
            server = build_server(self._Client(), doc=doc)
            server._tool_manager.get_tool(tool_name).fn(**call_kwargs)
        return captured

    def test_class_b_config_reaches_impl_when_host_omits_the_arg(self):
        for section, tool, impl, param, kwargs in self._CLASS_B_TOOLS:
            with self.subTest(tool=tool, param=param):
                doc = {"defaults": {section: {param: True}}}
                captured = self._invoke(section, tool, impl, param, kwargs, doc)
                self.assertIs(captured.get(param), True,
                              msg=f"{tool}.{param} did not receive the config default")

    def test_class_b_host_arg_still_beats_config(self):
        for section, tool, impl, param, kwargs in self._CLASS_B_TOOLS:
            with self.subTest(tool=tool, param=param):
                doc = {"defaults": {section: {param: True}}}
                captured = self._invoke(section, tool, impl, param,
                                        dict(kwargs, **{param: False}), doc)
                self.assertIs(captured.get(param), False)

    def test_no_config_no_injection(self):
        captured, spy = self._spy()
        self._invoke_image({}, spy)
        self.assertIsNone(captured["hide_watermark"])     # impl default applies
        self.assertIsNone(captured["safe_mode"])
        self.assertIsNone(captured["steps"])


if __name__ == "__main__":
    unittest.main()
