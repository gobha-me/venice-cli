"""Tests for the `venice mcp-serve` subcommand and MCPServer wiring.

Two layers: the missing-`mcp`-extra path runs everywhere (it patches the SDK out);
the `build_server` wiring test is skipped unless the `mcp` SDK is importable (it is
absent on Python 3.9, where the extra's environment marker excludes it).
"""
import argparse
import asyncio
import base64
import importlib.util
import inspect
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_HAS_MCP = importlib.util.find_spec("mcp") is not None

EXPECTED_TOOLS = {
    "venice_image", "venice_tts", "venice_sfx", "venice_music",
    "venice_upscale", "venice_bg_remove", "venice_chat",
    "venice_video", "venice_image_edit", "venice_vision",
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


class TestCommandWiring(unittest.TestCase):
    def test_parser_defaults_to_text_only_and_accepts_declaration(self):
        from venice import cli

        parser = cli.build_parser()
        self.assertFalse(parser.parse_args(["mcp-serve"]).host_image_content)
        self.assertTrue(
            parser.parse_args([
                "mcp-serve", "--host-image-content"
            ]).host_image_content
        )

    @unittest.skipUnless(_HAS_MCP, "mcp SDK not installed (expected on Python 3.9)")
    def test_run_forwards_the_startup_declaration(self):
        from venice.commands import mcp_serve
        from venice import mcp_server

        client = object()
        doc = {"defaults": {}}
        with mock.patch.object(mcp_serve._mcp, "import_mcp", return_value=object()), \
             mock.patch.object(mcp_serve, "build_client_from_auth", return_value=client), \
             mock.patch.object(mcp_serve.userconfig, "load_config", return_value=doc), \
             mock.patch.object(mcp_server, "serve") as serve, \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = mcp_serve._run(argparse.Namespace(host_image_content=True))
        self.assertEqual(rc, 0)
        serve.assert_called_once_with(
            client, doc=doc, host_image_content=True
        )


@unittest.skipUnless(_HAS_MCP, "mcp SDK not installed (expected on Python 3.9)")
class TestServerWiring(unittest.TestCase):
    def test_build_server_exposes_exactly_ten_tools(self):
        from venice.mcp_server import build_server

        class FakeClient:
            api_key = "fake"
            base_url = "https://api.venice.ai/api/v1"

        server = build_server(FakeClient())
        names = {t.name for t in server._tool_manager.list_tools()}
        self.assertEqual(names, EXPECTED_TOOLS)

    def test_v2_client_lists_and_calls_the_public_server_surface(self):
        from mcp import Client

        from venice.commands import _mcp
        from venice.mcp_server import build_server

        class FakeClient:
            api_key = "fake"
            base_url = "https://api.venice.ai/api/v1"

        server = build_server(FakeClient(), doc={})

        async def exercise():
            async with Client(server) as client:
                listed = await client.list_tools()
                names = {tool.name for tool in listed.tools}
                chat = next(tool for tool in listed.tools if tool.name == "venice_chat")
                result = await client.call_tool("venice_chat", {"message": "ping"})
                return names, chat.input_schema, result

        with mock.patch.object(
            _mcp,
            "chat_tool",
            return_value={"status": "ok", "content": "pong"},
        ) as impl:
            names, schema, result = asyncio.run(exercise())

        self.assertEqual(names, EXPECTED_TOOLS)
        self.assertEqual(schema["required"], ["message"])
        self.assertFalse(result.is_error)
        self.assertIsNone(result.structured_content)
        self.assertEqual(
            json.loads(result.content[0].text),
            {"status": "ok", "content": "pong"},
        )
        impl.assert_called_once()

    def test_native_vision_returns_prompt_then_exact_image_content(self):
        from mcp import Client

        from venice.commands import _mcp, _openai
        from venice.mcp_server import build_server

        class FakeClient:
            api_key = "fake"
            base_url = "https://api.venice.ai/api/v1"

        png = b"\x89PNG\r\n\x1a\nexact-pixels"
        with tempfile.TemporaryDirectory() as root:
            Path(root, "frame.png").write_bytes(png)
            server = build_server(
                FakeClient(), doc={}, root=root, host_image_content=True
            )

            async def exercise():
                async with Client(server) as client:
                    return await client.call_tool("venice_vision", {
                        "input_path": "frame.png",
                        "prompt": "Inspect the alignment.",
                    })

            with mock.patch.object(_mcp, "vision_tool") as delegate, \
                 mock.patch.object(_openai, "import_openai") as sdk_import:
                result = asyncio.run(exercise())

        self.assertFalse(result.is_error)
        self.assertEqual([block.type for block in result.content], ["text", "image"])
        self.assertEqual(result.content[0].text, "Inspect the alignment.")
        self.assertEqual(result.content[1].mime_type, "image/png")
        self.assertEqual(base64.b64decode(result.content[1].data), png)
        delegate.assert_not_called()
        sdk_import.assert_not_called()

    def test_auto_without_declaration_delegates_as_text(self):
        from mcp import Client

        from venice.commands import _mcp
        from venice.mcp_server import build_server

        class FakeClient:
            api_key = "fake"
            base_url = "https://api.venice.ai/api/v1"

        server = build_server(FakeClient(), doc={})

        async def exercise():
            async with Client(server) as client:
                return await client.call_tool("venice_vision", {
                    "input_path": "frame.png",
                    "prompt": "Read it",
                    "model": "vision-model",
                    "max_tokens": 77,
                })

        delegated = {
            "status": "ok", "content": "delegated answer", "model": "vision-model"
        }
        with mock.patch.object(_mcp, "vision_tool", return_value=delegated) as impl:
            result = asyncio.run(exercise())

        self.assertFalse(result.is_error)
        self.assertEqual([block.type for block in result.content], ["text"])
        self.assertEqual(json.loads(result.content[0].text), delegated)
        impl.assert_called_once()
        self.assertEqual(impl.call_args.kwargs["model"], "vision-model")
        self.assertEqual(impl.call_args.kwargs["max_tokens"], 77)

    def test_native_without_declaration_fails_closed(self):
        from venice.commands import _mcp
        from venice.mcp_server import build_server

        class FakeClient:
            api_key = "fake"
            base_url = "https://api.venice.ai/api/v1"

        with mock.patch.object(_mcp, "vision_tool") as impl:
            result = build_server(FakeClient(), doc={})._tool_manager.get_tool(
                "venice_vision"
            ).fn(input_path="frame.png", mode="native")
        self.assertTrue(result.is_error)
        self.assertIn("--host-image-content", result.content[0].text)
        impl.assert_not_called()

    def test_explicit_delegate_wins_over_host_declaration(self):
        from venice.commands import _mcp
        from venice.mcp_server import build_server

        class FakeClient:
            api_key = "fake"
            base_url = "https://api.venice.ai/api/v1"

        delegated = {"status": "ok", "content": "text"}
        with mock.patch.object(_mcp, "vision_tool", return_value=delegated) as impl:
            result = build_server(
                FakeClient(), doc={}, host_image_content=True
            )._tool_manager.get_tool("venice_vision").fn(
                input_path="frame.png", mode="delegate"
            )
        self.assertEqual(json.loads(result.content[0].text), delegated)
        impl.assert_called_once()

    def test_native_remote_url_is_rejected_without_fetch_or_delegate(self):
        from venice.commands import _mcp
        from venice.mcp_server import build_server

        class FakeClient:
            api_key = "fake"
            base_url = "https://api.venice.ai/api/v1"

        with mock.patch.object(_mcp, "vision_tool") as impl:
            result = build_server(
                FakeClient(), doc={}, host_image_content=True
            )._tool_manager.get_tool("venice_vision").fn(
                image_url="https://example.test/frame.png", mode="native"
            )
        self.assertTrue(result.is_error)
        self.assertIn("only input_path", result.content[0].text)
        impl.assert_not_called()

    def test_auto_remote_url_delegates_even_for_image_content_host(self):
        from venice.commands import _mcp
        from venice.mcp_server import build_server

        class FakeClient:
            api_key = "fake"
            base_url = "https://api.venice.ai/api/v1"

        delegated = {"status": "ok", "content": "remote description"}
        with mock.patch.object(_mcp, "vision_tool", return_value=delegated) as impl:
            result = build_server(
                FakeClient(), doc={}, host_image_content=True
            )._tool_manager.get_tool("venice_vision").fn(
                image_url="https://example.test/frame.png"
            )
        self.assertEqual(json.loads(result.content[0].text), delegated)
        impl.assert_called_once()

    def test_vision_config_defaults_and_explicit_overrides_reach_delegate(self):
        from venice.commands import _mcp
        from venice.mcp_server import build_server

        class FakeClient:
            api_key = "fake"
            base_url = "https://api.venice.ai/api/v1"

        doc = {"defaults": {"vision": {
            "mode": "delegate", "model": "configured", "max_tokens": 44,
        }}}
        delegated = {"status": "ok", "content": "description"}
        tool = build_server(
            FakeClient(), doc=doc, host_image_content=True
        )._tool_manager.get_tool("venice_vision").fn
        with mock.patch.object(_mcp, "vision_tool", return_value=delegated) as impl:
            tool(input_path="frame.png")
            tool(
                input_path="frame.png", mode="delegate",
                model="explicit", max_tokens=55,
            )
        first, second = impl.call_args_list
        self.assertEqual(first.kwargs["model"], "configured")
        self.assertEqual(first.kwargs["max_tokens"], 44)
        self.assertEqual(second.kwargs["model"], "explicit")
        self.assertEqual(second.kwargs["max_tokens"], 55)

    def test_native_vision_reuses_local_media_authority_failures(self):
        from venice.commands import _shared
        from venice.mcp_server import build_server

        class FakeClient:
            api_key = "fake"
            base_url = "https://api.venice.ai/api/v1"

        cases = {
            "empty.png": b"",
            "text.png": b"not an image",
        }
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            for name, data in cases.items():
                Path(root, name).write_bytes(data)
            outside = Path(out, "outside.png")
            outside.write_bytes(b"\x89PNG\r\n\x1a\nbody")
            Path(root, "escape.png").symlink_to(outside)
            Path(root, "large.png").write_bytes(b"\x89PNG\r\n\x1a\nbody")
            tool = build_server(
                FakeClient(), doc={}, root=root, host_image_content=True
            )._tool_manager.get_tool("venice_vision").fn
            results = [tool(input_path=name) for name in cases]
            results.append(tool(input_path="missing.png"))
            results.append(tool(input_path=outside.as_posix()))
            results.append(tool(input_path="escape.png"))
            with mock.patch.object(_shared, "MAX_IMAGE_BYTES", 8):
                results.append(tool(input_path="large.png"))
        for result in results:
            self.assertTrue(result.is_error)
            self.assertEqual([block.type for block in result.content], ["text"])

    def test_server_captures_startup_root_for_media_authority(self):
        from venice.mcp_server import build_server
        from venice.commands import _mcp, _shared

        class FakeClient:
            api_key = "fake"
            base_url = "https://api.venice.ai/api/v1"

        captured = {}

        def upscale_tool(client, input_path, *, path_authority=None, **kwargs):
            captured["authority"] = path_authority
            return {"status": "ok"}

        with tempfile.TemporaryDirectory() as root, \
             tempfile.TemporaryDirectory() as outside, \
             mock.patch.object(_mcp, "upscale_tool", upscale_tool):
            server = build_server(FakeClient(), doc={}, root=root)
            server._tool_manager.get_tool("venice_upscale").fn(input_path="frame.png")
            frame = Path(root) / "frame.png"
            frame.write_bytes(b"\x89PNG\r\n\x1a\nbody")
            resolved, mime = captured["authority"].resolve(
                frame, kind="image", max_bytes=1024
            )
            self.assertEqual(resolved, frame.resolve())
            self.assertEqual(mime, "image/png")
            denied = Path(outside) / "frame.png"
            denied.write_bytes(b"\x89PNG\r\n\x1a\nbody")
            with self.assertRaises(_shared.MediaPathError):
                captured["authority"].resolve(
                    denied, kind="image", max_bytes=1024
                )

    def test_upscale_schema_matches_current_contract(self):
        from venice.mcp_server import build_server

        class FakeClient:
            api_key = "fake"
            base_url = "https://api.venice.ai/api/v1"

        server = build_server(FakeClient(), doc={})
        props = server._tool_manager.get_tool("venice_upscale").parameters["properties"]
        self.assertEqual(
            set(props),
            {"input_path", "scale", "creativity", "output_dir", "confirm", "max_spend"},
        )
        self.assertEqual(props["scale"]["anyOf"][0]["enum"], [2, 4])
        creativity = props["creativity"]["anyOf"][0]
        self.assertEqual(creativity["minimum"], 0.0)
        self.assertEqual(creativity["maximum"], 0.02)

    def test_image_wrappers_expose_native_controls(self):
        from venice.mcp_server import build_server

        class FakeClient:
            api_key = "fake"
            base_url = "https://api.venice.ai/api/v1"

        server = build_server(FakeClient(), doc={})
        image_props = server._tool_manager.get_tool(
            "venice_image"
        ).parameters["properties"]
        for name in (
            "style_references", "aspect_ratio", "resolution",
            "embed_exif_metadata", "lora_strength", "quality",
            "enable_web_search", "disable_prompt_optimization_thinking",
            "enhance_prompt",
        ):
            self.assertIn(name, image_props)
        edit_props = server._tool_manager.get_tool(
            "venice_image_edit"
        ).parameters["properties"]
        for name in (
            "quality", "disable_prompt_optimization_thinking", "enhance_prompt"
        ):
            self.assertIn(name, edit_props)

    def test_vision_schema_exposes_only_the_declared_public_inputs(self):
        from venice.mcp_server import build_server

        class FakeClient:
            api_key = "fake"
            base_url = "https://api.venice.ai/api/v1"

        tool = build_server(FakeClient(), doc={})._tool_manager.get_tool(
            "venice_vision"
        )
        props = tool.parameters["properties"]
        self.assertEqual(
            set(props),
            {"input_path", "image_url", "prompt", "model", "max_tokens", "mode"},
        )
        self.assertEqual(
            props["mode"]["anyOf"][0]["enum"],
            ["auto", "native", "delegate"],
        )

    def test_retired_upscale_config_returns_error_without_delegating(self):
        from venice.mcp_server import build_server
        from venice.commands import _mcp

        class FakeClient:
            api_key = "fake"
            base_url = "https://api.venice.ai/api/v1"

        doc = {"defaults": {"upscale": {"enhance_creativity": 0.5}}}
        with mock.patch.object(_mcp, "upscale_tool") as impl:
            server = build_server(FakeClient(), doc=doc)
            result = server._tool_manager.get_tool("venice_upscale").fn(
                input_path="frame.png"
            )
        self.assertEqual(result["status"], "error")
        self.assertIn("defaults.upscale.enhance_creativity", result["message"])
        impl.assert_not_called()


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
    # (section, tool, impl, param, config_value, HOST_value, base_kwargs)
    # config_value and host_value MUST differ: seeding both with the same value
    # makes the host-beats-config assertion hold regardless of precedence.
    _CLASS_C_TOOLS = [
        ("image", "venice_image", "image_tool", "model", "hidream", "venice-sd35",
         dict(prompt="p")),
        ("image", "venice_image", "image_tool", "format", "webp", "jpeg",
         dict(prompt="p")),
        ("image", "venice_image", "image_tool", "variants", 3, 1, dict(prompt="p")),
        ("tts", "venice_tts", "tts_tool", "model", "tts-orpheus", "tts-xai-v1",
         dict(text="t")),
        ("tts", "venice_tts", "tts_tool", "format", "wav", "flac", dict(text="t")),
        ("sfx", "venice_sfx", "sfx_tool", "model", "mmaudio-v2-text-to-audio",
         "elevenlabs-sound-effects-v2", dict(prompt="p")),
        ("sfx", "venice_sfx", "sfx_tool", "duration", 12, 7, dict(prompt="p")),
        ("music", "venice_music", "music_tool", "model", "other-music", "cli-music",
         dict(prompt="p")),
        ("video", "venice_video", "video_tool", "duration", "10s", "8s",
         dict(prompt="p")),
        ("upscale", "venice_upscale", "upscale_tool", "scale", 3.0, 4.0,
         dict(input_path="in.png")),
        # #57 Class C2. `venice_video.max_wait` was the last wrapper param left
        # holding a concrete constant, so it was non-None on every call and
        # `_merged` (which drops only None) let it beat defaults.video.max_wait
        # while the CLI path looked correct. sfx/music expose no `max_wait`
        # param at all, so config is their only source and there is nothing to
        # tri-state -- see test_config's tool-path asymmetry test.
        ("video", "venice_video", "video_tool", "max_wait", 42.0, 120.0,
         dict(prompt="p")),
    ]

    def test_class_c_config_reaches_impl_when_host_omits_the_arg(self):
        for section, tool, impl, param, cfg, _host, kwargs in self._CLASS_C_TOOLS:
            with self.subTest(tool=tool, param=param):
                doc = {"defaults": {section: {param: cfg}}}
                captured = self._invoke(section, tool, impl, param, kwargs, doc)
                self.assertEqual(captured.get(param), cfg,
                                 msg=f"{tool}.{param} did not receive the "
                                     f"config default")

    def test_class_c_host_arg_still_beats_config(self):
        for section, tool, impl, param, cfg, host, kwargs in self._CLASS_C_TOOLS:
            with self.subTest(tool=tool, param=param):
                self.assertNotEqual(cfg, host, "table row would be vacuous")
                doc = {"defaults": {section: {param: cfg}}}
                captured = self._invoke(
                    section, tool, impl, param, {**kwargs, param: host}, doc)
                self.assertEqual(captured.get(param), host,
                                 msg=f"{tool}.{param}: config beat an explicit "
                                     "host argument")

    def test_class_c_wrapper_defaults_are_none(self):
        """The direct tripwire, needing no config doc at all: it fails the
        instant someone re-types a wrapper param back to a concrete literal,
        which is what made these ten unreachable from config in the first place.
        """
        import inspect as _inspect

        from venice.mcp_server import build_server

        # `doc={}`, never `doc=None`: None is the sentinel that makes
        # build_server call `userconfig.load_config()` and read the developer's
        # real ~/.config/venice/config.json. This module redirects no HOME, and
        # CONTRIBUTING requires hermetic tests. `{}` proves the same thing.
        server = build_server(self._Client(), doc={})
        for _section, tool, _impl, param, _cfg, _host, _kwargs in self._CLASS_C_TOOLS:
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
