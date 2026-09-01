"""Tests for the external MCP client (`venice chat --mcp`, issue #21).

Three layers:
- Pure helpers (spec resolution, name namespacing, result translation, side-effect
  classification) run everywhere -- they never import the `mcp` SDK.
- An import-purity test proves the module has no module-scope `import mcp` (the CI
  `package` job asserts a base install stays mcp-free).
- A real end-to-end `attach()` test spawns a tiny stdio MCP server subprocess; it is
  skipped unless the `mcp` SDK is importable (absent on Python 3.9).
"""
import asyncio
import contextlib
import importlib.util
import os
import subprocess
import sys
import unittest
from unittest import mock

from venice.commands import _mcp_client as mc

_HAS_MCP = importlib.util.find_spec("mcp") is not None
_FAKE_SERVER = os.path.join(os.path.dirname(__file__), "_mcp_fake_server.py")


# --- duck-typed stand-ins for the mcp SDK's result/annotation objects --------

class _Ann:
    def __init__(self, read_only_hint=None):
        self.read_only_hint = read_only_hint


class _Block:
    def __init__(self, text=None, type=None):
        self.text = text
        self.type = type


class _CallResult:
    def __init__(self, content=None, is_error=False, structured_content=None):
        self.content = content or []
        self.is_error = is_error
        self.structured_content = structured_content


class TestPureHelpers(unittest.TestCase):
    def test_resolve_specs_stdio_and_http(self):
        doc = {"mcpServers": {
            "fs": {"command": "echo", "args": ["x"]},
            "web": {"type": "http", "url": "http://h"},
        }}
        specs = mc.resolve_specs(["fs", "web"], doc)
        self.assertEqual(specs[0], ("fs", {"command": "echo", "args": ["x"]}))
        self.assertEqual(specs[1][0], "web")

    def test_resolve_specs_unknown_raises_and_lists_available(self):
        doc = {"mcpServers": {"fs": {"command": "echo"}}}
        with self.assertRaises(ValueError) as ctx:
            mc.resolve_specs(["nope"], doc)
        msg = str(ctx.exception)
        self.assertIn("nope", msg)
        self.assertIn("fs", msg)  # names what IS registered

    def test_resolve_specs_malformed_entry_raises(self):
        doc = {"mcpServers": {"bad": {"foo": 1}}}  # neither command nor url
        with self.assertRaises(ValueError):
            mc.resolve_specs(["bad"], doc)

    # --- #70: @secret:<name> references resolved at attach time -------------
    def test_resolve_secret_refs_substitutes_inline_token(self):
        with mock.patch.object(mc.auth, "load_secret", return_value="tok123") as ls:
            out = mc.resolve_secret_refs(
                {"Authorization": "Bearer @secret:cluster"}, where="headers")
        self.assertEqual(out, {"Authorization": "Bearer tok123"})
        ls.assert_called_once_with("cluster")

    def test_resolve_secret_refs_multiple_refs_in_one_value(self):
        # Separate the two refs with a char outside the name set (a hyphen would
        # be read as part of the first name, since [A-Za-z0-9_.-] allows it).
        with mock.patch.object(mc.auth, "load_secret",
                               side_effect=lambda n: {"a": "A", "b": "B"}[n]):
            out = mc.resolve_secret_refs(
                {"X": "@secret:a @secret:b"}, where="env")
        self.assertEqual(out, {"X": "A B"})

    def test_resolve_secret_refs_passes_through_plaintext(self):
        # No @secret: token -> value untouched and load_secret never consulted
        # (existing plaintext entries keep working).
        with mock.patch.object(mc.auth, "load_secret",
                               side_effect=AssertionError("must not be called")):
            out = mc.resolve_secret_refs(
                {"TOKEN": "literal-plaintext"}, where="env")
        self.assertEqual(out, {"TOKEN": "literal-plaintext"})

    def test_resolve_secret_refs_none_and_empty_are_noops(self):
        self.assertIsNone(mc.resolve_secret_refs(None, where="env"))
        self.assertEqual(mc.resolve_secret_refs({}, where="env"), {})

    def test_resolve_secret_refs_missing_secret_raises(self):
        with mock.patch.object(mc.auth, "load_secret", return_value=None):
            with self.assertRaises(ValueError) as ctx:
                mc.resolve_secret_refs(
                    {"Authorization": "Bearer @secret:cluster"}, where="headers")
        msg = str(ctx.exception)
        self.assertIn("cluster", msg)     # names the missing secret
        self.assertIn("headers", msg)     # names where it was referenced
        self.assertIn("venice secret set", msg)  # actionable hint

    def test_advertised_name_namespaces_and_sanitizes(self):
        self.assertEqual(mc._advertised_name("fs", "read", set()), "fs__read")
        self.assertEqual(mc._advertised_name("my server", "a/b", set()), "my_server__a_b")

    def test_advertised_name_de_collides(self):
        taken = set()
        a = mc._advertised_name("s", "t", taken)
        b = mc._advertised_name("s", "t", taken)
        self.assertEqual(a, "s__t")
        self.assertNotEqual(a, b)

    def test_advertised_name_truncates_to_64(self):
        name = mc._advertised_name("s" * 50, "t" * 50, set())
        self.assertLessEqual(len(name), 64)

    def test_advertised_name_truncation_still_unique(self):
        taken = set()
        a = mc._advertised_name("s" * 50, "t" * 50, taken)
        b = mc._advertised_name("s" * 50, "t" * 50, taken)
        self.assertNotEqual(a, b)
        self.assertLessEqual(len(b), 64)

    def test_is_side_effecting_defaults_true(self):
        self.assertTrue(mc._is_side_effecting(None))
        self.assertTrue(mc._is_side_effecting(_Ann(read_only_hint=None)))
        self.assertTrue(mc._is_side_effecting(_Ann(read_only_hint=False)))
        self.assertFalse(mc._is_side_effecting(_Ann(read_only_hint=True)))

    def test_translate_ok_joins_text(self):
        r = mc._translate_result(_CallResult([_Block(text="hi"), _Block(text="yo")]))
        self.assertEqual(r, {"status": "ok", "content": "hi\nyo"})

    def test_translate_marks_non_text(self):
        r = mc._translate_result(_CallResult([_Block(type="image")]))
        self.assertEqual(r["status"], "ok")
        self.assertIn("non-text content: image", r["content"])

    def test_translate_error(self):
        r = mc._translate_result(_CallResult([_Block(text="boom")], is_error=True))
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["message"], "boom")

    def test_translate_carries_structured(self):
        r = mc._translate_result(
            _CallResult([_Block(text="x")], structured_content={"a": 1})
        )
        self.assertEqual(r["structured"], {"a": 1})

    def test_clean_args_strips_controlled_keys(self):
        self.assertEqual(
            mc._clean_args({"a": 1, "confirm": True, "max_spend": 9, "output_dir": "/x"}),
            {"a": 1},
        )
        self.assertEqual(mc._clean_args("not-a-dict"), {})

    def test_resolve_timeout_precedence(self):
        self.assertEqual(mc._resolve_timeout(12, "NOPE_ENV", 30), 12.0)
        with mock.patch.dict(os.environ, {"T_ENV": "7"}):
            self.assertEqual(mc._resolve_timeout(None, "T_ENV", 30), 7.0)
        self.assertEqual(mc._resolve_timeout(None, "MISSING_ENV", 30), 30.0)

    def test_resolve_timeout_rejects_non_finite_selected_value(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(source="explicit", value=bad), self.assertRaises(ValueError):
                mc._resolve_timeout(bad, "MISSING_ENV", 30)
        with mock.patch.dict(os.environ, {"T_ENV": "nan"}):
            with self.assertRaises(ValueError):
                mc._resolve_timeout(None, "T_ENV", 30)


class TestImportClean(unittest.TestCase):
    def test_imports_without_the_mcp_sdk(self):
        """A fresh interpreter with `mcp` unavailable must still import
        `_mcp_client` -- proving no module-scope `import mcp` (guards the CI
        base-install purity assertion). Runs in a subprocess so it never perturbs
        this process's `sys.modules` for other tests."""
        src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
        code = (
            "import sys; sys.modules['mcp'] = None;"
            "import venice.commands._mcp_client as m;"
            "assert hasattr(m, 'attach') and hasattr(m, 'resolve_specs');"
            "print('import-clean-ok')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONPATH": src},
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("import-clean-ok", proc.stdout)


@unittest.skipUnless(_HAS_MCP, "mcp SDK not installed (expected on Python 3.9)")
class TestAttachIntegration(unittest.TestCase):
    """End-to-end against a real stdio MCP server subprocess (exercises the
    async->sync bridge AND subprocess teardown, the risky surface)."""

    def _specs(self):
        return [("fake", {"command": sys.executable, "args": [_FAKE_SERVER]})]

    def test_streamable_http_uses_v2_client_and_preserves_headers(self):
        class AsyncCM:
            def __init__(self, value):
                self.value = value

            async def __aenter__(self):
                return self.value

            async def __aexit__(self, exc_type, exc, tb):
                return False

        async def exercise():
            bridge = mc._Bridge([], connect_timeout=20, call_timeout=20)
            http_client = object()
            with mock.patch(
                "mcp.client.streamable_http.create_mcp_http_client",
                return_value=AsyncCM(http_client),
            ) as create_client, mock.patch(
                "mcp.client.streamable_http.streamable_http_client",
                return_value=AsyncCM(("read", "write")),
            ) as connect:
                async with contextlib.AsyncExitStack() as stack:
                    streams = await bridge._open_transport(
                        stack,
                        {
                            "url": "https://mcp.example.test/rpc",
                            "headers": {"Authorization": "Bearer fake-token"},
                        },
                    )
                create_client.assert_called_once_with(
                    {"Authorization": "Bearer fake-token"}
                )
                connect.assert_called_once_with(
                    "https://mcp.example.test/rpc", http_client=http_client
                )
                return streams

        self.assertEqual(asyncio.run(exercise()), ("read", "write"))

    def test_lists_namespaces_and_calls(self):
        with mc.attach(self._specs(), connect_timeout=20, call_timeout=20) as tools:
            disp = {t.name: t for t in tools}
            self.assertEqual(set(disp), {"fake__echo", "fake__write_note"})

            # read-only tool: paid=False, runs immediately
            echo = disp["fake__echo"]
            self.assertFalse(echo.paid)
            res = echo.invoke({"text": "hi"}, confirm=False)
            self.assertEqual(res["status"], "ok")
            self.assertIn("echo: hi", res["content"])

            # side-effecting tool: paid=True, gated without confirm, runs with it
            note = disp["fake__write_note"]
            self.assertTrue(note.paid)
            gated = note.invoke({"note": "n"}, confirm=False)
            self.assertEqual(gated["status"], "confirmation_required")
            ran = note.invoke({"note": "n"}, confirm=True)
            self.assertEqual(ran["status"], "ok")
            self.assertIn("wrote: n", ran["content"])

    def test_bad_command_raises_and_cleans_up(self):
        specs = [("bad", {"command": "venice-no-such-binary-xyz", "args": []})]
        with self.assertRaises(Exception):
            with mc.attach(specs, connect_timeout=5, call_timeout=5):
                self.fail("attach() should have raised during setup")


if __name__ == "__main__":
    unittest.main()
