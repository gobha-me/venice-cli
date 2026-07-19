"""Tests for the external MCP client (`venice chat --mcp`, issue #21).

Three layers:
- Pure helpers (spec resolution, name namespacing, result translation, side-effect
  classification) run everywhere -- they never import the `mcp` SDK.
- An import-purity test proves the module has no module-scope `import mcp` (the CI
  `package` job asserts a base install stays mcp-free).
- A real end-to-end `attach()` test spawns a tiny stdio MCP server subprocess; it is
  skipped unless the `mcp` SDK is importable (absent on Python 3.9).
"""
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
    def __init__(self, readOnlyHint=None):
        self.readOnlyHint = readOnlyHint


class _Block:
    def __init__(self, text=None, type=None):
        self.text = text
        self.type = type


class _CallResult:
    def __init__(self, content=None, isError=False, structuredContent=None):
        self.content = content or []
        self.isError = isError
        self.structuredContent = structuredContent


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
        self.assertTrue(mc._is_side_effecting(_Ann(readOnlyHint=None)))
        self.assertTrue(mc._is_side_effecting(_Ann(readOnlyHint=False)))
        self.assertFalse(mc._is_side_effecting(_Ann(readOnlyHint=True)))

    def test_translate_ok_joins_text(self):
        r = mc._translate_result(_CallResult([_Block(text="hi"), _Block(text="yo")]))
        self.assertEqual(r, {"status": "ok", "content": "hi\nyo"})

    def test_translate_marks_non_text(self):
        r = mc._translate_result(_CallResult([_Block(type="image")]))
        self.assertEqual(r["status"], "ok")
        self.assertIn("non-text content: image", r["content"])

    def test_translate_error(self):
        r = mc._translate_result(_CallResult([_Block(text="boom")], isError=True))
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["message"], "boom")

    def test_translate_carries_structured(self):
        r = mc._translate_result(
            _CallResult([_Block(text="x")], structuredContent={"a": 1})
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
