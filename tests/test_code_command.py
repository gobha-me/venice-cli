"""Unit tests for the `venice code` command harness (#30).

Drives `code._run` end-to-end with a faked OpenAI client and the free /models
catalog GET mocked (via urlopen) -- no network, no real key. Reuses the tool-call
fakes from `test_chat`. File writes land in a per-test tmpdir project root.
"""
import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from tests.test_chat import (
    FakeToolCompletion, _FnCall, _fake_openai_seq, _urlopen_ok,
)


def _code_args(**ov):
    base = dict(
        task=None, root=None, model=None, system=None, temperature=None,
        max_tokens=None, json=False, auto=None, manual=None, yes=None,
        plan_only=False, no_plan=False, no_verify=False, max_tool_calls=None,
        exec_timeout=None, interactive=False, resume=None, assets=None,
    )
    base.update(ov)
    return argparse.Namespace(**base)


def _write_call(cid, path, content):
    return _FnCall(cid, "write_file",
                   json.dumps({"path": path, "content": content}))


class TestCodeCommand(unittest.TestCase):
    def setUp(self):
        _cfg = mock.patch(
            "venice.userconfig.load_config",
            lambda *a, **k: {"version": 1, "mcpServers": {}, "defaults": {}},
        )
        _cfg.start()
        self.addCleanup(_cfg.stop)
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.realpath(self.tmp)
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def _run(self, args, seq, urlopen=None, stdout=None, stderr=None):
        from venice.commands import code
        fake, calls = _fake_openai_seq(seq)
        stdin = mock.MagicMock()
        stdin.isatty.return_value = False  # one-shot, non-interactive
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        urlopen or _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.object(sys, "stdin", stdin), \
             mock.patch.object(sys, "stdout", stdout or io.StringIO()), \
             mock.patch.object(sys, "stderr", stderr or io.StringIO()):
            rc = code._run(args)
        return rc, calls

    # --- plan-only ---
    def test_plan_only_prints_and_exits_without_executing(self):
        out = io.StringIO()
        seq = [FakeToolCompletion("1. do it\nAcceptance criteria:\n- works")]
        rc, calls = self._run(
            _code_args(task="do x", root=self.root, plan_only=True), seq, stdout=out)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)                 # only the plan turn
        self.assertEqual(calls[0]["tool_choice"], "none")
        self.assertIn("do it", out.getvalue())

    # --- autonomous happy path with a real file write ---
    def test_auto_executes_and_writes_file(self):
        seq = [
            FakeToolCompletion("plan: write hello"),                     # plan (none)
            FakeToolCompletion(tool_calls=[
                _write_call("c1", "hello.py", "def hi():\n    return 1\n")]),
            FakeToolCompletion("done -- wrote hello.py"),               # exec final
            FakeToolCompletion("- works: MET\nACCEPTANCE: PASS"),        # verify (none)
        ]
        rc, calls = self._run(
            _code_args(task="add hello", root=self.root, auto=True), seq)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0]["tool_choice"], "none")   # plan
        self.assertEqual(calls[1]["tool_choice"], "auto")   # execute loop
        self.assertEqual(calls[3]["tool_choice"], "none")   # acceptance check
        # auto -> confirm=True -> the write actually happened
        with open(os.path.join(self.root, "hello.py")) as f:
            self.assertEqual(f.read(), "def hi():\n    return 1\n")

    # --- --assets exposes the in-process asset tools ---
    def _exec_tool_names(self, calls):
        # the execute turn is the one advertising tools with tool_choice="auto"
        execs = [c for c in calls if c.get("tool_choice") == "auto"]
        self.assertTrue(execs, "no execute turn recorded")
        return {t["function"]["name"] for t in execs[0]["tools"]}

    def test_assets_flag_exposes_asset_tools(self):
        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion("nothing to do"),           # execute (auto), no calls
            FakeToolCompletion("ACCEPTANCE: PASS"),
        ]
        rc, calls = self._run(
            _code_args(task="draw", root=self.root, auto=True, assets=True), seq)
        self.assertEqual(rc, 0)
        names = self._exec_tool_names(calls)
        self.assertIn("venice_image", names)
        self.assertIn("venice_image_edit", names)
        self.assertNotIn("venice_chat", names)   # excluded by design

    def test_assets_absent_by_default(self):
        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion("nothing to do"),
            FakeToolCompletion("ACCEPTANCE: PASS"),
        ]
        rc, calls = self._run(
            _code_args(task="x", root=self.root, auto=True), seq)
        self.assertEqual(rc, 0)
        self.assertNotIn("venice_image", self._exec_tool_names(calls))

    def test_assets_tool_dispatches_with_confirm_under_auto(self):
        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion(tool_calls=[
                _FnCall("c1", "venice_image",
                        json.dumps({"prompt": "a hero sprite"}))]),
            FakeToolCompletion("made the sprite"),
            FakeToolCompletion("ACCEPTANCE: PASS"),
        ]
        with mock.patch(
            "venice.commands._mcp.image_tool",
            return_value={"status": "ok", "paths": ["/x.png"], "count": 1},
        ) as stub:
            rc, calls = self._run(
                _code_args(task="draw", root=self.root, auto=True, assets=True), seq)
        self.assertEqual(rc, 0)
        self.assertEqual(stub.call_count, 1)
        # --auto -> confirm=True bypasses the spend gate
        self.assertTrue(stub.call_args.kwargs.get("confirm"))
        # the model supplied only prompt; control kwargs are injected, not from args
        self.assertEqual(stub.call_args.kwargs.get("prompt"), "a hero sprite")

    def test_acceptance_fail_returns_1(self):
        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion("did nothing useful"),
            FakeToolCompletion("- works: NOT MET\nACCEPTANCE: FAIL"),
        ]
        rc, _calls = self._run(
            _code_args(task="x", root=self.root, auto=True), seq)
        self.assertEqual(rc, 1)

    # --- #37: the verdict parse is case/format-tolerant (no false-fail) ---
    def test_acceptance_pass_loose_parse(self):
        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion("did the work"),
            FakeToolCompletion("- works: MET\n**acceptance: pass**"),  # lower + markdown
        ]
        rc, calls = self._run(
            _code_args(task="x", root=self.root, auto=True), seq)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 3)                 # no re-prompt fired

    # --- #37: unparseable verdict -> re-prompt once -> recovers to exit 0 ---
    def test_acceptance_unknown_reprompt_recovers(self):
        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion("did the work"),
            FakeToolCompletion("All acceptance criteria are met."),    # no sentinel
            FakeToolCompletion("ACCEPTANCE: PASS"),                    # re-prompt reply
        ]
        rc, calls = self._run(
            _code_args(task="x", root=self.root, auto=True), seq)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 4)                 # extra re-prompt turn
        self.assertEqual(calls[3]["tool_choice"], "none")

    # --- #37: still no verdict after the re-prompt -> exit 10 + warning ---
    def test_acceptance_unknown_persists_exits_10(self):
        err = io.StringIO()
        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion("did the work"),
            FakeToolCompletion("All criteria met."),                   # no sentinel
            FakeToolCompletion("Looks good."),                         # still none
        ]
        rc, calls = self._run(
            _code_args(task="x", root=self.root, auto=True), seq, stderr=err)
        self.assertEqual(rc, 10)
        self.assertEqual(len(calls), 4)
        self.assertIn("exiting 10", err.getvalue())

    def test_json_verdict_unknown(self):
        out = io.StringIO()
        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion("did the work"),
            FakeToolCompletion("All criteria met."),
            FakeToolCompletion("Still looks good."),
        ]
        rc, _calls = self._run(
            _code_args(task="x", root=self.root, auto=True, json=True), seq, stdout=out)
        self.assertEqual(rc, 10)
        env = json.loads(out.getvalue())
        self.assertEqual(env["acceptance"]["verdict"], "unknown")
        self.assertIsNone(env["acceptance"]["passed"])

    # --- fail-safe: non-TTY without --auto aborts before any model call ---
    def test_non_tty_without_auto_aborts(self):
        err = io.StringIO()
        rc, calls = self._run(
            _code_args(task="x", root=self.root), [], stderr=err)
        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 0)                 # fail fast, no plan turn
        self.assertIn("--auto", err.getvalue())

    # --- capability guard: non-tool-calling model errors out ---
    def test_model_without_function_calling_errors(self):
        err = io.StringIO()
        rc, calls = self._run(
            _code_args(task="x", root=self.root, auto=True), [],
            urlopen=_urlopen_ok(fc=False), stderr=err)
        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 0)
        self.assertIn("does not support function calling", err.getvalue())

    # --- JSON envelope ---
    def test_json_envelope(self):
        out = io.StringIO()
        seq = [
            FakeToolCompletion("plan text"),
            FakeToolCompletion(tool_calls=[_write_call("c1", "n.py", "x=1\n")]),
            FakeToolCompletion("wrote n.py"),
            FakeToolCompletion("ACCEPTANCE: PASS"),
        ]
        rc, _calls = self._run(
            _code_args(task="x", root=self.root, auto=True, json=True), seq, stdout=out)
        self.assertEqual(rc, 0)
        env = json.loads(out.getvalue())
        self.assertEqual(env["mode"], "auto")
        self.assertEqual(env["plan"], "plan text")
        self.assertIn("wrote n.py", env["final"])
        self.assertTrue(env["acceptance"]["passed"])
        self.assertEqual(env["acceptance"]["verdict"], "pass")
        self.assertEqual(env["root"], self.root)

    # --- --no-plan skips plan + verify ---
    def test_no_plan_executes_directly(self):
        seq = [FakeToolCompletion("did it directly")]
        rc, calls = self._run(
            _code_args(task="x", root=self.root, auto=True, no_plan=True), seq)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)                 # no plan, no verify turn
        self.assertEqual(calls[0]["tool_choice"], "auto")

    def test_no_plan_with_plan_only_is_error(self):
        rc, calls = self._run(
            _code_args(task="x", root=self.root, no_plan=True, plan_only=True), [])
        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 0)

    # --- interactive routes to the REPL with an injected coding tools session ---
    def test_interactive_delegates_to_repl_with_tools_session(self):
        captured = {}

        def _fake_repl_run(args, oai, openai, client, models, model,
                           initial=None, *, tools_session=None, gen_kwargs=None,
                           label="venice chat"):
            captured["tools_session"] = tools_session
            captured["label"] = label
            captured["initial"] = initial
            return 0

        stdin = mock.MagicMock()
        stdin.isatty.return_value = True
        from venice.commands import code
        fake, _calls = _fake_openai_seq([])
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch("venice.commands._repl.run", _fake_repl_run), \
             mock.patch.object(sys, "stdin", stdin), \
             mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = code._run(_code_args(task="hi", root=self.root, interactive=True))
        self.assertEqual(rc, 0)
        self.assertIsNotNone(captured["tools_session"])
        self.assertEqual(captured["label"], "venice code")
        self.assertEqual(captured["initial"], "hi")


if __name__ == "__main__":
    unittest.main()
