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


# Auto-save is on by default (#47): keep this module hermetic (belt-and-suspenders
# on top of the per-run redirects) so no code._run ever writes to the real home.
_SESSIONS_TMP = None


def setUpModule():
    global _SESSIONS_TMP
    _SESSIONS_TMP = tempfile.mkdtemp()
    os.environ["VENICE_SESSIONS_DIR"] = _SESSIONS_TMP
    # #49: also redirect the global memory tier (see test_repl for the rationale).
    os.environ["VENICE_MEMORY_DIR"] = os.path.join(_SESSIONS_TMP, "memory")


def tearDownModule():
    os.environ.pop("VENICE_SESSIONS_DIR", None)
    os.environ.pop("VENICE_MEMORY_DIR", None)
    if _SESSIONS_TMP:
        __import__("shutil").rmtree(_SESSIONS_TMP, ignore_errors=True)


def _code_args(**ov):
    base = dict(
        task=None, root=None, model=None, system=None, temperature=None,
        max_tokens=None, json=False, auto=None, manual=None, yes=None,
        plan_only=False, no_plan=False, no_verify=False, max_tool_calls=None,
        exec_timeout=None, interactive=False, resume=None, assets=None,
        auto_compact=None, compact_threshold=None, compact_keep_turns=None,
        session_max_spend=None, cont=None, ephemeral=None,
    )
    base.update(ov)
    return argparse.Namespace(**base)


def _write_call(cid, path, content):
    return _FnCall(cid, "write_file",
                   json.dumps({"path": path, "content": content}))


def _mem_call(cid, name, content, scope="project"):
    return _FnCall(cid, "memory_write",
                   json.dumps({"name": name, "content": content, "scope": scope}))


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
        self._sess_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._sess_dir, ignore_errors=True))
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake",
                                          "VENICE_SESSIONS_DIR": self._sess_dir}), \
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

    # --- #76: cross-repo write protection wired through code._run ---
    def _sibling(self):
        other = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(other, ignore_errors=True))
        return other

    def test_write_outside_root_blocked_without_allow_root(self):
        other = self._sibling()
        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion(tool_calls=[
                _write_call("c1", os.path.join(other, "x.txt"), "hi")]),
            FakeToolCompletion("wrote it"),               # model's (wrong) final
            FakeToolCompletion("ACCEPTANCE: PASS"),
        ]
        rc, _calls = self._run(
            _code_args(task="write x", root=self.root, auto=True), seq)
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(os.path.join(other, "x.txt")))  # guard held

    def test_allow_root_flag_enables_cross_repo_write(self):
        other = self._sibling()
        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion(tool_calls=[
                _write_call("c1", os.path.join(other, "x.txt"), "hi")]),
            FakeToolCompletion("wrote it"),
            FakeToolCompletion("ACCEPTANCE: PASS"),
        ]
        rc, _calls = self._run(
            _code_args(task="write x", root=self.root, auto=True,
                       allow_root=[other]), seq)
        self.assertEqual(rc, 0)
        with open(os.path.join(other, "x.txt")) as f:
            self.assertEqual(f.read(), "hi")

    # --- --memory surfaces the memory/task rails and persists a note (#49) ---
    def test_memory_flag_writes_a_note(self):
        # `venice code` runs with cwd == root in practice; the memory project tier
        # discovers from cwd, so chdir into root for the run (and to keep the test's
        # .venice/ out of the repo).
        seq = [
            FakeToolCompletion("plan: remember the convention"),
            FakeToolCompletion(tool_calls=[
                _mem_call("c1", "conv", "use tabs", "project")]),
            FakeToolCompletion("done -- remembered"),
            FakeToolCompletion("- works: MET\nACCEPTANCE: PASS"),
        ]
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            rc, calls = self._run(
                _code_args(task="remember", root=self.root, auto=True, memory=True), seq)
        finally:
            os.chdir(cwd)
        self.assertEqual(rc, 0)
        store = os.path.join(self.root, ".venice", "memory", "memory.json")
        self.assertTrue(os.path.exists(store))
        with open(store) as f:
            self.assertEqual(json.load(f)["entries"]["conv"]["content"], "use tabs")

    def test_no_memory_flag_omits_the_tools(self):
        # Without --memory the model calling memory_write is an unknown tool -> the
        # loop returns a tool error, but no store is created.
        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion(tool_calls=[_mem_call("c1", "x", "y")]),
            FakeToolCompletion("done"),
            FakeToolCompletion("ACCEPTANCE: PASS"),
        ]
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            rc, calls = self._run(
                _code_args(task="x", root=self.root, auto=True), seq)
        finally:
            os.chdir(cwd)
        self.assertFalse(os.path.exists(os.path.join(self.root, ".venice", "memory")))

    # --- auto-compact (#48) ---
    def test_auto_compact_compacts_during_execute(self):
        # Plan turn, then several over-budget tool rounds so the history grows
        # past keep_turns -- the loop fires one tool-free summarization turn.
        # Post-compaction turns report small usage (the history is now short),
        # so compaction doesn't re-fire.
        over = {"prompt_tokens": 9000, "completion_tokens": 5, "total_tokens": 9005}
        under = {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}
        seq = [FakeToolCompletion("plan: read a file")]                # plan
        for i in range(4):                                             # exec rounds
            seq.append(FakeToolCompletion(
                tool_calls=[_FnCall(f"c{i}", "read_file", '{"path":"x"}')],
                usage=over))
        seq += [
            FakeToolCompletion("summary so far"),                      # compact turn
            FakeToolCompletion("done", usage=under),                   # exec final
            FakeToolCompletion("- works: MET\nACCEPTANCE: PASS"),      # verify
        ]
        rc, calls = self._run(
            _code_args(task="read x", root=self.root, auto=True,
                       auto_compact=True, compact_threshold=1000,
                       compact_keep_turns=1),
            seq)
        self.assertEqual(rc, 0)
        # At least one tool-free, tool_choice="none" summarization turn fired
        # (the plan and verify turns pass tools; only the compact turn omits
        # them). With several over-budget rounds it can fire more than once as
        # the history re-grows past the threshold -- that's expected.
        summary_turns = [c for c in calls
                         if c.get("tool_choice") == "none" and "tools" not in c]
        self.assertGreaterEqual(len(summary_turns), 1)
        # The summarization turn is self-contained: instruction system + the
        # flattened transcript, no tools, no tool_choice other than "none".
        st = summary_turns[0]
        self.assertEqual(st["messages"][0]["role"], "system")
        self.assertEqual(len(st["messages"]), 2)  # instruction + transcript only

    def test_auto_compact_off_by_default_no_compact_call(self):
        usage = {"prompt_tokens": 999999, "completion_tokens": 1,
                 "total_tokens": 1000000}
        seq = [FakeToolCompletion("plan")]
        for i in range(4):
            seq.append(FakeToolCompletion(
                tool_calls=[_FnCall(f"c{i}", "read_file", '{"path":"x"}')],
                usage=usage))
        seq += [FakeToolCompletion("done"), FakeToolCompletion("ACCEPTANCE: PASS")]
        rc, calls = self._run(
            _code_args(task="x", root=self.root, auto=True), seq)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 7)  # plan + 4 rounds + final + verify; no compact

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
        self.assertIn("venice_video", names)
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

    def test_video_asset_dispatches_with_confirm_under_auto(self):
        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion(tool_calls=[
                _FnCall("c1", "venice_video",
                        json.dumps({"prompt": "a koi pond at dawn"}))]),
            FakeToolCompletion("made the clip"),
            FakeToolCompletion("ACCEPTANCE: PASS"),
        ]
        with mock.patch(
            "venice.commands._mcp.video_tool",
            return_value={"status": "ok", "path": "/x.mp4", "bytes": 1},
        ) as stub:
            rc, calls = self._run(
                _code_args(task="film", root=self.root, auto=True, assets=True), seq)
        self.assertEqual(rc, 0)
        self.assertEqual(stub.call_count, 1)
        # --auto -> confirm=True bypasses the spend gate
        self.assertTrue(stub.call_args.kwargs.get("confirm"))
        self.assertEqual(stub.call_args.kwargs.get("prompt"), "a koi pond at dawn")

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
                           label="venice chat", max_tool_calls=8, session=None,
                           ephemeral=False, root=None, system_reseed=False):
            captured["tools_session"] = tools_session
            captured["label"] = label
            captured["initial"] = initial
            captured["max_tool_calls"] = max_tool_calls
            captured["root"] = root
            captured["system_reseed"] = system_reseed
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
        # code -i gets its higher default budget (25), not the chat REPL's 8
        self.assertEqual(captured["max_tool_calls"], code._DEFAULT_MAX_TOOL_CALLS)
        self.assertTrue(captured["system_reseed"])       # code always reseeds (#47)
        self.assertEqual(captured["root"], self.root)

    # --- session resume (#47) ---
    def _mk_zone(self):
        from venice.commands import _session
        zone = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(zone, ignore_errors=True))
        return zone, _session

    def _run_interactive(self, args, seq, inputs, zone):
        from venice.commands import code
        fake, calls = _fake_openai_seq(seq)
        stdin = mock.MagicMock()
        stdin.isatty.return_value = True
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake",
                                          "VENICE_SESSIONS_DIR": zone}), \
             mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch("builtins.input", side_effect=inputs), \
             mock.patch.object(sys, "stdin", stdin), \
             mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = code._run(args)
        return rc, calls

    def test_resume_rebuilds_system_prompt_against_new_root(self):
        zone, _session = self._mk_zone()
        stale = _session.new_session(
            "code", label="venice code", model="llama-3.3-70b",
            root="/nonexistent/oldroot",
            messages=[{"role": "system", "content": "STALE root=/nonexistent/oldroot"},
                      {"role": "user", "content": "prev"},
                      {"role": "assistant", "content": "ok"}],
        )
        with mock.patch.dict(os.environ, {"VENICE_SESSIONS_DIR": zone}):
            _session.save(stale)
        # Resume by id with an explicit --root: the leading system message must be
        # rebuilt against the NEW root, not the persisted stale one.
        rc, calls = self._run_interactive(
            _code_args(resume=stale.id, root=self.root, auto=True),
            [FakeToolCompletion("done")],           # one turn, no tool calls -> ends
            ["carry on", "/exit"], zone,
        )
        self.assertEqual(rc, 0)
        sysmsg = calls[0]["messages"][0]
        self.assertEqual(sysmsg["role"], "system")
        self.assertIn(self.root, sysmsg["content"])
        self.assertNotIn("/nonexistent/oldroot", sysmsg["content"])

    def test_resume_restores_saved_root_when_no_root_flag(self):
        zone, _session = self._mk_zone()
        saved_root = os.path.realpath(self.root)
        sess = _session.new_session(
            "code", label="venice code", model="llama-3.3-70b", root=saved_root,
            messages=[{"role": "user", "content": "prev"}],
        )
        with mock.patch.dict(os.environ, {"VENICE_SESSIONS_DIR": zone}):
            _session.save(sess)

        captured = {}

        def _fake_repl_run(a, *rest, root=None, session=None, **kw):
            captured["root"] = root
            captured["session_id"] = session.id if session else None
            return 0

        from venice.commands import code
        fake, _c = _fake_openai_seq([])
        stdin = mock.MagicMock()
        stdin.isatty.return_value = True
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake",
                                          "VENICE_SESSIONS_DIR": zone}), \
             mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch("venice.commands._repl.run", _fake_repl_run), \
             mock.patch.object(sys, "stdin", stdin), \
             mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = code._run(_code_args(cont=True))       # --continue, no --root
        self.assertEqual(rc, 0)
        self.assertEqual(captured["root"], saved_root)   # faithful restore
        self.assertEqual(captured["session_id"], sess.id)


class TestOneShotSteering(unittest.TestCase):
    """#78: one-shot `venice code` runs persist a steerable session + drain steers.

    Standalone harness (not a `TestCodeCommand` subclass, to avoid re-running its
    whole suite under this name).
    """

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

    def _run(self, args, seq):
        from venice.commands import code
        fake, calls = _fake_openai_seq(seq)
        stdin = mock.MagicMock()
        stdin.isatty.return_value = False  # one-shot, non-interactive
        self._sess_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._sess_dir, ignore_errors=True))
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake",
                                          "VENICE_SESSIONS_DIR": self._sess_dir}), \
             mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.object(sys, "stdin", stdin), \
             mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = code._run(args)
        return rc, calls

    def _sessions(self):
        """The persisted session files in this run's redirected zone (post-run)."""
        import glob
        return sorted(glob.glob(os.path.join(self._sess_dir, "*.json")))

    def test_auto_run_persists_a_code_session(self):
        seq = [
            FakeToolCompletion("plan: write hello"),
            FakeToolCompletion("done -- nothing to do"),
            FakeToolCompletion("- works: MET\nACCEPTANCE: PASS"),
        ]
        rc, calls = self._run(_code_args(task="do x", root=self.root, auto=True), seq)
        self.assertEqual(rc, 0)
        files = self._sessions()
        self.assertEqual(len(files), 1)                 # one-shot now leaves a session
        with open(files[0]) as f:
            doc = json.loads(f.read())
        self.assertEqual(doc["command"], "code")
        self.assertEqual(doc["root"], self.root)
        # the transcript was persisted (system + user task + assistant turns)
        self.assertTrue(any(m.get("role") == "assistant" for m in doc["messages"]))

    def test_ephemeral_run_persists_nothing(self):
        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion("done"),
            FakeToolCompletion("ACCEPTANCE: PASS"),
        ]
        rc, calls = self._run(
            _code_args(task="do x", root=self.root, auto=True, ephemeral=True), seq)
        self.assertEqual(rc, 0)
        self.assertEqual(self._sessions(), [])          # --ephemeral opts out

    def test_steer_deposited_at_execute_is_consumed(self):
        # A steer queued the instant the run becomes steerable (its session is first
        # saved, at Execute) must be drained at the execute loop's first checkpoint and
        # land in the persisted transcript -- proving the end-to-end wiring.
        from venice.commands import _session, _mailbox
        real_save = _session.save

        def _save_then_steer(sess):
            path = real_save(sess)
            if sess.command == "code" and not getattr(_save_then_steer, "fired", False):
                _save_then_steer.fired = True
                _mailbox.deposit(sess.id, "ALSO: update the changelog")
            return path

        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion("done -- and I saw the steer"),  # execute final
            FakeToolCompletion("ACCEPTANCE: PASS"),
        ]
        with mock.patch.object(_session, "save", _save_then_steer):
            rc, calls = self._run(_code_args(task="do x", root=self.root, auto=True), seq)
        self.assertEqual(rc, 0)
        with open(self._sessions()[0]) as f:
            doc = json.loads(f.read())
        steers = [m for m in doc["messages"] if m.get("role") == "user"
                  and "steering message received mid-run" in m.get("content", "")]
        self.assertEqual(len(steers), 1)
        self.assertIn("update the changelog", steers[0]["content"])


if __name__ == "__main__":
    unittest.main()
