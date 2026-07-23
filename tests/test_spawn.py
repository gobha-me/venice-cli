"""Unit tests for the scout subagent -- epic #52 slice 1 (context firewall).

Covers `_agent.run_scout` (the disposable read-only core), `_code.read_only_tools`
(the inner toolset + read-only/no-self-spawn invariant), `_code.scout_tool` (the
`venice_scout` Tool wrapper: budget clamp + error envelope), and the `venice code
--scout` wiring. Reuses the tool-call fakes from `test_chat`; no network, no real key.
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

from venice.commands import _agent, _code


# Belt-and-suspenders (mirrors test_code_command): redirect the session + global
# memory tiers to a tmp dir so nothing here can write to the real home.
_TMP = None


def setUpModule():
    global _TMP
    _TMP = tempfile.mkdtemp()
    os.environ["VENICE_SESSIONS_DIR"] = _TMP
    os.environ["VENICE_MEMORY_DIR"] = os.path.join(_TMP, "memory")


def tearDownModule():
    os.environ.pop("VENICE_SESSIONS_DIR", None)
    os.environ.pop("VENICE_MEMORY_DIR", None)
    if _TMP:
        __import__("shutil").rmtree(_TMP, ignore_errors=True)


def _read_call(cid, path="a.py"):
    return _FnCall(cid, "read_file", json.dumps({"path": path}))


def _grep_call(cid, pattern="x"):
    return _FnCall(cid, "grep", json.dumps({"pattern": pattern}))


_REPORT = (
    "FINDINGS: read_file is defined in _code.py.\n"
    "CONFIDENCE: high -- read the source.\n"
    "DEAD-ENDS: none.\n"
    "NOT CHECKED: the tests.\n"
    "VERIFIED-LIVE vs HYPOTHETICAL: live (read the file)."
)


class _ScoutBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.realpath(self.tmp)
        self.addCleanup(
            lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        with open(os.path.join(self.root, "a.py"), "w") as fh:
            fh.write("def f():\n    return 1\n")


class TestRunScout(_ScoutBase):
    def test_returns_structured_report_and_firewalls_stdout(self):
        seq = [
            FakeToolCompletion(tool_calls=[_read_call("c1")]),
            FakeToolCompletion(_REPORT),  # final answer (no tool_calls)
        ]
        fake, calls = _fake_openai_seq(seq)
        tools = _code.read_only_tools(self.root)  # client=None -> no project_search

        outer = io.StringIO()
        with mock.patch.object(sys, "stdout", outer):
            out = _agent.run_scout(
                fake, "m", "where is read_file defined?", tools, {},
                max_tool_calls=6)

        self.assertEqual(out["status"], "ok")
        for marker in ("FINDINGS:", "CONFIDENCE:", "DEAD-ENDS:",
                       "NOT CHECKED:", "VERIFIED-LIVE"):
            self.assertIn(marker, out["report"])
        self.assertEqual(out["tool_calls"], 1)
        self.assertFalse(out["truncated"])
        # Firewall: the nested report never reaches the caller's stdout.
        self.assertEqual(outer.getvalue(), "")

    def test_fresh_context_only_task_and_system(self):
        # The subagent must start from a clean slate -- system + user(task) only.
        seq = [FakeToolCompletion(_REPORT)]
        fake, calls = _fake_openai_seq(seq)
        _agent.run_scout(fake, "m", "investigate", _code.read_only_tools(self.root),
                         {}, max_tool_calls=4)
        first_msgs = calls[0]["messages"]
        self.assertEqual(len(first_msgs), 2)
        self.assertEqual(first_msgs[0]["role"], "system")
        self.assertIn("SCOUT", first_msgs[0]["content"])
        self.assertEqual(first_msgs[1], {"role": "user", "content": "investigate"})

    def test_focus_hint_threaded_into_system(self):
        seq = [FakeToolCompletion(_REPORT)]
        fake, calls = _fake_openai_seq(seq)
        _agent.run_scout(fake, "m", "t", _code.read_only_tools(self.root), {},
                         max_tool_calls=4, focus="src/venice/commands/_repl.py")
        self.assertIn("src/venice/commands/_repl.py",
                      calls[0]["messages"][0]["content"])

    def test_empty_task_short_circuits(self):
        fake, calls = _fake_openai_seq([])
        out = _agent.run_scout(fake, "m", "   ", _code.read_only_tools(self.root),
                               {}, max_tool_calls=4)
        self.assertEqual(out["status"], "error")
        self.assertEqual(calls, [])  # never called the model

    def test_budget_cap_forces_final_and_marks_truncated(self):
        # One turn asks for TWO tool calls but the cap is 1: the first runs, the
        # second is reported not-executed, and a tool_choice="none" turn wraps up.
        seq = [
            FakeToolCompletion(tool_calls=[_read_call("c1"), _grep_call("c2")]),
            FakeToolCompletion("partial report"),  # the forced final
        ]
        fake, calls = _fake_openai_seq(seq)
        with mock.patch.object(sys, "stderr", io.StringIO()):  # swallow cap notice
            out = _agent.run_scout(fake, "m", "t", _code.read_only_tools(self.root),
                                   {}, max_tool_calls=1)
        self.assertTrue(out["truncated"])
        self.assertEqual(out["report"], "partial report")
        self.assertEqual(out["tool_calls"], 1)
        self.assertEqual(calls[-1]["tool_choice"], "none")  # _force_final fired

    def test_recursion_guard_rejects_paid_or_scout_tools(self):
        schema = {"type": "object", "properties": {}}
        paid = _agent.Tool("write_file", "d", schema,
                           lambda a, *, confirm=False: {"status": "ok"},
                           paid=True, category="fs", tags=("write",))
        scout_named = _agent.Tool(_agent.SCOUT_TOOL_NAME, "d", schema,
                                  lambda a, *, confirm=False: {"status": "ok"},
                                  paid=False)
        with self.assertRaises(ValueError):
            _agent.run_scout(None, "m", "t", [paid], {}, max_tool_calls=3)
        with self.assertRaises(ValueError):
            _agent.run_scout(None, "m", "t", [scout_named], {}, max_tool_calls=3)


class TestReadOnlyTools(_ScoutBase):
    def test_inner_toolset_is_read_only_and_scout_free(self):
        names = {t.name for t in _code.read_only_tools(self.root)}
        self.assertLessEqual({"read_file", "list_dir", "grep", "git"}, names)
        self.assertTrue(
            names.isdisjoint({"write_file", "edit_file", "apply_patch",
                              "run", "reindex"}))
        self.assertNotIn(_agent.SCOUT_TOOL_NAME, names)

    def test_all_read_only_tools_are_free_and_read_tagged(self):
        for t in _code.read_only_tools(self.root):
            self.assertFalse(t.paid, f"{t.name} must be free")
            self.assertIn("read", t.tags, f"{t.name} must be read-tagged")
            self.assertNotIn("write", t.tags)
            self.assertNotIn("exec", t.tags)
            self.assertNotIn("mutate", t.tags)


class TestScoutTool(_ScoutBase):
    def test_builds_one_read_only_agent_tool(self):
        st = _code.scout_tool(None, "m", self.root, None, {})
        self.assertEqual(st.name, _agent.SCOUT_TOOL_NAME)
        self.assertFalse(st.paid)
        self.assertEqual(st.category, "agent")
        self.assertIn("read", st.tags)

    def test_not_a_code_tools_rail(self):
        # scout must never leak into the default coding toolset (only via --scout).
        names = {t.name for t in _code.code_tools(self.root, None)}
        self.assertNotIn(_agent.SCOUT_TOOL_NAME, names)

    def test_max_tool_calls_clamped_to_range(self):
        rec = {}

        def _rec_run_scout(oai, model, task, tools, base_kwargs, *,
                           max_tool_calls, **kw):
            rec["n"] = max_tool_calls
            return {"status": "ok", "report": "r", "tool_calls": 0,
                    "truncated": False}

        with mock.patch.object(_agent, "run_scout", _rec_run_scout):
            st = _code.scout_tool(None, "m", self.root, None, {})
            st.invoke({"task": "x", "max_tool_calls": 999})
            self.assertEqual(rec["n"], _code._SCOUT_HARD_CAP)
            st.invoke({"task": "x", "max_tool_calls": 0})
            self.assertEqual(rec["n"], _code._SCOUT_MAX_TOOL_CALLS)
            st.invoke({"task": "x", "max_tool_calls": -3})
            self.assertEqual(rec["n"], _code._SCOUT_MAX_TOOL_CALLS)
            st.invoke({"task": "x"})
            self.assertEqual(rec["n"], _code._SCOUT_MAX_TOOL_CALLS)

    def test_missing_task_is_error_envelope(self):
        st = _code.scout_tool(None, "m", self.root, None, {})
        out = st.invoke({"focus": "somewhere"})
        self.assertEqual(out["status"], "error")

    def test_nested_failure_becomes_error_envelope(self):
        fake = mock.MagicMock()
        fake.chat.completions.create.side_effect = RuntimeError("boom")
        st = _code.scout_tool(fake, "m", self.root, None, {})
        out = st.invoke({"task": "investigate"})
        self.assertEqual(out["status"], "error")
        self.assertIn("scout failed", out["message"])


def _code_args(**ov):
    base = dict(
        task=None, root=None, model=None, system=None, temperature=None,
        max_tokens=None, json=False, auto=None, manual=None, yes=None,
        plan_only=False, no_plan=False, no_verify=False, max_tool_calls=None,
        exec_timeout=None, interactive=False, resume=None, assets=None,
        scout=None, auto_compact=None, compact_threshold=None,
        compact_keep_turns=None, session_max_spend=None, cont=None, ephemeral=None,
    )
    base.update(ov)
    return argparse.Namespace(**base)


class TestScoutWiring(_ScoutBase):
    """`venice code --scout` folds the tool in and advertises it in the prompt."""

    def _run(self, args, seq, stdout=None):
        from venice.commands import code
        fake, calls = _fake_openai_seq(seq)
        stdin = mock.MagicMock()
        stdin.isatty.return_value = False
        cfg = mock.patch(
            "venice.userconfig.load_config",
            lambda *a, **k: {"version": 1, "mcpServers": {}, "defaults": {}})
        cfg.start()
        self.addCleanup(cfg.stop)
        sess = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(sess, ignore_errors=True))
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake",
                                          "VENICE_SESSIONS_DIR": sess}), \
             mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.object(sys, "stdin", stdin), \
             mock.patch.object(sys, "stdout", stdout or io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            rc = code._run(args)
        return rc, calls

    def test_scout_flag_advertises_venice_scout_in_prompt(self):
        # --plan-only: one tool_choice="none" plan turn, then exit 0 without executing.
        seq = [FakeToolCompletion("1. do it\nAcceptance criteria:\n- ok")]
        rc, calls = self._run(
            _code_args(task="do x", root=self.root, plan_only=True, scout=True), seq)
        self.assertEqual(rc, 0)
        system_msg = calls[0]["messages"][0]["content"]
        self.assertIn(_agent.SCOUT_TOOL_NAME, system_msg)

    def test_no_scout_flag_omits_the_tool(self):
        seq = [FakeToolCompletion("1. do it\nAcceptance criteria:\n- ok")]
        rc, calls = self._run(
            _code_args(task="do x", root=self.root, plan_only=True), seq)
        self.assertEqual(rc, 0)
        self.assertNotIn(_agent.SCOUT_TOOL_NAME, calls[0]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
