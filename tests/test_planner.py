"""Unit tests for the planner harness -- epic #52 planner slice.

`venice code --planner` turns the scout/spawn/memory rails into a coherent
decompose -> dispatch -> track -> MERGE workflow: it implies the three rails,
appends the PLANNER_PROTOCOL overlay to the system prompt, records every launched
scout/spawn dispatch into a session-shared list, and exposes `venice_merge` -- a
deterministic rollup of dispatch provenance + the #49 task checklist + structural
warnings (`_code.merge_summary`, the pure heart). The `--json` envelope carries the
same rollup under `planner`.

Mirrors test_spawn.py: reuses the tool-call fakes from test_chat; no network, no
real key. Classes that touch the #49 task store chdir into a throwaway project so
the cwd walk-up (`_memory._project_dir`) never sees the real repo.
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

from venice.commands import _agent, _code, _memory


# Belt-and-suspenders (mirrors test_spawn): redirect the session + global memory
# tiers to a tmp dir so nothing here can write to the real home.
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


_WREPORT = (
    "OUTCOME: done -- created new.py.\n"
    "CHANGES: wrote new.py; ran no commands.\n"
    "VERIFIED: re-read the file live.\n"
    "FOLLOW-UPS: none.\n"
    "BLOCKERS: none."
)


def _rec(seq=1, kind="spawn", role="code", task_id=None, status="ok",
         fields=None, truncated=False, spent=None):
    return {
        "seq": seq, "kind": kind, "role": role, "task_id": task_id,
        "task": "t", "status": status, "fields": fields,
        "tool_calls": 1, "truncated": truncated, "spent_usd": spent,
    }


class _ProjBase(unittest.TestCase):
    """chdir into a throwaway project dir so the #49 task store stays hermetic."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.realpath(self.tmp)
        self.addCleanup(
            lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self._cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(lambda: os.chdir(self._cwd))
        with open(os.path.join(self.root, "a.py"), "w") as fh:
            fh.write("def f():\n    return 1\n")


# --------------------------------------------------------------------------- #
# merge_summary: the pure, deterministic heart of the merge step.
# --------------------------------------------------------------------------- #
class TestMergeSummary(_ProjBase):
    def test_empty_ledger_is_ok_with_a_warning(self):
        out = _code.merge_summary([])
        self.assertEqual(out["totals"], {"dispatches": 0, "spawns": 0, "scouts": 0,
                                         "errors": 0, "spent_usd": 0.0})
        self.assertEqual(out["dispatches"], [])
        self.assertEqual(out["tasks"], [])
        self.assertTrue(any("no dispatches" in w for w in out["warnings"]))

    def test_fields_surface_verbatim_and_spend_totals(self):
        f1 = {"OUTCOME": "done", "BLOCKERS": "none"}
        f2 = {"FINDINGS": "x lives in y.py"}
        recs = [_rec(1, "spawn", "asset", fields=f1, spent=1.25),
                _rec(2, "scout", None, fields=f2, spent=None)]
        out = _code.merge_summary(recs)
        self.assertEqual(out["totals"]["dispatches"], 2)
        self.assertEqual(out["totals"]["spawns"], 1)
        self.assertEqual(out["totals"]["scouts"], 1)
        self.assertEqual(out["totals"]["errors"], 0)
        self.assertAlmostEqual(out["totals"]["spent_usd"], 1.25)
        self.assertIs(out["dispatches"][0]["fields"], f1)  # verbatim, not re-parsed
        self.assertIs(out["dispatches"][1]["fields"], f2)

    def test_done_and_linked_task_yields_no_warnings(self):
        t = _memory.add_task("create new.py")
        _memory.update_task(t["id"], status="done")
        out = _code.merge_summary([_rec(task_id=t["id"])])
        self.assertEqual(out["warnings"], [])
        self.assertEqual(out["tasks"][0]["status"], "done")

    def test_undone_task_warns_and_flags_never_dispatched(self):
        a = _memory.add_task("unit a")           # dispatched but left in_progress
        b = _memory.add_task("unit b")           # never dispatched, still pending
        _memory.update_task(a["id"], status="in_progress")
        out = _code.merge_summary([_rec(task_id=a["id"])])
        wa = [w for w in out["warnings"] if f"task {a['id']} " in w]
        wb = [w for w in out["warnings"] if f"task {b['id']} " in w]
        self.assertTrue(wa and "in_progress" in wa[0])
        self.assertNotIn("never dispatched", wa[0])
        self.assertTrue(wb and "pending" in wb[0] and "never dispatched" in wb[0])

    def test_unknown_task_id_warns(self):
        out = _code.merge_summary([_rec(task_id="99")])
        self.assertTrue(any("unknown task_id '99'" in w for w in out["warnings"]))

    def test_error_and_truncated_dispatches_flagged(self):
        recs = [_rec(1, status="error"), _rec(2, truncated=True)]
        out = _code.merge_summary(recs)
        self.assertEqual(out["totals"]["errors"], 1)
        self.assertTrue(any("status='error'" in w for w in out["warnings"]))
        self.assertTrue(any("tool-call cap" in w for w in out["warnings"]))

    def test_unreadable_task_store_warns_never_raises(self):
        with mock.patch.object(_memory, "list_tasks",
                               side_effect=RuntimeError("boom")):
            out = _code.merge_summary([])
        self.assertEqual(out["tasks"], [])
        self.assertTrue(any("task store unreadable" in w for w in out["warnings"]))


# --------------------------------------------------------------------------- #
# Dispatch recording in scout_tool / spawn_tool.
# --------------------------------------------------------------------------- #
class TestDispatchRecording(_ProjBase):
    def _ok_run(self, **extra):
        out = {"status": "ok", "report": "r", "tool_calls": 2, "truncated": False,
               "fields": {"OUTCOME": "done"}}
        out.update(extra)
        return lambda *a, **kw: dict(out)

    def test_spawn_appends_a_full_record_and_echoes_task_id(self):
        led = []
        tool, _ = self._paid()
        with mock.patch.object(_agent, "run_spawn", self._ok_run()):
            st = _code.spawn_tool(None, "m", {}, [tool], dispatches=led)
            out = st.invoke({"task": "unit a", "role": "code", "task_id": "1"})
        self.assertEqual(out["task_id"], "1")          # echoed on the report
        self.assertEqual(len(led), 1)
        rec = led[0]
        self.assertEqual((rec["seq"], rec["kind"], rec["role"], rec["task_id"]),
                         (1, "spawn", "code", "1"))
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["fields"], {"OUTCOME": "done"})
        self.assertEqual(rec["tool_calls"], 2)
        self.assertAlmostEqual(rec["spent_usd"], 0.0)  # default cap attaches provenance

    def test_scout_appends_a_record(self):
        led = []
        with mock.patch.object(_agent, "run_scout", self._ok_run()):
            st = _code.scout_tool(None, "m", self.root, None, {}, dispatches=led)
            st.invoke({"task": "where is f?", "task_id": "2"})
        self.assertEqual(len(led), 1)
        self.assertEqual((led[0]["kind"], led[0]["role"], led[0]["task_id"]),
                         ("scout", None, "2"))

    def test_default_no_dispatches_no_task_id_is_prior_behavior(self):
        # Regression pin: without dispatches= and without a task_id arg, the result
        # carries no task_id key and nothing is recorded anywhere.
        tool, _ = self._paid()
        with mock.patch.object(_agent, "run_spawn", self._ok_run()):
            out = _code.spawn_tool(None, "m", {}, [tool]).invoke(
                {"task": "unit a", "role": "code"})
        self.assertNotIn("task_id", out)
        with mock.patch.object(_agent, "run_scout", self._ok_run()):
            out = _code.scout_tool(None, "m", self.root, None, {}).invoke(
                {"task": "q"})
        self.assertNotIn("task_id", out)

    def test_error_envelope_dispatch_is_still_recorded(self):
        led = []
        fake = mock.MagicMock()
        fake.chat.completions.create.side_effect = RuntimeError("boom")
        st = _code.scout_tool(fake, "m", self.root, None, {}, dispatches=led)
        out = st.invoke({"task": "investigate", "task_id": "3"})
        self.assertEqual(out["status"], "error")
        self.assertEqual(len(led), 1)
        self.assertEqual(led[0]["status"], "error")
        self.assertEqual(led[0]["task_id"], "3")

    def test_validation_refusal_is_not_recorded(self):
        led = []
        st = _code.scout_tool(None, "m", self.root, None, {}, dispatches=led)
        self.assertEqual(st.invoke({"task": "  "})["status"], "error")
        self.assertEqual(led, [])                      # never launched -> no record

    def test_long_task_text_is_truncated_in_the_record(self):
        led = []
        with mock.patch.object(_agent, "run_scout", self._ok_run()):
            _code.scout_tool(None, "m", self.root, None, {},
                             dispatches=led).invoke({"task": "x" * 300})
        self.assertEqual(len(led[0]["task"]), _code._DISPATCH_TASK_CHARS + 3)
        self.assertTrue(led[0]["task"].endswith("..."))

    @staticmethod
    def _paid():
        seen = {}

        def inv(arguments, *, confirm=False):
            return {"status": "ok"}

        return _agent.Tool("write_file", "d", {"type": "object", "properties": {}},
                           inv, paid=True, category="fs", tags=("write",)), seen


# --------------------------------------------------------------------------- #
# merge_tool: shape + containment.
# --------------------------------------------------------------------------- #
class TestMergeTool(_ProjBase):
    def test_builds_one_free_read_only_agent_tool(self):
        mt = _code.merge_tool([])
        self.assertEqual(mt.name, _agent.MERGE_TOOL_NAME)
        self.assertFalse(mt.paid)
        self.assertEqual(mt.category, "agent")
        self.assertEqual(mt.tags, ("read",))
        # Not part of the plain coding toolset -- planner-only.
        self.assertNotIn(_agent.MERGE_TOOL_NAME,
                         {t.name for t in _code.code_tools(self.root, None)})

    def test_invoke_rolls_up_the_shared_list_live(self):
        led = []
        mt = _code.merge_tool(led)
        self.assertEqual(mt.invoke({})["totals"]["dispatches"], 0)
        led.append(_rec())                             # the list is shared, not copied
        out = mt.invoke({})
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["totals"]["dispatches"], 1)

    def test_worker_can_never_hold_a_merge_tool(self):
        # Belt (structural): spawn_tool's grant filter drops category "agent".
        rec = {}

        def spy(oai, model, task, tools, base_kwargs, *, max_tool_calls, **kw):
            rec["names"] = {t.name for t in tools}
            return {"status": "ok", "report": "r", "tool_calls": 0,
                    "truncated": False}

        parent = _code.code_tools(self.root, None) + [_code.merge_tool([])]
        with mock.patch.object(_agent, "run_spawn", spy):
            _code.spawn_tool(None, "m", {}, parent).invoke(
                {"task": "x", "role": "code"})
        self.assertNotIn(_agent.MERGE_TOOL_NAME, rec["names"])
        # Suspenders (runtime): run_spawn refuses a merge-named tool outright.
        schema = {"type": "object", "properties": {}}
        nested = _agent.Tool(_agent.MERGE_TOOL_NAME, "d", schema,
                             lambda a, *, confirm=False: {"status": "ok"},
                             paid=False, category="agent", tags=("read",))
        with self.assertRaises(ValueError):
            _agent.run_spawn(None, "m", "t", [nested], {}, max_tool_calls=3)


# --------------------------------------------------------------------------- #
# `venice code --planner` wiring.
# --------------------------------------------------------------------------- #
def _code_args(**ov):
    base = dict(
        task=None, root=None, model=None, system=None, temperature=None,
        max_tokens=None, json=False, auto=None, manual=None, yes=None,
        plan_only=False, no_plan=False, no_verify=False, max_tool_calls=None,
        exec_timeout=None, interactive=False, resume=None, assets=None,
        scout=None, spawn=None, spawn_max_spend=None, planner=None, memory=None,
        auto_compact=None, compact_threshold=None, compact_keep_turns=None,
        session_max_spend=None, cont=None, ephemeral=None,
    )
    base.update(ov)
    return argparse.Namespace(**base)


class _WiringBase(_ProjBase):
    def _run(self, args, seq, stdout=None, config=None):
        from venice.commands import code
        fake, calls = _fake_openai_seq(seq)
        stdin = mock.MagicMock()
        stdin.isatty.return_value = False
        doc = config or {"version": 1, "mcpServers": {}, "defaults": {}}
        cfg = mock.patch("venice.userconfig.load_config", lambda *a, **k: doc)
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


class TestPlannerWiring(_WiringBase):
    _PLAN = [FakeToolCompletion("1. do it\nAcceptance criteria:\n- ok")]

    def test_planner_implies_rails_and_advertises_the_protocol(self):
        rc, calls = self._run(
            _code_args(task="do x", root=self.root, plan_only=True, planner=True),
            list(self._PLAN))
        self.assertEqual(rc, 0)
        system_msg = calls[0]["messages"][0]["content"]
        for name in (_agent.SCOUT_TOOL_NAME, _agent.SPAWN_TOOL_NAME,
                     _agent.MERGE_TOOL_NAME, "task_add"):
            self.assertIn(name, system_msg)
        self.assertIn("PLANNER", system_msg)
        self.assertIn("MERGE SUMMARY", system_msg)

    def test_no_planner_flag_leaves_the_prompt_unchanged(self):
        rc, calls = self._run(
            _code_args(task="do x", root=self.root, plan_only=True),
            list(self._PLAN))
        self.assertEqual(rc, 0)
        system_msg = calls[0]["messages"][0]["content"]
        for absent in (_agent.MERGE_TOOL_NAME, "PLANNER", "MERGE SUMMARY"):
            self.assertNotIn(absent, system_msg)

    def test_config_default_enables_the_planner(self):
        doc = {"version": 1, "mcpServers": {},
               "defaults": {"code": {"planner": True}}}
        rc, calls = self._run(
            _code_args(task="do x", root=self.root, plan_only=True),
            list(self._PLAN), config=doc)
        self.assertEqual(rc, 0)
        self.assertIn(_agent.MERGE_TOOL_NAME, calls[0]["messages"][0]["content"])

    def test_spawn_max_spend_still_threads_under_planner(self):
        captured = {}
        real = _code.spawn_tool

        def spy(oai, model, base_kwargs, parent_tools, **kw):
            captured["max_spend"] = kw.get("max_spend")
            captured["dispatches"] = kw.get("dispatches")
            return real(oai, model, base_kwargs, parent_tools, **kw)

        with mock.patch.object(_code, "spawn_tool", spy):
            rc, _ = self._run(
                _code_args(task="do x", root=self.root, plan_only=True,
                           planner=True, spawn_max_spend=1.5), list(self._PLAN))
        self.assertEqual(rc, 0)
        self.assertEqual(captured["max_spend"], 1.5)
        self.assertEqual(captured["dispatches"], [])   # the shared list, wired in


class TestPlannerEndToEnd(_WiringBase):
    def test_full_protocol_round_trip_and_json_rollup(self):
        # One planner run: task_add -> in_progress -> venice_spawn(task_id=1) ->
        # (nested worker writes new.py, reports) -> done -> venice_merge -> final.
        seq = [
            FakeToolCompletion(tool_calls=[
                _FnCall("c1", "task_add", json.dumps({"text": "create new.py"}))]),
            FakeToolCompletion(tool_calls=[
                _FnCall("c2", "task_update",
                        json.dumps({"id": "1", "status": "in_progress"}))]),
            FakeToolCompletion(tool_calls=[
                _FnCall("c3", "venice_spawn",
                        json.dumps({"task": "create new.py with x = 1",
                                    "role": "code", "task_id": "1"}))]),
            # -- nested worker loop consumes the next two --
            FakeToolCompletion(tool_calls=[
                _FnCall("c4", "write_file",
                        json.dumps({"path": "new.py", "content": "x = 1\n"}))]),
            FakeToolCompletion(_WREPORT),
            # -- back in the planner --
            FakeToolCompletion(tool_calls=[
                _FnCall("c5", "task_update",
                        json.dumps({"id": "1", "status": "done"}))]),
            FakeToolCompletion(tool_calls=[_FnCall("c6", "venice_merge", "{}")]),
            FakeToolCompletion("MERGE SUMMARY: 1/1 units done; no blockers."),
        ]
        stdout = io.StringIO()
        rc, calls = self._run(
            _code_args(task="build it", root=self.root, no_plan=True, auto=True,
                       planner=True, json=True),
            seq, stdout=stdout)
        self.assertEqual(rc, 0)
        # The worker actually wrote the file (yes=True doer).
        with open(os.path.join(self.root, "new.py")) as fh:
            self.assertEqual(fh.read(), "x = 1\n")
        # The venice_merge tool message carried the rollup to the planner.
        merge_msgs = [m for c in calls for m in c["messages"]
                      if m.get("role") == "tool" and m.get("tool_call_id") == "c6"]
        rollup = json.loads(merge_msgs[0]["content"])
        self.assertEqual(rollup["status"], "ok")
        self.assertEqual(rollup["totals"]["spawns"], 1)
        self.assertEqual(rollup["dispatches"][0]["task_id"], "1")
        self.assertIn("OUTCOME", rollup["dispatches"][0]["fields"])
        # The --json envelope carries the same rollup structurally.
        envelope = json.loads(stdout.getvalue())
        planner = envelope["planner"]
        self.assertEqual(planner["totals"],
                         {"dispatches": 1, "spawns": 1, "scouts": 0, "errors": 0,
                          "spent_usd": 0.0})
        self.assertEqual(planner["warnings"], [])      # done + linked + ok
        self.assertEqual(planner["tasks"][0]["status"], "done")
        self.assertEqual(envelope["final"],
                         "MERGE SUMMARY: 1/1 units done; no blockers.")


if __name__ == "__main__":
    unittest.main()
