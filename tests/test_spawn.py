"""Unit tests for the scout + worker subagents -- epic #52 slices 1 & 2.

Slice 1 (context firewall): `_agent.run_scout` (the disposable read-only core),
`_code.read_only_tools` (the inner toolset + read-only/no-self-spawn invariant), and
`_code.scout_tool` (the `venice_scout` wrapper), plus the `venice code --scout` wiring.

Slice 2 (write-capable worker): `_agent.run_spawn` (the same disposable core past
read-only: allows paid/write tools, rejects only recursion), `_code.spawn_tool` (the
`venice_spawn` wrapper: role->category grant, blast-radius filtering, budget clamp,
error envelope), and the `venice code --spawn` wiring. Both share the private
`_agent._run_disposable` core, so the unchanged `TestRunScout` suite doubles as the
regression pin for that extraction. Reuses the tool-call fakes from `test_chat`; no
network, no real key.
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


def _write_call(cid, path="new.py", content="x = 1\n"):
    return _FnCall(cid, "write_file",
                   json.dumps({"path": path, "content": content}))


def _fake_paid_tool(name="venice_image", category="image", cost=0.90):
    """A paid Tool whose invoke records each `confirm` and returns a fixed cost.

    `cost=None` models a code-role paid tool (write_file/run) that reports no USD, so
    the spend meter never moves for it.
    """
    seen = {"confirms": [], "calls": 0}

    def inv(arguments, *, confirm=False):
        seen["confirms"].append(confirm)
        seen["calls"] += 1
        r = {"status": "ok", "paths": ["x"]}
        if cost is not None:
            r["cost_estimate_usd"] = cost
        return r

    tool = _agent.Tool(name, "gen", {"type": "object", "properties": {}}, inv,
                       paid=True, category=category, tags=("write",))
    return tool, seen


_REPORT = (
    "FINDINGS: read_file is defined in _code.py.\n"
    "CONFIDENCE: high -- read the source.\n"
    "DEAD-ENDS: none.\n"
    "NOT CHECKED: the tests.\n"
    "VERIFIED-LIVE vs HYPOTHETICAL: live (read the file)."
)

_WREPORT = (
    "OUTCOME: done -- created new.py.\n"
    "CHANGES: wrote new.py; ran no commands.\n"
    "VERIFIED: re-read the file live.\n"
    "FOLLOW-UPS: none.\n"
    "BLOCKERS: none."
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
        scout=None, spawn=None, spawn_max_spend=None, auto_compact=None,
        compact_threshold=None, compact_keep_turns=None, session_max_spend=None,
        cont=None, ephemeral=None,
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


# --------------------------------------------------------------------------- #
# Slice 2: the write-capable worker subagent (`venice_spawn`).
# --------------------------------------------------------------------------- #
class TestRunSpawn(_ScoutBase):
    def _granted(self, cats=("fs", "exec", "vcs")):
        # A worker's toolset = a category subset of the parent's real code tools, minus
        # the root-tagged attach_root -- exactly what `spawn_tool` grants.
        return [t for t in _code.code_tools(self.root, None)
                if t.category in cats and "root" not in t.tags]

    def test_returns_worker_report_and_firewalls_stdout(self):
        seq = [
            FakeToolCompletion(tool_calls=[_write_call("c1", "new.py")]),
            FakeToolCompletion(_WREPORT),  # final answer (no tool_calls)
        ]
        fake, calls = _fake_openai_seq(seq)
        outer = io.StringIO()
        with mock.patch.object(sys, "stdout", outer):
            out = _agent.run_spawn(
                fake, "m", "create new.py", self._granted(), {}, max_tool_calls=12)
        self.assertEqual(out["status"], "ok")
        for marker in ("OUTCOME:", "CHANGES:", "VERIFIED:", "FOLLOW-UPS:", "BLOCKERS:"):
            self.assertIn(marker, out["report"])
        self.assertEqual(out["tool_calls"], 1)
        self.assertFalse(out["truncated"])
        self.assertEqual(outer.getvalue(), "")  # firewall
        # A worker is a *doer*: the paid write actually happened (yes=True).
        self.assertTrue(os.path.exists(os.path.join(self.root, "new.py")))

    def test_guard_allows_paid_but_rejects_self_spawn(self):
        schema = {"type": "object", "properties": {}}
        paid = _agent.Tool("write_file", "d", schema,
                           lambda a, *, confirm=False: {"status": "ok"},
                           paid=True, category="fs", tags=("write",))
        # A paid/write tool is fine for a worker -- the model just returns a report.
        fake, _ = _fake_openai_seq([FakeToolCompletion(_WREPORT)])
        out = _agent.run_spawn(fake, "m", "t", [paid], {}, max_tool_calls=3)
        self.assertEqual(out["status"], "ok")
        # But a nested spawn/scout tool is rejected -- no nested subagents.
        for name in (_agent.SPAWN_TOOL_NAME, _agent.SCOUT_TOOL_NAME):
            nested = _agent.Tool(name, "d", schema,
                                 lambda a, *, confirm=False: {"status": "ok"},
                                 paid=False, category="agent", tags=("spawn",))
            with self.assertRaises(ValueError):
                _agent.run_spawn(None, "m", "t", [nested], {}, max_tool_calls=3)

    def test_fresh_context_focus_and_role_threaded(self):
        fake, calls = _fake_openai_seq([FakeToolCompletion(_WREPORT)])
        _agent.run_spawn(fake, "m", "do the thing", self._granted(), {},
                         max_tool_calls=5, focus="src/x.py", role="code")
        first = calls[0]["messages"]
        self.assertEqual(len(first), 2)
        self.assertEqual(first[0]["role"], "system")
        self.assertIn("WORKER", first[0]["content"])
        self.assertIn("Your role: code", first[0]["content"])   # role folded in
        self.assertIn("src/x.py", first[0]["content"])          # focus hint threaded
        self.assertEqual(first[1], {"role": "user", "content": "do the thing"})

    def test_budget_cap_forces_final_and_marks_truncated(self):
        seq = [
            FakeToolCompletion(tool_calls=[_write_call("c1", "w1.py"),
                                           _write_call("c2", "w2.py")]),
            FakeToolCompletion("partial"),  # the forced final
        ]
        fake, calls = _fake_openai_seq(seq)
        with mock.patch.object(sys, "stderr", io.StringIO()):
            out = _agent.run_spawn(fake, "m", "t", self._granted(), {},
                                   max_tool_calls=1)
        self.assertTrue(out["truncated"])
        self.assertEqual(out["tool_calls"], 1)
        self.assertEqual(calls[-1]["tool_choice"], "none")  # _force_final fired


class TestSpawnTool(_ScoutBase):
    def _grant_recorder(self):
        rec = {}

        def _rec(oai, model, task, tools, base_kwargs, *, max_tool_calls, **kw):
            rec["names"] = {t.name for t in tools}
            rec["cats"] = {t.category for t in tools}
            rec["n"] = max_tool_calls
            return {"status": "ok", "report": "r", "tool_calls": 0, "truncated": False}

        return rec, _rec

    def test_builds_one_write_capable_agent_tool(self):
        st = _code.spawn_tool(None, "m", {}, _code.code_tools(self.root, None))
        self.assertEqual(st.name, _agent.SPAWN_TOOL_NAME)
        self.assertFalse(st.paid)
        self.assertEqual(st.category, "agent")
        self.assertIn("write", st.tags)
        self.assertIn("spawn", st.tags)

    def test_not_a_code_tools_rail(self):
        names = {t.name for t in _code.code_tools(self.root, None)}
        self.assertNotIn(_agent.SPAWN_TOOL_NAME, names)

    def test_code_role_grants_write_exec_vcs(self):
        rec, _rec = self._grant_recorder()
        parent = _code.code_tools(self.root, None)
        with mock.patch.object(_agent, "run_spawn", _rec):
            _code.spawn_tool(None, "m", {}, parent).invoke(
                {"task": "x", "role": "code"})
        self.assertLessEqual(rec["cats"], {"fs", "exec", "vcs", "search"})
        self.assertIn("write_file", rec["names"])   # write-capable, not read-only
        self.assertIn("run", rec["names"])
        self.assertNotIn("attach_root", rec["names"])  # root-tagged -> excluded

    def test_agent_category_never_granted_even_if_requested(self):
        rec, _rec = self._grant_recorder()
        parent = list(_code.code_tools(self.root, None))
        parent.append(_code.scout_tool(None, "m", self.root, None, {}))
        parent.append(_agent.Tool(
            _agent.SPAWN_TOOL_NAME, "d", {"type": "object", "properties": {}},
            lambda a, *, confirm=False: {"status": "ok"},
            paid=False, category="agent", tags=("write", "spawn")))
        with mock.patch.object(_agent, "run_spawn", _rec):
            # Even an explicit categories override asking for 'agent' can't leak them.
            _code.spawn_tool(None, "m", {}, parent).invoke(
                {"task": "x", "categories": ["fs", "agent"]})
        self.assertNotIn(_agent.SCOUT_TOOL_NAME, rec["names"])
        self.assertNotIn(_agent.SPAWN_TOOL_NAME, rec["names"])
        self.assertIn("write_file", rec["names"])   # fs still granted

    def test_categories_override_is_intersected_with_parent(self):
        rec, _rec = self._grant_recorder()
        parent = _code.code_tools(self.root, None)  # no client -> no image tools
        with mock.patch.object(_agent, "run_spawn", _rec):
            _code.spawn_tool(None, "m", {}, parent).invoke(
                {"task": "x", "categories": ["fs", "image"]})
        self.assertIn("fs", rec["cats"])
        self.assertNotIn("image", rec["cats"])  # parent lacks image -> not granted

    def test_asset_role_without_media_is_empty_grant_error(self):
        parent = _code.code_tools(self.root, None)  # no client, no --assets
        out = _code.spawn_tool(None, "m", {}, parent).invoke(
            {"task": "make a card image", "role": "asset"})
        self.assertEqual(out["status"], "error")
        self.assertIn("no tools available", out["message"])

    def test_granted_write_tool_enforces_parent_roots(self):
        # The worker shares the parent's Roots-bound tool instances (#76): an out-of-root
        # write fails loudly and creates nothing.
        rec, _rec = self._grant_recorder()
        rec2 = {}

        def _capture(oai, model, task, tools, base_kwargs, *, max_tool_calls, **kw):
            rec2["tools"] = tools
            return {"status": "ok", "report": "r", "tool_calls": 0, "truncated": False}

        parent = _code.code_tools(self.root, None)
        with mock.patch.object(_agent, "run_spawn", _capture):
            _code.spawn_tool(None, "m", {}, parent).invoke(
                {"task": "x", "role": "code"})
        wf = next(t for t in rec2["tools"] if t.name == "write_file")
        outside = os.path.join(os.path.dirname(self.root), "escapee.py")
        out = wf.invoke({"path": outside, "content": "x"}, confirm=True)
        self.assertEqual(out["status"], "error")
        self.assertIn("escape", out["message"])
        self.assertFalse(os.path.exists(outside))

    def test_missing_task_is_error_envelope(self):
        st = _code.spawn_tool(None, "m", {}, _code.code_tools(self.root, None))
        self.assertEqual(st.invoke({"role": "code"})["status"], "error")

    def test_max_tool_calls_clamped_to_range(self):
        rec, _rec = self._grant_recorder()
        with mock.patch.object(_agent, "run_spawn", _rec):
            st = _code.spawn_tool(None, "m", {}, _code.code_tools(self.root, None))
            st.invoke({"task": "x", "max_tool_calls": 999})
            self.assertEqual(rec["n"], _code._SPAWN_HARD_CAP)
            st.invoke({"task": "x", "max_tool_calls": 0})
            self.assertEqual(rec["n"], _code._SPAWN_MAX_TOOL_CALLS)
            st.invoke({"task": "x"})
            self.assertEqual(rec["n"], _code._SPAWN_MAX_TOOL_CALLS)

    def test_nested_failure_becomes_error_envelope(self):
        fake = mock.MagicMock()
        fake.chat.completions.create.side_effect = RuntimeError("boom")
        st = _code.spawn_tool(fake, "m", {}, _code.code_tools(self.root, None))
        out = st.invoke({"task": "do x", "role": "code"})
        self.assertEqual(out["status"], "error")
        self.assertIn("spawn failed", out["message"])
        # Even on the error path the spend provenance rides along (default cap active).
        self.assertEqual(out["spent_usd"], 0.0)
        self.assertEqual(out["spend_cap_usd"], _code._SPAWN_MAX_SPEND)

    # ---- per-worker USD media spend cap (#52 spend slice) -------------------- #
    def _spend_run(self, media_name, n):
        """A run_spawn stub that fires the granted `media_name` tool `n` times
        (confirm=True, as the yes=True worker loop does) and reports the statuses."""
        def _run(oai, model, task, tools, base_kwargs, *, max_tool_calls, **kw):
            media = next(t for t in tools if t.name == media_name)
            statuses = [media.invoke({"prompt": "p"}, confirm=True)["status"]
                        for _ in range(n)]
            return {"status": "ok", "report": "OUTCOME: done", "tool_calls": n,
                    "truncated": False, "_statuses": statuses}
        return _run

    def test_asset_worker_media_spend_capped_and_provenance(self):
        tool, seen = _fake_paid_tool(cost=0.90)
        with mock.patch.object(_agent, "run_spawn",
                               self._spend_run("venice_image", 4)):
            out = _code.spawn_tool(None, "m", {}, [tool], max_spend=2.00).invoke(
                {"task": "make art", "role": "asset"})
        # 0.90, 1.80, 2.70 all pass the pre-call check; the 4th (2.70 >= 2.00) is refused.
        self.assertEqual(out["_statuses"], ["ok", "ok", "ok", "blocked"])
        self.assertEqual(seen["calls"], 3)            # blocked call never reached inner
        self.assertTrue(seen["confirms"] and all(seen["confirms"]))  # confirm forwarded
        self.assertAlmostEqual(out["spent_usd"], 2.70)               # handoff provenance
        self.assertEqual(out["spend_cap_usd"], 2.00)

    def test_blocked_envelope_has_status_blocked_not_error(self):
        tool, _seen = _fake_paid_tool(cost=5.0)  # one call overshoots a $2 cap
        captured = {}

        def _run(oai, model, task, tools, base_kwargs, *, max_tool_calls, **kw):
            media = next(t for t in tools if t.name == "venice_image")
            media.invoke({"prompt": "p"}, confirm=True)          # spends 5.0
            captured["second"] = media.invoke({"prompt": "p"}, confirm=True)
            return {"status": "ok", "report": "r", "tool_calls": 1, "truncated": False}

        with mock.patch.object(_agent, "run_spawn", _run):
            _code.spawn_tool(None, "m", {}, [tool], max_spend=2.00).invoke(
                {"task": "x", "role": "asset"})
        self.assertEqual(captured["second"]["status"], "blocked")
        self.assertIn("spend cap", captured["second"]["message"])

    def test_default_cap_applies_when_max_spend_unset(self):
        # No max_spend arg -> the module default _SPAWN_MAX_SPEND caps the worker.
        tool, _seen = _fake_paid_tool(cost=_code._SPAWN_MAX_SPEND + 1.0)
        with mock.patch.object(_agent, "run_spawn",
                               self._spend_run("venice_image", 2)):
            out = _code.spawn_tool(None, "m", {}, [tool]).invoke(
                {"task": "x", "role": "asset"})
        self.assertEqual(out["_statuses"], ["ok", "blocked"])  # 1st overshoots, 2nd refused
        self.assertEqual(out["spend_cap_usd"], _code._SPAWN_MAX_SPEND)

    def test_code_worker_spend_cap_never_bites(self):
        # A code-role paid tool reports no cost, so even a tiny cap never fires.
        tool, seen = _fake_paid_tool(name="write_file", category="fs", cost=None)
        with mock.patch.object(_agent, "run_spawn",
                               self._spend_run("write_file", 5)):
            out = _code.spawn_tool(None, "m", {}, [tool], max_spend=0.01).invoke(
                {"task": "x", "role": "code"})
        self.assertEqual(out["_statuses"], ["ok"] * 5)
        self.assertEqual(seen["calls"], 5)
        self.assertEqual(out["spent_usd"], 0.0)

    def test_spawn_max_spend_zero_disables_metering(self):
        # <= 0 => identity pass-through: granted holds the parent's own tool instances
        # (unwrapped) and no spend provenance is attached -- exact pre-slice behavior.
        tool, _seen = _fake_paid_tool(cost=0.90)
        rec = {}

        def _capture(oai, model, task, tools, base_kwargs, *, max_tool_calls, **kw):
            rec["media"] = next(t for t in tools if t.name == "venice_image")
            return {"status": "ok", "report": "r", "tool_calls": 0, "truncated": False}

        with mock.patch.object(_agent, "run_spawn", _capture):
            out = _code.spawn_tool(None, "m", {}, [tool], max_spend=0).invoke(
                {"task": "x", "role": "asset"})
        self.assertIs(rec["media"], tool)             # not wrapped -> same object
        self.assertNotIn("spent_usd", out)
        self.assertNotIn("spend_cap_usd", out)


class TestSpawnWiring(_ScoutBase):
    """`venice code --spawn` folds the tool in and advertises it in the prompt."""

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

    def test_spawn_flag_advertises_venice_spawn_in_prompt(self):
        seq = [FakeToolCompletion("1. do it\nAcceptance criteria:\n- ok")]
        rc, calls = self._run(
            _code_args(task="do x", root=self.root, plan_only=True, spawn=True), seq)
        self.assertEqual(rc, 0)
        self.assertIn(_agent.SPAWN_TOOL_NAME, calls[0]["messages"][0]["content"])

    def test_no_spawn_flag_omits_the_tool(self):
        seq = [FakeToolCompletion("1. do it\nAcceptance criteria:\n- ok")]
        rc, calls = self._run(
            _code_args(task="do x", root=self.root, plan_only=True), seq)
        self.assertEqual(rc, 0)
        self.assertNotIn(_agent.SPAWN_TOOL_NAME, calls[0]["messages"][0]["content"])

    def test_spawn_max_spend_threads_into_spawn_tool(self):
        # `venice code --spawn --spawn-max-spend X` reaches _code.spawn_tool(max_spend=X).
        captured = {}
        real = _code.spawn_tool

        def spy(oai, model, base_kwargs, parent_tools, **kw):
            captured["max_spend"] = kw.get("max_spend")
            return real(oai, model, base_kwargs, parent_tools, **kw)

        seq = [FakeToolCompletion("1. do it\nAcceptance criteria:\n- ok")]
        with mock.patch.object(_code, "spawn_tool", spy):
            rc, calls = self._run(
                _code_args(task="do x", root=self.root, plan_only=True,
                           spawn=True, spawn_max_spend=1.5), seq)
        self.assertEqual(rc, 0)
        self.assertEqual(captured["max_spend"], 1.5)


if __name__ == "__main__":
    unittest.main()
