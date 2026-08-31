"""Unit tests for the `venice code` command harness (#30).

Drives `code._run` end-to-end with a faked OpenAI client and the free /models
catalog GET mocked (via urlopen) -- no network, no real key. Reuses the tool-call
fakes from `test_chat`. File writes land in a per-test tmpdir project root.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_chat import (
    FakeToolCompletion, _FnCall, _fake_openai_seq, _urlopen_ok,
)
from venice.commands import _agent, _code, code as code_command


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
        session_max_spend=None, cache_guard=None, cont=None, ephemeral=None,
        review=None, review_model=None, review_rounds=None,   # #80 part 1a
        web_search=None, web_search_model=None,
        browser=None, browser_allow=None, browser_deny=None,
    )
    base.update(ov)
    return argparse.Namespace(**base)


def _write_call(cid, path, content):
    return _FnCall(cid, "write_file",
                   json.dumps({"path": path, "content": content}))


def _mem_call(cid, name, content, scope="project"):
    return _FnCall(cid, "memory_write",
                   json.dumps({"name": name, "content": content, "scope": scope}))


class TestAuxiliaryModelResolution(unittest.TestCase):
    _UNSET = object()
    MODELS = [
        {"id": "llama-author", "type": "text", "model_spec": {
            "traits": ["default"],
            "capabilities": {
                "supportsFunctionCalling": True, "supportsWebSearch": False,
            },
        }},
        {"id": "qwen-reviewer", "type": "text", "model_spec": {
            "traits": [], "capabilities": {"supportsFunctionCalling": True},
        }},
        {"id": "searcher", "type": "text", "model_spec": {
            "traits": [], "capabilities": {
                "supportsFunctionCalling": True, "supportsWebSearch": True,
            },
        }},
        {"id": "no-tools", "type": "text", "model_spec": {
            "traits": [], "capabilities": {
                "supportsFunctionCalling": False, "supportsWebSearch": False,
            },
        }},
    ]

    @staticmethod
    def _args(**overrides):
        values = {
            "review": None, "review_model": None,
            "web_search": None, "web_search_model": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def _resolve(self, args, models=_UNSET):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            state, rc = code_command._resolve_auxiliary_models(
                args, self.MODELS if models is self._UNSET else models, "llama-author",
            )
        return state, rc, err.getvalue()

    def test_auto_picks_are_announced_and_structured(self):
        state, rc, err = self._resolve(self._args(review=True, web_search=True))
        self.assertIsNone(rc)
        self.assertEqual(state["review_model"], "qwen-reviewer")
        self.assertEqual(state["web_search_model"], "searcher")
        self.assertEqual(state["resolved_models"], {
            "review": {"id": "qwen-reviewer", "source": "auto"},
            "web_search": {"id": "searcher", "source": "auto"},
        })
        self.assertIn("review model: qwen-reviewer (source: auto)", err)
        self.assertIn("web-search model: searcher (source: auto)", err)

    def test_flag_and_exact_config_provenance_are_distinct(self):
        args = self._args(
            review=True, review_model="qwen-reviewer",
            web_search=True, web_search_model="searcher",
            _config_sources={
                "web_search_model": "defaults.code.web_search_model",
            },
        )
        state, rc, err = self._resolve(args)
        self.assertIsNone(rc)
        self.assertEqual(state["resolved_models"], {
            "review": {"id": "qwen-reviewer", "source": "flag"},
            "web_search": {
                "id": "searcher", "source": "config",
                "config_key": "defaults.code.web_search_model",
            },
        })
        self.assertIn("source: flag --review-model", err)
        self.assertIn("source: config defaults.code.web_search_model", err)

    def test_unknown_review_model_exits_six(self):
        state, rc, err = self._resolve(
            self._args(review=True, review_model="typo-reviewer"),
        )
        self.assertIsNone(state)
        self.assertEqual(rc, 6)
        self.assertIn("unknown review model 'typo-reviewer'", err)

    def test_unknown_config_web_model_names_the_durable_fix(self):
        args = self._args(
            web_search=True, web_search_model="retired-searcher",
            _config_sources={
                "web_search_model": "defaults.code.web_search_model",
            },
        )
        state, rc, err = self._resolve(args)
        self.assertIsNone(state)
        self.assertEqual(rc, 6)
        self.assertIn("unknown web-search model 'retired-searcher'", err)
        self.assertIn("venice config unset defaults.code.web_search_model", err)

    def test_known_incapable_models_fail_at_startup(self):
        cases = [
            (self._args(review=True, review_model="no-tools"),
             "does not support function calling"),
            (self._args(web_search=True, web_search_model="no-tools"),
             "does not advertise supportsWebSearch"),
        ]
        for args, message in cases:
            with self.subTest(message=message):
                state, rc, err = self._resolve(args)
                self.assertIsNone(state)
                self.assertEqual(rc, 2)
                self.assertIn(message, err)

    def test_no_capable_web_model_fails_before_tool_use(self):
        models = [self.MODELS[0], self.MODELS[1], self.MODELS[3]]
        state, rc, err = self._resolve(self._args(web_search=True), models=models)
        self.assertIsNone(state)
        self.assertEqual(rc, 2)
        self.assertIn("no web-search-capable model available", err)

    def test_unverifiable_catalog_preserves_attempt_anyway_behavior(self):
        state, rc, err = self._resolve(
            self._args(review=True, web_search=True), models=None,
        )
        self.assertIsNone(rc)
        self.assertEqual(state["review_model"], "llama-author")
        self.assertEqual(state["web_search_model"], "llama-author")
        self.assertIn("could not verify function-calling support", err)

    def test_disabled_rails_ignore_stale_unused_defaults(self):
        args = self._args(
            review_model="retired-reviewer", web_search_model="retired-searcher",
            _config_sources={
                "review_model": "defaults.code.review_model",
                "web_search_model": "defaults.code.web_search_model",
            },
        )
        state, rc, err = self._resolve(args)
        self.assertIsNone(rc)
        self.assertEqual(state["resolved_models"], {})
        self.assertEqual(err, "")


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

    def test_browser_flag_fails_before_sdk_auth_or_network(self):
        from venice.commands import code
        err = io.StringIO()
        with mock.patch.object(code._openai, "import_openai") as import_openai, \
                mock.patch.object(code, "build_client_from_auth") as build_client, \
                mock.patch("venice.client.urllib.request.urlopen") as urlopen, \
                mock.patch.object(sys, "stderr", err):
            rc = code._run(_code_args(
                task="fetch a page", root=self.root, browser=True, auto=True))
        self.assertEqual(rc, 2)
        self.assertIn("temporarily disabled for security", err.getvalue())
        self.assertIn("GHSA-mqjr-2vh8-6fvg", err.getvalue())
        import_openai.assert_not_called()
        build_client.assert_not_called()
        urlopen.assert_not_called()

    # --- plan-only ---
    def test_plan_only_prints_and_exits_without_executing(self):
        out = io.StringIO()
        plan = "1. do it\nAcceptance criteria:\n- works"
        seq = [FakeToolCompletion(plan)]
        rc, calls = self._run(
            _code_args(task="do x", root=self.root, plan_only=True), seq, stdout=out)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)                 # only the plan turn
        self.assertEqual(calls[0]["tool_choice"], "none")
        self.assertEqual(out.getvalue(), plan + "\n")

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
        plan_request = calls[0]["messages"]
        execute_request = calls[1]["messages"]
        self.assertEqual(execute_request[:len(plan_request)], plan_request)
        self.assertEqual(
            plan_request[-1],
            {"role": "user", "content": code_command._PLAN_INSTRUCTION},
        )
        # #128: every phase of one top-level conversation shares backend affinity.
        keys = [c["extra_body"]["prompt_cache_key"] for c in calls]
        self.assertTrue(keys[0].startswith("venice-"))
        self.assertEqual(len(set(keys)), 1)
        # auto -> confirm=True -> the write actually happened
        with open(os.path.join(self.root, "hello.py")) as f:
            self.assertEqual(f.read(), "def hi():\n    return 1\n")

    def test_plan_reasoning_replays_and_survives_the_session_round_trip(self):
        seq = [
            FakeToolCompletion(
                "plan: inspect then fix",
                reasoning_content="plan reasoning \u2603",
            ),
            FakeToolCompletion("done"),
            FakeToolCompletion("ACCEPTANCE: PASS"),
        ]
        rc, calls = self._run(
            _code_args(task="fix it", root=self.root, auto=True), seq,
        )
        self.assertEqual(rc, 0)

        plan_in_execute = [
            message for message in calls[1]["messages"]
            if message.get("role") == "assistant"
        ][0]
        self.assertEqual(plan_in_execute["content"], "plan: inspect then fix")
        self.assertEqual(plan_in_execute["reasoning_content"], "plan reasoning \u2603")

        session_files = [
            os.path.join(self._sess_dir, name)
            for name in os.listdir(self._sess_dir)
            if name.endswith(".json")
        ]
        self.assertEqual(len(session_files), 1)
        with open(session_files[0]) as fh:
            saved = json.load(fh)
        saved_messages = saved["messages"]
        plan_index = next(
            i for i, message in enumerate(saved_messages)
            if message.get("content") == "plan: inspect then fix"
        )
        self.assertEqual(
            saved_messages[plan_index - 1],
            {"role": "user", "content": code_command._PLAN_INSTRUCTION},
        )
        saved_plan = saved_messages[plan_index]
        self.assertEqual(saved_plan["reasoning_content"], "plan reasoning \u2603")

    def test_replanning_requests_remain_exact_prefixes(self):
        seq = [
            FakeToolCompletion("plan: first pass"),
            FakeToolCompletion(
                "plan: revised pass",
                reasoning_content="revised reasoning",
            ),
            FakeToolCompletion("done"),
            FakeToolCompletion("ACCEPTANCE: PASS"),
        ]
        with mock.patch.object(code_command, "_decide_mode", return_value="prompt"), \
                mock.patch("builtins.input", side_effect=[
                    "edit", "cover the edge case", "auto",
                ]):
            rc, calls = self._run(
                _code_args(task="fix it", root=self.root), seq,
            )
        self.assertEqual(rc, 0)

        first_plan = calls[0]["messages"]
        revised_plan = calls[1]["messages"]
        execution = calls[2]["messages"]
        self.assertEqual(revised_plan[:len(first_plan)], first_plan)
        self.assertEqual(execution[:len(revised_plan)], revised_plan)
        self.assertEqual(
            first_plan[-1],
            {"role": "user", "content": code_command._PLAN_INSTRUCTION},
        )
        self.assertEqual(
            revised_plan[-1],
            {"role": "user", "content": code_command._PLAN_INSTRUCTION},
        )
        revised_reply = execution[len(revised_plan)]
        self.assertEqual(revised_reply["content"], "plan: revised pass")
        self.assertEqual(revised_reply["reasoning_content"], "revised reasoning")

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
        authority = stub.call_args.kwargs.get("path_authority")
        self.assertIsNotNone(authority)
        frame = Path(self.root) / "frame.png"
        frame.write_bytes(b"\x89PNG\r\n\x1a\nbody")
        resolved, mime = authority.resolve(
            frame, kind="image", max_bytes=1024
        )
        self.assertEqual(resolved, frame.resolve())
        self.assertEqual(mime, "image/png")

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

    def test_acceptance_retry_replays_the_complete_assistant_message(self):
        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion("did the work"),
            FakeToolCompletion(
                "All acceptance criteria are met.",
                reasoning_details=[{"type": "summary", "text": "checked"}],
            ),
            FakeToolCompletion("ACCEPTANCE: PASS"),
        ]
        rc, calls = self._run(
            _code_args(task="x", root=self.root, auto=True), seq,
        )
        self.assertEqual(rc, 0)
        replayed = calls[3]["messages"][-2]
        self.assertEqual(replayed["content"], "All acceptance criteria are met.")
        self.assertEqual(
            replayed["reasoning_details"],
            [{"type": "summary", "text": "checked"}],
        )

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

    def test_usage_raw_never_writes_to_stdout(self):
        # #98: the raw-usage dump is a DIAGNOSTIC. `venice code --json` is piped into
        # jq, so a stray `print()` without file=sys.stderr silently breaks every
        # caller. Same run as test_json_verdict_unknown, with the dump enabled.
        out, err = io.StringIO(), io.StringIO()
        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion("did the work"),
            FakeToolCompletion("All criteria met."),
            FakeToolCompletion("Still looks good."),
        ]
        with mock.patch.dict(os.environ, {"VENICE_USAGE_RAW": "1"}):
            rc, _calls = self._run(
                _code_args(task="x", root=self.root, auto=True, json=True),
                seq, stdout=out, stderr=err)
        self.assertEqual(rc, 10)
        env = json.loads(out.getvalue())       # stdout still machine-readable
        self.assertEqual(env["acceptance"]["verdict"], "unknown")
        self.assertIn("usage-raw:", err.getvalue())   # ...and the dump did happen

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
        self.assertEqual(env["resolved_models"], {})

    def test_plan_only_json_records_auxiliary_model_provenance(self):
        out, err = io.StringIO(), io.StringIO()
        rc, calls = self._run(
            _code_args(
                task="x", root=self.root, plan_only=True, json=True,
                review=True, review_model="llama-3.3-70b",
            ),
            [FakeToolCompletion("plan text")], stdout=out, stderr=err,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(out.getvalue())["resolved_models"], {
            "review": {"id": "llama-3.3-70b", "source": "flag"},
        })
        self.assertIn("source: flag --review-model", err.getvalue())

    def test_full_json_and_session_share_resolved_models(self):
        out = io.StringIO()
        selection = {
            "review": {"id": "llama-3.3-70b", "source": "flag"},
        }
        seq = [
            FakeToolCompletion("plan text"),
            FakeToolCompletion("done"),
            FakeToolCompletion("ACCEPTANCE: PASS"),
        ]
        rc, _calls = self._run(
            _code_args(
                task="x", root=self.root, auto=True, json=True,
                review=True, review_model="llama-3.3-70b",
            ),
            seq, stdout=out,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out.getvalue())["resolved_models"], selection)
        session_files = list(Path(self._sess_dir).glob("*.json"))
        self.assertEqual(len(session_files), 1)
        self.assertEqual(
            json.loads(session_files[0].read_text())["resolved_models"], selection,
        )

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

        # `**kw`: a rigid stub here breaks on every new keyword-only factory kwarg
        # `_run` learns to forward, which is a failure of the stub and not of the code.
        def _fake_repl_run(args, oai, openai, client, models, model,
                           initial=None, *, tools_session=None, gen_kwargs=None,
                           label="venice chat", max_tool_calls=8, session=None,
                           ephemeral=False, root=None, system_reseed=False, **kw):
            captured["tools_session"] = tools_session
            captured["label"] = label
            captured["initial"] = initial
            captured["max_tool_calls"] = max_tool_calls
            captured["root"] = root
            captured["system_reseed"] = system_reseed
            captured["ledger"] = kw.get("ledger")
            captured["resolved_models"] = kw.get("resolved_models")
            captured["gen_kwargs"] = gen_kwargs
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
        self.assertEqual(captured["resolved_models"], {})
        # #117: the REPL is HANDED the hoisted ledger rather than building its own,
        # because the rails already mirror into that object.
        self.assertIsNotNone(captured["ledger"])
        self.assertTrue(
            captured["gen_kwargs"]["extra_body"]["prompt_cache_key"].startswith(
                "venice-"))

    # --- session resume (#47) ---
    def _mk_zone(self):
        from venice.commands import _session
        zone = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(zone, ignore_errors=True))
        return zone, _session

    def _run_interactive(self, args, seq, inputs, zone, *, stderr=None):
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
             mock.patch.object(sys, "stderr", stderr or io.StringIO()):
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
        err = io.StringIO()
        rc, calls = self._run_interactive(
            _code_args(resume=stale.id, root=self.root, auto=True),
            [FakeToolCompletion("done")],           # one turn, no tool calls -> ends
            ["carry on", "/exit"], zone, stderr=err,
        )
        self.assertEqual(rc, 0)
        sysmsg = calls[0]["messages"][0]
        self.assertEqual(sysmsg["role"], "system")
        self.assertIn(self.root, sysmsg["content"])
        self.assertNotIn("/nonexistent/oldroot", sysmsg["content"])
        self.assertIn(
            "code: resumed with a different system prompt; prompt cache will be cold",
            err.getvalue(),
        )
        with open(Path(zone) / f"{stale.id}.json") as f:
            usage = json.load(f)["usage"]
        self.assertEqual(
            usage["context_events"][0], {"kind": "resume_reseed", "after_n": 0}
        )
        self.assertNotIn("STALE", json.dumps(usage["context_events"]))

    def test_identical_resume_prompt_emits_no_warning_or_event(self):
        zone, _session = self._mk_zone()
        rc, _ = self._run_interactive(
            _code_args(task="first", root=self.root, interactive=True, auto=True),
            [FakeToolCompletion("done")], ["/exit"], zone,
        )
        self.assertEqual(rc, 0)
        saved = json.loads(next(Path(zone).glob("*.json")).read_text())

        err = io.StringIO()
        rc2, _ = self._run_interactive(
            _code_args(resume=saved["id"], root=self.root, auto=True),
            [], ["/exit"], zone, stderr=err,
        )
        self.assertEqual(rc2, 0)
        self.assertNotIn("prompt cache will be cold", err.getvalue())
        persisted = json.loads((Path(zone) / f"{saved['id']}.json").read_text())
        self.assertNotIn(
            "resume_reseed",
            [event.get("kind") for event in persisted["usage"]["context_events"]],
        )

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

    def test_prompt_cache_key_survives_save_reset_and_resume(self):
        zone, _session = self._mk_zone()
        rc, first_calls = self._run_interactive(
            _code_args(task="first", root=self.root, interactive=True, auto=True),
            [FakeToolCompletion("done")], ["/exit"], zone,
        )
        self.assertEqual(rc, 0)
        first_key = first_calls[0]["extra_body"]["prompt_cache_key"]

        with mock.patch.dict(os.environ, {"VENICE_SESSIONS_DIR": zone}):
            saved = _session.load(_session.most_recent("code").id, "code")
        self.assertEqual(
            saved.gen_kwargs["extra_body"]["prompt_cache_key"], first_key)

        rc, resumed_calls = self._run_interactive(
            _code_args(resume=saved.id, root=self.root, auto=True),
            [FakeToolCompletion("done again")],
            ["/reset", "second", "/exit"], zone,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(
            resumed_calls[0]["extra_body"]["prompt_cache_key"], first_key)

    def test_old_session_without_a_cache_key_is_upgraded_on_resume(self):
        zone, _session = self._mk_zone()
        old = _session.new_session(
            "code", label="venice code", model="llama-3.3-70b", root=self.root,
            gen_kwargs={"temperature": 0.4},
            messages=[{"role": "system", "content": "old"}],
        )
        with mock.patch.dict(os.environ, {"VENICE_SESSIONS_DIR": zone}):
            _session.save(old)

        rc, calls = self._run_interactive(
            _code_args(resume=old.id, root=self.root, auto=True),
            [FakeToolCompletion("continued")], ["continue", "/exit"], zone,
        )
        self.assertEqual(rc, 0)
        key = calls[0]["extra_body"]["prompt_cache_key"]
        self.assertTrue(key.startswith("venice-"))
        with mock.patch.dict(os.environ, {"VENICE_SESSIONS_DIR": zone}):
            upgraded = _session.load(old.id, "code")
        self.assertEqual(
            upgraded.gen_kwargs["extra_body"]["prompt_cache_key"], key)


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


class TestAttachedCtrlCSteering(unittest.TestCase):
    """#79: on an attached tty, `code` wraps the execute loop in `pause_and_steer` and
    aborts cleanly on Ctrl+C. The real signal/prompt machinery is covered by
    `test_steer`; here we assert the wiring (enabled flag, injected steer, exit 130).
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

    def _run(self, args, seq, pause_cm):
        from venice.commands import code, _steer
        fake, calls = _fake_openai_seq(seq)
        stdin = mock.MagicMock()
        stdin.isatty.return_value = True  # attached terminal
        self._sess_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._sess_dir, ignore_errors=True))
        err = io.StringIO()
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake",
                                          "VENICE_SESSIONS_DIR": self._sess_dir}), \
             mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.object(_steer, "pause_and_steer", pause_cm), \
             mock.patch.object(sys, "stdin", stdin), \
             mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", err):
            rc = code._run(args)
        self._err = err.getvalue()
        return rc, calls

    def _sessions(self):
        import glob
        return sorted(glob.glob(os.path.join(self._sess_dir, "*.json")))

    def test_attached_tty_enables_steering_and_injects(self):
        from contextlib import contextmanager
        seen = {}

        @contextmanager
        def _inject_once(session_id, *, enabled):
            seen["enabled"] = enabled
            seen["sid"] = session_id
            state = {"n": 0}

            def _drain():
                state["n"] += 1
                return ["reprioritize: fix the #3 bug first"] if state["n"] == 1 else []

            yield _drain

        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion("done -- and I saw the steer"),  # execute final
            FakeToolCompletion("ACCEPTANCE: PASS"),
        ]
        rc, _ = self._run(_code_args(task="do x", root=self.root, auto=True), seq,
                          _inject_once)
        self.assertEqual(rc, 0)
        self.assertTrue(seen["enabled"])       # tty + not --json -> steering armed
        self.assertIsNotNone(seen["sid"])      # non-ephemeral run has a session id
        with open(self._sessions()[0]) as f:
            doc = json.loads(f.read())
        steers = [m for m in doc["messages"] if m.get("role") == "user"
                  and "steering message received mid-run" in m.get("content", "")]
        self.assertEqual(len(steers), 1)
        self.assertIn("fix the #3 bug", steers[0]["content"])

    def test_json_mode_disables_prompt_steering(self):
        from contextlib import contextmanager
        seen = {}

        @contextmanager
        def _capture(session_id, *, enabled):
            seen["enabled"] = enabled
            yield (lambda: [])

        seq = [
            FakeToolCompletion("plan"),
            FakeToolCompletion("done"),
            FakeToolCompletion("ACCEPTANCE: PASS"),
        ]
        rc, _ = self._run(_code_args(task="do x", root=self.root, auto=True, json=True),
                          seq, _capture)
        self.assertEqual(rc, 0)
        self.assertFalse(seen["enabled"])  # --json is machine-facing: no tty prompt

    def test_ctrlc_at_prompt_aborts_exit_130(self):
        from contextlib import contextmanager

        @contextmanager
        def _abort(session_id, *, enabled):
            def _drain():
                raise KeyboardInterrupt  # Ctrl+C at the steer prompt / 2nd Ctrl+C

            yield _drain

        seq = [FakeToolCompletion("plan")]  # execute create is never reached
        rc, _ = self._run(_code_args(task="do x", root=self.root, auto=True), seq, _abort)
        self.assertEqual(rc, 130)                 # documented Ctrl-C exit code
        self.assertIn("aborted", self._err)
        self.assertEqual(len(self._sessions()), 1)  # partial transcript still saved


# --------------------------------------------------------------------------- #
# #81: the usage + wall-clock surface
# --------------------------------------------------------------------------- #
class TestCodeUsageSurface(unittest.TestCase):
    """`venice code` reports what a run cost and how long it kept you waiting."""

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
        self._sess_dir = tempfile.mkdtemp()
        self.addCleanup(
            lambda: __import__("shutil").rmtree(self._sess_dir, ignore_errors=True))

    def _run(self, args, seq, steer=None):
        from venice.commands import code
        fake, calls = _fake_openai_seq(seq)
        stdin = mock.MagicMock()
        stdin.isatty.return_value = False
        self._out, self._errbuf = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.dict(
                os.environ, {"VENICE_API_KEY": "fake",
                             "VENICE_SESSIONS_DIR": self._sess_dir}))
            st.enter_context(mock.patch("venice.client.urllib.request.urlopen",
                                        _urlopen_ok()))
            st.enter_context(mock.patch("openai.OpenAI", return_value=fake))
            st.enter_context(mock.patch.object(sys, "stdin", stdin))
            st.enter_context(mock.patch.object(sys, "stdout", self._out))
            st.enter_context(mock.patch.object(sys, "stderr", self._errbuf))
            if steer is not None:
                st.enter_context(mock.patch(
                    "venice.commands._steer.pause_and_steer", steer))
            rc = code._run(args)
        self._err = self._errbuf.getvalue()
        return rc, calls

    def _sessions(self):
        import glob
        return sorted(glob.glob(os.path.join(self._sess_dir, "*.json")))

    def _usage(self, n):
        """A usage blob on the nth API call, so per-call totals are distinguishable."""
        return {"prompt_tokens": n * 100, "completion_tokens": n}

    def _full_seq(self):
        return [
            FakeToolCompletion("plan text", usage=self._usage(1)),     # plan turn
            FakeToolCompletion("done", usage=self._usage(2)),          # exec final
            FakeToolCompletion("ACCEPTANCE: PASS", usage=self._usage(4)),  # verify
        ]

    def test_json_envelope_carries_usage(self):
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True, json=True), self._full_seq())
        self.assertEqual(rc, 0)
        usage = json.loads(self._out.getvalue())["usage"]
        self.assertEqual(usage["turns"], 1)          # one run == one blocked window
        self.assertGreaterEqual(usage["elapsed_seconds"], 0)

    def test_plan_and_acceptance_turns_are_counted(self):
        # They run outside `run_loop` and carry the whole transcript as prompt, so a
        # total that skipped them would understate the run badly. 100 + 200 + 400.
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True, json=True), self._full_seq())
        self.assertEqual(rc, 0)
        usage = json.loads(self._out.getvalue())["usage"]
        self.assertEqual(usage["prompt_tokens"], 700)
        self.assertEqual(usage["completion_tokens"], 7)

    def test_default_run_meters_without_a_spend_cap(self):
        # The gap this closed: `ledger_from_args` returns None unless
        # --session-max-spend is set, so a plain run metered nothing at all.
        args = _code_args(task="x", root=self.root, auto=True, json=True)
        self.assertIsNone(getattr(args, "session_max_spend", None))
        rc, _ = self._run(args, self._full_seq())
        self.assertEqual(rc, 0)
        self.assertGreater(json.loads(self._out.getvalue())["usage"]["prompt_tokens"], 0)

    def test_footer_on_stderr_and_stdout_stays_clean(self):
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True), self._full_seq())
        self.assertEqual(rc, 0)
        self.assertRegex(self._err, r"code: .*wall")
        # `venice code | ...` must keep getting only the deliverable.
        self.assertNotIn("wall", self._out.getvalue())

    def test_footer_suppressed_under_json(self):
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True, json=True), self._full_seq())
        self.assertEqual(rc, 0)
        self.assertNotIn("wall", self._err)
        self.assertNotIn("tools", self._err)    # #82 rides the same suppression
        json.loads(self._out.getvalue())        # envelope still parses

    # -- the cache hit rate reaches the footer and the envelope (#100) ------- #
    #
    # The regression these exist to catch is the one that motivated the ticket: a
    # prompt-cache collapse is a silent 3-5x cost event, and before this the number
    # lived only behind a REPL slash command a one-shot run is never in.

    def _cached_seq(self):
        """The same three turns, but the provider reports its cache buckets."""
        def u(n, cached):
            return {"prompt_tokens": n * 100, "completion_tokens": n,
                    "prompt_tokens_details": {"cached_tokens": cached}}
        return [
            FakeToolCompletion("plan text", usage=u(1, 50)),
            FakeToolCompletion("done", usage=u(2, 100)),
            FakeToolCompletion("ACCEPTANCE: PASS", usage=u(4, 200)),
        ]

    def test_footer_reports_an_unknown_cache_state(self):
        # `_usage()` carries no `prompt_tokens_details` -- the shape the spec's own
        # example ships. The footer must say so rather than print a 0.0% that reads
        # as a measured total miss (#98's contract, now on this surface).
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True), self._full_seq())
        self.assertEqual(rc, 0)
        self.assertIn("cache n/a", self._err)
        self.assertNotIn("0.0% hit", self._err)

    def test_footer_reports_a_real_hit_rate(self):
        # 350 cached of 700 prompt across the three turns.
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True), self._cached_seq())
        self.assertEqual(rc, 0)
        footer = [ln for ln in self._err.splitlines() if ln.startswith("code: ")
                  and "wall" in ln]
        self.assertEqual(len(footer), 1)
        self.assertTrue(footer[0].endswith("cache 50.0% hit"), footer[0])

    def test_json_envelope_carries_the_cache_hit_rate(self):
        # So a pipeline can alert on a collapse without re-deriving it from two
        # counters -- and gets `null`, never a fabricated 0.0, when it is unknown.
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True, json=True),
            self._cached_seq())
        self.assertEqual(rc, 0)
        self.assertEqual(
            json.loads(self._out.getvalue())["usage"]["cache_hit_percent"], 50.0)
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True, json=True), self._full_seq())
        self.assertEqual(rc, 0)
        usage = json.loads(self._out.getvalue())["usage"]
        self.assertIn("cache_hit_percent", usage)
        self.assertIsNone(usage["cache_hit_percent"])

    # -- per-tool timing reaches the footer and the envelope (#82) ---------- #

    def _tool_seq(self):
        """The same run, but the exec turn dispatches two real `write_file` calls."""
        return [
            FakeToolCompletion("plan text", usage=self._usage(1)),
            FakeToolCompletion(tool_calls=[_write_call("c1", "a.txt", "x"),
                                           _write_call("c2", "b.txt", "y")],
                               usage=self._usage(2)),
            FakeToolCompletion("done", usage=self._usage(2)),
            FakeToolCompletion("ACCEPTANCE: PASS", usage=self._usage(4)),
        ]

    def _footer(self):
        lines = [ln for ln in self._err.splitlines()
                 if ln.startswith("code: ") and "wall" in ln]
        self.assertEqual(len(lines), 1, self._err)
        return lines[0]

    def test_footer_carries_the_tools_clause(self):
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True), self._tool_seq())
        self.assertEqual(rc, 0)
        # The clause lives INSIDE the wall field: " -- " stays the top-level boundary.
        self.assertRegex(self._footer(),
                         r"^code: [\d.]+s wall \([\d.]+s tools\) -- cost: ")

    def test_footer_has_no_tools_clause_when_no_tool_ran(self):
        # assertIn("wall", ...) cannot see a wrongly-appended " (0.0s tools)" -- a
        # substring match is blind to a suffix, so pin the shape of the whole field.
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True), self._full_seq())
        self.assertEqual(rc, 0)
        self.assertRegex(self._footer(), r"^code: [\d.]+s wall -- cost: ")
        self.assertNotIn("tools", self._footer())

    def test_json_envelope_carries_the_tools_block(self):
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True, json=True), self._tool_seq())
        self.assertEqual(rc, 0)
        usage = json.loads(self._out.getvalue())["usage"]
        self.assertEqual(usage["tools"]["write_file"]["calls"], 2)
        self.assertGreaterEqual(usage["tool_seconds"], 0)

    def test_session_file_carries_the_tools_block(self):
        # Same contract as the cache rate: `jq .usage` agrees on envelope and session.
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True), self._tool_seq())
        self.assertEqual(rc, 0)
        with open(self._sessions()[0]) as f:
            self.assertEqual(json.load(f)["usage"]["tools"]["write_file"]["calls"], 2)

    def test_session_file_carries_the_cache_hit_rate(self):
        # Same shape as the envelope: `jq .usage` agrees on both (code.py's
        # stated contract), which is what made the 08-03 archaeology possible.
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True), self._cached_seq())
        self.assertEqual(rc, 0)
        with open(self._sessions()[0]) as f:
            self.assertEqual(json.load(f)["usage"]["cache_hit_percent"], 50.0)

    def test_session_file_carries_usage(self):
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True), self._full_seq())
        self.assertEqual(rc, 0)
        with open(self._sessions()[0]) as f:
            usage = json.load(f)["usage"]
        self.assertEqual(usage["turns"], 1)
        self.assertEqual(usage["prompt_tokens"], 700)

    def test_session_file_carries_the_call_trace_in_call_order(self):
        # #99: three API calls -> three rows, and `_usage(n)` makes them individually
        # identifiable, so this pins that row order matches CALL order rather than
        # happening to look right because every row is identical.
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True), self._full_seq())
        self.assertEqual(rc, 0)
        with open(self._sessions()[0]) as f:
            usage = json.load(f)["usage"]
        self.assertEqual(usage["api_calls_total"], 3)
        self.assertEqual([r["n"] for r in usage["api_calls"]], [1, 2, 3])
        self.assertEqual([r["prompt_tokens"] for r in usage["api_calls"]],
                         [100, 200, 400])
        # `_usage()` carries no cache block, so every row must say UNKNOWN, not 0.
        self.assertEqual([r["cache_read_tokens"] for r in usage["api_calls"]],
                         [None, None, None])

    def test_every_recorded_call_carries_a_stamped_window(self):
        # THE test that kills "a call site forgot to bracket its create()". The plan
        # and acceptance turns run OUTSIDE run_loop (`_no_tool_turn`), so a bracket
        # added only to the loop would leave the two largest rows reading n/a.
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True), self._full_seq())
        self.assertEqual(rc, 0)
        with open(self._sessions()[0]) as f:
            rows = json.load(f)["usage"]["api_calls"]
        for row in rows:
            self.assertIsNotNone(row["seconds"], f"call #{row['n']} was not bracketed")
            self.assertGreaterEqual(row["seconds"], 0.0)

    def test_the_trace_distinguishes_api_calls_from_operator_turns(self):
        # `usage.turns` is "one time the CLI made you wait" (one whole `code` run);
        # `api_calls_total` is one model call. Both live in the same dict, so pin the
        # distinction into the artifact -- otherwise the next reader collapses them.
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True), self._full_seq())
        self.assertEqual(rc, 0)
        with open(self._sessions()[0]) as f:
            usage = json.load(f)["usage"]
        self.assertEqual(usage["turns"], 1)
        self.assertEqual(usage["api_calls_total"], 3)

    def test_json_envelope_and_session_file_agree_on_the_trace(self):
        # The four-surfaces contract: `--json` and the session save must not drift.
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True, json=True),
            self._full_seq())
        self.assertEqual(rc, 0)
        envelope = json.loads(self._out.getvalue())["usage"]
        with open(self._sessions()[0]) as f:
            persisted = json.load(f)["usage"]
        self.assertEqual(envelope["api_calls"], persisted["api_calls"])
        self.assertEqual(envelope["api_calls_total"], persisted["api_calls_total"])
        self.assertEqual(envelope["context_events"], persisted["context_events"])
        self.assertEqual(envelope["buckets"], persisted["buckets"])   # #117

    # --- #117: the hoisted ledger is the one that gets reported -------------

    def test_a_run_without_rails_reports_no_buckets(self):
        # The canary, and the control for the two tests below: an over-eager bucket
        # write would show up here, where nothing off-loop happened at all.
        rc, _ = self._run(
            _code_args(task="x", root=self.root, auto=True), self._full_seq())
        self.assertEqual(rc, 0)
        with open(self._sessions()[0]) as f:
            self.assertEqual(json.load(f)["usage"]["buckets"], {})

    def test_scout_spend_reaches_the_reported_usage(self):
        # END TO END for the hoist: the rails are handed a ledger built in `_run`,
        # long before `_run_oneshot` would have made one. If `_run_oneshot` built its
        # own instead of adopting, every rail would mirror into an orphan and this
        # bucket would simply be missing -- no error, no other failing test.
        def _run_scout(oai, model, task, tools, base_kwargs, *, max_tool_calls,
                       ledger=None, **kw):
            ledger.record({"prompt_tokens": 4321, "completion_tokens": 21})
            return {"status": "ok", "report": "r", "tool_calls": 1, "truncated": False}

        seq = [
            FakeToolCompletion("plan text", usage=self._usage(1)),
            FakeToolCompletion(None, tool_calls=[_FnCall(
                "c1", "venice_scout", json.dumps({"task": "look"}))],
                usage=self._usage(2)),
            FakeToolCompletion("done", usage=self._usage(3)),
            FakeToolCompletion("ACCEPTANCE: PASS", usage=self._usage(4)),
        ]
        with mock.patch.object(_agent, "run_scout", _run_scout):
            rc, _ = self._run(
                _code_args(task="x", root=self.root, auto=True, scout=True), seq)
        self.assertEqual(rc, 0)
        with open(self._sessions()[0]) as f:
            usage = json.load(f)["usage"]
        # Membership FIRST: if the bucket is missing (the mirror dropped, or a second
        # ledger built downstream), a bare subscript raises KeyError and the run reports
        # an ERROR, which reads like a broken test rather than a caught regression.
        self.assertIn("scout", usage["buckets"])
        self.assertEqual(usage["buckets"]["scout"]["calls"], 1)
        self.assertEqual(usage["buckets"]["scout"]["prompt_tokens"], 4321)
        # ...and it stayed OUT of the main-loop trace, which is the whole partition.
        self.assertEqual(usage["api_calls_total"], len(usage["api_calls"]))
        self.assertNotIn(4321, [r["prompt_tokens"] for r in usage["api_calls"]])

    def test_the_rails_and_the_reporter_share_one_ledger_object(self):
        # The one-object rule, by IDENTITY rather than by value: two ledgers with
        # equal contents would satisfy a value assertion while still losing every
        # subsequent rail write to whichever object is not reported.
        seen = {}
        real = _code.scout_tool

        def _capture(*a, **kw):
            seen["rails"] = kw.get("parent_ledger")
            return real(*a, **kw)

        def _fake_repl_run(*a, **kw):
            seen["repl"] = kw.get("ledger")
            return 0

        with mock.patch.object(_code, "scout_tool", _capture), \
             mock.patch("venice.commands._repl.run", _fake_repl_run):
            stdin = mock.MagicMock()
            stdin.isatty.return_value = True
            fake, _calls = _fake_openai_seq([])
            with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
                 mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
                 mock.patch("openai.OpenAI", return_value=fake), \
                 mock.patch.object(sys, "stdin", stdin), \
                 mock.patch.object(sys, "stdout", io.StringIO()), \
                 mock.patch.object(sys, "stderr", io.StringIO()):
                from venice.commands import code
                code._run(_code_args(task="hi", root=self.root, interactive=True,
                                     scout=True))
        self.assertIsNotNone(seen["rails"])
        self.assertIs(seen["rails"], seen["repl"])

    def test_ctrlc_run_reports_time_and_persists_usage(self):
        # The run an operator most wants a cost readout for: they sat through it and
        # then killed it. A happy-path-only footer would hide exactly this one.
        from contextlib import contextmanager

        @contextmanager
        def _abort(session_id, *, enabled):
            def _drain():
                raise KeyboardInterrupt

            yield _drain

        seq = [FakeToolCompletion("plan", usage=self._usage(3))]
        rc, _ = self._run(_code_args(task="x", root=self.root, auto=True), seq,
                          steer=_abort)
        self.assertEqual(rc, 130)
        self.assertRegex(self._err, r"code: .*wall")
        with open(self._sessions()[0]) as f:
            usage = json.load(f)["usage"]
        self.assertEqual(usage["turns"], 1)          # stamped BEFORE the snapshot
        self.assertEqual(usage["prompt_tokens"], 300)

    def test_plan_only_json_carries_the_plan_turns_usage(self):
        seq = [FakeToolCompletion("just the plan", usage=self._usage(5))]
        rc, _ = self._run(
            _code_args(task="x", root=self.root, plan_only=True, json=True), seq)
        self.assertEqual(rc, 0)
        usage = json.loads(self._out.getvalue())["usage"]
        self.assertEqual(usage["prompt_tokens"], 500)
        self.assertEqual(usage["turns"], 1)

    def test_api_error_run_still_reports_time(self):
        import openai

        fake = mock.MagicMock()
        fake.chat.completions.create.side_effect = openai.OpenAIError("boom")
        stdin = mock.MagicMock()
        stdin.isatty.return_value = False
        err = io.StringIO()
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake",
                                          "VENICE_SESSIONS_DIR": self._sess_dir}), \
             mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.object(sys, "stdin", stdin), \
             mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", err):
            from venice.commands import code
            code._run(_code_args(task="x", root=self.root, auto=True))
        self.assertRegex(err.getvalue(), r"code: .*wall")

    def test_fail_safe_abort_reports_nothing(self):
        # No model call happened, so "0.0s wall -- cost: $0.0000" would be noise
        # dressed as data. The footer must not fire before the first call.
        stdin = mock.MagicMock()
        stdin.isatty.return_value = False       # non-tty + no --auto -> fail safe
        err = io.StringIO()
        fake, _calls = _fake_openai_seq([])
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake",
                                          "VENICE_SESSIONS_DIR": self._sess_dir}), \
             mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.object(sys, "stdin", stdin), \
             mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", err):
            from venice.commands import code
            rc = code._run(_code_args(task="x", root=self.root))
        self.assertEqual(rc, 2)
        self.assertIn("refusing to run unattended", err.getvalue())
        self.assertNotIn("wall", err.getvalue())


if __name__ == "__main__":
    unittest.main()
