"""Unit tests for the interactive `venice chat` REPL (issue #22).

Drives `chat._run` in interactive mode with scripted `input()` lines and a fake
OpenAI client, exactly like `test_chat.py` (no network, no real key). Reuses that
module's fakes so the two stay in lock-step.
"""
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_chat import (
    FakeChunk,
    FakeToolCompletion,
    _FnCall,
    _MCP_PRESENT,
    _fake_attach_cm,
    _fake_openai_seq,
    _fake_tool,
    _urlopen_ok,
    _args,
)

from venice.commands import _repl  # noqa: E402

_EMPTY_CFG = {"version": 1, "mcpServers": {}, "defaults": {}}


def _run_repl(args, results, inputs, *, stdout=None, stderr=None,
              urlopen=None, stdin=None, cfg=None, attach=None, mcp_probe=_MCP_PRESENT):
    """Run the REPL: `results` are returned by successive create() calls,
    `inputs` are fed to input(). Returns (rc, fake_client, recorded_calls).

    `cfg` overrides the (empty) config doc; `attach` patches the MCP client seam;
    `mcp_probe` is what `import_mcp` returns (SDK-independent, like test_chat)."""
    from venice.commands import chat
    fake, calls = _fake_openai_seq(results)
    with contextlib.ExitStack() as st:
        st.enter_context(mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}))
        st.enter_context(mock.patch("venice.userconfig.load_config",
                                    lambda *a, **k: cfg or _EMPTY_CFG))
        st.enter_context(mock.patch("venice.client.urllib.request.urlopen",
                                    urlopen or _urlopen_ok()))
        st.enter_context(mock.patch("openai.OpenAI", return_value=fake))
        st.enter_context(mock.patch("venice.commands._mcp.import_mcp",
                                    return_value=mcp_probe))
        st.enter_context(mock.patch("builtins.input", side_effect=inputs))
        st.enter_context(mock.patch.object(sys, "stdin", stdin or io.StringIO("")))
        st.enter_context(mock.patch.object(sys, "stdout", stdout or io.StringIO()))
        st.enter_context(mock.patch.object(sys, "stderr", stderr or io.StringIO()))
        if attach is not None:
            st.enter_context(mock.patch("venice.commands._mcp_client.attach", attach))
        rc = chat._run(args)
    return rc, fake, calls


class TestRepl(unittest.TestCase):

    def test_multi_turn_carries_context(self):
        out = io.StringIO()
        results = [
            [FakeChunk("Hi there"),
             FakeChunk(usage={"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2})],
            [FakeChunk("Doing well")],
        ]
        rc, fake, calls = _run_repl(
            _args(interactive=True),
            results, ["hello", "how are you", "/exit"], stdout=out,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)
        # second turn carries the full prior history (context across turns)
        roles = [m["role"] for m in calls[1]["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user"])
        self.assertEqual(calls[1]["messages"][0]["content"], "hello")
        self.assertEqual(calls[1]["messages"][1]["content"], "Hi there")
        self.assertEqual(calls[1]["messages"][2]["content"], "how are you")
        # both replies were streamed to stdout
        self.assertIn("Hi there", out.getvalue())
        self.assertIn("Doing well", out.getvalue())

    def test_system_prompt_seeded(self):
        rc, fake, calls = _run_repl(
            _args(interactive=True, system="be terse"),
            [[FakeChunk("yo")]], ["hey", "/exit"],
        )
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0]["messages"][0],
                         {"role": "system", "content": "be terse"})
        self.assertEqual(calls[0]["messages"][1],
                         {"role": "user", "content": "hey"})

    def test_reset_keeps_system_clears_rest(self):
        rc, fake, calls = _run_repl(
            _args(interactive=True, system="sys"),
            [[FakeChunk("a")], [FakeChunk("b")]],
            ["one", "/reset", "two", "/exit"],
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)
        roles = [m["role"] for m in calls[1]["messages"]]
        self.assertEqual(roles, ["system", "user"])
        self.assertEqual(calls[1]["messages"][1]["content"], "two")

    def test_slash_system_sets_prompt(self):
        rc, fake, calls = _run_repl(
            _args(interactive=True),
            [[FakeChunk("ahoy")]], ["/system you are a pirate", "arr", "/exit"],
        )
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0]["messages"][0],
                         {"role": "system", "content": "you are a pirate"})
        self.assertEqual(calls[0]["messages"][1]["content"], "arr")

    def test_slash_model_switches(self):
        rc, fake, calls = _run_repl(
            _args(interactive=True),
            [[FakeChunk("ok")]], ["/model venice-uncensored", "hi", "/exit"],
        )
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0]["model"], "venice-uncensored")

    def test_slash_model_unknown_keeps_current(self):
        err = io.StringIO()
        rc, fake, calls = _run_repl(
            _args(interactive=True),
            [[FakeChunk("ok")]], ["/model nope-model", "hi", "/exit"],
            stderr=err,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0]["model"], "llama-3.3-70b")  # default kept
        self.assertIn("nope-model", err.getvalue())

    def test_empty_lines_skipped(self):
        rc, fake, calls = _run_repl(
            _args(interactive=True),
            [[FakeChunk("ok")]], ["", "   ", "real", "/exit"],
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)  # only "real" produced a turn

    def test_eof_exits_clean(self):
        rc, fake, calls = _run_repl(
            _args(interactive=True),
            [[FakeChunk("yo")]], ["hi", EOFError],
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)

    def test_auto_interactive_on_tty(self):
        # No message + stdin is a TTY -> REPL (was exit 2 before #22).
        fake_stdin = mock.MagicMock()
        fake_stdin.isatty.return_value = True
        rc, fake, calls = _run_repl(
            _args(message=None),  # no --interactive flag: detected via the TTY
            [[FakeChunk("hey")]], ["hi", "/exit"], stdin=fake_stdin,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)

    def test_save_then_resume_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "session.json")
            # session 1: one turn, then /save
            rc, fake, calls = _run_repl(
                _args(interactive=True),
                [[FakeChunk("noted")]], ["remember X", "/save " + path, "/exit"],
            )
            self.assertEqual(rc, 0)
            saved = json.loads(Path(path).read_text())
            self.assertEqual(saved[0], {"role": "user", "content": "remember X"})
            self.assertEqual(saved[1]["role"], "assistant")
            self.assertEqual(saved[1]["content"], "noted")

            # session 2: --resume, next turn sees the restored context
            rc2, fake2, calls2 = _run_repl(
                _args(interactive=True, resume=path),
                [[FakeChunk("you said X")]], ["what did I say", "/exit"],
            )
            self.assertEqual(rc2, 0)
            roles = [m["role"] for m in calls2[0]["messages"]]
            self.assertEqual(roles, ["user", "assistant", "user"])
            self.assertEqual(calls2[0]["messages"][0]["content"], "remember X")

    def test_bad_resume_exit_2(self):
        err = io.StringIO()
        rc, fake, calls = _run_repl(
            _args(interactive=True, resume="/no/such/file.json"),
            [], ["/exit"], stderr=err,
        )
        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 0)
        self.assertIn("transcript", err.getvalue())

    def _resume_history(self, tmpdir, pairs=6):
        """Write a resumable transcript of `pairs` user/assistant turns."""
        path = Path(tmpdir) / "session.json"
        msgs = []
        for i in range(pairs):
            msgs.append({"role": "user", "content": f"u{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        path.write_text(json.dumps(msgs))
        return str(path)

    def test_slash_compact_summarizes_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            resume = self._resume_history(d, pairs=6)
            err = io.StringIO()
            # 1st create(): the /compact summarization turn (a plain completion).
            rc, fake, calls = _run_repl(
                _args(interactive=True, resume=resume),
                [FakeToolCompletion("we discussed u0..u5")],
                ["/compact 2", "/exit"], stderr=err,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["tool_choice"], "none")
            self.assertIn("compacted:", err.getvalue())

    def test_slash_compact_nothing_to_do(self):
        # A fresh session has nothing to compact: no API call, a note instead.
        err = io.StringIO()
        rc, fake, calls = _run_repl(
            _args(interactive=True),
            [], ["/compact", "/exit"], stderr=err,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 0)
        self.assertIn("nothing to compact", err.getvalue())

    def test_slash_cost_without_cap_reports_no_tracking(self):
        err = io.StringIO()
        rc, fake, calls = _run_repl(
            _args(interactive=True),
            [], ["/cost", "/exit"], stderr=err,
        )
        self.assertEqual(rc, 0)
        self.assertIn("no session cost tracking", err.getvalue())

    def test_slash_cost_with_cap_reports_running_total(self):
        # Turn 1 streams a reply with usage; /cost then reports it. The test
        # catalog advertises no pricing, so the ledger is unpriced (tokens only).
        err = io.StringIO()
        results = [[FakeChunk("hi"),
                    FakeChunk(usage={"prompt_tokens": 1000, "completion_tokens": 500,
                                     "total_tokens": 1500})]]
        rc, fake, calls = _run_repl(
            _args(interactive=True, session_max_spend=1.0),
            results, ["hey", "/cost", "/exit"], stderr=err,
            urlopen=_urlopen_ok(),
        )
        self.assertEqual(rc, 0)
        self.assertIn("unpriced", err.getvalue())
        self.assertIn("prompt=1000", err.getvalue())
        self.assertIn("completion=500", err.getvalue())

    def test_slash_compact_then_turn_sees_summary(self):
        with tempfile.TemporaryDirectory() as d:
            resume = self._resume_history(d, pairs=6)
            results = [
                FakeToolCompletion("summary of u0..u3"),  # the /compact turn
                [FakeChunk("reply")],                      # the next chat turn
            ]
            rc, fake, calls = _run_repl(
                _args(interactive=True, resume=resume),
                results, ["/compact 2", "next question", "/exit"],
                stderr=io.StringIO(),
            )
            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 2)
            # The chat turn's history carries the summary as a system message
            # plus the kept tail -- not the original six pairs.
            msgs = calls[1]["messages"]
            self.assertEqual(msgs[0]["role"], "system")
            self.assertIn("summary of u0..u3", msgs[0]["content"])
            self.assertLess(len(msgs), 13)

    def test_auto_compact_fires_before_overbudget_turn(self):
        with tempfile.TemporaryDirectory() as d:
            resume = self._resume_history(d, pairs=6)
            err = io.StringIO()
            results = [
                # Turn 1: a normal streamed reply whose usage crosses the budget.
                [FakeChunk("r1"),
                 FakeChunk(usage={"prompt_tokens": 5000, "completion_tokens": 2,
                                  "total_tokens": 5002})],
                # Turn 2's auto-compact summarization call.
                FakeToolCompletion("compact summary"),
                # Turn 2 itself.
                [FakeChunk("r2")],
            ]
            rc, fake, calls = _run_repl(
                _args(interactive=True, resume=resume,
                      auto_compact=True, compact_threshold=1000,
                      compact_keep_turns=2),
                results, ["first", "second", "/exit"], stderr=err,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[1]["tool_choice"], "none")  # the compact call
            self.assertIn("auto-compacted", err.getvalue())
            # Turn 2 saw the compacted history.
            msgs = calls[2]["messages"]
            self.assertTrue(any(
                m.get("role") == "system" and "compact summary" in str(m.get("content"))
                for m in msgs
            ))

    def test_malformed_resume_exit_2(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.json"
            path.write_text('{"not": "a list"}')
            err = io.StringIO()
            rc, fake, calls = _run_repl(
                _args(interactive=True, resume=str(path)),
                [], ["/exit"], stderr=err,
            )
        self.assertEqual(rc, 2)
        self.assertIn("list of message objects", err.getvalue())

    def test_tools_turn_runs_agent_loop(self):
        out = io.StringIO()
        seq = [
            FakeToolCompletion(tool_calls=[
                _FnCall("c1", "venice_chat", '{"message": "hola"}')]),
            FakeToolCompletion("final"),
        ]
        with mock.patch(
            "venice.commands._mcp.chat_tool",
            return_value={"status": "ok", "content": "hola"},
        ):
            rc, fake, calls = _run_repl(
                _args(interactive=True, tools=True),
                seq, ["do it", "/exit"], stdout=out,
            )
        self.assertEqual(rc, 0)
        self.assertIn("final", out.getvalue())
        # the tool round-trip is in the history the second create() saw
        tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("hola", tool_msgs[0]["content"])

    def test_mcp_repl_attaches_external_tools(self):
        # `--mcp NAME` turns the REPL into an agent session; the remote tool is
        # advertised alongside the built-ins and its result flows back.
        out = io.StringIO()
        seq = [
            FakeToolCompletion(tool_calls=[
                _FnCall("c1", "fs__read", '{"path": "/x"}')]),
            FakeToolCompletion("done"),
        ]
        cfg = {"version": 1, "mcpServers": {"fs": {"command": "srv"}}, "defaults": {}}
        attach = _fake_attach_cm(
            [_fake_tool("fs__read", {"status": "ok", "content": "data"})]
        )
        rc, fake, calls = _run_repl(
            _args(interactive=True, mcp=["fs"]),
            seq, ["do it", "/exit"], stdout=out, cfg=cfg, attach=attach,
        )
        self.assertEqual(rc, 0)
        names = {t["function"]["name"] for t in calls[0]["tools"]}
        self.assertIn("fs__read", names)
        self.assertIn("venice_image", names)
        tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
        self.assertIn("data", tool_msgs[0]["content"])

    def test_ctrl_c_aborts_turn_keeps_session(self):
        # First turn is interrupted mid-flight; the session survives and the
        # aborted turn is rolled out of history so the next turn is clean.
        from venice.commands import chat
        calls = []
        seq = [KeyboardInterrupt(), [FakeChunk("recovered")]]
        fake = mock.MagicMock()

        def _create(**kw):
            snap = dict(kw)
            if "messages" in snap:
                snap["messages"] = copy.deepcopy(snap["messages"])
            calls.append(snap)
            item = seq.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        fake.chat.completions.create.side_effect = _create
        err = io.StringIO()
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.userconfig.load_config", lambda *a, **k: _EMPTY_CFG), \
             mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch("builtins.input", side_effect=["boom", "again", "/exit"]), \
             mock.patch.object(sys, "stdin", io.StringIO("")), \
             mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", err):
            rc = chat._run(_args(interactive=True))
        self.assertEqual(rc, 0)
        self.assertIn("aborted", err.getvalue())
        # the interrupted first turn left no residue -> second call sees only "again"
        self.assertEqual([m["role"] for m in calls[1]["messages"]], ["user"])
        self.assertEqual(calls[1]["messages"][0]["content"], "again")

    # ------------------------------------------------------------------ #
    # #39: /models listing + bare /model listing
    # ------------------------------------------------------------------ #
    def test_slash_models_lists_catalog(self):
        err = io.StringIO()
        rc, fake, calls = _run_repl(
            _args(interactive=True), [], ["/models", "/exit"], stderr=err,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 0)  # listing never calls the model
        out = err.getvalue()
        self.assertIn("llama-3.3-70b", out)
        self.assertIn("venice-uncensored", out)
        self.assertIn("(default)", out)           # default-trait model marked
        self.assertIn("* llama-3.3-70b", out)     # current model marked

    def test_bare_model_shows_current_and_lists(self):
        err = io.StringIO()
        rc, fake, calls = _run_repl(
            _args(interactive=True), [], ["/model", "/exit"], stderr=err,
        )
        self.assertEqual(rc, 0)
        out = err.getvalue()
        self.assertIn("model: llama-3.3-70b", out)  # current still shown
        self.assertIn("venice-uncensored", out)     # ...plus the catalog list

    # ------------------------------------------------------------------ #
    # #55: /auto and /manual toggle per-turn auto-accept; banner shows state
    # ------------------------------------------------------------------ #
    def test_slash_auto_and_manual_toggle_state(self):
        state = {"model": "m", "tools": [], "tools_on": True, "yes": False,
                 "max_tool_calls": 8}
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            _repl._dispatch_slash("/auto", [], state, _args(interactive=True), [])
            self.assertTrue(state["yes"])
            _repl._dispatch_slash("/manual", [], state, _args(interactive=True), [])
            self.assertFalse(state["yes"])
        self.assertIn("auto-accept on", err.getvalue())
        self.assertIn("auto-accept off", err.getvalue())

    def test_slash_auto_noop_without_tools(self):
        state = {"model": "m", "tools": None, "tools_on": False, "yes": False,
                 "max_tool_calls": 8}
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            _repl._dispatch_slash("/auto", [], state, _args(interactive=True), [])
        self.assertFalse(state["yes"])            # nothing to auto-accept
        self.assertIn("no tools", err.getvalue())

    def test_banner_shows_auto_off_by_default_with_tools(self):
        err = io.StringIO()
        _run_repl(_args(interactive=True, tools=True), [], ["/exit"], stderr=err)
        self.assertIn("auto-accept off", err.getvalue())

    def test_banner_shows_auto_on_with_yes_flag(self):
        err = io.StringIO()
        _run_repl(_args(interactive=True, tools=True, yes=True), [], ["/exit"], stderr=err)
        self.assertIn("auto-accept on", err.getvalue())


class _FakeRL:
    """Minimal readline stand-in: the completer needs only these two hooks."""

    def __init__(self, buffer, begidx, *, doc=""):
        self._buffer = buffer
        self._begidx = begidx
        self.__doc__ = doc

    def get_line_buffer(self):
        return self._buffer

    def get_begidx(self):
        return self._begidx


_CATALOG = [
    {"id": "llama-3.3-70b", "model_spec": {"traits": ["default"]}},
    {"id": "venice-uncensored", "model_spec": {"traits": []}},
]


class TestReplCompletion(unittest.TestCase):
    """#40: the readline completer (unit-tested via an injected fake rl)."""

    def _complete_all(self, buffer, begidx, text):
        comp = _repl._make_completer(_CATALOG, _FakeRL(buffer, begidx))
        out, state = [], 0
        while (m := comp(text, state)) is not None:
            out.append(m)
            state += 1
        return out

    def test_commands_include_models(self):
        self.assertIn("/models", _repl._COMMANDS)

    def test_commands_include_auto_manual(self):
        self.assertIn("/auto", _repl._COMMANDS)
        self.assertIn("/manual", _repl._COMMANDS)

    def test_completes_slash_command(self):
        self.assertEqual(self._complete_all("/mo", 0, "/mo"), ["/model", "/models"])

    def test_completes_model_id_after_model(self):
        self.assertEqual(
            self._complete_all("/model ven", 7, "ven"), ["venice-uncensored"]
        )

    def test_bare_model_space_lists_all_ids(self):
        self.assertEqual(
            self._complete_all("/model ", 7, ""),
            ["llama-3.3-70b", "venice-uncensored"],
        )

    def test_no_completion_off_slash_lines(self):
        self.assertEqual(self._complete_all("hello wor", 6, "wor"), [])

    def test_other_commands_get_no_model_ids(self):
        # after `/save ` we offer nothing (filename completion is deferred)
        self.assertEqual(self._complete_all("/save ", 6, ""), [])

    def test_completer_tolerates_empty_catalog(self):
        comp = _repl._make_completer(None, _FakeRL("/model ", 7))
        self.assertIsNone(comp("", 0))  # no ids, no crash

    def test_install_completer_restores_on_exit(self):
        class RL:
            def __init__(self):
                self.completer = "PREV"
                self.delims = "PREVDELIMS"
                self.bound = None
                self.__doc__ = ""  # not libedit -> "tab: complete"

            def get_completer(self):
                return self.completer

            def get_completer_delims(self):
                return self.delims

            def set_completer(self, c):
                self.completer = c

            def set_completer_delims(self, d):
                self.delims = d

            def parse_and_bind(self, s):
                self.bound = s

        rl = RL()
        with contextlib.ExitStack() as stack:
            _repl._install_completer(rl, _CATALOG, stack)
            self.assertTrue(callable(rl.completer))   # our completer is installed
            self.assertEqual(rl.delims, " \t\n")
            self.assertEqual(rl.bound, "tab: complete")
        # leaving the REPL restores the prior completer + delims (no leak)
        self.assertEqual(rl.completer, "PREV")
        self.assertEqual(rl.delims, "PREVDELIMS")


if __name__ == "__main__":
    unittest.main()
