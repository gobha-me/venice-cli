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
    _fake_attach_cm,
    _fake_openai_seq,
    _fake_tool,
    _urlopen_ok,
    _args,
)

_EMPTY_CFG = {"version": 1, "mcpServers": {}, "defaults": {}}


def _run_repl(args, results, inputs, *, stdout=None, stderr=None,
              urlopen=None, stdin=None, cfg=None, attach=None):
    """Run the REPL: `results` are returned by successive create() calls,
    `inputs` are fed to input(). Returns (rc, fake_client, recorded_calls).

    `cfg` overrides the (empty) config doc; `attach` patches the MCP client seam."""
    from venice.commands import chat
    fake, calls = _fake_openai_seq(results)
    with contextlib.ExitStack() as st:
        st.enter_context(mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}))
        st.enter_context(mock.patch("venice.userconfig.load_config",
                                    lambda *a, **k: cfg or _EMPTY_CFG))
        st.enter_context(mock.patch("venice.client.urllib.request.urlopen",
                                    urlopen or _urlopen_ok()))
        st.enter_context(mock.patch("openai.OpenAI", return_value=fake))
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


if __name__ == "__main__":
    unittest.main()
