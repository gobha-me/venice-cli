"""Unit tests for the interactive `venice chat` REPL (issue #22).

Drives `chat._run` in interactive mode with scripted `input()` lines and a fake
OpenAI client, exactly like `test_chat.py` (no network, no real key). Reuses that
module's fakes so the two stay in lock-step.
"""
import contextlib
import copy
import io
import itertools
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

from venice.commands import _agent, _compact, _repl  # noqa: E402

_EMPTY_CFG = {"version": 1, "mcpServers": {}, "defaults": {}}

# Auto-save is on by default (#47). Point the whole module's session store at a
# throwaway dir so any test that drives the REPL (even via an inline harness that
# doesn't set VENICE_SESSIONS_DIR itself) never writes to ~/.config/venice/sessions.
_SESSIONS_TMP = None


def setUpModule():
    global _SESSIONS_TMP
    _SESSIONS_TMP = tempfile.mkdtemp()
    os.environ["VENICE_SESSIONS_DIR"] = _SESSIONS_TMP
    # #49: also redirect the global memory tier so a REPL that ever writes memory
    # can't leak to ~/.config/venice/memory (belt-and-suspenders, like sessions).
    os.environ["VENICE_MEMORY_DIR"] = os.path.join(_SESSIONS_TMP, "memory")


def tearDownModule():
    os.environ.pop("VENICE_SESSIONS_DIR", None)
    os.environ.pop("VENICE_MEMORY_DIR", None)
    if _SESSIONS_TMP:
        __import__("shutil").rmtree(_SESSIONS_TMP, ignore_errors=True)


class _InterruptingStderr(io.StringIO):
    """A stderr whose next write raises KeyboardInterrupt, once (#92).

    Reproduces the exact interleaving in the reported traceback: `input()` raises
    EOFError, and the SIGINT is delivered on the very next bytecode -- the
    `print(file=sys.stderr)` that opens the ^D handler. Arm it from the `input`
    side_effect immediately before raising EOFError. `fired` exists so a test can
    prove the interrupt actually happened rather than passing vacuously.
    """

    def __init__(self):
        super().__init__()
        self._armed = False
        self.fired = False

    def arm(self):
        self._armed = True

    def write(self, s):
        if self._armed:
            self._armed = False
            self.fired = True
            raise KeyboardInterrupt
        return super().write(s)


def _fake_clock(step=1.5):
    """Patch `_repl`'s monotonic clock with a deterministic counter (#81).

    A *callable*, not a `side_effect` list: the number of clock reads is an
    implementation detail, and a list would raise StopIteration the moment the
    production code took one more or one fewer -- turning a timing test into a
    tripwire on unrelated refactors. A counter just yields a different (still
    non-zero, still monotonic) duration.
    """
    ticks = itertools.count(0.0, step)
    return mock.patch("venice.commands._repl.time.monotonic",
                      side_effect=lambda: next(ticks))


def _run_repl(args, results, inputs, *, stdout=None, stderr=None,
              urlopen=None, stdin=None, cfg=None, attach=None, mcp_probe=_MCP_PRESENT,
              sessions_dir=None):
    """Run the REPL: `results` are returned by successive create() calls,
    `inputs` are fed to input(). Returns (rc, fake_client, recorded_calls).

    `cfg` overrides the (empty) config doc; `attach` patches the MCP client seam;
    `mcp_probe` is what `import_mcp` returns (SDK-independent, like test_chat).
    `sessions_dir` pins the session store (#47) -- pass the same dir across two
    calls to test resume-by-id; omit for a throwaway per-call store."""
    from venice.commands import chat
    fake, calls = _fake_openai_seq(results)
    with contextlib.ExitStack() as st:
        # Auto-save is on by default (#47) -- keep it hermetic: point the session
        # store at a throwaway dir so tests never touch ~/.config/venice/sessions.
        _sess_dir = sessions_dir or st.enter_context(tempfile.TemporaryDirectory())
        st.enter_context(mock.patch.dict(
            os.environ,
            {"VENICE_API_KEY": "fake", "VENICE_SESSIONS_DIR": _sess_dir},
        ))
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

    # ----------------------------------------------------------------- #
    # #92: the ^D teardown used to be an unguarded window
    # ----------------------------------------------------------------- #
    def test_eof_autosaves_a_slash_only_edit(self):
        # /model commits no turn, so nothing autosaves during the session -- the ^D
        # exit is the only flush. Baseline for the interrupted case below.
        with tempfile.TemporaryDirectory() as d:
            rc, fake, calls = _run_repl(
                _args(interactive=True),
                [], ["/model venice-uncensored", EOFError], sessions_dir=d,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 0)          # no turn was ever taken
            env = json.loads(list(Path(d).glob("*.json"))[0].read_text())
            self.assertEqual(env["model"], "venice-uncensored")

    def test_ctrl_c_inside_eof_teardown_still_autosaves(self):
        """A SIGINT delivered *inside* the ^D handler (#92).

        The regression test. `except KeyboardInterrupt` is a sibling clause of
        `except EOFError`, so it never guarded the handler's own body: the interrupt
        escaped `run()` entirely, printed a traceback out of `cli.main`, and -- the
        part that wasn't cosmetic -- skipped the `_autosave` that sat on the next
        line. The `/model` switch below is what used to be lost.
        """
        with tempfile.TemporaryDirectory() as d:
            err = _InterruptingStderr()
            lines = iter(["/model venice-uncensored"])

            def _input(prompt=""):
                try:
                    return next(lines)
                except StopIteration:
                    # ^D, and a SIGINT that lands on the handler's first statement.
                    err.arm()
                    raise EOFError

            rc, fake, calls = _run_repl(
                _args(interactive=True), [], _input, sessions_dir=d, stderr=err,
            )
            self.assertEqual(rc, 130)
            self.assertIn("aborted", err.getvalue())
            self.assertTrue(err.fired, "the fake stderr never raised; test is vacuous")
            # The point of the fix: the session survived the interrupt.
            env = json.loads(list(Path(d).glob("*.json"))[0].read_text())
            self.assertEqual(env["model"], "venice-uncensored")

    def test_ephemeral_ctrl_c_teardown_writes_nothing(self):
        # The `finally` runs, but --ephemeral has no active session, so _autosave
        # no-ops. 130 without a file.
        with tempfile.TemporaryDirectory() as d:
            err = _InterruptingStderr()

            def _input(prompt=""):
                err.arm()
                raise EOFError

            rc, fake, calls = _run_repl(
                _args(interactive=True, ephemeral=True), [], _input,
                sessions_dir=d, stderr=err,
            )
            self.assertEqual(rc, 130)
            self.assertEqual(list(Path(d).glob("*.json")), [])

    def test_exit_autosaves_exactly_once(self):
        # The flush moved into a `finally`; the inline copies were dropped. Guard
        # against it being done twice (or the /exit copy being left behind).
        with tempfile.TemporaryDirectory() as d:
            saves = []
            real_save = _repl._session.save
            with mock.patch.object(_repl._session, "save",
                                   side_effect=lambda s: (saves.append(s.id),
                                                          real_save(s))[1]):
                rc, fake, calls = _run_repl(
                    _args(interactive=True), [], ["/model venice-uncensored", "/exit"],
                    sessions_dir=d,
                )
            self.assertEqual(rc, 0)
            self.assertEqual(len(saves), 1)

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

    def test_autosave_and_resume_by_id(self):
        # A session auto-saves every turn (no /save); --resume <id> restores it (#47).
        with tempfile.TemporaryDirectory() as d:
            rc, fake, calls = _run_repl(
                _args(interactive=True),
                [[FakeChunk("noted")]], ["remember X", "/exit"],
                sessions_dir=d,
            )
            self.assertEqual(rc, 0)
            files = list(Path(d).glob("*.json"))
            self.assertEqual(len(files), 1)              # one session auto-saved
            env = json.loads(files[0].read_text())
            sid = env["id"]
            self.assertEqual(env["command"], "chat")
            self.assertEqual(env["messages"][0]["content"], "remember X")
            self.assertEqual(env["messages"][1]["content"], "noted")

            rc2, fake2, calls2 = _run_repl(
                _args(interactive=True, resume=sid),
                [[FakeChunk("you said X")]], ["what did I say", "/exit"],
                sessions_dir=d,
            )
            self.assertEqual(rc2, 0)
            roles = [m["role"] for m in calls2[0]["messages"]]
            self.assertEqual(roles, ["user", "assistant", "user"])
            self.assertEqual(calls2[0]["messages"][0]["content"], "remember X")
            # resume updates the SAME file in place (id kept); now 4 messages.
            self.assertEqual(len(list(Path(d).glob("*.json"))), 1)
            env2 = json.loads((Path(d) / (sid + ".json")).read_text())
            self.assertEqual(len(env2["messages"]), 4)

    def test_streamed_reasoning_and_signature_replay_save_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            results = [
                [
                    FakeChunk(
                        reasoning_content="think ",
                        thought_signature="sig-",
                    ),
                    FakeChunk(
                        "answer",
                        reasoning_content="carefully",
                        thought_signature="exact",
                    ),
                ],
                [FakeChunk("follow-up")],
            ]
            rc, _fake, calls = _run_repl(
                _args(interactive=True),
                results,
                ["first", "second", "/exit"],
                sessions_dir=d,
            )
            self.assertEqual(rc, 0)
            replayed = calls[1]["messages"][1]
            self.assertEqual(replayed["content"], "answer")
            self.assertEqual(replayed["reasoning_content"], "think carefully")
            self.assertEqual(replayed["thought_signature"], "sig-exact")

            session_path = list(Path(d).glob("*.json"))[0]
            saved = json.loads(session_path.read_text())
            self.assertEqual(
                saved["messages"][1]["reasoning_content"],
                "think carefully",
            )
            self.assertEqual(
                saved["messages"][1]["thought_signature"],
                "sig-exact",
            )

            rc2, _fake2, calls2 = _run_repl(
                _args(interactive=True, resume=saved["id"]),
                [[FakeChunk("resumed")]],
                ["third", "/exit"],
                sessions_dir=d,
            )
            self.assertEqual(rc2, 0)
            self.assertEqual(
                calls2[0]["messages"][1]["thought_signature"],
                "sig-exact",
            )

    def test_ephemeral_writes_no_file(self):
        with tempfile.TemporaryDirectory() as d:
            rc, fake, calls = _run_repl(
                _args(interactive=True, ephemeral=True),
                [[FakeChunk("hi")]], ["yo", "/exit"],
                sessions_dir=d,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(list(Path(d).glob("*.json")), [])

    def test_ephemeral_on_resume_loads_but_does_not_write_back(self):
        # --ephemeral must suppress auto-save even when resuming (read-only resume).
        with tempfile.TemporaryDirectory() as d:
            rc, _f, _c = _run_repl(
                _args(interactive=True),
                [[FakeChunk("noted")]], ["remember X", "/exit"], sessions_dir=d,
            )
            self.assertEqual(rc, 0)
            path = list(Path(d).glob("*.json"))[0]
            sid = json.loads(path.read_text())["id"]
            before = path.read_bytes()

            rc2, _f2, calls2 = _run_repl(
                _args(interactive=True, resume=sid, ephemeral=True),
                [[FakeChunk("you said X")]], ["what did I say", "/exit"],
                sessions_dir=d,
            )
            self.assertEqual(rc2, 0)
            # context WAS restored (resume still loads) ...
            self.assertEqual(calls2[0]["messages"][0]["content"], "remember X")
            # ... but nothing was written back: same file, unchanged, no new session.
            self.assertEqual(len(list(Path(d).glob("*.json"))), 1)
            self.assertEqual(path.read_bytes(), before)

    def test_resume_restores_model_unless_overridden(self):
        with tempfile.TemporaryDirectory() as d:
            rc, fake, calls = _run_repl(
                _args(interactive=True, model="venice-uncensored"),
                [[FakeChunk("ok")]], ["hi", "/exit"],
                sessions_dir=d,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(calls[0]["model"], "venice-uncensored")   # non-default
            sid = json.loads(list(Path(d).glob("*.json"))[0].read_text())["id"]

            # resume WITHOUT --model -> the saved model drives the turn
            rc2, fake2, calls2 = _run_repl(
                _args(interactive=True, resume=sid),
                [[FakeChunk("ok2")]], ["again", "/exit"],
                sessions_dir=d,
            )
            self.assertEqual(calls2[0]["model"], "venice-uncensored")

            # explicit --model on resume overrides the saved one (precedence).
            rc3, fake3, calls3 = _run_repl(
                _args(interactive=True, resume=sid, model="llama-3.3-70b"),
                [[FakeChunk("ok3")]], ["again2", "/exit"],
                sessions_dir=d,
            )
            self.assertEqual(calls3[0]["model"], "llama-3.3-70b")

    def test_usage_carries_across_resume(self):
        with tempfile.TemporaryDirectory() as d:
            rc, fake, calls = _run_repl(
                _args(interactive=True),
                [[FakeChunk("a"),
                  FakeChunk(usage={"prompt_tokens": 100, "completion_tokens": 10,
                                   "total_tokens": 110})]],
                ["hi", "/exit"],
                sessions_dir=d,
            )
            self.assertEqual(rc, 0)
            env = json.loads(list(Path(d).glob("*.json"))[0].read_text())
            self.assertEqual(env["usage"]["prompt_tokens"], 100)
            sid = env["id"]

            err = io.StringIO()
            rc2, fake2, calls2 = _run_repl(
                _args(interactive=True, resume=sid),
                [[FakeChunk("b"),
                  FakeChunk(usage={"prompt_tokens": 50, "completion_tokens": 5,
                                   "total_tokens": 55})]],
                ["more", "/usage", "/exit"], stderr=err,
                sessions_dir=d,
            )
            self.assertEqual(rc2, 0)
            # cumulative prompt tokens = 100 (restored) + 50 (this turn) = 150
            self.assertIn("150", err.getvalue())

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

    def test_slash_cost_without_cap_reports_running_total(self):
        # The REPL ledger is always-on now (#75): /cost reports a total even
        # without --session-max-spend -- just no cap line.
        err = io.StringIO()
        rc, fake, calls = _run_repl(
            _args(interactive=True),
            [], ["/cost", "/exit"], stderr=err,
        )
        self.assertEqual(rc, 0)
        out = err.getvalue()
        self.assertNotIn("no session cost tracking", out)
        self.assertIn("cost:", out)
        self.assertNotIn("cap", out)  # no cap set -> no cap line

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

    def test_slash_usage_reports_cache_split(self):
        # A cache-heavy streamed turn, then /usage: the breakdown keeps the
        # cached vs uncached input split distinct (#75).
        err = io.StringIO()
        results = [[FakeChunk("hi"),
                    FakeChunk(usage={
                        "prompt_tokens": 10000, "completion_tokens": 500,
                        "total_tokens": 10500,
                        "prompt_tokens_details": {"cached_tokens": 9000},
                    })]]
        rc, fake, calls = _run_repl(
            _args(interactive=True),
            results, ["hey", "/usage", "/exit"], stderr=err,
            urlopen=_urlopen_ok(),
        )
        self.assertEqual(rc, 0)
        out = err.getvalue()
        self.assertIn("session usage:", out)
        self.assertIn("9,000 cache-read", out)
        self.assertIn("1,000 uncached", out)
        self.assertIn("cache hit rate: 90.0%", out)

    def test_slash_usage_shows_the_tool_breakdown(self):
        # #82 end-to-end, in-process: a real tools turn dispatches `venice_chat`
        # through `run_loop`, and /usage reports the row. The only test proving the
        # sink survives the whole path rather than just `_run_one_call` in isolation.
        err = io.StringIO()
        seq = [
            FakeToolCompletion(tool_calls=[
                _FnCall("c1", "venice_chat", '{"message": "hola"}')]),
            FakeToolCompletion("final", usage={"prompt_tokens": 100,
                                               "completion_tokens": 5}),
        ]
        with mock.patch(
            "venice.commands._mcp.chat_tool",
            return_value={"status": "ok", "content": "hola"},
        ):
            rc, fake, calls = _run_repl(
                _args(interactive=True, tools=True),
                seq, ["do it", "/usage", "/exit"], stderr=err,
            )
        self.assertEqual(rc, 0)
        out = err.getvalue()
        self.assertIn("  tools ", out)
        self.assertRegex(out, r"\n    venice_chat\s+[\d.]+s\s+1 call\(s\)")

    def test_slash_usage_reports_unknown_cache_state(self):
        # #98 end-to-end, in-process: a streamed turn whose usage carries no
        # `prompt_tokens_details` at all must reach /usage as "unknown", not 0%.
        # This is the only test proving the distinction survives the real path --
        # chunk.usage -> SDK object -> model_dump() -> CostLedger.record().
        err = io.StringIO()
        results = [[FakeChunk("hi"),
                    FakeChunk(usage={
                        "prompt_tokens": 10000, "completion_tokens": 500,
                        "total_tokens": 10500,
                    })]]
        rc, fake, calls = _run_repl(
            _args(interactive=True),
            results, ["hey", "/usage", "/exit"], stderr=err,
            urlopen=_urlopen_ok(),
        )
        self.assertEqual(rc, 0)
        out = err.getvalue()
        self.assertIn("cache hit rate: n/a (no cache fields reported)", out)
        self.assertIn("cache breakdown not reported", out)
        self.assertNotIn("cache hit rate: 0.0%", out)

    def test_slash_usage_shows_the_call_trace(self):
        # #99 end-to-end, in-process: a streamed turn's usage reaches /usage as a
        # per-call row, and the row's window is stamped -- the only test proving the
        # `_stream_turn` bracket survives chunk.usage -> model_dump() -> record().
        err = io.StringIO()
        results = [[FakeChunk("hi"),
                    FakeChunk(usage={
                        "prompt_tokens": 10000, "completion_tokens": 500,
                        "total_tokens": 10500,
                        "prompt_tokens_details": {"cached_tokens": 9000},
                    })]]
        rc, fake, calls = _run_repl(
            _args(interactive=True),
            results, ["hey", "/usage", "/exit"], stderr=err,
            urlopen=_urlopen_ok(),
        )
        self.assertEqual(rc, 0)
        out = err.getvalue()
        self.assertIn("  calls ", out)
        self.assertRegex(out, r"\n    #1\s+10,000 in\s+90% cached\s+500 out\s+[\d.]+s")
        # Not `n/a`: the streamed path brackets its own window.
        self.assertNotRegex(out, r"\n    #1[^\n]*out\s+n/a")

    def test_a_tools_turn_records_more_calls_than_turns(self):
        # `usage.turns` counts operator waits, `api_calls_total` counts model calls.
        # One REPL turn that dispatches a tool makes TWO API calls -- pin the
        # distinction end-to-end so the two counters cannot be collapsed later.
        err = io.StringIO()
        seq = [
            FakeToolCompletion(tool_calls=[
                _FnCall("c1", "venice_chat", '{"message": "hola"}')]),
            FakeToolCompletion("final", usage={"prompt_tokens": 100,
                                               "completion_tokens": 5}),
        ]
        with mock.patch(
            "venice.commands._mcp.chat_tool",
            return_value={"status": "ok", "content": "hola"},
        ):
            rc, fake, calls = _run_repl(
                _args(interactive=True, tools=True),
                seq, ["do it", "/usage", "/exit"], stderr=err,
            )
        self.assertEqual(rc, 0)
        out = err.getvalue()
        self.assertRegex(out, r"\n  calls[^\n]*across 2 API call\(s\)")
        self.assertRegex(out, r"\n  wall[^\n]*over 1 turn\(s\)")

    def test_slash_compact_records_a_manual_context_event(self):
        # `/compact` bypasses `maybe_compact` entirely, so it is the call site most
        # likely to be missed. `trigger` is what tells a self-inflicted sawtooth from
        # an automatic one afterwards.
        with tempfile.TemporaryDirectory() as d:
            resume = self._resume_history(d, pairs=6)
            err = io.StringIO()
            rc, fake, calls = _run_repl(
                _args(interactive=True, resume=resume),
                [FakeToolCompletion("we discussed u0..u5")],
                ["/compact 2", "/usage", "/exit"], stderr=err,
            )
            self.assertEqual(rc, 0)
            out = err.getvalue()
            # Whole line, not a prefix regex: `assertRegex` here was blind to
            # anything appended after "msgs", which is exactly where #101's cost
            # clause lands. The fake carries no usage, so the cost is falsy and the
            # clause must be ABSENT -- "$0.0000" would be #98's fabricated zero.
            self.assertIn(
                "    -- compacted (manual) after #0: 12 -> 5 msgs,"
                " ~60 -> ~38 tok est",
                out.split("\n"))
            # No budget in play, so the estimate must not claim to be measured.
            self.assertNotIn("measured before", out)

    def test_slash_compact_clears_the_stale_observed_count(self):
        # #116: `/compact` used to hand-copy the budget reset that `maybe_compact` also
        # hand-copied. `compact_messages` owns it now, and nothing pinned this path
        # before -- a reset that stayed behind in only one of the two files would leave
        # `Budget.over` reading a count larger than the history it now describes, so the
        # very next turn compacts again. Sawtooth, forever.
        held = []
        real = _compact.budget_from_args

        def _capture(args):
            b = real(args) or _compact.Budget(threshold_tokens=10**9, keep_turns=2)
            b.last_prompt_tokens = 88110
            held.append(b)
            return b

        with tempfile.TemporaryDirectory() as d:
            resume = self._resume_history(d, pairs=6)
            with mock.patch.object(_compact, "budget_from_args", _capture):
                rc, fake, calls = _run_repl(
                    _args(interactive=True, resume=resume),
                    [FakeToolCompletion("we discussed u0..u5")],
                    ["/compact 2", "/exit"], stderr=io.StringIO(),
                )
            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 1)  # the compaction actually ran
            self.assertIsNone(held[0].last_prompt_tokens)

    def test_slash_usage_before_any_turn(self):
        err = io.StringIO()
        rc, fake, calls = _run_repl(
            _args(interactive=True),
            [], ["/usage", "/exit"], stderr=err,
        )
        self.assertEqual(rc, 0)
        self.assertIn("no usage recorded yet", err.getvalue())

    # -- wall-clock per turn (#81) ------------------------------------------ #

    def test_turn_records_its_wall_clock_window(self):
        # `_do_turn` brackets "input -> able to type again"; /usage reports the total.
        err = io.StringIO()
        with _fake_clock():
            rc, fake, calls = _run_repl(
                _args(interactive=True),
                [[FakeChunk("hi"),
                  FakeChunk(usage={"prompt_tokens": 10, "completion_tokens": 2})]],
                ["hey", "/usage", "/exit"], stderr=err, urlopen=_urlopen_ok(),
            )
        self.assertEqual(rc, 0)
        out = err.getvalue()
        self.assertIn("wall", out)
        self.assertIn("over 1 turn(s)", out)

    def test_aborted_turn_is_still_timed(self):
        # The stamp lives in `finally`, not `else`: a turn that burned time and then
        # raised still burned it, and those are the turns worth knowing about.
        # Inline harness (not _run_repl) because the shared fake returns its queued
        # items rather than raising -- same shape as test_ctrl_c_aborts_turn_keeps_session.
        from venice.commands import chat
        fake = mock.MagicMock()

        def _create(**kw):
            raise KeyboardInterrupt()

        fake.chat.completions.create.side_effect = _create
        err = io.StringIO()
        with _fake_clock(), \
             mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.userconfig.load_config", lambda *a, **k: _EMPTY_CFG), \
             mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch("builtins.input", side_effect=["boom", "/usage", "/exit"]), \
             mock.patch.object(sys, "stdin", io.StringIO("")), \
             mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", err):
            rc = chat._run(_args(interactive=True))
        self.assertEqual(rc, 0)
        out = err.getvalue()
        self.assertIn("[turn aborted]", out)
        self.assertIn("over 1 turn(s)", out)

    def test_spend_gated_turn_is_still_timed(self):
        # The gate returns early, BEFORE the try -- so only a `finally` on the outer
        # wrapper covers it. Driven through `_do_turn` directly (the harness the #79
        # steering test uses) because the shared fake catalog is unpriced, so a
        # cap can't be tripped through the full REPL.
        import openai
        from venice.commands import chat, _agent as _ag

        led = _ag.CostLedger(max_spend=0.01)
        led.bind_pricing({"input": {"usd": 1000.0}, "output": {"usd": 1000.0}})
        led.record({"prompt_tokens": 100000, "completion_tokens": 100000})
        self.assertTrue(led.over())          # precondition, not the assertion

        fake = mock.MagicMock()
        state = {"model": "m", "tools": [], "tools_on": False, "yes": True,
                 "max_tool_calls": 0, "session": None, "ledger": led}
        err = io.StringIO()
        with _fake_clock(), mock.patch.object(sys, "stderr", err):
            _repl._do_turn(fake, openai, chat, "do it", [], {}, state,
                           _args(interactive=True))
        self.assertIn("max-spend reached", err.getvalue())     # the turn was skipped
        fake.chat.completions.create.assert_not_called()       # ...genuinely skipped
        self.assertEqual(led.turns, 1)                         # ...and still counted
        self.assertGreater(led.elapsed_seconds, 0)

    def test_a_capped_session_does_not_compact_before_refusing_the_turn(self):
        # #101: compaction costs money now, so the spend gate has to run FIRST. With
        # the old order a capped session paid to summarize and then refused the very
        # turn the summary was for -- and did it again on the next message, and the
        # next, spending without bound on work it would never use. `run_loop` has
        # always gated first; this is the REPL agreeing.
        import openai
        from venice.commands import chat, _agent as _ag

        led = _ag.CostLedger(max_spend=0.01)
        led.bind_pricing({"input": {"usd": 1000.0}, "output": {"usd": 1000.0}})
        led.record({"prompt_tokens": 100000, "completion_tokens": 100000})
        self.assertTrue(led.over())          # precondition, not the assertion

        # A budget that is over on any history at all, so compaction WOULD fire.
        budget = _compact.Budget(threshold_tokens=1, keep_turns=1)
        msgs = []
        for i in range(6):
            msgs.append({"role": "user", "content": f"u{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})

        fake = mock.MagicMock()
        state = {"model": "m", "tools": [], "tools_on": False, "yes": True,
                 "max_tool_calls": 0, "session": None, "ledger": led,
                 "budget": budget}
        err = io.StringIO()
        with _fake_clock(), mock.patch.object(sys, "stderr", err):
            _repl._do_turn(fake, openai, chat, "do it", msgs, {}, state,
                           _args(interactive=True))
        self.assertIn("max-spend reached", err.getvalue())
        # THE assertion: not one API call, so not one cent. The summarization call
        # would have been the only `create()` here.
        self.assertEqual(fake.chat.completions.create.call_count, 0)
        self.assertEqual(led.bucket_calls("compaction"), 0)
        self.assertNotIn("auto-compacted", err.getvalue())

    def _turn_state(self, ledger, session=None):
        return {"model": "m", "tools": [], "tools_on": False, "yes": True,
                "max_tool_calls": 0, "session": session, "ledger": ledger}

    def test_unexpected_exception_still_records_the_time_it_burned(self):
        # THIS is what the `finally` buys. The two `except` clauses in `_turn` return
        # normally, so they'd be timed by a plain sequential stamp too; only an
        # exception no handler catches proves the stamp is unconditional. A turn that
        # burned 40s and then blew up still burned 40s.
        import openai
        from venice.commands import chat, _agent as _ag

        led = _ag.CostLedger()
        fake = mock.MagicMock()
        fake.chat.completions.create.side_effect = RuntimeError("boom")
        with _fake_clock(), mock.patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(RuntimeError):        # propagates, as before
                _repl._do_turn(fake, openai, chat, "do it", [], {},
                               self._turn_state(led), _args(interactive=True))
        self.assertEqual(led.turns, 1)
        self.assertGreater(led.elapsed_seconds, 0)

    def test_committed_turn_persists_its_own_elapsed_not_the_previous_one(self):
        # The guard for the `_turn`/`_do_turn` split. Python runs a try's `else`
        # BEFORE its `finally`, so autosaving inside the timed body would persist
        # every session's elapsed exactly one turn behind its token counts.
        # Driven per-turn rather than through the REPL because `/exit` autosaves
        # too, which would mask a one-turn lag at the only moment we could see it.
        import openai
        from venice.commands import chat, _agent as _ag, _session

        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"VENICE_SESSIONS_DIR": d}):
                led = _ag.CostLedger()
                sess = _session.new_session("chat", model="m")
                state = self._turn_state(led, session=sess)
                fake = mock.MagicMock()
                fake.chat.completions.create.side_effect = lambda **kw: iter(
                    [FakeChunk("ok"), FakeChunk(usage={"prompt_tokens": 3,
                                                       "completion_tokens": 1})]
                )
                with _fake_clock(), mock.patch.object(sys, "stdout", io.StringIO()), \
                        mock.patch.object(sys, "stderr", io.StringIO()):
                    _repl._do_turn(fake, openai, chat, "one", [], {}, state,
                                   _args(interactive=True))
                # After turn ONE the file must already carry turn one's clock.
                usage = json.loads((Path(d) / (sess.id + ".json")).read_text())["usage"]
                self.assertEqual(usage["turns"], 1)
                self.assertGreater(usage["elapsed_seconds"], 0)

    def test_aborted_turn_is_not_autosaved(self):
        # The other half of the `else` -> `return False` conversion: an aborted turn
        # rolls its messages back, so persisting it would write a session for a turn
        # that left no trace. Pre-#81 the `else` clause gave this for free.
        import openai
        from venice.commands import chat, _agent as _ag, _session

        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"VENICE_SESSIONS_DIR": d}):
                fake = mock.MagicMock()
                fake.chat.completions.create.side_effect = KeyboardInterrupt()
                sess = _session.new_session("chat", model="m")
                err = io.StringIO()
                with _fake_clock(), mock.patch.object(sys, "stderr", err):
                    _repl._do_turn(fake, openai, chat, "do it", [], {},
                                   self._turn_state(_ag.CostLedger(), session=sess),
                                   _args(interactive=True))
                self.assertIn("[turn aborted]", err.getvalue())
                self.assertEqual(list(Path(d).glob("*.json")), [])

    def test_spend_gated_turn_is_not_autosaved(self):
        # The gate's `return False` keeps a skipped turn out of the session, matching
        # the pre-#81 behaviour where the gate returned before the try's `else`.
        import openai
        from venice.commands import chat, _agent as _ag, _session

        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"VENICE_SESSIONS_DIR": d}):
                led = _ag.CostLedger(max_spend=0.01)
                led.bind_pricing({"input": {"usd": 1000.0}, "output": {"usd": 1000.0}})
                led.record({"prompt_tokens": 100000, "completion_tokens": 100000})
                sess = _session.new_session("chat", model="m")
                with _fake_clock(), mock.patch.object(sys, "stderr", io.StringIO()):
                    _repl._do_turn(mock.MagicMock(), openai, chat, "do it", [], {},
                                   self._turn_state(led, session=sess),
                                   _args(interactive=True))
                self.assertEqual(list(Path(d).glob("*.json")), [])

    def test_resumed_session_accumulates_wall_clock(self):
        # restore() is additive, so `--resume` reports the total time this session
        # has kept you waiting -- not just the latest leg.
        with tempfile.TemporaryDirectory() as d:
            with _fake_clock():
                rc, _, _ = _run_repl(
                    _args(interactive=True),
                    [[FakeChunk("noted")]], ["remember X", "/exit"],
                    sessions_dir=d, stderr=io.StringIO(),
                )
            self.assertEqual(rc, 0)
            sid = json.loads(list(Path(d).glob("*.json"))[0].read_text())["id"]
            err = io.StringIO()
            with _fake_clock():
                rc2, _, _ = _run_repl(
                    _args(interactive=True, resume=sid),
                    [[FakeChunk("you said X")]], ["what did I say", "/usage", "/exit"],
                    sessions_dir=d, stderr=err,
                )
            self.assertEqual(rc2, 0)
            self.assertIn("over 2 turn(s)", err.getvalue())

    def test_resume_replaces_stale_auxiliary_model_metadata(self):
        from venice.commands import _session

        selection = {"review": {"id": "reviewer", "source": "auto"}}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, {"VENICE_SESSIONS_DIR": d}), \
                mock.patch("builtins.input", return_value="/exit"), \
                mock.patch.object(sys, "stdout", io.StringIO()), \
                mock.patch.object(sys, "stderr", io.StringIO()):
            session = _session.new_session(
                "code", model="author", resolved_models={
                    "web_search": {"id": "stale", "source": "auto"},
                },
            )
            rc = _repl.run(
                _args(interactive=True), mock.MagicMock(), mock.MagicMock(),
                None, [], "author", session=session, resolved_models=selection,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(_session.load(session.id, "code").resolved_models,
                             selection)

            rc = _repl.run(
                _args(interactive=True), mock.MagicMock(), mock.MagicMock(),
                None, [], "author", session=session, resolved_models={},
            )
            self.assertEqual(rc, 0)
            self.assertEqual(_session.load(session.id, "code").resolved_models, {})

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

    def test_ctrlc_steer_injected_midturn_commits(self):
        # #79: on an attached tty the first Ctrl+C injects a steer at the checkpoint and
        # the turn CONTINUES (a second Ctrl+C would hit the rollback above). The real
        # signal/prompt machinery is in test_steer; here pause_and_steer is faked to
        # yield an injecting drain so we prove _do_turn wires it and commits the turn.
        import contextlib as _ctx
        import openai
        from venice.commands import chat, _steer
        from tests.test_agent import _free_tool, _tty

        @_ctx.contextmanager
        def _inject(session_id, *, enabled):
            n = {"i": 0}

            def _drain():
                n["i"] += 1
                return ["also: add a regression test"] if n["i"] == 1 else []

            yield _drain

        fake = mock.MagicMock()
        fake.chat.completions.create.side_effect = [
            FakeToolCompletion("done -- and I saw the steer"),
        ]
        messages = []
        state = {"model": "m", "tools": [_free_tool()], "tools_on": True,
                 "yes": True, "max_tool_calls": 0, "session": None}
        with mock.patch.object(_steer, "pause_and_steer", _inject), \
             mock.patch.object(sys, "stdin", _tty()), \
             mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", io.StringIO()):
            _repl._do_turn(fake, openai, chat, "do it", messages, {}, state,
                           _args(interactive=True))
        # committed, not rolled back: the final assistant turn survives...
        self.assertTrue(any(m.get("role") == "assistant" for m in messages))
        # ...and the injected steer is a tagged user turn in history.
        steers = [m for m in messages if m.get("role") == "user"
                  and "steering message received mid-run" in m.get("content", "")]
        self.assertEqual(len(steers), 1)
        self.assertIn("regression test", steers[0]["content"])

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

    # ------------------------------------------------------------------ #
    # #68: /persona loads a file-backed system prompt; lists; rejects escapes
    # ------------------------------------------------------------------ #
    def _personas_dir(self, **files):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name) / "personas"
        d.mkdir()
        for name, text in files.items():
            (d / name).write_text(text, encoding="utf-8")
        p = mock.patch("venice.config.PERSONAS_DIR", d)
        p.start()
        self.addCleanup(p.stop)
        return d

    def _state(self):
        return {"model": "m", "tools": [], "tools_on": True, "yes": False,
                "max_tool_calls": 8}

    def test_slash_persona_loads_file_as_system(self):
        self._personas_dir(**{"pirate.md": "You are a terse pirate."})
        messages = []
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            _repl._dispatch_slash("/persona pirate", messages, self._state(),
                                  _args(interactive=True), [])
        self.assertEqual(messages[0],
                         {"role": "system", "content": "You are a terse pirate."})
        self.assertIn("loaded", err.getvalue())

    def test_slash_persona_replaces_existing_system_keeps_history(self):
        self._personas_dir(**{"pirate.md": "pirate"})
        messages = [{"role": "system", "content": "old"},
                    {"role": "user", "content": "hi"}]
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            _repl._dispatch_slash("/persona pirate", messages, self._state(),
                                  _args(interactive=True), [])
        self.assertEqual(messages[0], {"role": "system", "content": "pirate"})
        self.assertEqual(messages[1], {"role": "user", "content": "hi"})  # kept

    def test_slash_persona_no_arg_lists(self):
        self._personas_dir(**{"pirate.md": "You are a pirate.",
                              "coach.txt": "Motivate."})
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            _repl._dispatch_slash("/persona", [], self._state(),
                                  _args(interactive=True), [])
        out = err.getvalue()
        self.assertIn("pirate", out)
        self.assertIn("You are a pirate.", out)
        self.assertIn("coach", out)

    def test_slash_persona_no_arg_empty_dir(self):
        self._personas_dir()
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            _repl._dispatch_slash("/persona", [], self._state(),
                                  _args(interactive=True), [])
        self.assertIn("no personas yet", err.getvalue())

    def test_slash_persona_traversal_rejected(self):
        d = self._personas_dir()
        (d.parent / "credentials").write_text("SECRET", encoding="utf-8")
        messages = []
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            _repl._dispatch_slash("/persona ../credentials", messages, self._state(),
                                  _args(interactive=True), [])
        self.assertEqual(messages, [])                 # nothing loaded
        self.assertNotIn("SECRET", err.getvalue())     # secret never read/printed
        self.assertIn("/persona:", err.getvalue())     # a friendly error instead

    def test_slash_persona_missing_name(self):
        self._personas_dir()
        messages = []
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            _repl._dispatch_slash("/persona ghost", messages, self._state(),
                                  _args(interactive=True), [])
        self.assertEqual(messages, [])
        self.assertIn("ghost", err.getvalue())

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

    def test_commands_include_cost_usage(self):
        # #75 drift guard: /usage and /cost stay in the completion tuple and help.
        self.assertIn("/usage", _repl._COMMANDS)
        self.assertIn("/cost", _repl._COMMANDS)
        self.assertIn("/usage", _repl._HELP)

    def test_commands_include_paste_edit(self):
        self.assertIn("/paste", _repl._COMMANDS)
        self.assertIn("/edit", _repl._COMMANDS)

    def test_completes_paste(self):
        self.assertEqual(self._complete_all("/pa", 0, "/pa"), ["/paste"])

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


class TestReplMultiline(unittest.TestCase):
    """#65: /paste block mode and /edit ($EDITOR) multi-line composition. Both
    are handled in the run() loop and submit exactly one normal turn."""

    # ---- /paste ------------------------------------------------------- #
    def test_paste_composes_multiline_turn(self):
        rc, fake, calls = _run_repl(
            _args(interactive=True),
            [[FakeChunk("ok")]],
            ["/paste", "line one", "line two", "/end", "/exit"],
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["messages"][-1],
                         {"role": "user", "content": "line one\nline two"})

    def test_paste_preserves_indentation(self):
        rc, fake, calls = _run_repl(
            _args(interactive=True),
            [[FakeChunk("ok")]],
            ["/paste", "def f():", "    return 1", "/end", "/exit"],
        )
        self.assertEqual(calls[0]["messages"][-1]["content"],
                         "def f():\n    return 1")

    def test_paste_cancel_sends_nothing(self):
        rc, fake, calls = _run_repl(
            _args(interactive=True), [],
            ["/paste", "junk", "/cancel", "/exit"],
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 0)

    def test_paste_empty_block_sends_nothing(self):
        rc, fake, calls = _run_repl(
            _args(interactive=True), [],
            ["/paste", "/end", "/exit"],
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 0)

    # ---- /edit -------------------------------------------------------- #
    @staticmethod
    def _editor_writing(text, rc=0):
        """A fake subprocess.call that simulates the user saving `text` (the temp
        path is the last argv element) then leaving the editor with code `rc`."""
        def _call(cmd, *a, **k):
            with open(cmd[-1], "w", encoding="utf-8") as fh:
                fh.write(text)
            return rc
        return _call

    def test_edit_composes_turn(self):
        with mock.patch.dict(os.environ, {"EDITOR": "true"}), \
             mock.patch("venice.commands._repl.subprocess.call",
                        self._editor_writing("edited one\nedited two\n")):
            rc, fake, calls = _run_repl(
                _args(interactive=True), [[FakeChunk("ok")]], ["/edit", "/exit"],
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["messages"][-1],
                         {"role": "user", "content": "edited one\nedited two"})

    def test_edit_preseeds_buffer_with_inline_text(self):
        seen = {}

        def _call(cmd, *a, **k):
            with open(cmd[-1], encoding="utf-8") as fh:
                seen["seed"] = fh.read()
            with open(cmd[-1], "w", encoding="utf-8") as fh:
                fh.write(seen["seed"] + " and more")
            return 0

        with mock.patch.dict(os.environ, {"EDITOR": "true"}), \
             mock.patch("venice.commands._repl.subprocess.call", _call):
            rc, fake, calls = _run_repl(
                _args(interactive=True), [[FakeChunk("ok")]],
                ["/edit draft text", "/exit"],
            )
        self.assertEqual(seen["seed"], "draft text")
        self.assertEqual(calls[0]["messages"][-1]["content"], "draft text and more")

    def test_edit_empty_buffer_sends_nothing(self):
        with mock.patch.dict(os.environ, {"EDITOR": "true"}), \
             mock.patch("venice.commands._repl.subprocess.call",
                        self._editor_writing("   \n")):
            rc, fake, calls = _run_repl(
                _args(interactive=True), [], ["/edit", "/exit"],
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 0)

    def test_edit_nonzero_exit_aborts(self):
        with mock.patch.dict(os.environ, {"EDITOR": "true"}), \
             mock.patch("venice.commands._repl.subprocess.call",
                        self._editor_writing("would-be text", rc=1)):
            rc, fake, calls = _run_repl(
                _args(interactive=True), [], ["/edit", "/exit"],
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 0)

    def test_edit_no_editor_found_is_graceful(self):
        def _call(cmd, *a, **k):
            raise FileNotFoundError(cmd[0])

        err = io.StringIO()
        with mock.patch.dict(os.environ, {"EDITOR": "nope-no-such-editor"}), \
             mock.patch("venice.commands._repl.subprocess.call", _call):
            rc, fake, calls = _run_repl(
                _args(interactive=True), [], ["/edit", "/exit"], stderr=err,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 0)
        self.assertIn("editor", err.getvalue().lower())


class TestReplLedgerAdoption(unittest.TestCase):
    """#117: `_repl.run(ledger=...)` ADOPTS the caller's ledger by identity.

    `venice code` must hand this in: its subagent rails were bound to that object at
    tool-factory time, long before the REPL starts. If `run` built its own instead,
    every rail would mirror into an orphan -- no error, no crash, just money silently
    missing from `/usage` and the session file.

    Driving the REAL `_repl.run` is the point. The `code.py` side is covered by an
    `assertIs` test that mocks `_repl.run` away, which by construction cannot see what
    happens INSIDE it -- a mutation making this line build a fresh ledger survived the
    entire suite until this class existed.
    """

    def _drive(self, ledger):
        fake, calls = _fake_openai_seq([[FakeChunk(
            "hi", usage={"prompt_tokens": 40, "completion_tokens": 2,
                         "total_tokens": 42})]])
        import openai as openai_mod
        with contextlib.ExitStack() as st:
            sess_dir = st.enter_context(tempfile.TemporaryDirectory())
            st.enter_context(mock.patch.dict(
                os.environ, {"VENICE_API_KEY": "fake",
                             "VENICE_SESSIONS_DIR": sess_dir}))
            st.enter_context(mock.patch("venice.userconfig.load_config",
                                        lambda *a, **k: _EMPTY_CFG))
            st.enter_context(mock.patch("builtins.input", side_effect=["/exit"]))
            st.enter_context(mock.patch.object(sys, "stdin", io.StringIO("")))
            st.enter_context(mock.patch.object(sys, "stdout", io.StringIO()))
            st.enter_context(mock.patch.object(sys, "stderr", io.StringIO()))
            rc = _repl.run(_args(interactive=True), fake, openai_mod, None,
                           [{"id": "llama-3.3-70b"}], "llama-3.3-70b",
                           initial="hello", ledger=ledger)
        return rc, calls

    def test_an_injected_ledger_is_the_one_that_meters_the_turn(self):
        L = _agent.CostLedger()
        rc, calls = self._drive(L)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(L.prompt_tokens, 40)      # the injected object, not a fresh one
        self.assertEqual(L.api_calls_total, 1)

    def test_rail_spend_banked_before_the_repl_starts_is_still_reported(self):
        # The real sequence: `code.py` builds the ledger, the rails bind to it, and
        # only then does the REPL open. Pre-existing bucket state must survive.
        L = _agent.CostLedger()
        L.record_bucket("scout", cost=0.01, prompt_tokens=1234)
        rc, _ = self._drive(L)
        self.assertEqual(rc, 0)
        self.assertEqual(L.buckets["scout"]["prompt_tokens"], 1234)
        self.assertAlmostEqual(L.billed_total(), 0.01)

    def test_no_injected_ledger_still_builds_one(self):
        # `venice chat` passes nothing and must keep working.
        rc, calls = self._drive(None)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
