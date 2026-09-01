"""Unit tests for `venice chat`.

Mocks the OpenAI client (chat completions) and the free /models catalog GET
(via urlopen). No network, no real key. The openai package must be importable
(pip install -e ".[openai]").
"""
import argparse
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

from tests.test_client import FakeResp


def _args(**ov):
    base = dict(
        message=None, system=None, persona=None, model=None, temperature=None,
        max_tokens=None, stream=True, json=False,
        web_search=None, web_citations=False, web_scraping=False,
        character=None, no_venice_system_prompt=False,
        strip_thinking=False, no_thinking=False, x_search=False,
        # --- agent / tools (#15) ---
        tools=None, tool=None, max_tool_calls=None,
        max_spend=None, yes=None, output=None,
        # --- shell exec tool (#33) ---
        shell=None, shell_allow=None, shell_deny=None, shell_unrestricted=None,
        # --- browser egress rail (#71/#127) ---
        browser=None, browser_allow=None, browser_deny=None,
        browser_private_host=None, browser_private_range=None,
        # --- external MCP client (#21) ---
        mcp=None, no_mcp=False,
        # --- interactive / REPL (#22) ---
        interactive=False, resume=None,
        # --- session store (#47) ---
        cont=None, ephemeral=None,
        # --- auto-compaction (#48) ---
        auto_compact=None, compact_threshold=None, compact_keep_turns=None,
        # --- session spend cap (#66) ---
        session_max_spend=None,
    )
    base.update(ov)
    return argparse.Namespace(**base)


# Auto-save is on by default (#47): keep this module hermetic even though its
# one-shot paths don't persist -- an interactive chat._run here would otherwise
# write to ~/.config/venice/sessions.
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


# --- fake OpenAI response objects ---

class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class FakeCompletion:
    def __init__(self, content, venice_parameters=None, usage=None):
        self.choices = [_Choice(content)]
        self.venice_parameters = venice_parameters
        self.usage = usage
        self._dump = {
            "choices": [{"message": {"content": content}}],
            "venice_parameters": venice_parameters,
        }
        if usage is not None:
            self._dump["usage"] = usage

    def model_dump(self):
        return self._dump


class _Delta:
    def __init__(self, content, **fields):
        self.content = content
        for name, value in fields.items():
            setattr(self, name, value)


class _StreamChoice:
    def __init__(self, content, **fields):
        self.delta = _Delta(content, **fields)


class FakeChunk:
    def __init__(self, content=None, usage=None, venice_parameters=None, **delta_fields):
        self.choices = (
            [_StreamChoice(content, **delta_fields)]
            if content is not None or delta_fields else []
        )
        self.usage = usage
        self.venice_parameters = venice_parameters


# --- catalog GET mock: two text models, one with the `default` trait ---

def _text_payload(fc=True):
    """Catalog with a `default`-trait model. `fc` sets supportsFunctionCalling."""
    return json.dumps({
        "object": "list",
        "data": [
            {"id": "llama-3.3-70b", "type": "text",
             "model_spec": {"traits": ["default"],
                            "capabilities": {"supportsFunctionCalling": fc}}},
            {"id": "venice-uncensored", "type": "text",
             "model_spec": {"traits": [],
                            "capabilities": {"supportsFunctionCalling": False}}},
        ],
    }).encode()


def _urlopen_ok(fc=True):
    def _u(req, timeout=None):
        return FakeResp(200, _text_payload(fc), "application/json")
    return _u


def _fake_openai(result):
    """Return (fake_client, captured_kwargs). create() records its kwargs."""
    captured = {}
    fake = mock.MagicMock()

    def _create(**kwargs):
        captured.clear()
        captured.update(kwargs)
        if kwargs.get("stream"):
            return iter(result)
        return result

    fake.chat.completions.create.side_effect = _create
    return fake, captured


# --- fakes for the tool-calling (agent) loop ---

class _FnRef:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FnCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.type = "function"
        self.function = _FnRef(name, arguments)


class _ToolMsg:
    def __init__(self, content=None, tool_calls=None, **fields):
        self.content = content
        self.tool_calls = tool_calls
        for name, value in fields.items():
            setattr(self, name, value)


class _ToolChoice:
    def __init__(self, msg):
        self.message = msg


class FakeToolCompletion:
    """A completion whose message may carry tool_calls (None => a final answer)."""

    def __init__(self, content=None, tool_calls=None, venice_parameters=None,
                 usage=None, **message_fields):
        self.choices = [_ToolChoice(_ToolMsg(content, tool_calls, **message_fields))]
        self.venice_parameters = venice_parameters
        self.usage = usage

    def model_dump(self):
        return {"choices": [{"message": {"content": self.choices[0].message.content}}]}


def _fake_openai_seq(results):
    """create() returns successive `results`; records every call's kwargs.

    `messages` is deep-copied per call because the loop mutates one list in place,
    so a shallow record would show every call the final state.
    """
    calls = []
    fake = mock.MagicMock()
    seq = list(results)

    def _create(**kwargs):
        snap = dict(kwargs)
        if "messages" in snap:
            snap["messages"] = copy.deepcopy(snap["messages"])
        calls.append(snap)
        return seq.pop(0)

    fake.chat.completions.create.side_effect = _create
    return fake, calls


class TestChat(unittest.TestCase):

    def setUp(self):
        # Hermetic: never read the developer's real ~/.config/venice/config.json.
        _cfg = mock.patch(
            "venice.userconfig.load_config",
            lambda *a, **k: {"version": 1, "mcpServers": {}, "defaults": {}},
        )
        _cfg.start()
        self.addCleanup(_cfg.stop)

    def _run(self, args, result, stdout=None, stderr=None, side_effect=None):
        from venice.commands import chat
        fake, captured = _fake_openai(result)
        if side_effect is not None:
            fake.chat.completions.create.side_effect = side_effect
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.object(sys, "stdout", stdout or io.StringIO()), \
             mock.patch.object(sys, "stderr", stderr or io.StringIO()):
            rc = chat._run(args)
        return rc, fake, captured

    def test_browser_flag_implies_agent_tools(self):
        from venice.commands import chat
        args = _args(message="fetch a page", browser=True, stream=False)
        with mock.patch.object(chat, "_run_agent", return_value=0) as agent_loop:
            rc, _fake, _captured = self._run(args, FakeCompletion("unused"))
        self.assertEqual(rc, 0)
        self.assertTrue(args.tools)
        agent_loop.assert_called_once()

    def test_reply_printed_non_stream(self):
        out = io.StringIO()
        rc, fake, captured = self._run(
            _args(message="hi", stream=False), FakeCompletion("hello there"), stdout=out
        )
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue().strip(), "hello there")
        # default model resolved from the `default`-trait catalog entry
        self.assertEqual(captured["model"], "llama-3.3-70b")
        self.assertEqual(captured["messages"][-1], {"role": "user", "content": "hi"})

    def test_non_stream_reports_usage_on_stderr(self):
        out, err = io.StringIO(), io.StringIO()
        rc, _fake, _captured = self._run(
            _args(message="hi", stream=False),
            FakeCompletion("hello", usage={
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
            }),
            stdout=out,
            stderr=err,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue(), "hello\n")
        self.assertEqual(err.getvalue(), "usage: prompt=11 completion=3 total=14\n")

    def test_non_stream_without_usage_keeps_stderr_quiet(self):
        err = io.StringIO()
        rc, _fake, _captured = self._run(
            _args(message="hi", stream=False),
            FakeCompletion("hello"),
            stderr=err,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(err.getvalue(), "")

    def test_config_default_model_applied(self):
        cfg = {"version": 1, "mcpServers": {},
               "defaults": {"chat": {"model": "venice-uncensored"}}}
        with mock.patch("venice.userconfig.load_config", lambda *a, **k: cfg):
            rc, fake, captured = self._run(
                _args(message="hi", stream=False), FakeCompletion("ok")
            )
        self.assertEqual(rc, 0)
        # config default used instead of the catalog `default`-trait model
        self.assertEqual(captured["model"], "venice-uncensored")

    def test_explicit_model_overrides_config_default(self):
        cfg = {"version": 1, "mcpServers": {},
               "defaults": {"chat": {"model": "venice-uncensored"}}}
        with mock.patch("venice.userconfig.load_config", lambda *a, **k: cfg):
            rc, fake, captured = self._run(
                _args(message="hi", model="llama-3.3-70b", stream=False),
                FakeCompletion("ok"),
            )
        self.assertEqual(rc, 0)
        self.assertEqual(captured["model"], "llama-3.3-70b")

    def test_system_prompt_and_model(self):
        rc, fake, captured = self._run(
            _args(message="hi", system="be terse", model="venice-uncensored",
                  stream=False),
            FakeCompletion("ok"),
        )
        self.assertEqual(rc, 0)
        self.assertEqual(captured["model"], "venice-uncensored")
        self.assertEqual(captured["messages"][0], {"role": "system", "content": "be terse"})

    # --- personas (#68): --persona / defaults.chat.persona seed args.system ---

    def _personas_dir(self, **files):
        """A temp personas dir patched in as config.PERSONAS_DIR; returns its Path."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        for name, text in files.items():
            (d / name).write_text(text, encoding="utf-8")
        p = mock.patch("venice.config.PERSONAS_DIR", d)
        p.start()
        self.addCleanup(p.stop)
        return d

    def test_persona_flag_seeds_system(self):
        self._personas_dir(**{"pirate.md": "You are a terse pirate."})
        rc, fake, captured = self._run(
            _args(message="hi", persona="pirate", stream=False), FakeCompletion("ok")
        )
        self.assertEqual(rc, 0)
        self.assertEqual(
            captured["messages"][0],
            {"role": "system", "content": "You are a terse pirate."},
        )

    def test_persona_config_default_applied(self):
        self._personas_dir(**{"pirate.md": "Arr."})
        cfg = {"version": 1, "mcpServers": {},
               "defaults": {"chat": {"persona": "pirate"}}}
        with mock.patch("venice.userconfig.load_config", lambda *a, **k: cfg):
            rc, fake, captured = self._run(
                _args(message="hi", stream=False), FakeCompletion("ok")
            )
        self.assertEqual(rc, 0)
        self.assertEqual(captured["messages"][0],
                         {"role": "system", "content": "Arr."})

    def test_explicit_system_beats_persona(self):
        self._personas_dir(**{"pirate.md": "Arr."})
        rc, fake, captured = self._run(
            _args(message="hi", system="be terse", persona="pirate", stream=False),
            FakeCompletion("ok"),
        )
        self.assertEqual(rc, 0)
        self.assertEqual(captured["messages"][0],
                         {"role": "system", "content": "be terse"})

    def test_persona_not_found_errors(self):
        self._personas_dir()
        err = io.StringIO()
        rc, fake, captured = self._run(
            _args(message="hi", persona="ghost", stream=False),
            FakeCompletion("ok"), stderr=err,
        )
        self.assertEqual(rc, 2)
        self.assertIn("ghost", err.getvalue())

    def test_persona_traversal_rejected(self):
        self._personas_dir()
        err = io.StringIO()
        rc, fake, captured = self._run(
            _args(message="hi", persona="../credentials", stream=False),
            FakeCompletion("ok"), stderr=err,
        )
        self.assertEqual(rc, 2)
        self.assertIn("invalid persona", err.getvalue())

    def test_stdin_dash_becomes_message(self):
        with mock.patch.object(sys, "stdin", io.StringIO("piped question")):
            rc, fake, captured = self._run(
                _args(message="-", stream=False), FakeCompletion("answer")
            )
        self.assertEqual(rc, 0)
        self.assertEqual(captured["messages"][-1]["content"], "piped question")

    def test_piped_stdin_no_arg(self):
        with mock.patch.object(sys, "stdin", io.StringIO("from pipe")):
            rc, fake, captured = self._run(
                _args(message=None, stream=False), FakeCompletion("answer")
            )
        self.assertEqual(rc, 0)
        self.assertEqual(captured["messages"][-1]["content"], "from pipe")

    def test_no_message_non_tty_exit_2(self):
        # No positional message and stdin is not a TTY with nothing piped: there
        # is nothing to send and it isn't interactive, so exit 2. (No message on
        # a *TTY* now drops into the REPL instead -- see test_repl.py.)
        err = io.StringIO()
        with mock.patch.object(sys, "stdin", io.StringIO("")):
            rc, fake, captured = self._run(
                _args(message=None), FakeCompletion("x"), stderr=err
            )
        self.assertEqual(rc, 2)
        self.assertEqual(fake.chat.completions.create.call_count, 0)

    def test_json_dumps_raw_and_forces_non_stream(self):
        out, err = io.StringIO(), io.StringIO()
        rc, fake, captured = self._run(
            _args(message="hi", json=True),  # stream default True
            FakeCompletion(
                "hello",
                venice_parameters={"enable_web_search": "on"},
                usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            ),
            stdout=out,
            stderr=err,
        )
        self.assertEqual(rc, 0)
        doc = json.loads(out.getvalue())
        self.assertEqual(doc["choices"][0]["message"]["content"], "hello")
        self.assertEqual(doc["venice_parameters"], {"enable_web_search": "on"})
        self.assertEqual(doc["usage"]["total_tokens"], 3)
        self.assertEqual(err.getvalue(), "")
        # --json must not stream
        self.assertNotIn("stream", captured)

    def test_streaming_increments(self):
        out = io.StringIO()
        chunks = [
            FakeChunk("Hel"),
            FakeChunk("lo"),
            FakeChunk(usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}),
        ]
        rc, fake, captured = self._run(
            _args(message="hi", stream=True), chunks, stdout=out
        )
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue().strip(), "Hello")
        self.assertTrue(captured["stream"])
        self.assertEqual(captured["stream_options"], {"include_usage": True})

    def test_non_stream_interrupt_prints_command_notice_and_exits_130(self):
        for args in (
            _args(message="hi", stream=False),
            _args(message="hi", json=True),
        ):
            with self.subTest(json=args.json):
                out, err = io.StringIO(), io.StringIO()
                rc, _fake, _captured = self._run(
                    args,
                    FakeCompletion("unused"),
                    stdout=out,
                    stderr=err,
                    side_effect=KeyboardInterrupt,
                )
                self.assertEqual(rc, 130)
                self.assertEqual(out.getvalue(), "")
                self.assertEqual(err.getvalue(), "\nchat: aborted\n")

    def test_stream_interrupt_warns_that_stdout_may_be_partial(self):
        def interrupted_stream():
            yield FakeChunk("PARTIAL-FROM-FAKE")
            raise KeyboardInterrupt

        out, err = io.StringIO(), io.StringIO()
        rc, _fake, _captured = self._run(
            _args(message="hi", stream=True),
            interrupted_stream(),
            stdout=out,
            stderr=err,
        )
        self.assertEqual(rc, 130)
        self.assertEqual(out.getvalue(), "PARTIAL-FROM-FAKE")
        self.assertEqual(
            err.getvalue(),
            "\nchat: aborted (partial output may appear above)\n",
        )

    def test_venice_parameters_extra_body(self):
        rc, fake, captured = self._run(
            _args(
                message="hi", stream=False,
                web_search="on", web_citations=True, web_scraping=True,
                character="venice", no_venice_system_prompt=True,
                strip_thinking=True, no_thinking=True, x_search=True,
            ),
            FakeCompletion("ok"),
        )
        self.assertEqual(rc, 0)
        self.assertEqual(captured["extra_body"], {"venice_parameters": {
            "enable_web_search": "on",
            "enable_web_citations": True,
            "enable_web_scraping": True,
            "character_slug": "venice",
            "include_venice_system_prompt": False,
            "strip_thinking_response": True,
            "disable_thinking": True,
            "enable_x_search": True,
        }})

    def test_no_extensions_omits_extra_body(self):
        rc, fake, captured = self._run(
            _args(message="hi", stream=False), FakeCompletion("ok")
        )
        self.assertEqual(rc, 0)
        self.assertNotIn("extra_body", captured)

    def test_citations_printed_to_stderr(self):
        err = io.StringIO()
        resp = FakeCompletion("blue sky", venice_parameters={
            "web_search_citations": [
                {"title": "Why the sky is blue", "url": "http://example.com/sky",
                 "date": "2026-01-01"},
            ],
        })
        rc, fake, captured = self._run(
            _args(message="why is the sky blue", stream=False, web_search="on"),
            resp, stderr=err,
        )
        self.assertEqual(rc, 0)
        text = err.getvalue()
        self.assertIn("Sources:", text)
        self.assertIn("Why the sky is blue", text)
        self.assertIn("http://example.com/sky", text)

    def test_bad_model_exit_6_before_call(self):
        err = io.StringIO()
        rc, fake, captured = self._run(
            _args(message="hi", model="no-such-model"),
            FakeCompletion("x"), stderr=err,
        )
        self.assertEqual(rc, 6)
        self.assertEqual(fake.chat.completions.create.call_count, 0)
        self.assertIn("no-such-model", err.getvalue())

    def test_missing_openai_exit_2(self):
        from venice.commands import chat
        err = io.StringIO()
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch.dict(sys.modules, {"openai": None}), \
             mock.patch.object(sys, "stderr", err):
            rc = chat._run(_args(message="hi"))
        self.assertEqual(rc, 2)
        self.assertIn("openai", err.getvalue())


class TestChatAgent(unittest.TestCase):
    """The `--tools` function-calling agent loop (#15)."""

    def setUp(self):
        _cfg = mock.patch(
            "venice.userconfig.load_config",
            lambda *a, **k: {"version": 1, "mcpServers": {}, "defaults": {}},
        )
        _cfg.start()
        self.addCleanup(_cfg.stop)

    def _run_seq(self, args, results, stdout=None, stderr=None, urlopen=None):
        from venice.commands import chat
        fake, calls = _fake_openai_seq(results)
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        urlopen or _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.object(sys, "stdout", stdout or io.StringIO()), \
             mock.patch.object(sys, "stderr", stderr or io.StringIO()):
            rc = chat._run(args)
        return rc, fake, calls

    def test_two_step_tool_loop(self):
        out = io.StringIO()
        seq = [
            FakeToolCompletion(tool_calls=[
                _FnCall("call_1", "venice_chat", '{"message": "say hola"}')]),
            FakeToolCompletion("final answer"),
        ]
        with mock.patch(
            "venice.commands._mcp.chat_tool",
            return_value={"status": "ok", "content": "hola", "model": "m"},
        ) as stub:
            rc, fake, calls = self._run_seq(
                _args(message="hi", tools=True, stream=False), seq, stdout=out
            )
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue().strip(), "final answer")
        # the tool impl received the model's parsed arguments
        _pos, kw = stub.call_args
        self.assertEqual(kw.get("message"), "say hola")
        # first turn advertised tools + tool_choice=auto
        self.assertIn("tools", calls[0])
        self.assertEqual(calls[0]["tool_choice"], "auto")
        # second turn carries the tool result with the matching id
        tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], "call_1")
        self.assertIn("hola", tool_msgs[0]["content"])

    # --- shell exec tool (#33) ---

    def test_shell_flag_implies_tools_and_invokes_shell(self):
        out = io.StringIO()
        seq = [
            FakeToolCompletion(tool_calls=[
                _FnCall("call_1", "shell", '{"command": "echo hola"}')]),
            FakeToolCompletion("done"),
        ]
        # --shell alone (no --tools), auto-approved via --yes with an allowlist so
        # the loud-unrestricted guard doesn't trip. The shell tool runs `echo` in cwd.
        rc, fake, calls = self._run_seq(
            _args(message="hi", stream=False, shell=True, shell_allow=["echo"],
                  yes=True),
            seq, stdout=out,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue().strip(), "done")
        # first turn advertised a `shell` tool (implies-tools worked)
        names = [t["function"]["name"] for t in calls[0]["tools"]]
        self.assertIn("shell", names)
        # the tool result carried the command output back to the model
        tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
        self.assertIn("hola", tool_msgs[0]["content"])

    def test_shell_deny_refuses_command(self):
        out = io.StringIO()
        seq = [
            FakeToolCompletion(tool_calls=[
                _FnCall("call_1", "shell", '{"command": "sudo reboot"}')]),
            FakeToolCompletion("understood"),
        ]
        rc, fake, calls = self._run_seq(
            _args(message="hi", stream=False, shell=True, shell_deny=["sudo"],
                  yes=True, shell_unrestricted=True),
            seq, stdout=out,
        )
        self.assertEqual(rc, 0)
        tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
        self.assertIn("deny", tool_msgs[0]["content"])

    def test_shell_unrestricted_with_yes_requires_ack(self):
        # Empty allowlist + --yes without --shell-unrestricted -> refuse (exit 2).
        err = io.StringIO()
        rc, fake, calls = self._run_seq(
            _args(message="hi", stream=False, shell=True, yes=True),
            [FakeCompletion("unused")], stderr=err,
        )
        self.assertEqual(rc, 2)
        self.assertIn("shell-unrestricted", err.getvalue())

    def test_shell_unrestricted_ack_allows_empty_allowlist(self):
        out = io.StringIO()
        rc, fake, calls = self._run_seq(
            _args(message="hi", stream=False, shell=True, yes=True,
                  shell_unrestricted=True),
            [FakeToolCompletion("ok")], stdout=out,
        )
        self.assertEqual(rc, 0)

    def test_shell_flags_parse(self):
        from venice import cli
        args = cli.build_parser().parse_args([
            "chat", "hi", "--shell",
            "--shell-allow", "git", "--shell-allow", "ls",
            "--shell-deny", "rm *", "--shell-unrestricted",
        ])
        self.assertTrue(args.shell)
        self.assertEqual(args.shell_allow, ["git", "ls"])
        self.assertEqual(args.shell_deny, ["rm *"])
        self.assertTrue(args.shell_unrestricted)
        # --exec is an alias for --shell
        self.assertTrue(cli.build_parser().parse_args(["chat", "hi", "--exec"]).shell)

    def test_shell_config_policy_reaches_tool(self):
        # A config `shell.deny` (no CLI flag) still scopes the tool.
        cfg = {"version": 1, "mcpServers": {}, "defaults": {},
               "shell": {"deny": ["sudo"]}}
        out = io.StringIO()
        seq = [
            FakeToolCompletion(tool_calls=[
                _FnCall("call_1", "shell", '{"command": "sudo rm -rf /"}')]),
            FakeToolCompletion("noted"),
        ]
        with mock.patch("venice.userconfig.load_config", lambda *a, **k: cfg):
            rc, fake, calls = self._run_seq(
                _args(message="hi", stream=False, shell=True, shell_allow=["sudo"],
                      yes=True),
                seq, stdout=out,
            )
        self.assertEqual(rc, 0)
        tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
        self.assertIn("deny", tool_msgs[0]["content"])

    def test_tools_auto_compact_hands_budget_to_loop(self):
        # #48 parity: chat --tools must honor --auto-compact by giving run_loop a
        # Budget (like code / the REPL do). Non-interactive single turns rarely
        # compact, but the flag has to reach the loop -- spy on run_loop to prove
        # the wiring without depending on compaction actually firing.
        from venice.commands import _agent, _compact
        captured = {}

        def _spy(*a, **kw):
            captured["budget"] = kw.get("budget")
            return 0

        with mock.patch.object(_agent, "run_loop", _spy):
            rc, _fake, _calls = self._run_seq(
                _args(message="hi", tools=True, stream=False, auto_compact=True,
                      compact_threshold=1234, compact_keep_turns=3),
                [],
            )
        self.assertEqual(rc, 0)
        self.assertIsInstance(captured["budget"], _compact.Budget)
        self.assertEqual(captured["budget"].threshold_tokens, 1234)
        self.assertEqual(captured["budget"].keep_turns, 3)

    def test_tools_without_auto_compact_passes_no_budget(self):
        from venice.commands import _agent
        captured = {}

        def _spy(*a, **kw):
            captured["budget"] = kw.get("budget")
            return 0

        with mock.patch.object(_agent, "run_loop", _spy):
            rc, _fake, _calls = self._run_seq(
                _args(message="hi", tools=True, stream=False), [],
            )
        self.assertEqual(rc, 0)
        self.assertIsNone(captured["budget"])

    def test_tools_session_max_spend_hands_ledger_to_loop(self):
        # #66: chat --tools must honor --session-max-spend by giving run_loop a
        # CostLedger bound to the session model's catalog pricing.
        from venice.commands import _agent
        captured = {}

        def _spy(*a, **kw):
            captured["ledger"] = kw.get("ledger")
            return 0

        with mock.patch.object(_agent, "run_loop", _spy):
            rc, _fake, _calls = self._run_seq(
                _args(message="hi", tools=True, stream=False,
                      session_max_spend=1.25),
                [],
            )
        self.assertEqual(rc, 0)
        self.assertIsInstance(captured["ledger"], _agent.CostLedger)
        self.assertEqual(captured["ledger"].max_spend, 1.25)

    def test_tools_without_session_max_spend_passes_an_uncapped_ledger(self):
        # #86 inverted this: it used to assert `ledger is None`, which is exactly
        # why a default `chat --tools` run metered nothing. The ledger is now
        # always-on -- but the invariant the old test really cared about still
        # holds and is what's pinned here: no flag => no cap => no spend gating.
        # `over()` short-circuits on a None cap, so metering costs nothing.
        from venice.commands import _agent
        captured = {}

        def _spy(*a, **kw):
            captured["ledger"] = kw.get("ledger")
            return 0

        with mock.patch.object(_agent, "run_loop", _spy):
            rc, _fake, _calls = self._run_seq(
                _args(message="hi", tools=True, stream=False), [],
            )
        self.assertEqual(rc, 0)
        self.assertIsInstance(captured["ledger"], _agent.CostLedger)
        self.assertIsNone(captured["ledger"].max_spend)
        self.assertFalse(captured["ledger"].over())
        self.assertFalse(captured["ledger"].over_tokens())

    def test_capability_degrade_to_plain_chat(self):
        err = io.StringIO()
        rc, fake, calls = self._run_seq(
            _args(message="hi", tools=True, stream=False),
            [FakeCompletion("plain reply")],
            stderr=err, urlopen=_urlopen_ok(fc=False),
        )
        self.assertEqual(rc, 0)
        # loop not entered: the single create advertised no tools
        self.assertNotIn("tools", calls[0])
        self.assertIn("does not support function calling", err.getvalue())

    def test_max_tool_calls_cap(self):
        out, err = io.StringIO(), io.StringIO()
        seq = [
            FakeToolCompletion(tool_calls=[
                _FnCall("c1", "venice_chat", '{"message": "x"}')]),
            FakeToolCompletion(tool_calls=[
                _FnCall("c2", "venice_chat", '{"message": "x"}')]),
            FakeToolCompletion("done"),
        ]
        with mock.patch(
            "venice.commands._mcp.chat_tool",
            return_value={"status": "ok", "content": "x"},
        ) as stub:
            rc, fake, calls = self._run_seq(
                _args(message="hi", tools=True, stream=False, max_tool_calls=2),
                seq, stdout=out, stderr=err,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(stub.call_count, 2)
        self.assertEqual(calls[-1]["tool_choice"], "none")  # forced final answer
        self.assertEqual(out.getvalue().strip(), "done")
        self.assertIn("max-tool-calls", err.getvalue())

    def test_spend_gate_yes_auto_approves(self):
        seq = [
            FakeToolCompletion(tool_calls=[
                _FnCall("c1", "venice_image", '{"prompt": "a cat"}')]),
            FakeToolCompletion("described the cat"),
        ]
        with mock.patch(
            "venice.commands._mcp.image_tool",
            return_value={"status": "ok", "paths": ["/x.png"]},
        ) as stub:
            rc, fake, calls = self._run_seq(
                _args(message="hi", tools=True, stream=False, yes=True), seq
            )
        self.assertEqual(rc, 0)
        _pos, kw = stub.call_args
        self.assertTrue(kw.get("confirm"))  # --yes -> confirm=True
        self.assertEqual(kw.get("prompt"), "a cat")

    def test_spend_gate_non_tty_feeds_confirmation_back(self):
        seq = [
            FakeToolCompletion(tool_calls=[
                _FnCall("c1", "venice_image", '{"prompt": "a cat"}')]),
            FakeToolCompletion("could not afford it"),
        ]
        gate = {"status": "confirmation_required", "message": "over cap",
                "estimated_cost_usd": 5.0, "max_spend_usd": 0.1}
        fake_stdin = mock.MagicMock()
        fake_stdin.isatty.return_value = False
        with mock.patch(
            "venice.commands._mcp.image_tool", return_value=gate
        ) as stub, mock.patch.object(sys, "stdin", fake_stdin):
            rc, fake, calls = self._run_seq(
                _args(message="hi", tools=True, stream=False), seq
            )
        self.assertEqual(rc, 0)
        self.assertEqual(stub.call_count, 1)  # not re-invoked without approval
        tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
        self.assertIn("confirmation_required", tool_msgs[0]["content"])

    def test_paid_tool_schema_excludes_control_kwargs(self):
        rc, fake, calls = self._run_seq(
            _args(message="hi", tools=True, stream=False),
            [FakeToolCompletion("no tools needed")],
        )
        self.assertEqual(rc, 0)
        tools = calls[0]["tools"]
        for t in tools:
            props = t["function"]["parameters"].get("properties", {})
            for banned in ("confirm", "max_spend", "output_dir"):
                self.assertNotIn(
                    banned, props,
                    f"{t['function']['name']} leaks control kwarg {banned}",
                )
        names = {t["function"]["name"] for t in tools}
        # media/chat + project_search + reindex + models(+details) + vision
        # + job status/result
        self.assertEqual(len(names), 14)
        self.assertIn("project_search", names)
        self.assertIn("reindex", names)
        self.assertIn("venice_models", names)
        self.assertIn("venice_model_details", names)
        self.assertIn("venice_vision", names)
        self.assertIn("venice_job_status", names)
        self.assertIn("venice_job_result", names)

    def test_tool_error_surfaced_not_fatal(self):
        seq = [
            FakeToolCompletion(tool_calls=[
                _FnCall("c1", "venice_chat", '{"message": "x"}')]),
            FakeToolCompletion("recovered"),
        ]
        with mock.patch(
            "venice.commands._mcp.chat_tool",
            return_value={"status": "error", "message": "boom"},
        ):
            rc, fake, calls = self._run_seq(
                _args(message="hi", tools=True, stream=False), seq
            )
        self.assertEqual(rc, 0)
        tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
        self.assertIn("boom", tool_msgs[0]["content"])

    def test_openai_error_is_fatal(self):
        import openai
        from venice.commands import chat
        err = io.StringIO()
        fake = mock.MagicMock()
        fake.chat.completions.create.side_effect = openai.OpenAIError("boom")
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.object(sys, "stdout", io.StringIO()), \
             mock.patch.object(sys, "stderr", err):
            rc = chat._run(_args(message="hi", tools=True, stream=False))
        self.assertEqual(rc, 5)

    def test_unknown_tool_subset_exit_2(self):
        err = io.StringIO()
        rc, fake, calls = self._run_seq(
            _args(message="hi", tools=True, stream=False, tool=["venice_nope"]),
            [FakeToolCompletion("unused")], stderr=err,
        )
        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 0)  # never reached the model
        self.assertIn("venice_nope", err.getvalue())

    def test_tools_off_leaves_one_shot_unchanged(self):
        rc, fake, calls = self._run_seq(
            _args(message="hi", stream=False), [FakeCompletion("plain")]
        )
        self.assertEqual(rc, 0)
        self.assertNotIn("tools", calls[0])


# --- external MCP client wiring (#21) ---

# A truthy stand-in for the `mcp` SDK module so wiring tests are independent of
# whether the real SDK is installed (it isn't on Python 3.9). `import_mcp` is
# patched to return this; the wiring never uses the module beyond a None check.
_MCP_PRESENT = object()


def _fake_tool(name, result, *, paid=False):
    from venice.commands import _agent
    return _agent.Tool(
        name=name, description="fake mcp tool",
        parameters={"type": "object", "properties": {}},
        invoke=lambda arguments, *, confirm=False: result, paid=paid,
    )


def _fake_attach_cm(tools):
    """A stand-in for `_mcp_client.attach`: a context manager yielding `tools`."""
    @contextlib.contextmanager
    def _attach(specs, **kwargs):
        _attach.specs = specs
        yield tools
    _attach.specs = None
    return _attach


class TestChatUsageSurface(unittest.TestCase):
    """A one-shot `venice chat --tools` run reports what it cost (#86).

    Sister to `TestCodeUsageSurface` in test_code_command.py (#81). The gap: the
    ledger was `ledger_from_args`, i.e. None unless `--session-max-spend`, so a
    default run metered nothing and had no object to report from.
    """

    def setUp(self):
        _cfg = mock.patch(
            "venice.userconfig.load_config",
            lambda *a, **k: {"version": 1, "mcpServers": {}, "defaults": {}},
        )
        _cfg.start()
        self.addCleanup(_cfg.stop)
        self._out = io.StringIO()
        self._errbuf = io.StringIO()

    @property
    def _err(self):
        return self._errbuf.getvalue()

    def _run(self, args, results, *, urlopen=None, side_effect=None):
        from venice.commands import chat
        fake, calls = _fake_openai_seq(results)
        if side_effect is not None:
            fake.chat.completions.create.side_effect = side_effect
        stdin = mock.MagicMock()
        stdin.isatty.return_value = False  # `wants_interactive` reads this
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}))
            st.enter_context(mock.patch("venice.client.urllib.request.urlopen",
                                        urlopen or _urlopen_ok()))
            st.enter_context(mock.patch("openai.OpenAI", return_value=fake))
            st.enter_context(mock.patch.object(sys, "stdin", stdin))
            st.enter_context(mock.patch.object(sys, "stdout", self._out))
            st.enter_context(mock.patch.object(sys, "stderr", self._errbuf))
            rc = chat._run(args)
        return rc, fake, calls

    @staticmethod
    def _usage(n):
        """Distinct per-call blobs, so a failure names *which* call went missing."""
        return {"prompt_tokens": n * 100, "completion_tokens": n}

    def _tool_seq(self, *ns):
        """A tool-calling run of len(ns) turns, the last one final."""
        seq = []
        for i, n in enumerate(ns):
            last = i == len(ns) - 1
            seq.append(FakeToolCompletion(
                content="final answer" if last else None,
                tool_calls=None if last else [
                    _FnCall(f"call_{i}", "venice_chat", '{"message": "hi"}')],
                usage=self._usage(n),
            ))
        return seq

    @staticmethod
    @contextlib.contextmanager
    def _stub_tool():
        with mock.patch("venice.commands._mcp.chat_tool",
                        return_value={"status": "ok", "content": "hola", "model": "m"}):
            yield

    # --- the fix itself ---

    def test_default_tools_run_meters_without_a_spend_cap(self):
        """THE regression test: `usage_ledger`, not `ledger_from_args`."""
        args = _args(message="hi", tools=True, stream=False)
        self.assertIsNone(getattr(args, "session_max_spend", None))  # the precondition
        with self._stub_tool():
            rc, _f, _c = self._run(args, self._tool_seq(1, 2))
        self.assertEqual(rc, 0)
        self.assertRegex(self._err, r"chat: .*wall")
        self.assertRegex(self._err, r"prompt=300\b")

    def test_footer_totals_every_turn_not_just_the_final(self):
        """700/7 across three turns -- not the raw final-turn usage (which is 400/4)."""
        with self._stub_tool():
            rc, _f, _c = self._run(
                _args(message="hi", tools=True, stream=False), self._tool_seq(1, 2, 4))
        self.assertEqual(rc, 0)
        # `assertIn("completion=7", ...)` would also pass on "completion=700" --
        # and the unpriced summary has no closing paren to anchor on. Use \b.
        self.assertRegex(self._err, r"prompt=700\b")
        self.assertRegex(self._err, r"completion=7\b")

    def test_footer_on_stderr_and_stdout_stays_clean(self):
        with self._stub_tool():
            rc, _f, _c = self._run(
                _args(message="hi", tools=True, stream=False), self._tool_seq(1, 2))
        self.assertEqual(rc, 0)
        self.assertRegex(self._err, r"chat: .*wall")
        self.assertEqual(self._out.getvalue().strip(), "final answer")
        self.assertNotIn("wall", self._out.getvalue())

    def test_json_envelope_totals_every_turn_and_stamps_window(self):
        """The aggregate replaces the misleading final-turn SDK usage (#88)."""
        seq = self._tool_seq(1, 2, 4)
        seq[-1].venice_parameters = {"enable_web_search": "on"}
        with self._stub_tool():
            rc, _f, _c = self._run(
                _args(message="hi", tools=True, stream=False, json=True),
                seq)
        self.assertEqual(rc, 0)
        doc = json.loads(self._out.getvalue())
        self.assertEqual(set(doc), {"final", "usage", "venice_parameters"})
        self.assertEqual(doc["final"], "final answer")
        self.assertEqual(doc["venice_parameters"], {"enable_web_search": "on"})
        self.assertEqual(doc["usage"]["prompt_tokens"], 700)
        self.assertEqual(doc["usage"]["completion_tokens"], 7)
        self.assertEqual(doc["usage"]["api_calls_total"], 3)
        self.assertEqual(doc["usage"]["turns"], 1)
        self.assertGreaterEqual(doc["usage"]["elapsed_seconds"], 0)
        self.assertNotRegex(self._err, r"chat: .*wall")

    def test_json_envelope_uses_null_for_absent_venice_parameters(self):
        with self._stub_tool():
            rc, _f, _c = self._run(
                _args(message="hi", tools=True, stream=False, json=True),
                self._tool_seq(1, 2),
            )
        self.assertEqual(rc, 0)
        doc = json.loads(self._out.getvalue())
        self.assertIsNone(doc["venice_parameters"])
        self.assertNotIn("choices", doc)

    # --- the cache hit rate on the footer (#100) ---

    def test_footer_reports_an_unknown_cache_state(self):
        # `_usage()` carries no `prompt_tokens_details`, so the honest answer is
        # "nobody looked" -- not a 0.0% that reads as a measured total miss.
        with self._stub_tool():
            rc, _f, _c = self._run(
                _args(message="hi", tools=True, stream=False), self._tool_seq(1, 2))
        self.assertEqual(rc, 0)
        self.assertIn("cache n/a", self._err)
        self.assertNotIn("0.0% hit", self._err)

    def test_footer_reports_a_real_hit_rate(self):
        # 150 cached of 300 prompt. The human footer and #88's JSON ledger must agree.
        seq = self._tool_seq(1, 2)
        for turn, cached in zip(seq, (50, 100)):
            turn.usage["prompt_tokens_details"] = {"cached_tokens": cached}
        with self._stub_tool():
            rc, _f, _c = self._run(
                _args(message="hi", tools=True, stream=False), seq)
        self.assertEqual(rc, 0)
        footer = [ln for ln in self._err.splitlines()
                  if ln.startswith("chat: ") and "wall" in ln]
        self.assertEqual(len(footer), 1)
        self.assertTrue(footer[0].endswith("cache 50.0% hit"), footer[0])

    def test_footer_carries_the_tools_clause(self):
        # #82. `_tool_seq` dispatches one `venice_chat` call before the final turn,
        # so the clause must appear -- INSIDE the wall field, keeping " -- " as the
        # top-level boundary.
        with self._stub_tool():
            rc, _f, _c = self._run(
                _args(message="hi", tools=True, stream=False), self._tool_seq(1, 2))
        self.assertEqual(rc, 0)
        footer = [ln for ln in self._err.splitlines()
                  if ln.startswith("chat: ") and "wall" in ln]
        self.assertEqual(len(footer), 1)
        self.assertRegex(footer[0], r"^chat: [\d.]+s wall \([\d.]+s tools\) -- cost: ")

    def test_footer_has_no_tools_clause_when_no_tool_ran(self):
        # A one-turn run: the model answers without dispatching. `assertIn("wall")`
        # is blind to a wrongly-appended " (0.0s tools)" -- pin the whole field.
        with self._stub_tool():
            rc, _f, _c = self._run(
                _args(message="hi", tools=True, stream=False), self._tool_seq(1))
        self.assertEqual(rc, 0)
        footer = [ln for ln in self._err.splitlines()
                  if ln.startswith("chat: ") and "wall" in ln]
        self.assertEqual(len(footer), 1)
        self.assertRegex(footer[0], r"^chat: [\d.]+s wall -- cost: ")
        self.assertNotIn("tools", footer[0])

    def test_plain_chat_footer_has_no_cache_claim(self):
        # The non-`--tools` path renders `_print_usage` off the raw SDK blob and
        # never touches a ledger. Out of scope for #100 (that is #90) -- pinned so
        # it is a decision, not a surprise.
        rc, _f, _c = self._run(
            _args(message="hi", stream=False),
            [FakeToolCompletion("hello", usage=self._usage(1))])
        self.assertEqual(rc, 0)
        self.assertNotIn("cache", self._err)

    # --- the exits that must still report ---

    def test_api_error_run_still_reports_time(self):
        import openai
        rc, _f, _c = self._run(_args(message="hi", tools=True, stream=False), [],
                               side_effect=openai.OpenAIError("boom"))
        self.assertEqual(rc, 5)
        self.assertRegex(self._err, r"chat: .*wall")

    def test_json_api_error_emits_no_partial_envelope_and_reports_time(self):
        import openai
        rc, _f, _c = self._run(
            _args(message="hi", tools=True, stream=False, json=True), [],
            side_effect=openai.OpenAIError("boom"),
        )
        self.assertEqual(rc, 5)
        self.assertEqual(self._out.getvalue(), "")
        self.assertRegex(self._err, r"chat: .*wall")

    def test_ctrlc_reports_time_and_exits_130(self):
        # One paid tool-calling turn, then Ctrl+C mid-loop -- the run whose cost you
        # most want to see. Before #86 this was an uncaught traceback.
        seq = [FakeToolCompletion(
            tool_calls=[_FnCall("call_0", "venice_chat", '{"message": "hi"}')],
            usage=self._usage(3))]

        def _create(**kwargs):
            if seq:
                return seq.pop(0)
            raise KeyboardInterrupt

        with self._stub_tool():
            rc, _f, _c = self._run(_args(message="hi", tools=True, stream=False),
                                   [], side_effect=_create)
        self.assertEqual(rc, 130)
        self.assertIn("chat: aborted", self._err)
        self.assertRegex(self._err, r"chat: .*wall")

    def test_json_ctrlc_emits_no_partial_envelope_and_reports_time(self):
        seq = [FakeToolCompletion(
            tool_calls=[_FnCall("call_0", "venice_chat", '{"message": "hi"}')],
            usage=self._usage(3))]

        def _create(**kwargs):
            if seq:
                return seq.pop(0)
            raise KeyboardInterrupt

        with self._stub_tool():
            rc, _f, _c = self._run(
                _args(message="hi", tools=True, stream=False, json=True), [],
                side_effect=_create,
            )
        self.assertEqual(rc, 130)
        self.assertEqual(self._out.getvalue(), "")
        self.assertRegex(self._err, r"chat: .*wall")
        self.assertRegex(self._err, r"prompt=300\b")  # the turn you did pay for

    # --- deliberate absence: no model call => no footer ---

    def test_degraded_run_reports_nothing(self):
        """No ledger is minted on the degrade path, so no orphan 0.0s footer."""
        rc, _f, _c = self._run(
            _args(message="hi", tools=True, stream=False),
            [FakeToolCompletion("plain answer")], urlopen=_urlopen_ok(fc=False))
        self.assertEqual(rc, 0)
        self.assertNotIn("wall", self._err)

    def test_bad_tool_subset_reports_nothing(self):
        rc, _f, _c = self._run(
            _args(message="hi", tools=True, tool=["nope"], stream=False), [])
        self.assertEqual(rc, 2)
        self.assertNotIn("wall", self._err)

    def test_plain_chat_gets_no_footer(self):
        """`_print_usage` is untouched; the ledger never reaches the plain path."""
        chunks = [FakeChunk("hi"), FakeChunk(None, usage={
            "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})]
        rc, _f, _c = self._run(_args(message="hi", stream=True), [chunks])
        self.assertEqual(rc, 0)
        self.assertIn("usage: prompt=5", self._err)
        self.assertNotIn("wall", self._err)

    # --- the guard itself ---

    def test_finish_is_idempotent(self):
        import time as _time

        from venice.commands import _agent, chat
        led = _agent.CostLedger()
        t0 = _time.monotonic()
        with mock.patch.object(sys, "stderr", self._errbuf):
            chat._finish(led, t0)
            chat._finish(led, t0)
        self.assertEqual(led.turns, 1)
        self.assertEqual(self._err.count("wall"), 1)  # and only one footer

    def test_finish_is_inert_without_a_ledger(self):
        from venice.commands import chat
        with mock.patch.object(sys, "stderr", self._errbuf):
            chat._finish(None, 0.0)
        self.assertEqual(self._err, "")


class TestChatMcp(unittest.TestCase):
    """`venice chat --mcp NAME` attaches external tools behind the agent loop."""

    _CFG = {"version": 1,
            "mcpServers": {"fs": {"command": "srv", "args": []}},
            "defaults": {}}

    def _run_seq(self, args, results, *, cfg=None, attach=None, mcp_probe=_MCP_PRESENT,
                 stdin_tty=None, stdout=None, stderr=None, urlopen=None):
        from venice.commands import chat
        fake, calls = _fake_openai_seq(results)
        cfg = self._CFG if cfg is None else cfg
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}))
            st.enter_context(mock.patch("venice.userconfig.load_config",
                                        lambda *a, **k: cfg))
            st.enter_context(mock.patch("venice.client.urllib.request.urlopen",
                                        urlopen or _urlopen_ok()))
            st.enter_context(mock.patch("openai.OpenAI", return_value=fake))
            # SDK-independent: pretend the [mcp] extra is (or isn't) present.
            st.enter_context(mock.patch("venice.commands._mcp.import_mcp",
                                        return_value=mcp_probe))
            st.enter_context(mock.patch.object(sys, "stdout", stdout or io.StringIO()))
            st.enter_context(mock.patch.object(sys, "stderr", stderr or io.StringIO()))
            if attach is not None:
                st.enter_context(mock.patch("venice.commands._mcp_client.attach", attach))
            if stdin_tty is not None:
                fs = mock.MagicMock()
                fs.isatty.return_value = stdin_tty
                st.enter_context(mock.patch.object(sys, "stdin", fs))
            rc = chat._run(args)
        return rc, fake, calls

    def _tool_names(self, call):
        return {t["function"]["name"] for t in call["tools"]}

    def test_mcp_tools_concatenated_and_dispatched(self):
        # --mcp alone (no --tools) still enters the agent loop.
        seq = [
            FakeToolCompletion(tool_calls=[
                _FnCall("c1", "fs__read", '{"path": "/etc/hosts"}')]),
            FakeToolCompletion("read it"),
        ]
        attach = _fake_attach_cm([_fake_tool("fs__read", {"status": "ok", "content": "127.0.0.1"})])
        out = io.StringIO()
        rc, fake, calls = self._run_seq(
            _args(message="hi", mcp=["fs"], stream=False), seq, attach=attach, stdout=out
        )
        self.assertEqual(rc, 0)
        names = self._tool_names(calls[0])
        self.assertIn("fs__read", names)          # remote tool advertised
        self.assertIn("venice_image", names)      # alongside the built-ins
        self.assertEqual(len(names), 15)          # 14 built-ins + 1 remote
        self.assertEqual(attach.specs, [("fs", {"command": "srv", "args": []})])
        tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
        self.assertIn("127.0.0.1", tool_msgs[0]["content"])
        self.assertEqual(out.getvalue().strip(), "read it")

    def test_no_mcp_disables_attach(self):
        attach = mock.MagicMock()
        rc, fake, calls = self._run_seq(
            _args(message="hi", tools=True, mcp=["fs"], no_mcp=True, stream=False),
            [FakeToolCompletion("plain agent")], attach=attach,
        )
        self.assertEqual(rc, 0)
        attach.assert_not_called()
        self.assertEqual(len(self._tool_names(calls[0])), 14)  # built-ins only

    def test_unknown_mcp_server_exits_2_before_model(self):
        attach = mock.MagicMock()
        err = io.StringIO()
        rc, fake, calls = self._run_seq(
            _args(message="hi", mcp=["ghost"], stream=False),
            [FakeToolCompletion("unreached")],
            cfg={"version": 1, "mcpServers": {}, "defaults": {}},
            attach=attach, stderr=err,
        )
        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 0)       # never reached the model
        attach.assert_not_called()
        self.assertIn("unknown MCP server", err.getvalue())

    def test_missing_mcp_extra_exits_2(self):
        err = io.StringIO()
        rc, fake, calls = self._run_seq(
            _args(message="hi", mcp=["fs"], stream=False),
            [FakeToolCompletion("unreached")], stderr=err, mcp_probe=None,
        )
        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 0)

    def test_side_effecting_remote_tool_gated_non_tty(self):
        def se_invoke(arguments, *, confirm=False):
            return ({"status": "ok", "content": "wrote"} if confirm
                    else {"status": "confirmation_required", "message": "gate"})
        from venice.commands import _agent
        tool = _agent.Tool(name="fs__write", description="w",
                           parameters={"type": "object", "properties": {}},
                           invoke=se_invoke, paid=True)
        seq = [
            FakeToolCompletion(tool_calls=[
                _FnCall("c1", "fs__write", '{"path": "/x", "data": "y"}')]),
            FakeToolCompletion("declined, adapting"),
        ]
        rc, fake, calls = self._run_seq(
            _args(message="hi", mcp=["fs"], stream=False), seq,
            attach=_fake_attach_cm([tool]), stdin_tty=False,
        )
        self.assertEqual(rc, 0)
        tool_msgs = [m for m in calls[1]["messages"] if m.get("role") == "tool"]
        self.assertIn("confirmation_required", tool_msgs[0]["content"])

    def test_side_effecting_remote_tool_runs_under_yes(self):
        seen = {}
        def se_invoke(arguments, *, confirm=False):
            seen["confirm"] = confirm
            return {"status": "ok", "content": "wrote"}
        from venice.commands import _agent
        tool = _agent.Tool(name="fs__write", description="w",
                           parameters={"type": "object", "properties": {}},
                           invoke=se_invoke, paid=True)
        seq = [
            FakeToolCompletion(tool_calls=[
                _FnCall("c1", "fs__write", '{"path": "/x"}')]),
            FakeToolCompletion("done"),
        ]
        rc, fake, calls = self._run_seq(
            _args(message="hi", mcp=["fs"], yes=True, stream=False), seq,
            attach=_fake_attach_cm([tool]),
        )
        self.assertEqual(rc, 0)
        self.assertTrue(seen["confirm"])  # --yes -> confirm=True bypasses the gate

    def test_config_default_mcp_attaches(self):
        cfg = {"version": 1,
               "mcpServers": {"fs": {"command": "srv"}},
               "defaults": {"chat": {"mcp": ["fs"]}}}
        attach = _fake_attach_cm([_fake_tool("fs__read", {"status": "ok", "content": "x"})])
        rc, fake, calls = self._run_seq(
            _args(message="hi", tools=True, stream=False),  # no --mcp on CLI
            [FakeToolCompletion("hi")], cfg=cfg, attach=attach,
        )
        self.assertEqual(rc, 0)
        self.assertIn("fs__read", self._tool_names(calls[0]))
        self.assertEqual(attach.specs, [("fs", {"command": "srv"})])


class TestBuiltinToolsRegistry(unittest.TestCase):
    """`_agent.builtin_tools` source-selection (backs `venice code --assets`, #45)."""

    def test_only_none_stays_eight(self):
        # chat's default advertisement must not grow when code gains asset tools
        from venice.commands import _agent
        names = {t.name for t in _agent.builtin_tools(object())}
        # +venice_models +model_details +vision +job_status +job_result +reindex
        self.assertEqual(len(names), 14)
        self.assertIn("reindex", names)
        self.assertIn("venice_models", names)
        self.assertIn("venice_model_details", names)
        self.assertIn("venice_vision", names)
        self.assertIn("venice_job_status", names)
        self.assertIn("venice_job_result", names)
        self.assertNotIn("venice_image_edit", names)

    def test_only_can_select_code_asset_extra(self):
        from venice.commands import _agent
        tools = _agent.builtin_tools(object(), only={"venice_image_edit"})
        self.assertEqual([t.name for t in tools], ["venice_image_edit"])
        self.assertTrue(tools[0].paid)

    def test_only_mixes_builtins_and_extras(self):
        from venice.commands import _agent
        names = {t.name for t in _agent.builtin_tools(
            object(), only={"venice_image", "venice_image_edit"})}
        self.assertEqual(names, {"venice_image", "venice_image_edit"})

    def test_image_edit_schema_excludes_controlled(self):
        from venice.commands import _agent
        props = _agent._IMAGE_EDIT_SCHEMA["properties"]
        for banned in ("confirm", "max_spend", "output_dir"):
            self.assertNotIn(banned, props)
        self.assertEqual(_agent._IMAGE_EDIT_SCHEMA.get("required"), ["prompt"])

    def test_image_schema_exposes_safety_flags(self):
        # #61: the agent must be able to toggle safe_mode/hide_watermark per call
        # (parity with venice_image_edit's safe_mode, renamed off no_safe_mode
        # in #57 Class B so both image tools spell the property the same way).
        from venice.commands import _agent
        props = _agent._IMAGE_SCHEMA["properties"]
        self.assertIn("safe_mode", props)
        self.assertIn("hide_watermark", props)
        self.assertEqual(props["safe_mode"]["type"], "boolean")
        self.assertEqual(props["hide_watermark"]["type"], "boolean")
        for banned in ("confirm", "max_spend", "output_dir"):
            self.assertNotIn(banned, props)
        self.assertEqual(_agent._IMAGE_SCHEMA.get("required"), ["prompt"])

    def test_video_schema_excludes_controlled(self):
        from venice.commands import _agent
        props = _agent._VIDEO_SCHEMA["properties"]
        for banned in ("confirm", "max_spend", "output_dir"):
            self.assertNotIn(banned, props)
        self.assertEqual(_agent._VIDEO_SCHEMA.get("required"), ["prompt"])


if __name__ == "__main__":
    unittest.main()
