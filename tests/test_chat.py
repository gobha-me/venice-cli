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
import unittest
from unittest import mock

from tests.test_client import FakeResp


def _args(**ov):
    base = dict(
        message=None, system=None, model=None, temperature=None,
        max_tokens=None, stream=True, json=False,
        web_search=None, web_citations=False, web_scraping=False,
        character=None, no_venice_system_prompt=False,
        strip_thinking=False, no_thinking=False, x_search=False,
        # --- agent / tools (#15) ---
        tools=None, tool=None, max_tool_calls=None,
        max_spend=None, yes=None, output=None,
        # --- external MCP client (#21) ---
        mcp=None, no_mcp=False,
        # --- interactive / REPL (#22) ---
        interactive=False, resume=None,
    )
    base.update(ov)
    return argparse.Namespace(**base)


# --- fake OpenAI response objects ---

class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class FakeCompletion:
    def __init__(self, content, venice_parameters=None):
        self.choices = [_Choice(content)]
        self.venice_parameters = venice_parameters
        self._dump = {
            "choices": [{"message": {"content": content}}],
            "venice_parameters": venice_parameters,
        }

    def model_dump(self):
        return self._dump


class _Delta:
    def __init__(self, content):
        self.content = content


class _StreamChoice:
    def __init__(self, content):
        self.delta = _Delta(content)


class FakeChunk:
    def __init__(self, content=None, usage=None, venice_parameters=None):
        self.choices = [_StreamChoice(content)] if content is not None else []
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
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _ToolChoice:
    def __init__(self, msg):
        self.message = msg


class FakeToolCompletion:
    """A completion whose message may carry tool_calls (None => a final answer)."""

    def __init__(self, content=None, tool_calls=None, venice_parameters=None):
        self.choices = [_ToolChoice(_ToolMsg(content, tool_calls))]
        self.venice_parameters = venice_parameters

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

    def _run(self, args, result, stdout=None, stderr=None):
        from venice.commands import chat
        fake, captured = _fake_openai(result)
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.object(sys, "stdout", stdout or io.StringIO()), \
             mock.patch.object(sys, "stderr", stderr or io.StringIO()):
            rc = chat._run(args)
        return rc, fake, captured

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
        out = io.StringIO()
        rc, fake, captured = self._run(
            _args(message="hi", json=True),  # stream default True
            FakeCompletion("hello", venice_parameters={"enable_web_search": "on"}),
            stdout=out,
        )
        self.assertEqual(rc, 0)
        doc = json.loads(out.getvalue())
        self.assertEqual(doc["choices"][0]["message"]["content"], "hello")
        self.assertEqual(doc["venice_parameters"], {"enable_web_search": "on"})
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
        self.assertEqual(len(names), 10)  # media/chat + project_search + models(+details)
        self.assertIn("project_search", names)
        self.assertIn("venice_models", names)
        self.assertIn("venice_model_details", names)

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
        self.assertEqual(len(names), 11)          # 10 built-ins + 1 remote
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
        self.assertEqual(len(self._tool_names(calls[0])), 10)  # built-ins only

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
        self.assertEqual(len(names), 10)  # +venice_models +venice_model_details (free)
        self.assertIn("venice_models", names)
        self.assertIn("venice_model_details", names)
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

    def test_video_schema_excludes_controlled(self):
        from venice.commands import _agent
        props = _agent._VIDEO_SCHEMA["properties"]
        for banned in ("confirm", "max_spend", "output_dir"):
            self.assertNotIn(banned, props)
        self.assertEqual(_agent._VIDEO_SCHEMA.get("required"), ["prompt"])


if __name__ == "__main__":
    unittest.main()
