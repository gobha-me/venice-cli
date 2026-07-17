"""Unit tests for `venice chat`.

Mocks the OpenAI client (chat completions) and the free /models catalog GET
(via urlopen). No network, no real key. The openai package must be importable
(pip install -e ".[openai]").
"""
import argparse
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

def _text_payload():
    return json.dumps({
        "object": "list",
        "data": [
            {"id": "llama-3.3-70b", "type": "text",
             "model_spec": {"traits": ["default"]}},
            {"id": "venice-uncensored", "type": "text",
             "model_spec": {"traits": []}},
        ],
    }).encode()


def _urlopen_ok():
    def _u(req, timeout=None):
        return FakeResp(200, _text_payload(), "application/json")
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


class TestChat(unittest.TestCase):

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

    def test_no_message_exit_2(self):
        fake_stdin = mock.MagicMock()
        fake_stdin.isatty.return_value = True
        err = io.StringIO()
        with mock.patch.object(sys, "stdin", fake_stdin):
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


if __name__ == "__main__":
    unittest.main()
