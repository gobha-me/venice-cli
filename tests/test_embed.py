"""Unit tests for `venice embed`.

Mocks the OpenAI client (embeddings.create) and the free
/models?type=embedding catalog GET (via urlopen). No network, no real key.
The openai package must be importable (pip install -e ".[openai]").
"""
import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from tests.test_client import FakeResp


def _args(**ov):
    base = dict(
        text=None, from_file=None, model=None,
        dimensions=None, encoding_format=None, json=False,
    )
    base.update(ov)
    return argparse.Namespace(**base)


# --- fake OpenAI embeddings response objects ---

class _Item:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class FakeEmbeddings:
    def __init__(self, vectors, model="text-embedding-bge-m3"):
        # vectors: list of (index, embedding) preserving arbitrary order
        self.data = [_Item(i, v) for i, v in vectors]
        self.model = model
        self.usage = {"prompt_tokens": 4, "total_tokens": 4}
        self._dump = {
            "object": "list",
            "model": model,
            "data": [{"index": i, "embedding": v} for i, v in vectors],
            "usage": self.usage,
        }

    def model_dump(self):
        return self._dump


# --- catalog GET mock: two embedding models, one with the `default` trait ---

def _embedding_payload():
    return json.dumps({
        "object": "list",
        "data": [
            {"id": "text-embedding-bge-m3", "type": "embedding",
             "model_spec": {"traits": ["default"]}},
            {"id": "text-embedding-qwen3-8b", "type": "embedding",
             "model_spec": {"traits": []}},
        ],
    }).encode()


def _urlopen_ok():
    def _u(req, timeout=None):
        return FakeResp(200, _embedding_payload(), "application/json")
    return _u


def _fake_openai(result):
    """Return (fake_client, captured_kwargs). create() records its kwargs."""
    captured = {}
    fake = mock.MagicMock()

    def _create(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return result

    fake.embeddings.create.side_effect = _create
    return fake, captured


class TestEmbed(unittest.TestCase):

    def _run(self, args, result, stdout=None, stderr=None):
        from venice.commands import embed
        fake, captured = _fake_openai(result)
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.object(sys, "stdout", stdout or io.StringIO()), \
             mock.patch.object(sys, "stderr", stderr or io.StringIO()):
            rc = embed._run(args)
        return rc, fake, captured

    def test_single_text_ndjson(self):
        out = io.StringIO()
        rc, fake, captured = self._run(
            _args(text="hello"), FakeEmbeddings([(0, [0.1, 0.2, 0.3])]), stdout=out
        )
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue().strip(), "[0.1, 0.2, 0.3]")
        # default model resolved from the `default`-trait catalog entry
        self.assertEqual(captured["model"], "text-embedding-bge-m3")
        # single input is a plain string, not a list
        self.assertEqual(captured["input"], "hello")

    def test_stdin_dash_becomes_input(self):
        with mock.patch.object(sys, "stdin", io.StringIO("piped text")):
            rc, fake, captured = self._run(
                _args(text="-"), FakeEmbeddings([(0, [1.0])])
            )
        self.assertEqual(rc, 0)
        self.assertEqual(captured["input"], "piped text")

    def test_piped_stdin_no_arg(self):
        with mock.patch.object(sys, "stdin", io.StringIO("from pipe")):
            rc, fake, captured = self._run(
                _args(text=None), FakeEmbeddings([(0, [1.0])])
            )
        self.assertEqual(rc, 0)
        self.assertEqual(captured["input"], "from pipe")

    def test_from_file_batch_index_order(self):
        out = io.StringIO()
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as fh:
            fh.write("first\n\nsecond\nthird\n")
            path = fh.name
        try:
            # return vectors out of order to prove we sort by index
            result = FakeEmbeddings([(2, [3.0]), (0, [1.0]), (1, [2.0])])
            rc, fake, captured = self._run(
                _args(from_file=path), result, stdout=out
            )
        finally:
            os.unlink(path)
        self.assertEqual(rc, 0)
        # blank line dropped -> three inputs as a list
        self.assertEqual(captured["input"], ["first", "second", "third"])
        self.assertEqual(
            out.getvalue().strip().splitlines(), ["[1.0]", "[2.0]", "[3.0]"]
        )

    def test_dimensions_and_encoding_passthrough(self):
        rc, fake, captured = self._run(
            _args(text="hi", dimensions=256, encoding_format="base64"),
            FakeEmbeddings([(0, ["QUJD"])]),
        )
        self.assertEqual(rc, 0)
        self.assertEqual(captured["dimensions"], 256)
        self.assertEqual(captured["encoding_format"], "base64")

    def test_optional_flags_omitted_when_unset(self):
        rc, fake, captured = self._run(
            _args(text="hi"), FakeEmbeddings([(0, [1.0])])
        )
        self.assertEqual(rc, 0)
        self.assertNotIn("dimensions", captured)
        self.assertNotIn("encoding_format", captured)

    def test_json_dumps_full_response(self):
        out = io.StringIO()
        rc, fake, captured = self._run(
            _args(text="hi", json=True),
            FakeEmbeddings([(0, [0.5])]),
            stdout=out,
        )
        self.assertEqual(rc, 0)
        doc = json.loads(out.getvalue())
        self.assertEqual(doc["model"], "text-embedding-bge-m3")
        self.assertIn("usage", doc)
        self.assertEqual(doc["data"][0]["embedding"], [0.5])

    def test_bad_model_exit_6_before_call(self):
        err = io.StringIO()
        rc, fake, captured = self._run(
            _args(text="hi", model="no-such-model"),
            FakeEmbeddings([(0, [1.0])]), stderr=err,
        )
        self.assertEqual(rc, 6)
        self.assertEqual(fake.embeddings.create.call_count, 0)
        self.assertIn("no-such-model", err.getvalue())

    def test_no_input_exit_2(self):
        fake_stdin = mock.MagicMock()
        fake_stdin.isatty.return_value = True
        err = io.StringIO()
        with mock.patch.object(sys, "stdin", fake_stdin):
            rc, fake, captured = self._run(
                _args(text=None), FakeEmbeddings([(0, [1.0])]), stderr=err
            )
        self.assertEqual(rc, 2)
        self.assertEqual(fake.embeddings.create.call_count, 0)

    def test_text_and_from_file_conflict_exit_2(self):
        err = io.StringIO()
        rc, fake, captured = self._run(
            _args(text="hi", from_file="/tmp/whatever.txt"),
            FakeEmbeddings([(0, [1.0])]), stderr=err,
        )
        self.assertEqual(rc, 2)
        self.assertEqual(fake.embeddings.create.call_count, 0)
        self.assertIn("not both", err.getvalue())

    def test_missing_openai_exit_2(self):
        from venice.commands import embed
        err = io.StringIO()
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch.dict(sys.modules, {"openai": None}), \
             mock.patch.object(sys, "stderr", err):
            rc = embed._run(_args(text="hi"))
        self.assertEqual(rc, 2)
        self.assertIn("openai", err.getvalue())


if __name__ == "__main__":
    unittest.main()
