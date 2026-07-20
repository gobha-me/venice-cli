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
        embed_base_url=None, embed_model=None,
        embed_ca_bundle=None, embed_insecure=False,
    )
    base.update(ov)
    return argparse.Namespace(**base)


def _clean_doc():
    """A defaults-free config doc so apply_defaults is a no-op (hermetic)."""
    return {"version": 1, "mcpServers": {}, "defaults": {}}


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
             mock.patch("venice.userconfig.load_config", return_value=_clean_doc()), \
             mock.patch("venice.client.urllib.request.urlopen", _urlopen_ok()), \
             mock.patch("openai.OpenAI", return_value=fake), \
             mock.patch.object(sys, "stdout", stdout or io.StringIO()), \
             mock.patch.object(sys, "stderr", stderr or io.StringIO()):
            rc = embed._run(args)
        return rc, fake, captured

    def _run_alt(self, args, result, *, extra_env=None, doc=None,
                 stdout=None, stderr=None):
        """Run the alternate-backend path. VENICE_API_KEY and the embed env
        vars are stripped (extra_env re-adds what a case needs), so a passing
        run proves no Venice key was required. `built` captures the OpenAI()
        constructor kwargs; `urlopen` asserts the Venice catalog GET is skipped.
        """
        from venice.commands import embed
        fake, captured = _fake_openai(result)
        built = {}

        def _ctor(**kwargs):
            built.update(kwargs)
            return fake

        urlopen = mock.MagicMock()
        env = {
            k: v for k, v in os.environ.items()
            if k not in ("VENICE_API_KEY", "VENICE_EMBED_BASE_URL",
                         "VENICE_EMBED_API_KEY", "VENICE_EMBED_CA_BUNDLE")
        }
        env.update(extra_env or {})
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("venice.userconfig.load_config",
                        return_value=doc or _clean_doc()), \
             mock.patch("venice.client.urllib.request.urlopen", urlopen), \
             mock.patch("openai.OpenAI", side_effect=_ctor), \
             mock.patch.object(sys, "stdout", stdout or io.StringIO()), \
             mock.patch.object(sys, "stderr", stderr or io.StringIO()):
            rc = embed._run(args)
        return rc, built, captured, urlopen

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
             mock.patch("venice.userconfig.load_config", return_value=_clean_doc()), \
             mock.patch.dict(sys.modules, {"openai": None}), \
             mock.patch.object(sys, "stderr", err):
            rc = embed._run(_args(text="hi"))
        self.assertEqual(rc, 2)
        self.assertIn("openai", err.getvalue())

    # --- alternate / local OpenAI-compatible backend (#23) ---

    def test_alt_backend_skips_catalog_and_needs_no_venice_key(self):
        out = io.StringIO()
        rc, built, captured, urlopen = self._run_alt(
            _args(text="hi", embed_base_url="http://localhost:1234/v1",
                  embed_model="local-embed"),
            FakeEmbeddings([(0, [0.1, 0.2])]), stdout=out,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue().strip(), "[0.1, 0.2]")
        # SDK pointed at the alt endpoint, model taken as given...
        self.assertEqual(built["base_url"], "http://localhost:1234/v1")
        self.assertEqual(captured["model"], "local-embed")
        # ...no key needed -> placeholder; and no Venice catalog GET happened.
        self.assertEqual(built["api_key"], "not-needed")
        self.assertEqual(urlopen.call_count, 0)

    def test_alt_backend_requires_embed_model_exit_2(self):
        err = io.StringIO()
        rc, built, captured, urlopen = self._run_alt(
            _args(text="hi", embed_base_url="http://localhost:1234/v1"),
            FakeEmbeddings([(0, [1.0])]), stderr=err,
        )
        self.assertEqual(rc, 2)
        self.assertIn("--embed-model is required", err.getvalue())
        self.assertEqual(built, {})  # SDK never constructed
        self.assertEqual(urlopen.call_count, 0)

    def test_embed_base_url_from_env_selects_alt_path(self):
        rc, built, captured, urlopen = self._run_alt(
            _args(text="hi", embed_model="local-embed"),  # no CLI base url
            FakeEmbeddings([(0, [1.0])]),
            extra_env={"VENICE_EMBED_BASE_URL": "http://env-host/v1"},
        )
        self.assertEqual(rc, 0)
        self.assertEqual(built["base_url"], "http://env-host/v1")
        self.assertEqual(urlopen.call_count, 0)

    def test_embed_api_key_from_env_reaches_sdk(self):
        rc, built, captured, urlopen = self._run_alt(
            _args(text="hi", embed_base_url="http://localhost:1234/v1",
                  embed_model="local-embed"),
            FakeEmbeddings([(0, [1.0])]),
            extra_env={"VENICE_EMBED_API_KEY": "local-test-key"},
        )
        self.assertEqual(rc, 0)
        self.assertEqual(built["api_key"], "local-test-key")

    def test_cli_base_url_beats_env_and_config(self):
        doc = {"version": 1, "mcpServers": {},
               "defaults": {"embed": {"embed_base_url": "http://config-host/v1"}}}
        rc, built, captured, urlopen = self._run_alt(
            _args(text="hi", embed_base_url="http://cli-host/v1",
                  embed_model="local-embed"),
            FakeEmbeddings([(0, [1.0])]),
            extra_env={"VENICE_EMBED_BASE_URL": "http://env-host/v1"},
            doc=doc,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(built["base_url"], "http://cli-host/v1")

    def test_config_default_supplies_alt_backend(self):
        doc = {"version": 1, "mcpServers": {}, "defaults": {"embed": {
            "embed_base_url": "http://config-host/v1", "embed_model": "cfg-embed"}}}
        rc, built, captured, urlopen = self._run_alt(
            _args(text="hi"),  # nothing on CLI or env
            FakeEmbeddings([(0, [1.0])]), doc=doc,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(built["base_url"], "http://config-host/v1")
        self.assertEqual(captured["model"], "cfg-embed")
        self.assertEqual(urlopen.call_count, 0)

    # --- TLS override for the alternate backend (#28) ---

    def test_ca_bundle_reaches_sdk_on_alt_path(self):
        with mock.patch("httpx.Client", return_value="httpx-sentinel") as Hx:
            rc, built, captured, urlopen = self._run_alt(
                _args(text="hi", embed_base_url="https://embed.local/v1",
                      embed_model="local-embed", embed_ca_bundle="/ca.pem"),
                FakeEmbeddings([(0, [1.0])]),
            )
        self.assertEqual(rc, 0)
        Hx.assert_called_once_with(verify="/ca.pem")
        self.assertEqual(built["http_client"], "httpx-sentinel")

    def test_insecure_disables_verification_and_warns(self):
        err = io.StringIO()
        with mock.patch("httpx.Client", return_value="httpx-sentinel") as Hx:
            rc, built, captured, urlopen = self._run_alt(
                _args(text="hi", embed_base_url="https://embed.local/v1",
                      embed_model="local-embed", embed_insecure=True),
                FakeEmbeddings([(0, [1.0])]), stderr=err,
            )
        self.assertEqual(rc, 0)
        Hx.assert_called_once_with(verify=False)
        self.assertEqual(built["http_client"], "httpx-sentinel")
        self.assertIn("TLS verification disabled", err.getvalue())

    def test_ca_bundle_from_env_on_alt_path(self):
        with mock.patch("httpx.Client", return_value="httpx-sentinel") as Hx:
            rc, built, captured, urlopen = self._run_alt(
                _args(text="hi", embed_base_url="https://embed.local/v1",
                      embed_model="local-embed"),
                FakeEmbeddings([(0, [1.0])]),
                extra_env={"VENICE_EMBED_CA_BUNDLE": "/env-ca.pem"},
            )
        self.assertEqual(rc, 0)
        Hx.assert_called_once_with(verify="/env-ca.pem")

    def test_ca_bundle_from_config_on_alt_path(self):
        doc = {"version": 1, "mcpServers": {}, "defaults": {"embed": {
            "embed_ca_bundle": "/cfg-ca.pem"}}}
        with mock.patch("httpx.Client", return_value="httpx-sentinel") as Hx:
            rc, built, captured, urlopen = self._run_alt(
                _args(text="hi", embed_base_url="https://embed.local/v1",
                      embed_model="local-embed"),
                FakeEmbeddings([(0, [1.0])]), doc=doc,
            )
        self.assertEqual(rc, 0)
        Hx.assert_called_once_with(verify="/cfg-ca.pem")

    def test_insecure_without_base_url_is_rejected(self):
        err = io.StringIO()
        rc, fake, captured = self._run(
            _args(text="hi", embed_insecure=True), FakeEmbeddings([(0, [1.0])]),
            stderr=err,
        )
        self.assertEqual(rc, 2)
        self.assertEqual(fake.embeddings.create.call_count, 0)
        self.assertIn("only apply with --embed-base-url", err.getvalue())

    def test_ca_bundle_without_base_url_is_rejected(self):
        err = io.StringIO()
        rc, fake, captured = self._run(
            _args(text="hi", embed_ca_bundle="/ca.pem"),
            FakeEmbeddings([(0, [1.0])]), stderr=err,
        )
        self.assertEqual(rc, 2)
        self.assertEqual(fake.embeddings.create.call_count, 0)
        self.assertIn("only apply with --embed-base-url", err.getvalue())

    def test_ca_bundle_and_insecure_are_mutually_exclusive(self):
        from venice.commands import embed
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        embed.register(sub)
        with mock.patch.object(sys, "stderr", io.StringIO()), \
             self.assertRaises(SystemExit):
            parser.parse_args(
                ["embed", "--embed-ca-bundle", "/ca.pem", "--embed-insecure"]
            )


if __name__ == "__main__":
    unittest.main()
