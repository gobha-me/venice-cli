"""Unit tests for `venice models` (mocks urlopen)."""
import argparse
import io
import json
import os
import sys
import unittest
from unittest import mock

from tests.test_client import FakeResp


def _args(**ov):
    base = dict(slug=None, type=None, detail=False, json=False)
    base.update(ov)
    return argparse.Namespace(**base)


def _models_payload(type_):
    ids_by_type = {
        "text": ["zai-org-glm-5-1", "claude-opus-4-7"],
        "music": ["elevenlabs-sound-effects-v2", "mmaudio-v2-text-to-audio"],
        "asr": ["nvidia-parakeet-tdt-0.6b-v3"],
        "inpaint": ["flux-dev-inpainting"],
        "future-media": ["tomorrows-model"],
        "alpha-future": ["next-weeks-model"],
    }
    if type_ == "all":
        typed_ids = [
            (model_type, mid)
            for model_type, ids in ids_by_type.items()
            for mid in ids
        ]
    elif type_ == "code":
        # `code` is a special filter over models whose native response type is
        # still `text`.
        typed_ids = [("text", "zai-org-glm-5-1")]
    else:
        typed_ids = [(type_, mid) for mid in ids_by_type.get(type_, [])]
    return json.dumps({
        "object": "list",
        "data": [
            {
                "id": mid,
                "type": model_type,
                "model_spec": {
                    "name": mid.upper(),
                    "pricing": {"input": {"usd": 1.5}, "output": {"usd": 4.0}},
                    "capabilities": {"supportsWebSearch": True, "supportsVision": False},
                },
            }
            for model_type, mid in typed_ids
        ],
    }).encode()


def _fake_urlopen_factory(calls=None):
    """Return a urlopen mock that routes /models?type=X to the right payload."""
    def _urlopen(req, timeout=None):
        url = req.full_url
        # Cheap parse: find ?type=
        type_ = ""
        if "type=" in url:
            type_ = url.split("type=", 1)[1].split("&", 1)[0]
        if calls is not None:
            calls.append(type_)
        return FakeResp(200, _models_payload(type_), "application/json")
    return _urlopen


class TestModels(unittest.TestCase):

    def test_parser_accepts_current_asr_and_inpaint_types(self):
        from venice.commands import models

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        models.register(subparsers)
        for model_type in ("asr", "inpaint"):
            with self.subTest(model_type=model_type):
                args = parser.parse_args(["models", "--type", model_type])
                self.assertEqual(args.type, model_type)

    def test_default_prints_counts_by_type(self):
        from venice.commands import models

        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", _fake_urlopen_factory()), \
             mock.patch.object(sys, "stdout", buf):
            rc = models._run(_args())
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("TYPE", out)
        self.assertIn("COUNT", out)
        self.assertIn("text", out)
        self.assertIn("music", out)
        self.assertIn("asr", out)
        self.assertIn("inpaint", out)
        self.assertIn("future-media", out)
        self.assertRegex(out, r"(?m)^asr\s+1$")
        self.assertRegex(out, r"(?m)^inpaint\s+1$")
        self.assertIn("TOTAL", out)

    def test_type_filter_lists_ids(self):
        from venice.commands import models

        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", _fake_urlopen_factory()), \
             mock.patch.object(sys, "stdout", buf):
            rc = models._run(_args(type="music"))
        self.assertEqual(rc, 0)
        lines = [l for l in buf.getvalue().splitlines() if l.strip()]
        self.assertIn("elevenlabs-sound-effects-v2", lines)
        self.assertIn("mmaudio-v2-text-to-audio", lines)

    def test_current_types_filter_and_emit_json(self):
        from venice.commands import models

        expected = {
            "asr": "nvidia-parakeet-tdt-0.6b-v3",
            "inpaint": "flux-dev-inpainting",
        }
        for model_type, model_id in expected.items():
            with self.subTest(model_type=model_type):
                buf = io.StringIO()
                with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
                     mock.patch("venice.client.urllib.request.urlopen",
                                _fake_urlopen_factory()), \
                     mock.patch.object(sys, "stdout", buf):
                    rc = models._run(_args(type=model_type, json=True))
                self.assertEqual(rc, 0)
                self.assertEqual(json.loads(buf.getvalue())[0]["id"], model_id)

    def test_all_uses_aggregate_catalog_and_preserves_future_types(self):
        from venice.commands import models

        calls = []
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        _fake_urlopen_factory(calls)), \
             mock.patch.object(sys, "stdout", buf):
            rc = models._run(_args(type="all", json=True))
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ["all", "code"])
        by_type = json.loads(buf.getvalue())
        self.assertEqual(by_type["asr"][0]["id"], "nvidia-parakeet-tdt-0.6b-v3")
        self.assertEqual(by_type["inpaint"][0]["id"], "flux-dev-inpainting")
        self.assertEqual(by_type["future-media"][0]["id"], "tomorrows-model")
        self.assertEqual(by_type["code"][0]["id"], "zai-org-glm-5-1")
        self.assertEqual(list(by_type)[-2:], ["alpha-future", "future-media"])

    def test_detail_includes_pricing_and_caps(self):
        from venice.commands import models

        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", _fake_urlopen_factory()), \
             mock.patch.object(sys, "stdout", buf):
            rc = models._run(_args(type="music", detail=True))
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("pricing:", out)
        self.assertIn("$1.5", out)
        self.assertIn("cache: not advertised", out)
        self.assertIn("capabilities:", out)
        self.assertIn("WebSearch", out)

    def test_cache_discount_formats_without_mutating_raw_pricing(self):
        from venice.commands import models

        cases = (
            ({"input": {"usd": 3.75}, "cache_input": {"usd": 0.375}},
             "10.0x discount"),
            ({"input": {"usd": 3.75}, "cache_input": {"usd": 0}},
             "free cache input"),
            ({"input": {"usd": 0.375}, "cache_input": {"usd": 3.75}},
             "advertised (no discount)"),
            ({"input": {"usd": 3.75}}, "not advertised"),
            ({"input": {"usd": 3.75}, "cache_input": {"usd": "bad"}},
             "advertised (discount unavailable)"),
        )
        for pricing, expected in cases:
            with self.subTest(expected=expected):
                before = json.loads(json.dumps(pricing))
                self.assertEqual(models._format_cache(pricing), expected)
                self.assertEqual(pricing, before)

    def test_slug_lookup_prints_json_for_one(self):
        from venice.commands import models

        calls = []
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        _fake_urlopen_factory(calls)), \
             mock.patch.object(sys, "stdout", buf):
            rc = models._run(_args(slug="mmaudio-v2-text-to-audio"))
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ["all"])
        doc = json.loads(buf.getvalue())
        self.assertEqual(doc["id"], "mmaudio-v2-text-to-audio")
        self.assertEqual(doc["type"], "music")
        self.assertNotIn("cache", doc)  # raw catalog JSON gains no derived field

    def test_current_type_slugs_are_found_in_the_all_catalog(self):
        from venice.commands import models

        for slug, model_type in (
            ("nvidia-parakeet-tdt-0.6b-v3", "asr"),
            ("flux-dev-inpainting", "inpaint"),
        ):
            with self.subTest(model_type=model_type):
                calls = []
                buf = io.StringIO()
                with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
                     mock.patch("venice.client.urllib.request.urlopen",
                                _fake_urlopen_factory(calls)), \
                     mock.patch.object(sys, "stdout", buf):
                    rc = models._run(_args(slug=slug))
                self.assertEqual(rc, 0)
                self.assertEqual(calls, ["all"])
                self.assertEqual(json.loads(buf.getvalue())["type"], model_type)

    def test_slug_not_found_returns_exit_6(self):
        from venice.commands import models

        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", _fake_urlopen_factory()), \
             mock.patch.object(sys, "stdout", buf):
            rc = models._run(_args(slug="no-such-model"))
        self.assertEqual(rc, 6)


if __name__ == "__main__":
    unittest.main()
