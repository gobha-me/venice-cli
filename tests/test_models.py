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
    if type_ == "music":
        ids = ["elevenlabs-sound-effects-v2", "mmaudio-v2-text-to-audio"]
    elif type_ == "text":
        ids = ["zai-org-glm-5-1", "claude-opus-4-7"]
    else:
        ids = []
    return json.dumps({
        "object": "list",
        "data": [
            {
                "id": mid,
                "type": type_,
                "model_spec": {
                    "name": mid.upper(),
                    "pricing": {"input": {"usd": 1.5}, "output": {"usd": 4.0}},
                    "capabilities": {"supportsWebSearch": True, "supportsVision": False},
                },
            }
            for mid in ids
        ],
    }).encode()


def _fake_urlopen_factory():
    """Return a urlopen mock that routes /models?type=X to the right payload."""
    def _urlopen(req, timeout=None):
        url = req.full_url
        # Cheap parse: find ?type=
        type_ = ""
        if "type=" in url:
            type_ = url.split("type=", 1)[1].split("&", 1)[0]
        return FakeResp(200, _models_payload(type_), "application/json")
    return _urlopen


class TestModels(unittest.TestCase):

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
        # text and music have 2 each (per payload factory); others 0
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
        self.assertIn("capabilities:", out)
        self.assertIn("WebSearch", out)

    def test_slug_lookup_prints_json_for_one(self):
        from venice.commands import models

        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", _fake_urlopen_factory()), \
             mock.patch.object(sys, "stdout", buf):
            rc = models._run(_args(slug="mmaudio-v2-text-to-audio"))
        self.assertEqual(rc, 0)
        doc = json.loads(buf.getvalue())
        self.assertEqual(doc["id"], "mmaudio-v2-text-to-audio")
        self.assertEqual(doc["type"], "music")

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
