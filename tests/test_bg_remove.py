"""Unit tests for `venice bg-remove` (mocks urlopen)."""
import argparse
import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

from tests.test_client import FakeResp

SOURCE_PNG = b"SOURCEPNG"
NOBG_PNG = b"TRANSPARENTPNGBYTES"


def _args(**ov):
    base = dict(
        input=Path("in.png"),
        image_url=None,
        output=None,
        yes=True,
        dry_run=False,
        max_spend=None,
        no_balance=True,
        command="bg-remove",
    )
    base.update(ov)
    return argparse.Namespace(**base)


def _http_error(code):
    body = json.dumps({"code": "ERR", "message": "nope"}).encode()

    def boom(*a, **kw):
        raise HTTPError(
            url="https://api.venice.ai/api/v1/image/background-remove",
            code=code,
            msg="err",
            hdrs={"Content-Type": "application/json"},  # type: ignore[arg-type]
            fp=io.BytesIO(body),
        )

    return boom


class TestBgRemoveFlow(unittest.TestCase):

    def setUp(self):
        # Hermetic: never read the developer's real ~/.config/venice/config.json.
        _cfg = mock.patch(
            "venice.userconfig.load_config",
            lambda *a, **k: {"version": 1, "mcpServers": {}, "defaults": {}},
        )
        _cfg.start()
        self.addCleanup(_cfg.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        Path("in.png").write_bytes(SOURCE_PNG)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_file_input_writes_png_and_sends_base64(self):
        from venice.commands import bg_remove

        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return FakeResp(200, NOBG_PNG, "image/png")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = bg_remove._run(_args())

        self.assertEqual(rc, 0)
        self.assertTrue(captured["url"].endswith("/image/background-remove"))
        self.assertEqual(base64.b64decode(captured["body"]["image"]), SOURCE_PNG)
        self.assertNotIn("image_url", captured["body"])
        out = Path("in-nobg.png")
        self.assertTrue(out.exists())
        self.assertEqual(out.read_bytes(), NOBG_PNG)

    def test_image_url_builds_url_body(self):
        from venice.commands import bg_remove

        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return FakeResp(200, NOBG_PNG, "image/png")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = bg_remove._run(_args(input=None,
                                      image_url="https://x.test/a.jpg"))

        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"], {"image_url": "https://x.test/a.jpg"})
        self.assertNotIn("image", captured["body"])
        self.assertTrue(Path("venice-nobg.png").exists())

    def test_output_flag_names_file(self):
        from venice.commands import bg_remove

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        lambda *a, **kw: FakeResp(200, NOBG_PNG, "image/png")):
            rc = bg_remove._run(_args(output=Path("alpha.png")))

        self.assertEqual(rc, 0)
        self.assertTrue(Path("alpha.png").exists())
        self.assertEqual(Path("alpha.png").read_bytes(), NOBG_PNG)

    def test_dry_run_makes_no_call_and_no_file(self):
        from venice.commands import bg_remove

        def explode(*a, **kw):
            raise AssertionError("dry-run must not call the API")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", explode):
            rc = bg_remove._run(_args(dry_run=True))

        self.assertEqual(rc, 0)
        self.assertFalse(Path("in-nobg.png").exists())

    def test_neither_source_returns_2(self):
        from venice.commands import bg_remove

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}):
            rc = bg_remove._run(_args(input=None, image_url=None))
        self.assertEqual(rc, 2)

    def test_missing_input_file_returns_2(self):
        from venice.commands import bg_remove

        def explode(*a, **kw):
            raise AssertionError("must not call the API on bad input")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", explode):
            rc = bg_remove._run(_args(input=Path("nope.png")))

        self.assertEqual(rc, 2)

    def test_402_maps_to_1(self):
        from venice.commands import bg_remove

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", _http_error(402)):
            rc = bg_remove._run(_args())
        self.assertEqual(rc, 1)

    def test_400_maps_to_2(self):
        from venice.commands import bg_remove

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", _http_error(400)):
            rc = bg_remove._run(_args())
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
