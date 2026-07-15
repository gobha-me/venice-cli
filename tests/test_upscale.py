"""Unit tests for `venice upscale` (mocks urlopen)."""
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
UPSCALED_PNG = b"UPSCALEDPNGBYTES"


def _args(**ov):
    base = dict(
        input=Path("in.png"),
        scale=2.0,
        enhance=False,
        enhance_creativity=None,
        enhance_prompt=None,
        replication=None,
        output=None,
        yes=True,
        dry_run=False,
        max_spend=None,
        no_balance=True,
        command="upscale",
    )
    base.update(ov)
    return argparse.Namespace(**base)


def _http_error(code):
    body = json.dumps({"code": "ERR", "message": "nope"}).encode()

    def boom(*a, **kw):
        raise HTTPError(
            url="https://api.venice.ai/api/v1/image/upscale",
            code=code,
            msg="err",
            hdrs={"Content-Type": "application/json"},  # type: ignore[arg-type]
            fp=io.BytesIO(body),
        )

    return boom


class TestUpscaleFlow(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        Path("in.png").write_bytes(SOURCE_PNG)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_upscale_writes_png_and_sends_base64(self):
        from venice.commands import upscale

        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return FakeResp(200, UPSCALED_PNG, "image/png")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = upscale._run(_args())

        self.assertEqual(rc, 0)
        self.assertTrue(captured["url"].endswith("/image/upscale"))
        self.assertEqual(base64.b64decode(captured["body"]["image"]), SOURCE_PNG)
        self.assertEqual(captured["body"]["scale"], 2.0)
        self.assertEqual(captured["body"]["enhance"], False)
        out = Path("in-upscaled.png")
        self.assertTrue(out.exists())
        self.assertEqual(out.read_bytes(), UPSCALED_PNG)

    def test_optional_params_included_when_set(self):
        from venice.commands import upscale

        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return FakeResp(200, UPSCALED_PNG, "image/png")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = upscale._run(_args(
                enhance=True, enhance_creativity=0.7,
                enhance_prompt="gold", replication=0.2,
            ))

        self.assertEqual(rc, 0)
        b = captured["body"]
        self.assertEqual(b["enhance"], True)
        self.assertEqual(b["enhanceCreativity"], 0.7)
        self.assertEqual(b["enhancePrompt"], "gold")
        self.assertEqual(b["replication"], 0.2)

    def test_output_flag_names_file(self):
        from venice.commands import upscale

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        lambda *a, **kw: FakeResp(200, UPSCALED_PNG, "image/png")):
            rc = upscale._run(_args(output=Path("big.png")))

        self.assertEqual(rc, 0)
        self.assertTrue(Path("big.png").exists())
        self.assertEqual(Path("big.png").read_bytes(), UPSCALED_PNG)

    def test_dry_run_makes_no_call_and_no_file(self):
        from venice.commands import upscale

        def explode(*a, **kw):
            raise AssertionError("dry-run must not call the API")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", explode):
            rc = upscale._run(_args(dry_run=True))

        self.assertEqual(rc, 0)
        self.assertFalse(Path("in-upscaled.png").exists())

    def test_missing_input_returns_2(self):
        from venice.commands import upscale

        def explode(*a, **kw):
            raise AssertionError("must not call the API on bad input")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", explode):
            rc = upscale._run(_args(input=Path("nope.png")))

        self.assertEqual(rc, 2)

    def test_scale_out_of_range_returns_2(self):
        from venice.commands import upscale

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}):
            rc = upscale._run(_args(scale=5.0))
        self.assertEqual(rc, 2)

    def test_scale_one_without_enhance_returns_2(self):
        from venice.commands import upscale

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}):
            rc = upscale._run(_args(scale=1.0, enhance=False))
        self.assertEqual(rc, 2)

    def test_scale_one_with_enhance_ok(self):
        from venice.commands import upscale

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        lambda *a, **kw: FakeResp(200, UPSCALED_PNG, "image/png")):
            rc = upscale._run(_args(scale=1.0, enhance=True))
        self.assertEqual(rc, 0)
        self.assertTrue(Path("in-upscaled.png").exists())

    def test_402_maps_to_1(self):
        from venice.commands import upscale

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", _http_error(402)):
            rc = upscale._run(_args())
        self.assertEqual(rc, 1)

    def test_400_maps_to_2(self):
        from venice.commands import upscale

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", _http_error(400)):
            rc = upscale._run(_args())
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
