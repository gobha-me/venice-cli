"""Unit tests for `venice image-edit` (mocks urlopen)."""
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
MASK_PNG = b"MASKPNG"
LAYER2_PNG = b"LAYER2PNG"
EDITED_PNG = b"EDITEDPNGBYTES"


def _args(**ov):
    base = dict(
        input=Path("in.png"),
        image_url=None,
        prompt="change the sky to a sunrise",
        layer=None,
        model=None,
        aspect_ratio=None,
        resolution=None,
        output_format=None,
        no_safe_mode=False,
        output=None,
        yes=True,
        dry_run=False,
        max_spend=None,
        no_balance=True,
        command="image-edit",
    )
    base.update(ov)
    return argparse.Namespace(**base)


def _http_error(code):
    body = json.dumps({"code": "ERR", "message": "nope"}).encode()

    def boom(*a, **kw):
        raise HTTPError(
            url="https://api.venice.ai/api/v1/image/edit",
            code=code,
            msg="err",
            hdrs={"Content-Type": "application/json"},  # type: ignore[arg-type]
            fp=io.BytesIO(body),
        )

    return boom


class TestImageEditFlow(unittest.TestCase):

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
        Path("mask.png").write_bytes(MASK_PNG)
        Path("layer2.png").write_bytes(LAYER2_PNG)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def _run(self, args, resp=None):
        from venice.commands import image_edit

        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return resp if resp is not None else FakeResp(200, EDITED_PNG, "image/png")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = image_edit._run(args)
        return rc, captured

    def test_file_input_writes_png_and_sends_base64(self):
        rc, cap = self._run(_args())

        self.assertEqual(rc, 0)
        self.assertTrue(cap["url"].endswith("/image/edit"))
        self.assertEqual(base64.b64decode(cap["body"]["image"]), SOURCE_PNG)
        self.assertEqual(cap["body"]["prompt"], "change the sky to a sunrise")
        self.assertNotIn("images", cap["body"])
        out = Path("in-edit.png")
        self.assertTrue(out.exists())
        self.assertEqual(out.read_bytes(), EDITED_PNG)

    def test_image_url_goes_in_image_field(self):
        # /image/edit takes a URL in the *same* `image` field (not `image_url`).
        rc, cap = self._run(_args(input=None, image_url="https://x.test/a.jpg"))

        self.assertEqual(rc, 0)
        self.assertEqual(cap["body"]["image"], "https://x.test/a.jpg")
        self.assertNotIn("image_url", cap["body"])
        self.assertTrue(cap["url"].endswith("/image/edit"))
        self.assertTrue(Path("venice-edit.png").exists())

    def test_layer_routes_to_multi_edit(self):
        rc, cap = self._run(_args(layer=[Path("mask.png")]))

        self.assertEqual(rc, 0)
        self.assertTrue(cap["url"].endswith("/image/multi-edit"))
        self.assertNotIn("image", cap["body"])
        imgs = cap["body"]["images"]
        self.assertEqual(len(imgs), 2)
        self.assertEqual(base64.b64decode(imgs[0]), SOURCE_PNG)
        self.assertEqual(base64.b64decode(imgs[1]), MASK_PNG)

    def test_two_layers_ordered_base_first(self):
        rc, cap = self._run(_args(layer=[Path("mask.png"), Path("layer2.png")]))

        self.assertEqual(rc, 0)
        imgs = cap["body"]["images"]
        self.assertEqual([base64.b64decode(i) for i in imgs],
                         [SOURCE_PNG, MASK_PNG, LAYER2_PNG])

    def test_output_format_jpeg_names_jpg_and_sets_field(self):
        rc, cap = self._run(_args(output_format="jpeg"))

        self.assertEqual(rc, 0)
        self.assertEqual(cap["body"]["output_format"], "jpeg")
        self.assertTrue(Path("in-edit.jpg").exists())

    def test_model_maps_to_model_for_single_edit(self):
        rc, cap = self._run(_args(model="qwen-edit"))
        self.assertEqual(rc, 0)
        self.assertEqual(cap["body"]["model"], "qwen-edit")
        self.assertNotIn("modelId", cap["body"])

    def test_model_maps_to_modelId_for_multi_edit(self):
        rc, cap = self._run(_args(model="qwen-edit", layer=[Path("mask.png")]))
        self.assertEqual(rc, 0)
        self.assertEqual(cap["body"]["modelId"], "qwen-edit")
        self.assertNotIn("model", cap["body"])

    def test_no_safe_mode_sends_false_else_omitted(self):
        _, cap = self._run(_args(no_safe_mode=True))
        self.assertIs(cap["body"]["safe_mode"], False)

        _, cap2 = self._run(_args())
        self.assertNotIn("safe_mode", cap2["body"])

    def test_aspect_ratio_and_resolution_pass_through(self):
        _, cap = self._run(_args(aspect_ratio="16:9", resolution="2K"))
        self.assertEqual(cap["body"]["aspect_ratio"], "16:9")
        self.assertEqual(cap["body"]["resolution"], "2K")

    def test_output_flag_names_file(self):
        rc, _ = self._run(_args(output=Path("edited.png")))
        self.assertEqual(rc, 0)
        self.assertEqual(Path("edited.png").read_bytes(), EDITED_PNG)

    def test_dry_run_makes_no_call_and_no_file(self):
        from venice.commands import image_edit

        def explode(*a, **kw):
            raise AssertionError("dry-run must not call the API")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", explode):
            rc = image_edit._run(_args(dry_run=True))

        self.assertEqual(rc, 0)
        self.assertFalse(Path("in-edit.png").exists())

    def test_neither_source_returns_2(self):
        from venice.commands import image_edit
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}):
            rc = image_edit._run(_args(input=None, image_url=None))
        self.assertEqual(rc, 2)

    def test_both_sources_returns_2(self):
        from venice.commands import image_edit
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}):
            rc = image_edit._run(_args(image_url="https://x.test/a.jpg"))
        self.assertEqual(rc, 2)

    def test_missing_prompt_returns_2(self):
        from venice.commands import image_edit
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}):
            rc = image_edit._run(_args(prompt=None))
        self.assertEqual(rc, 2)

    def test_too_many_layers_returns_2(self):
        from venice.commands import image_edit

        def explode(*a, **kw):
            raise AssertionError("must not call the API on bad input")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", explode):
            rc = image_edit._run(_args(
                layer=[Path("mask.png"), Path("layer2.png"), Path("in.png")]))
        self.assertEqual(rc, 2)

    def test_missing_input_file_returns_2(self):
        from venice.commands import image_edit

        def explode(*a, **kw):
            raise AssertionError("must not call the API on bad input")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", explode):
            rc = image_edit._run(_args(input=Path("nope.png")))
        self.assertEqual(rc, 2)

    def test_missing_layer_file_returns_2(self):
        from venice.commands import image_edit

        def explode(*a, **kw):
            raise AssertionError("must not call the API on bad input")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", explode):
            rc = image_edit._run(_args(layer=[Path("nope.png")]))
        self.assertEqual(rc, 2)

    def test_402_maps_to_1(self):
        from venice.commands import image_edit
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", _http_error(402)):
            rc = image_edit._run(_args())
        self.assertEqual(rc, 1)

    def test_400_maps_to_2(self):
        from venice.commands import image_edit
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", _http_error(400)):
            rc = image_edit._run(_args())
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
