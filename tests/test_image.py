"""Unit tests for `venice image` (mocks urlopen)."""
import argparse
import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_client import FakeResp

FAKE_PNG = b"FAKEPNG"


def _build_args(**ov):
    base = dict(
        prompt="a fierce red dragon",
        from_file=None,
        model="venice-sd35",
        format="png",
        width=None,
        height=None,
        aspect_ratio=None,
        resolution=None,
        negative_prompt=None,
        seed=None,
        cfg_scale=None,
        steps=None,
        style_preset=None,
        variants=1,
        safe_mode=True,
        hide_watermark=False,
        name=None,
        output=None,
        yes=True,
        dry_run=False,
        max_spend=None,
        no_balance=True,
        command="image",
    )
    base.update(ov)
    return argparse.Namespace(**base)


def _image_models_payload():
    """Mimics /models?type=image with one priced model."""
    return json.dumps({
        "object": "list",
        "data": [
            {
                "id": "venice-sd35",
                "type": "image",
                "model_spec": {
                    "name": "Venice SD3.5",
                    "pricing": {"image": {"usd": 0.01}},
                },
            },
        ],
    }).encode()


def _gen_payload(n=1):
    b64 = base64.b64encode(FAKE_PNG).decode()
    return json.dumps({
        "id": "generate-image-1",
        "images": [b64 for _ in range(n)],
        "timing": {"total": 1000},
    }).encode()


class TestImageFlow(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_generate_writes_png(self):
        from venice.commands import image

        responses = iter([
            FakeResp(200, _image_models_payload(), "application/json"),
            FakeResp(200, _gen_payload(1), "application/json"),
        ])
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        lambda *a, **kw: next(responses)):
            rc = image._run(_build_args())
        self.assertEqual(rc, 0)
        written = sorted(Path(".").glob("venice-image-*.png"))
        self.assertEqual(len(written), 1, f"expected 1 png, got {written}")
        self.assertEqual(written[0].read_bytes(), FAKE_PNG)

    def test_variants_write_multiple_numbered_files(self):
        from venice.commands import image

        responses = iter([
            FakeResp(200, _image_models_payload(), "application/json"),
            FakeResp(200, _gen_payload(3), "application/json"),
        ])
        captured = {}

        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/image/generate"):
                captured["body"] = json.loads(req.data.decode("utf-8"))
            return next(responses)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = image._run(_build_args(variants=3))
        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"]["variants"], 3)
        written = sorted(Path(".").glob("venice-image-*.png"))
        self.assertEqual(len(written), 3, f"expected 3 pngs, got {written}")
        self.assertTrue(any(p.name.endswith("-1.png") for p in written))
        self.assertTrue(any(p.name.endswith("-3.png") for p in written))

    def test_name_controls_filename(self):
        from venice.commands import image

        responses = iter([
            FakeResp(200, _image_models_payload(), "application/json"),
            FakeResp(200, _gen_payload(1), "application/json"),
        ])
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        lambda *a, **kw: next(responses)):
            rc = image._run(_build_args(name="Fire Dragon"))
        self.assertEqual(rc, 0)
        self.assertTrue(Path("fire-dragon.png").exists())

    def test_dry_run_does_not_call_generate(self):
        from venice.commands import image

        calls = []
        responses = iter([FakeResp(200, _image_models_payload(), "application/json")])

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            return next(responses)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = image._run(_build_args(dry_run=True))
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].endswith("/models?type=image"))
        self.assertEqual(list(Path(".").glob("*.png")), [])

    def test_max_spend_aborts_when_estimate_too_high(self):
        from venice.commands import image

        # 4 variants * $0.01 = $0.04; cap at $0.02 -> abort.
        responses = iter([FakeResp(200, _image_models_payload(), "application/json")])
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        lambda *a, **kw: next(responses)):
            rc = image._run(_build_args(variants=4, max_spend=0.02))
        self.assertEqual(rc, 1)
        self.assertEqual(list(Path(".").glob("*.png")), [])

    def test_missing_prompt_returns_exit_2(self):
        from venice.commands import image

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}):
            rc = image._run(_build_args(prompt=None))
        self.assertEqual(rc, 2)

    def test_variants_out_of_range_returns_exit_2(self):
        from venice.commands import image

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}):
            rc = image._run(_build_args(variants=5))
        self.assertEqual(rc, 2)

    def test_batch_from_file_writes_named_files(self):
        from venice.commands import image

        batch = Path(self.tmp.name) / "cards.tsv"
        batch.write_text(
            "fire-dragon\tA fierce red dragon breathing flame\n"
            "# a comment line\n"
            "\n"
            "An ancient stone golem\n",
            encoding="utf-8",
        )
        responses = iter([
            FakeResp(200, _image_models_payload(), "application/json"),
            FakeResp(200, _gen_payload(1), "application/json"),
            FakeResp(200, _gen_payload(1), "application/json"),
        ])
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        lambda *a, **kw: next(responses)):
            rc = image._run(_build_args(prompt=None, from_file=batch))
        self.assertEqual(rc, 0)
        self.assertTrue(Path("fire-dragon.png").exists())
        # second line has no explicit name -> slug of first ~4 words.
        self.assertTrue(Path("an-ancient-stone-golem.png").exists())

    def test_body_includes_passthrough_params(self):
        from venice.commands import image

        responses = iter([
            FakeResp(200, _image_models_payload(), "application/json"),
            FakeResp(200, _gen_payload(1), "application/json"),
        ])
        captured = {}

        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/image/generate"):
                captured["body"] = json.loads(req.data.decode("utf-8"))
            return next(responses)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = image._run(_build_args(
                negative_prompt="blurry", seed=42, cfg_scale=7.5,
                steps=20, style_preset="3D Model", width=768, height=1024,
                safe_mode=False, hide_watermark=True,
            ))
        self.assertEqual(rc, 0)
        b = captured["body"]
        self.assertEqual(b["negative_prompt"], "blurry")
        self.assertEqual(b["seed"], 42)
        self.assertEqual(b["cfg_scale"], 7.5)
        self.assertEqual(b["steps"], 20)
        self.assertEqual(b["style_preset"], "3D Model")
        self.assertEqual(b["width"], 768)
        self.assertEqual(b["height"], 1024)
        self.assertEqual(b["safe_mode"], False)
        self.assertEqual(b["hide_watermark"], True)
        self.assertEqual(b["format"], "png")
        self.assertNotIn("variants", b)  # omitted when 1

    def test_hide_watermark_defaults_false_in_body(self):
        from venice.commands import image

        responses = iter([
            FakeResp(200, _image_models_payload(), "application/json"),
            FakeResp(200, _gen_payload(1), "application/json"),
        ])
        captured = {}

        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/image/generate"):
                captured["body"] = json.loads(req.data.decode("utf-8"))
            return next(responses)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = image._run(_build_args())
        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"]["hide_watermark"], False)
        self.assertEqual(captured["body"]["safe_mode"], True)


if __name__ == "__main__":
    unittest.main()
