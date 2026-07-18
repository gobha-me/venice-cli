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
        style_prefix=None,
        preset=None,
        preset_file=None,
        seed=None,
        cfg_scale=None,
        steps=None,
        style_preset=None,
        variants=1,
        safe_mode=True,
        hide_watermark=False,
        name=None,
        output=None,
        save_json=False,
        from_json=None,
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


def _gen_payload_resolved(n=1, seed=998319):
    """Like _gen_payload but echoes resolved params at request.data (incl. the
    seed Venice picked), as the real /image/generate response does."""
    b64 = base64.b64encode(FAKE_PNG).decode()
    return json.dumps({
        "id": "generate-image-1",
        "images": [b64 for _ in range(n)],
        "request": {"data": {
            "model": "venice-sd35",
            "prompt": "a fierce red dragon",
            "format": "png",
            "seed": seed,
            "steps": 20,
            "variants": n,
        }},
        "timing": {"total": 1000},
    }).encode()


class TestImageFlow(unittest.TestCase):

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

    def test_style_prefix_prepended(self):
        from venice.commands import image

        body = image._build_body(
            "a fierce red dragon", _build_args(style_prefix="EPIC,"))
        self.assertEqual(body["prompt"], "EPIC, a fierce red dragon")

    def test_style_prefix_applies_in_batch(self):
        from venice.commands import image

        batch = Path(self.tmp.name) / "cards.tsv"
        batch.write_text(
            "fire-dragon\tA fierce red dragon\n"
            "stone-golem\tAn ancient stone golem\n",
            encoding="utf-8",
        )
        responses = iter([
            FakeResp(200, _image_models_payload(), "application/json"),
            FakeResp(200, _gen_payload(1), "application/json"),
            FakeResp(200, _gen_payload(1), "application/json"),
        ])
        bodies = []

        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/image/generate"):
                bodies.append(json.loads(req.data.decode("utf-8")))
            return next(responses)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = image._run(_build_args(
                prompt=None, from_file=batch, style_prefix="EPIC,"))
        self.assertEqual(rc, 0)
        self.assertEqual(len(bodies), 2)
        self.assertTrue(all(b["prompt"].startswith("EPIC, ") for b in bodies))
        # Filenames come from the per-card prompt, not the shared prefix.
        self.assertTrue(Path("fire-dragon.png").exists())

    def test_negative_prompt_batch_wide(self):
        from venice.commands import image

        batch = Path(self.tmp.name) / "cards.tsv"
        batch.write_text(
            "fire-dragon\tA fierce red dragon\n"
            "stone-golem\tAn ancient stone golem\n",
            encoding="utf-8",
        )
        responses = iter([
            FakeResp(200, _image_models_payload(), "application/json"),
            FakeResp(200, _gen_payload(1), "application/json"),
            FakeResp(200, _gen_payload(1), "application/json"),
        ])
        bodies = []

        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/image/generate"):
                bodies.append(json.loads(req.data.decode("utf-8")))
            return next(responses)

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = image._run(_build_args(
                prompt=None, from_file=batch, negative_prompt="text, watermark"))
        self.assertEqual(rc, 0)
        self.assertEqual(len(bodies), 2)
        self.assertTrue(
            all(b["negative_prompt"] == "text, watermark" for b in bodies))

    def test_preset_resolves_style_and_negative(self):
        from venice.commands import image

        preset_file = Path("presets.json")
        preset_file.write_text(json.dumps({
            "frontline": {
                "style_prefix": "dark fantasy oil painting",
                "negative_prompt": "text, watermark, blurry",
            },
        }), encoding="utf-8")
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
                prompt="a knight", preset="frontline", preset_file=preset_file))
        self.assertEqual(rc, 0)
        b = captured["body"]
        self.assertEqual(b["prompt"], "dark fantasy oil painting a knight")
        self.assertEqual(b["negative_prompt"], "text, watermark, blurry")

    def test_cli_overrides_preset(self):
        from venice.commands import image

        preset_file = Path("presets.json")
        preset_file.write_text(json.dumps({
            "frontline": {
                "style_prefix": "dark fantasy oil painting",
                "negative_prompt": "text, watermark, blurry",
            },
        }), encoding="utf-8")
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
                prompt="a knight", preset="frontline", preset_file=preset_file,
                style_prefix="watercolor sketch"))
        self.assertEqual(rc, 0)
        b = captured["body"]
        self.assertEqual(b["prompt"], "watercolor sketch a knight")
        # negative_prompt not overridden -> still from the preset.
        self.assertEqual(b["negative_prompt"], "text, watermark, blurry")

    def test_preset_unknown_name_returns_exit_2(self):
        from venice.commands import image

        preset_file = Path("presets.json")
        preset_file.write_text(json.dumps({"frontline": {}}), encoding="utf-8")
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}):
            rc = image._run(_build_args(
                prompt="a knight", preset="nope", preset_file=preset_file))
        self.assertEqual(rc, 2)

    def test_preset_missing_file_returns_exit_2(self):
        from venice.commands import image

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}):
            rc = image._run(_build_args(
                prompt="a knight", preset="frontline",
                preset_file=Path("does-not-exist.json")))
        self.assertEqual(rc, 2)

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

    def test_no_sidecar_by_default(self):
        from venice.commands import image

        responses = iter([
            FakeResp(200, _image_models_payload(), "application/json"),
            FakeResp(200, _gen_payload_resolved(1), "application/json"),
        ])
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        lambda *a, **kw: next(responses)):
            rc = image._run(_build_args())
        self.assertEqual(rc, 0)
        self.assertEqual(list(Path(".").glob("*.json")), [])

    def test_save_json_writes_sidecar(self):
        from venice.commands import image

        responses = iter([
            FakeResp(200, _image_models_payload(), "application/json"),
            FakeResp(200, _gen_payload_resolved(1, seed=998319), "application/json"),
        ])
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        lambda *a, **kw: next(responses)):
            rc = image._run(_build_args(save_json=True))
        self.assertEqual(rc, 0)
        sidecars = sorted(Path(".").glob("venice-image-*.json"))
        self.assertEqual(len(sidecars), 1, f"expected 1 sidecar, got {sidecars}")
        spec = json.loads(sidecars[0].read_text())
        self.assertEqual(spec["seed"], 998319)  # resolved seed captured
        self.assertNotIn("variants", spec)  # normalized to a single image

    def test_save_json_multivariant_writes_one_sidecar(self):
        from venice.commands import image

        responses = iter([
            FakeResp(200, _image_models_payload(), "application/json"),
            FakeResp(200, _gen_payload_resolved(3), "application/json"),
        ])
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen",
                        lambda *a, **kw: next(responses)):
            rc = image._run(_build_args(save_json=True, variants=3))
        self.assertEqual(rc, 0)
        pngs = sorted(Path(".").glob("venice-image-*.png"))
        sidecars = sorted(Path(".").glob("venice-image-*.json"))
        self.assertEqual(len(pngs), 3)
        # Only one call-level seed backs all variants and it reproduces the
        # first one, so exactly one sidecar is written -- next to variant 1.
        self.assertEqual(len(sidecars), 1, f"expected 1 sidecar, got {sidecars}")
        self.assertTrue(sidecars[0].name.endswith("-1.json"))
        self.assertNotIn("variants", json.loads(sidecars[0].read_text()))

    def test_replay_from_json_regenerates(self):
        from venice.commands import image

        sidecar = Path(self.tmp.name) / "card.json"
        sidecar.write_text(json.dumps({
            "model": "venice-sd35",
            "prompt": "a saved dragon",
            "format": "png",
            "seed": 998319,
            "steps": 20,
        }), encoding="utf-8")

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
            rc = image._run(_build_args(prompt=None, from_json=sidecar))
        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"]["seed"], 998319)
        self.assertEqual(captured["body"]["prompt"], "a saved dragon")
        self.assertEqual(captured["body"]["steps"], 20)
        self.assertEqual(len(list(Path(".").glob("*.png"))), 1)

    def test_replay_cli_override(self):
        from venice.commands import image

        sidecar = Path(self.tmp.name) / "card.json"
        sidecar.write_text(json.dumps({
            "model": "venice-sd35",
            "prompt": "a saved dragon",
            "format": "png",
            "seed": 998319,
            "steps": 20,
        }), encoding="utf-8")

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
                prompt="a different dragon", from_json=sidecar, steps=40))
        self.assertEqual(rc, 0)
        self.assertEqual(captured["body"]["steps"], 40)  # CLI overrides JSON
        self.assertEqual(captured["body"]["prompt"], "a different dragon")
        self.assertEqual(captured["body"]["seed"], 998319)  # unchanged from JSON

    def test_replay_invalid_json_returns_exit_2(self):
        from venice.commands import image

        sidecar = Path(self.tmp.name) / "bad.json"
        sidecar.write_text("{not valid json", encoding="utf-8")
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}):
            rc = image._run(_build_args(prompt=None, from_json=sidecar))
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
