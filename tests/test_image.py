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
    # #57 Class C1: model/format/variants default None on the parser now, so the
    # fixture matches. `_run` resolves them (apply_defaults -> apply_literals),
    # which turns every _run-based test below into passive proof the fallback
    # fires. Callers that reach past `_run` into `_build_body` want a RESOLVED
    # namespace -- use `_resolved_args` for those.
    base = dict(
        prompt="a fierce red dragon",
        from_file=None,
        model=None,
        format=None,
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
        style_references=None,
        embed_exif_metadata=None,
        lora_strength=None,
        quality=None,
        enable_web_search=None,
        disable_prompt_optimization_thinking=None,
        enhance_prompt=None,
        variants=None,
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


def _apply_class_c_literals(args):
    """Mirror `image._run`'s built-in-literal layer (#57 Class C1)."""
    from venice import userconfig
    from venice.commands import image

    userconfig.apply_literals(
        args,
        model=image.DEFAULT_IMAGE_MODEL,
        format=image.DEFAULT_FORMAT,
        variants=image.DEFAULT_VARIANTS,
    )
    return args


def _resolved_args(**ov):
    """A namespace as `_build_body` sees it: after `_run`'s resolution layers.

    `_build_body` legitimately assumes model/format/variants are already
    resolved -- hardening it with its own fallback would put a second copy of
    each literal in a second place and mask an ordering mistake in `_run`.
    """
    return _apply_class_c_literals(_build_args(**ov))


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


def _unpriced_image_models_payload():
    return json.dumps({
        "object": "list",
        "data": [{
            "id": "venice-sd35",
            "type": "image",
            "model_spec": {"name": "Venice SD3.5", "pricing": {}},
        }],
    }).encode()


def _native_image_entry(*, max_refs=2, strength=True):
    return {
        "id": "venice-sd35",
        "type": "image",
        "model_spec": {
            "name": "Venice SD3.5",
            "supportsStyleReferences": True,
            "constraints": {
                "maxStyleReferences": max_refs,
                "supportsStyleReferenceStrength": strength,
                "qualities": ["low", "medium", "high"],
                "defaultQuality": "medium",
                "defaultResolution": "1K",
            },
            "pricing": {
                "quality": {
                    "1K": {
                        "low": {"usd": 0.01},
                        "medium": {"usd": 0.02},
                        "high": {"usd": 0.03},
                    }
                }
            },
        },
    }


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

    def test_max_spend_with_unknown_price_fails_closed_before_generate(self):
        from venice.commands import image

        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            return FakeResp(200, _unpriced_image_models_payload(), "application/json")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = image._run(_build_args(max_spend=0.02))
        self.assertEqual(rc, 1)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].endswith("/models?type=image"))
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

    def test_body_includes_native_generation_controls(self):
        from venice.commands import image

        refs = [{"image": "https://x.test/style.png", "strength": 0.75}]
        body = image._build_body("p", _resolved_args(
            style_references=refs,
            embed_exif_metadata=False,
            lora_strength=65,
            quality="high",
            enable_web_search=True,
            disable_prompt_optimization_thinking=False,
            enhance_prompt=True,
        ))
        self.assertEqual(body["style_references"], refs)
        self.assertIs(body["embed_exif_metadata"], False)
        self.assertEqual(body["lora_strength"], 65)
        self.assertEqual(body["quality"], "high")
        self.assertIs(body["enable_web_search"], True)
        self.assertIs(body["disable_prompt_optimization_thinking"], False)
        self.assertIs(body["enhance_prompt"], True)

    def test_style_prefix_prepended(self):
        from venice.commands import image

        body = image._build_body(
            "a fierce red dragon", _resolved_args(style_prefix="EPIC,"))
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
        self.assertEqual(sidecars[0].stat().st_mode & 0o777, 0o600)

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


class TestClassCReplayPrecedence(unittest.TestCase):
    """#57 Class C1: model/format/variants moved off the parser to `default=None`,
    with the literal applied in `_run` AFTER the --from-json replay merge.

    `_apply_replay` decides "did the user set this?" by diffing against a virgin
    parser namespace. Applying the literals before the merge would make every
    field read as explicit and the sidecar would be ignored entirely -- silently,
    and with no other failing test. These cases pin the full ladder:

        explicit CLI flag > defaults.image.* > sidecar > built-in literal
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.cwd)
        self.sidecar = Path(self.tmp.name) / "card.json"
        self.sidecar.write_text(json.dumps({
            "model": "sidecar-model",
            "prompt": "a saved dragon",
            "format": "jpeg",
            "variants": 3,
            "seed": 998319,
        }), encoding="utf-8")

    def _body(self, doc=None, **arg_overrides):
        """Run the real handler and return the /image/generate request body."""
        from venice.commands import image

        responses = iter([
            FakeResp(200, _image_models_payload(), "application/json"),
            FakeResp(200, _gen_payload(4), "application/json"),
        ])
        captured = {}

        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/image/generate"):
                captured["body"] = json.loads(req.data.decode("utf-8"))
            return next(responses)

        doc = doc or {"version": 1, "mcpServers": {}, "defaults": {}}
        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.userconfig.load_config", lambda *a, **k: doc), \
             mock.patch("venice.client.urllib.request.urlopen", fake_urlopen):
            rc = image._run(_build_args(
                prompt=None, from_json=self.sidecar, **arg_overrides))
        self.assertEqual(rc, 0)
        return captured["body"]

    def test_cli_flag_beats_everything(self):
        doc = {"version": 1, "mcpServers": {},
               "defaults": {"image": {"model": "cfg-model"}}}
        self.assertEqual(self._body(doc, model="cli-model")["model"], "cli-model")

    def test_config_beats_the_sidecar(self):
        """Matches Class A's width/height exactly: `apply_defaults` runs before
        the merge, so a config value is indistinguishable from a typed flag."""
        doc = {"version": 1, "mcpServers": {},
               "defaults": {"image": {"model": "cfg-model"}}}
        self.assertEqual(self._body(doc)["model"], "cfg-model")

    def test_sidecar_beats_the_literal(self):
        body = self._body()
        self.assertEqual(body["model"], "sidecar-model")
        self.assertEqual(body["format"], "jpeg")

    def test_literal_applies_when_the_sidecar_omits_the_field(self):
        from venice.commands import image

        self.sidecar.write_text(json.dumps({"prompt": "p", "seed": 1}),
                                encoding="utf-8")
        body = self._body()
        self.assertEqual(body["model"], image.DEFAULT_IMAGE_MODEL)
        self.assertEqual(body["format"], image.DEFAULT_FORMAT)

    def test_explicit_flag_equal_to_the_literal_beats_the_sidecar(self):
        """The bug this slice fixes. Before Class C the typed value equalled the
        parser default, so `_apply_replay` read it as unset and the sidecar
        silently overrode a flag the user had actually typed."""
        from venice.commands import image

        body = self._body(model=image.DEFAULT_IMAGE_MODEL)
        self.assertEqual(body["model"], image.DEFAULT_IMAGE_MODEL)

    def test_config_variants_does_not_multiply_a_replay(self):
        """`_sidecar_params` drops `variants` so a sidecar reproduces the ONE
        saved image. A standing `defaults.image.variants` must not silently turn
        `--from-json` into an N-image charge."""
        doc = {"version": 1, "mcpServers": {},
               "defaults": {"image": {"variants": 4}}}
        body = self._body(doc)
        self.assertNotIn("variants", body)  # omitted when 1

    def test_explicit_variants_still_multiplies_a_replay(self):
        """The other half: the user can still ask for N on a replay."""
        body = self._body(variants=2)
        self.assertEqual(body["variants"], 2)

    def test_variants_from_config_survives_the_range_check(self):
        """Proves the literal lands after the merge but before the range check --
        an out-of-range config value must exit 2, not TypeError on None."""
        from venice.commands import image

        doc = {"version": 1, "mcpServers": {},
               "defaults": {"image": {"variants": 9}}}

        def no_network(req, timeout=None):
            # The range check returns 2 before the client is ever built, so this
            # should never fire -- but it must be patched anyway. Nothing pins
            # that ordering, and an unmocked urlopen here would POST to the real
            # api.venice.ai from `make test` the day the check moves.
            raise AssertionError(f"unexpected network call: {req.full_url}")

        with mock.patch.dict(os.environ, {"VENICE_API_KEY": "fake"}), \
             mock.patch("venice.userconfig.load_config", lambda *a, **k: doc), \
             mock.patch("venice.client.urllib.request.urlopen", no_network):
            rc = image._run(_build_args(prompt="p", from_json=None))
        self.assertEqual(rc, 2)


class TestHideWatermarkConfig(unittest.TestCase):
    """`--hide-watermark` is tri-state + config-backable (issue #56)."""

    def _parse(self, *argv):
        from venice.cli import build_parser
        return build_parser().parse_args(["image", "p", *argv])

    def test_flag_is_tristate(self):
        self.assertIsNone(self._parse().hide_watermark)          # unset
        self.assertTrue(self._parse("--hide-watermark").hide_watermark)
        self.assertFalse(self._parse("--no-hide-watermark").hide_watermark)

    def test_body_none_keeps_watermark(self):
        from venice.commands import image
        body = image._build_body("p", _resolved_args(hide_watermark=None))
        self.assertEqual(body["hide_watermark"], False)  # None -> keep watermark

    def test_config_default_hides_watermark(self):
        from venice.commands import image
        from venice import userconfig
        doc = {"version": 1, "mcpServers": {},
               "defaults": {"image": {"hide_watermark": True}}}
        args = self._parse()  # nothing on the CLI
        userconfig.apply_defaults(args, "image", doc)
        _apply_class_c_literals(args)  # mirror _run: _build_body needs variants
        self.assertTrue(args.hide_watermark)
        self.assertEqual(image._build_body("p", args)["hide_watermark"], True)

    def test_cli_no_hide_overrides_config_default(self):
        from venice import userconfig
        doc = {"version": 1, "mcpServers": {},
               "defaults": {"image": {"hide_watermark": True}}}
        args = self._parse("--no-hide-watermark")  # explicit off wins
        userconfig.apply_defaults(args, "image", doc)
        self.assertFalse(args.hide_watermark)

    def test_config_string_value_coerced(self):
        from venice import userconfig
        doc = {"version": 1, "mcpServers": {},
               "defaults": {"image": {"hide_watermark": "true"}}}
        args = self._parse()
        userconfig.apply_defaults(args, "image", doc)
        self.assertTrue(args.hide_watermark)  # _as_bool coerces "true"


class TestSafeModeConfig(unittest.TestCase):
    """`--safe-mode` is tri-state + config-backable (issue #57 Class B)."""

    def _parse(self, *argv):
        from venice.cli import build_parser
        return build_parser().parse_args(["image", "p", *argv])

    def test_flag_is_tristate(self):
        self.assertIsNone(self._parse().safe_mode)              # unset
        self.assertTrue(self._parse("--safe-mode").safe_mode)
        self.assertFalse(self._parse("--no-safe-mode").safe_mode)

    def test_body_none_stays_safe(self):
        from venice.commands import image
        body = image._build_body("p", _resolved_args(safe_mode=None))
        self.assertEqual(body["safe_mode"], True)  # None -> stay safe

    def test_config_default_disables_safe_mode(self):
        from venice.commands import image
        from venice import userconfig
        doc = {"version": 1, "mcpServers": {},
               "defaults": {"image": {"safe_mode": False}}}
        args = self._parse()  # nothing on the CLI
        userconfig.apply_defaults(args, "image", doc)
        _apply_class_c_literals(args)  # mirror _run: _build_body needs variants
        self.assertFalse(args.safe_mode)
        self.assertEqual(image._build_body("p", args)["safe_mode"], False)

    def test_cli_safe_mode_overrides_config(self):
        from venice import userconfig
        doc = {"version": 1, "mcpServers": {},
               "defaults": {"image": {"safe_mode": False}}}
        args = self._parse("--safe-mode")  # explicit on wins
        userconfig.apply_defaults(args, "image", doc)
        self.assertTrue(args.safe_mode)

    def test_config_string_value_coerced(self):
        from venice import userconfig
        doc = {"version": 1, "mcpServers": {},
               "defaults": {"image": {"safe_mode": "false"}}}
        args = self._parse()
        userconfig.apply_defaults(args, "image", doc)
        self.assertFalse(args.safe_mode)  # _as_bool coerces "false"


class TestImagePricingValidation(unittest.TestCase):
    def test_non_finite_catalog_price_is_rejected(self):
        from venice.commands import image

        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad), self.assertRaisesRegex(
                ValueError, "invalid image price"
            ):
                image._usd_from_pricing({"image": {"usd": bad}})

    def test_quality_matrix_uses_exact_resolution_and_quality(self):
        from venice.commands import image

        entry = _native_image_entry()
        self.assertEqual(
            image._price_from_entry(entry, resolution="1K", quality="high"),
            0.03,
        )
        self.assertIsNone(
            image._price_from_entry(entry, resolution="2K", quality="high")
        )


class TestNativeImageControls(unittest.TestCase):
    def test_local_style_reference_becomes_bounded_data_url(self):
        from venice.commands import image

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "style.png"
            raw = b"\x89PNG\r\n\x1a\nSTYLE"
            path.write_bytes(raw)
            refs = image.resolve_style_references([
                json.dumps({"image": str(path), "strength": 0.5})
            ])
        self.assertEqual(refs[0]["strength"], 0.5)
        self.assertTrue(refs[0]["image"].startswith("data:image/png;base64,"))
        self.assertEqual(
            base64.b64decode(refs[0]["image"].split(",", 1)[1]), raw
        )

    def test_raw_base64_style_reference_is_preserved(self):
        from venice.commands import image

        raw = base64.b64encode(b"\x89PNG\r\n\x1a\nSTYLE").decode()
        refs = image.resolve_style_references([{"image": raw}])
        self.assertEqual(refs, [{"image": raw}])

    def test_style_reference_rejects_unknown_fields_and_strength_range(self):
        from venice.commands import image

        with self.assertRaisesRegex(ValueError, "unknown field"):
            image.resolve_style_references([
                {"image": "https://x.test/a.png", "weight": 1}
            ])
        with self.assertRaisesRegex(ValueError, "between 0.1 and 1"):
            image.resolve_style_references([
                {"image": "https://x.test/a.png", "strength": 1.1}
            ])
        fake_data = (
            "data:image/png;base64,"
            + base64.b64encode(b"not an image").decode()
        )
        with self.assertRaisesRegex(ValueError, "recognized image"):
            image.resolve_style_references([{"image": fake_data}])

    def test_catalog_enforces_reference_count_and_strength_support(self):
        from venice.commands import image

        args = _resolved_args(style_references=[
            {"image": "https://x.test/a.png"},
            {"image": "https://x.test/b.png"},
        ])
        with self.assertRaisesRegex(ValueError, "at most 1"):
            image._validate_model_controls(_native_image_entry(max_refs=1), args)

        args.style_references = [
            {"image": "https://x.test/a.png", "strength": 0.5}
        ]
        with self.assertRaisesRegex(ValueError, "does not support.*strength"):
            image._validate_model_controls(
                _native_image_entry(strength=False), args
            )

    def test_quality_and_unknown_addons_are_fail_closed_for_cost(self):
        from venice.commands import image

        args = _resolved_args(quality="high")
        with mock.patch.object(image._models, "catalog", return_value=[
            _native_image_entry()
        ]):
            self.assertEqual(image.prepare_request(object(), args), 0.03)
            args.enable_web_search = True
            self.assertIsNone(image.prepare_request(object(), args))
            args.enable_web_search = False
            args.enhance_prompt = True
            self.assertIsNone(image.prepare_request(object(), args))

    def test_native_controls_round_trip_through_replay(self):
        from venice.commands import image

        with tempfile.TemporaryDirectory() as td:
            sidecar = Path(td) / "native.json"
            refs = [{"image": "https://x.test/style.png", "strength": 0.6}]
            sidecar.write_text(json.dumps({
                "prompt": "saved prompt",
                "style_references": refs,
                "embed_exif_metadata": False,
                "lora_strength": 40,
                "quality": "high",
                "enable_web_search": True,
                "disable_prompt_optimization_thinking": False,
                "enhance_prompt": True,
            }), encoding="utf-8")
            merged = image._apply_replay(_build_args(
                prompt=None, from_json=sidecar
            ))
        self.assertEqual(merged.style_references, refs)
        self.assertIs(merged.embed_exif_metadata, False)
        self.assertEqual(merged.lora_strength, 40)
        self.assertEqual(merged.quality, "high")
        self.assertIs(merged.enable_web_search, True)
        self.assertIs(merged.disable_prompt_optimization_thinking, False)
        self.assertIs(merged.enhance_prompt, True)

    def test_sidecar_keeps_native_fields_the_response_does_not_echo(self):
        from venice.commands import image

        body = image._build_body("p", _resolved_args(
            quality="high",
            embed_exif_metadata=False,
            enhance_prompt=True,
        ))
        params = image._sidecar_params({
            "request": {"data": {"seed": 1234}}
        }, body)
        self.assertEqual(params["seed"], 1234)
        self.assertEqual(params["quality"], "high")
        self.assertIs(params["embed_exif_metadata"], False)
        self.assertIs(params["enhance_prompt"], True)


if __name__ == "__main__":
    unittest.main()
