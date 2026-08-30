"""Offline OpenAPI inventory and request-builder contract tests (#150)."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
import types
import unittest
from pathlib import Path
from unittest import mock

from venice.commands import _audio, bg_remove, chat, embed, image, image_edit, music, sfx, tts, upscale, video


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "contracts" / "openapi" / "venice.lock.json"
MANIFEST_PATH = ROOT / "contracts" / "openapi" / "implementation.json"

_SPEC = importlib.util.spec_from_file_location(
    "venice_openapi_contract", ROOT / "scripts" / "openapi_contract.py"
)
assert _SPEC is not None and _SPEC.loader is not None
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)


def _namespace(**values):
    return types.SimpleNamespace(**values)


def _implemented_fields(manifest, operation):
    bodies = manifest["operations"][operation]["request_bodies"]
    body = bodies.get("application/json")
    return set(body["implemented"]) if body else set()


class CommittedInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_manifest_completely_classifies_the_locked_contract(self):
        self.assertEqual(contract.validate_manifest(self.lock, self.manifest), [])
        self.assertEqual(len(self.lock["implemented_operations"]), 18)

    def test_required_request_fields_cannot_be_marked_unsupported(self):
        manifest = copy.deepcopy(self.manifest)
        body = manifest["operations"]["POST /audio/speech"]["request_bodies"][
            "application/json"
        ]
        body["implemented"].remove("input")
        body["intentional_omissions"]["input"] = "not supported"
        errors = contract.validate_manifest(self.lock, manifest)
        self.assertTrue(
            any("required fields cannot be omitted" in error for error in errors),
            errors,
        )

    def test_contract_files_are_canonical_json(self):
        for path in (LOCK_PATH, MANIFEST_PATH):
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(path.read_bytes(), contract._canonical_bytes(value))

    def test_model_specific_constraints_point_back_to_models(self):
        constraints = {
            operation: body["model_catalog_constraints"]
            for operation, body in self.manifest["operations"].items()
            if body["model_catalog_constraints"]
        }
        self.assertIn("POST /audio/speech", constraints)
        self.assertIn("POST /audio/queue", constraints)
        self.assertIn("POST /video/queue", constraints)
        self.assertIn("POST /image/generate", constraints)
        self.assertIn("POST /image/multi-edit", constraints)
        for mapping in constraints.values():
            self.assertTrue(all(value.startswith("/models:") for value in mapping.values()))

    def test_source_endpoint_literals_are_declared(self):
        upstream_paths = {
            item["key"].split(" ", 1)[1] for item in self.lock["all_operations"]
        }
        declared_paths = {
            key.split(" ", 1)[1] for key in self.manifest["operations"]
        }
        used_paths = set()
        for path in (ROOT / "src" / "venice").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value in upstream_paths
                ):
                    used_paths.add(node.value)
        self.assertEqual(used_paths - declared_paths, set())
        self.assertIn("POST /chat/completions", self.manifest["operations"])
        self.assertIn("POST /embeddings", self.manifest["operations"])

    def test_mcp_and_agent_share_every_model_facing_media_contract(self):
        operations = self.manifest["operations"]
        for key, entry in operations.items():
            if key.startswith("GET ") or key in {
                "POST /chat/completions",
                "POST /embeddings",
            }:
                continue
            surfaces = {consumer.split(":", 1)[0] for consumer in entry["consumers"]}
            self.assertIn("cli", surfaces, key)
            self.assertIn("mcp", surfaces, key)
            self.assertIn("agent", surfaces, key)

    def test_actual_request_builders_match_manifest_field_sets(self):
        image_args = _namespace(
            model="image-model", format="png", safe_mode=True,
            hide_watermark=True, variants=2, width=1024, height=768,
            aspect_ratio="4:3", resolution="1K", negative_prompt="none",
            seed=7, cfg_scale=4.0, steps=20, style_preset="photo",
            style_references=[{"image": "data:image/png;base64,eA==", "strength": 0.5}],
            embed_exif_metadata=True, lora_strength=80, quality="high",
            enable_web_search=True, disable_prompt_optimization_thinking=True,
            enhance_prompt=True, style_prefix=None,
        )
        edit_common = dict(
            prompt="edit", model="edit-model", aspect_ratio="1:1",
            resolution="1K", output_format="png", quality="high",
            safe_mode=True, disable_prompt_optimization_thinking=True,
            enhance_prompt=True,
        )
        chat_args = _namespace(
            system="system", temperature=0.2, max_tokens=100,
            web_search="on", web_citations=True, web_scraping=True,
            character="slug", no_venice_system_prompt=True,
            strip_thinking=True, no_thinking=True, x_search=True,
        )
        chat_body = chat._build_kwargs(chat_args, "text-model", "hello")
        chat_fields = set(chat_body)
        if "extra_body" in chat_fields:
            chat_fields.remove("extra_body")
            chat_fields.update(chat_body["extra_body"])
        chat_fields.update({"stream", "stream_options", "tools", "tool_choice"})

        music_args = _namespace(
            model="music-model", prompt="song", duration=60,
            instrumental=True, lyrics=None, speed=1.2,
        )
        music_lyrics_args = _namespace(
            model="music-model", prompt="song", duration=60,
            instrumental=False, lyrics="lyrics", speed=None,
        )
        shared_video = {
            "duration": "5s", "resolution": "720p",
            "aspect_ratio": "16:9", "audio": False,
        }
        video_media = {
            "audio_url": "data:audio/wav;base64,eA==",
            "elements": [{"reference_image_urls": ["data:image/png;base64,eA=="]}],
            "end_image_url": "data:image/png;base64,eA==",
            "image_url": "data:image/png;base64,eA==",
            "reference_audio_urls": ["data:audio/wav;base64,eA=="],
            "reference_document_urls": ["data:application/pdf;base64,eA=="],
            "reference_image_urls": ["data:image/png;base64,eA=="],
            "reference_video_urls": ["data:video/mp4;base64,eA=="],
            "scene_image_urls": ["data:image/png;base64,eA=="],
            "video_url": "data:video/mp4;base64,eA==",
        }

        with mock.patch("venice.commands.bg_remove.encode_base64", return_value="base64"):
            bg_fields = set(bg_remove._build_body(
                _namespace(input=Path("image.png"), image_url=None)
            ))
        bg_fields.update(bg_remove._build_body(
            _namespace(input=None, image_url="https://example.test/image.png")
        ))

        actual = {
            "POST /audio/complete": set(_audio.job_body("m", "q")),
            "POST /audio/queue": set(sfx.queue_body("m", "p", 5))
                | set(music.queue_body(music_args))
                | set(music.queue_body(music_lyrics_args)),
            "POST /audio/quote": set(sfx.quote_body("m", 5))
                | set(music.quote_body("m", 60)),
            "POST /audio/retrieve": set(_audio.job_body("m", "q")),
            "POST /audio/speech": set(tts.request_body("hello", "m", "wav", "v", 1.0)),
            "POST /chat/completions": chat_fields,
            "POST /embeddings": set(embed._build_kwargs(
                _namespace(dimensions=256, encoding_format="float"), "m", ["x"]
            )),
            "POST /image/background-remove": bg_fields,
            "POST /image/edit": set(image_edit._build_body(
                _namespace(**edit_common), "base", []
            )[1]),
            "POST /image/generate": set(image._build_body("prompt", image_args)),
            "POST /image/multi-edit": set(image_edit._build_body(
                _namespace(**edit_common), "base", ["layer"]
            )[1]),
            "POST /image/upscale": set(upscale._build_body(
                _namespace(scale=4.0, creativity=0.5), "base64"
            )),
            "POST /video/complete": set(video.job_body("m", "q")),
            "POST /video/queue": set(video.queue_body(
                "m", "prompt", shared_video, video_media, "none"
            )),
            "POST /video/quote": set(video.quote_body(
                "m", shared_video,
                {"video_url": "data:video/mp4;base64,eA==", "reference_video_total_duration": 5},
            )),
            "POST /video/retrieve": set(video.job_body("m", "q")),
        }
        for operation, fields in actual.items():
            self.assertEqual(fields, _implemented_fields(self.manifest, operation), operation)


class DriftClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    def _changed(self):
        changed = copy.deepcopy(self.baseline)
        changed["source"]["sha256"] = "f" * 64
        return changed

    def test_unsupported_addition_is_informational(self):
        changed = self._changed()
        changed["all_operations"].append(
            {"key": "GET /new-unwrapped-endpoint", "operation_id": "newEndpoint"}
        )
        report = contract.compare_locks(self.baseline, changed)
        self.assertEqual(report["implemented_changes"], [])
        self.assertEqual(report["unsupported_added"], ["GET /new-unwrapped-endpoint"])

    def test_added_required_field_is_breaking(self):
        changed = self._changed()
        schema = changed["implemented_operations"]["POST /audio/speech"][
            "request_bodies"
        ]["application/json"]
        schema["required"].append("voice")
        report = contract.compare_locks(self.baseline, changed)
        self.assertTrue(any(level == "breaking" for level, *_ in report["implemented_changes"]))

    def test_optional_field_addition_requires_review(self):
        changed = self._changed()
        properties = changed["implemented_operations"]["POST /audio/speech"][
            "request_bodies"
        ]["application/json"]["properties"]
        properties["new_optional"] = {"type": "string"}
        report = contract.compare_locks(self.baseline, changed)
        self.assertTrue(any(level == "review" for level, *_ in report["implemented_changes"]))

    def test_enum_narrowing_is_breaking(self):
        changed = self._changed()
        schema = changed["implemented_operations"]["GET /models"]["parameters"][0]["schema"]
        branch = next(item for item in schema["anyOf"] if "asr" in item.get("enum", []))
        branch["enum"].remove("asr")
        report = contract.compare_locks(self.baseline, changed)
        self.assertTrue(any(level == "breaking" for level, *_ in report["implemented_changes"]))

    def test_unresolved_reference_fails_closed(self):
        with self.assertRaises(contract.ContractError):
            contract.normalize_schema({"$ref": "#/components/schemas/Missing"}, {})


if __name__ == "__main__":
    unittest.main()
