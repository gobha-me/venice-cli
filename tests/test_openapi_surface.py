"""Full API inventory completeness and response-field regression checks."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class SurfaceTests(unittest.TestCase):
    def test_current_snapshot_covers_every_operation_and_curated_classification(self):
        surface = json.loads((ROOT / "contracts/openapi/surface.json").read_text())
        lock = json.loads((ROOT / "contracts/openapi/venice.lock.json").read_text())
        manifest = json.loads((ROOT / "contracts/openapi/implementation.json").read_text())
        self.assertEqual(surface["source"], lock["source"])
        self.assertEqual(set(surface["operations"]), {item["key"] for item in lock["all_operations"]})
        self.assertEqual(len(surface["operations"]), 55)
        for key, operation in surface["operations"].items():
            with self.subTest(operation=key):
                self.assertTrue(operation["responses"])
                if key in manifest["operations"]:
                    self.assertEqual(operation["coverage"], manifest["operations"][key])
                    self.assertEqual(operation["request_bodies"], lock["implemented_operations"][key]["request_bodies"])
                else:
                    self.assertTrue(operation["coverage"]["unsupported"])

    def test_builder_retains_nested_response_fields_for_unsupported_operations(self):
        spec = importlib.util.spec_from_file_location("audit_surface", ROOT / "scripts/openapi_surface.py")
        module = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
            spec.loader.exec_module(module)
        raw = b'''openapi: 3.0.0
info: {version: "test"}
paths:
  /supported:
    get:
      responses:
        "200":
          content:
            application/json:
              schema: {type: object}
  /unsupported:
    get:
      responses:
        "200":
          content:
            application/json:
              schema:
                type: object
                properties:
                  added_field:
                    type: array
                    items:
                      type: object
                      properties:
                        nested: {type: string, enum: [a, b]}
'''
        manifest = {"operations": {"GET /supported": {"consumers": ["cli:test"]}}}
        built = module.build_surface(raw, manifest)
        response = built["operations"]["GET /unsupported"]["responses"]["200"]
        nested = response["content"]["application/json"]["properties"]["added_field"]["items"]["properties"]["nested"]
        self.assertEqual(nested, {"type": "string", "enum": ["a", "b"]})
        self.assertIn("unsupported", built["operations"]["GET /unsupported"]["coverage"])
        self.assertIn("## `GET /unsupported`", module.render_surface(built))
