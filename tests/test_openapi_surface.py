"""Full API-surface and MCP-surface inventory checks (r1 audit).

Test author of record: Claude (production change 1a844f2 is Codex's).

Three layers, from cheapest to most faithful:

1. The committed snapshots (surface.json/.md, mcp-surface.json) are mutually
   consistent with the lock and manifest -- stdlib only.
2. ``scripts/openapi_surface.py`` really builds and renders that shape, driven
   by a synthetic spec (JSON is valid YAML, so specs are built as dicts) --
   needs PyYAML, like the scripts themselves.
3. The MCP tool *registrations* in the server source match the committed
   mcp-surface.json profiles: statically (AST, no SDK -- so a dropped or extra
   tool fails even on Python 3.9 / without extras), and, with the SDK installed,
   by regenerating the inventory through real MCP sessions and comparing it
   byte-for-byte with the committed file.

Nothing here touches the network; ``fetch_spec`` is never called.
"""
import ast
import asyncio
import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONTRACTS = ROOT / "contracts" / "openapi"
UNSUPPORTED = "No dedicated client command or MCP tool; generic HTTP transport only."
PROFILES = ("http_media", "http_minimal", "stdio")

_HAS_YAML = importlib.util.find_spec("yaml") is not None
_HAS_MCP = importlib.util.find_spec("mcp") is not None


def _read(name):
    return json.loads((CONTRACTS / name).read_text(encoding="utf-8"))


def _load_script(name):
    """Import scripts/<name>.py by path with scripts/ importable (as its CLI does)."""
    spec = importlib.util.spec_from_file_location(f"r1_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
        spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# 1. Committed snapshots
# --------------------------------------------------------------------------- #

class CommittedSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.surface = _read("surface.json")
        cls.lock = _read("venice.lock.json")
        cls.manifest = _read("implementation.json")

    def test_header_matches_lock(self):
        self.assertEqual(self.surface["format"], 1)
        self.assertEqual(self.surface["source"], self.lock["source"])
        self.assertEqual(set(self.surface), {"format", "source", "operations"})

    def test_every_upstream_operation_is_inventoried_exactly_once(self):
        upstream = [item["key"] for item in self.lock["all_operations"]]
        self.assertEqual(len(upstream), len(set(upstream)))
        self.assertEqual(set(self.surface["operations"]), set(upstream))
        # Snapshot pin: a refresh that changes the operation count must update
        # this deliberately, not slip through as "still consistent".
        self.assertEqual(len(self.surface["operations"]), 55)

    def test_every_implemented_operation_is_in_the_surface(self):
        self.assertLessEqual(set(self.manifest["operations"]), set(self.surface["operations"]))
        self.assertEqual(set(self.manifest["operations"]), set(self.lock["implemented_operations"]))

    def test_operation_ids_match_upstream(self):
        ids = {item["key"]: item["operation_id"] for item in self.lock["all_operations"]}
        for key, operation in self.surface["operations"].items():
            with self.subTest(operation=key):
                self.assertEqual(operation["operation_id"], ids[key])

    def test_coverage_and_request_shapes(self):
        implemented = self.lock["implemented_operations"]
        unsupported = 0
        for key, operation in self.surface["operations"].items():
            with self.subTest(operation=key):
                self.assertIsInstance(operation["request_body_required"], bool)
                if key in self.manifest["operations"]:
                    self.assertEqual(operation["coverage"], self.manifest["operations"][key])
                    self.assertEqual(operation["request_bodies"], implemented[key]["request_bodies"])
                    self.assertEqual(operation["parameters"], implemented[key]["parameters"])
                    self.assertNotIn("unsupported", operation["coverage"])
                else:
                    unsupported += 1
                    self.assertEqual(operation["coverage"], {"unsupported": UNSUPPORTED})
                    if operation["request_bodies"]:
                        self.assertTrue(all(isinstance(s, dict)
                                            for s in operation["request_bodies"].values()))
        self.assertEqual(unsupported, len(self.surface["operations"]) - len(implemented))
        self.assertGreater(unsupported, 0)

    def test_every_operation_declares_responses_with_content_and_headers(self):
        for key, operation in self.surface["operations"].items():
            with self.subTest(operation=key):
                self.assertTrue(operation["responses"])
                for status, response in operation["responses"].items():
                    self.assertIsInstance(status, str)
                    self.assertEqual(set(response), {"content", "headers"})
                    self.assertIsInstance(response["content"], dict)
                    self.assertIsInstance(response["headers"], dict)

    @unittest.skipUnless(_HAS_YAML, "PyYAML missing: pip install -r scripts/openapi-requirements.txt")
    def test_committed_markdown_is_the_rendering_of_committed_json(self):
        module = _load_script("openapi_surface")
        # surface.json is written with sorted keys; the Markdown keeps the
        # upstream inventory order build_surface produced (the lock's order).
        ordered = dict(self.surface, operations={
            item["key"]: self.surface["operations"][item["key"]]
            for item in self.lock["all_operations"]
        })
        rendered = module.render_surface(ordered)
        committed = (CONTRACTS / "surface.md").read_text(encoding="utf-8")
        self.assertEqual(committed, rendered)


class CommittedMcpSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mcp = _read("mcp-surface.json")

    def test_profiles(self):
        self.assertEqual(set(self.mcp), set(PROFILES))

    def test_profile_relationships(self):
        stdio, minimal, media = (set(self.mcp[p]) for p in ("stdio", "http_minimal", "http_media"))
        self.assertEqual(minimal, {"venice_chat", "venice_vision"})
        self.assertLess(minimal, stdio)
        self.assertEqual(media - stdio, {
            "venice_media_import", "venice_media_delete",
            "venice_job_status", "venice_job_result",
        })
        self.assertEqual(stdio - media, set())

    def test_entries_are_complete_tool_descriptors(self):
        for profile, tools in self.mcp.items():
            for name, tool in tools.items():
                with self.subTest(profile=profile, tool=name):
                    self.assertEqual(tool["name"], name)
                    self.assertTrue(tool.get("description"))
                    self.assertEqual(tool["input_schema"]["type"], "object")
                    if profile != "stdio":
                        self.assertIn("annotations", tool)


# --------------------------------------------------------------------------- #
# 2. scripts/openapi_surface.py against a synthetic spec
# --------------------------------------------------------------------------- #

def _spec():
    """A small OpenAPI document exercising every branch build_surface has."""
    return {
        "openapi": "3.0.0",
        "info": {"version": "r1-test"},
        "components": {
            "responses": {"Err": {
                "description": "error",
                "content": {"application/json": {"schema": {"$ref": "#/components/schemas/E"}}},
                "headers": {"X-Ref": {"$ref": "#/components/headers/Trace"}},
            }},
            "headers": {"Trace": {"schema": {"type": "string", "maxLength": 64}}},
            "schemas": {"E": {"type": "object", "properties": {"code": {"type": "string"}}}},
            "requestBodies": {"Job": {"required": True, "content": {"application/json": {
                "schema": {"type": "object", "required": ["id"], "properties": {
                    "id": {"type": "string"}, "extra": {"type": "integer"},
                }},
            }}}},
        },
        "paths": {
            "/supported": {
                "parameters": [{"in": "query", "name": "limit", "required": True,
                                "schema": {"type": "integer"}}],
                "get": {"operationId": "getSupported", "responses": {
                    "200": {"description": "ok", "content": {"application/json": {
                        "schema": {"type": "object"}}},
                        "headers": {"X-Count": {"schema": {"type": "integer"}}}},
                    "400": {"$ref": "#/components/responses/Err"},
                }},
                "post": {"operationId": "postSupported",
                         "requestBody": {"$ref": "#/components/requestBodies/Job"},
                         "responses": {"200": {"description": "no content"}}},
            },
            "/unsupported": {"put": {
                "operationId": "putUnsupported",
                "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
                "responses": {"200": {"description": "ok", "content": {"application/json": {
                    "schema": {"type": "object", "properties": {"added": {
                        "type": "array", "items": {"type": "object", "properties": {
                            "nested": {"type": "string", "enum": ["a", "b"]},
                        }},
                    }}},
                }}}},
            }},
            "/bare": {"delete": {}},
        },
    }


def _manifest():
    return {"operations": {
        "GET /supported": {"consumers": ["cli:one", "mcp:two"], "request_bodies": {}},
        "POST /supported": {"consumers": ["cli:post"], "request_bodies": {"application/json": {
            "implemented": ["id"], "intentional_omissions": {"extra": "server-side only"},
        }}},
    }}


@unittest.skipUnless(_HAS_YAML, "PyYAML missing: pip install -r scripts/openapi-requirements.txt")
class BuildSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_script("openapi_surface")

    def build(self, spec=None, manifest=None):
        raw = json.dumps(spec or _spec()).encode()
        return raw, self.module.build_surface(raw, manifest or _manifest())

    def test_inventory_covers_all_operations_and_source(self):
        raw, built = self.build()
        self.assertEqual(set(built), {"format", "source", "operations"})
        self.assertEqual(built["format"], 1)
        self.assertEqual(list(built["operations"]), [
            "DELETE /bare", "GET /supported", "POST /supported", "PUT /unsupported",
        ])
        self.assertEqual(built["source"]["version"], "r1-test")
        self.assertEqual(built["source"]["sha256"], self.module.contract.hashlib.sha256(raw).hexdigest())

    def test_coverage_comes_from_manifest_or_is_marked_unsupported(self):
        _raw, built = self.build()
        ops = built["operations"]
        self.assertEqual(ops["GET /supported"]["coverage"], _manifest()["operations"]["GET /supported"])
        self.assertEqual(ops["PUT /unsupported"]["coverage"], {"unsupported": UNSUPPORTED})
        self.assertEqual(ops["DELETE /bare"]["coverage"], {"unsupported": UNSUPPORTED})

    def test_request_body_required_follows_refs_and_defaults_false(self):
        _raw, built = self.build()
        ops = built["operations"]
        self.assertIs(ops["POST /supported"]["request_body_required"], True)
        self.assertIs(ops["PUT /unsupported"]["request_body_required"], False)
        self.assertIs(ops["GET /supported"]["request_body_required"], False)
        self.assertEqual(ops["POST /supported"]["request_bodies"]["application/json"]["required"],
                         ["id"])

    def test_responses_resolve_refs_headers_and_stringify_status(self):
        _raw, built = self.build()
        get = built["operations"]["GET /supported"]["responses"]
        self.assertEqual(list(get), ["200", "400"])
        self.assertEqual(get["200"]["headers"], {"X-Count": {"type": "integer"}})
        self.assertEqual(get["400"]["content"]["application/json"]["properties"],
                         {"code": {"type": "string"}})
        self.assertEqual(get["400"]["headers"], {"X-Ref": {"type": "string", "maxLength": 64}})
        post = built["operations"]["POST /supported"]["responses"]
        self.assertEqual(post, {"200": {"content": {}, "headers": {}}})
        self.assertEqual(built["operations"]["DELETE /bare"]["responses"], {})

    def test_unquoted_yaml_status_codes_become_string_keys(self):
        # JSON specs can only carry string keys; real Swagger YAML often has bare
        # `200:` which PyYAML loads as int. The snapshot must not mix key types.
        raw = (b"openapi: 3.0.0\ninfo: {version: v}\npaths:\n  /x:\n    get:\n"
               b"      responses:\n        200: {description: ok}\n"
               b"        404: {description: missing}\n")
        built = self.module.build_surface(raw, {"operations": {"GET /x": {"consumers": ["cli:x"]}}})
        responses = built["operations"]["GET /x"]["responses"]
        self.assertEqual(list(responses), ["200", "404"])
        self.assertEqual(json.loads(json.dumps(built, sort_keys=True)), built)

    def test_nested_response_fields_of_unsupported_operations_are_retained(self):
        _raw, built = self.build()
        schema = built["operations"]["PUT /unsupported"]["responses"]["200"]["content"]["application/json"]
        nested = schema["properties"]["added"]["items"]["properties"]["nested"]
        self.assertEqual(nested, {"type": "string", "enum": ["a", "b"]})

    def test_parameters_are_inherited_from_the_path_item(self):
        _raw, built = self.build()
        self.assertEqual(built["operations"]["GET /supported"]["parameters"], [
            {"in": "query", "name": "limit", "required": True, "schema": {"type": "integer"}},
        ])

    def test_manifest_operation_missing_upstream_is_an_error(self):
        manifest = _manifest()
        manifest["operations"]["GET /gone"] = {"consumers": ["cli:x"]}
        with self.assertRaises(self.module.contract.ContractError):
            self.build(manifest=manifest)

    def test_inputs_are_not_mutated(self):
        spec, manifest = _spec(), _manifest()
        before = copy.deepcopy((spec, manifest))
        self.module.build_surface(json.dumps(spec).encode(), manifest)
        self.assertEqual((spec, manifest), before)

    def test_render(self):
        _raw, built = self.build()
        built["operations"]["PUT /unsupported"]["coverage"]["request_bodies"] = {
            "application/json": {"unsupported": "Body not modelled."}}
        text = self.module.render_surface(built)
        self.assertTrue(text.startswith("# Public API surface inventory\n\nSource: "
                                        f"{built['source']['url']} (version `r1-test`).\n"))
        self.assertTrue(text.endswith("Responses: `200`.\n"))
        self.assertFalse(text.endswith("\n\n"))
        sections = text.split("\n## ")[1:]
        self.assertEqual([s.split("\n", 1)[0] for s in sections], [
            "`DELETE /bare`", "`GET /supported`", "`POST /supported`", "`PUT /unsupported`",
        ])
        bare, get, post, put = sections
        self.assertIn(UNSUPPORTED, bare)
        self.assertIn("Responses: .", bare)
        self.assertNotIn("Parameters:", bare)
        self.assertIn("Consumers: `cli:one`, `mcp:two`.", get)
        self.assertIn("Parameters: `query:limit` (required).", get)
        self.assertIn("Responses: `200`, `400`.", get)
        self.assertIn("Request `application/json`:\n\n- `extra`: server-side only\n- `id`: implemented\n",
                      post)
        self.assertIn("Body not modelled.", put)
        self.assertIn("No named schema properties published.", put)

    def test_render_marks_undeclared_fields_unsupported_and_optional_params(self):
        spec = _spec()
        spec["paths"]["/supported"]["parameters"][0]["required"] = False
        manifest = _manifest()
        manifest["operations"]["POST /supported"]["request_bodies"]["application/json"] = {
            "implemented": [], "intentional_omissions": {}}
        _raw, built = self.build(spec, manifest)
        text = self.module.render_surface(built)
        self.assertIn("- `extra`: unsupported\n- `id`: unsupported\n", text)
        self.assertIn("Parameters: `query:limit`.", text)

    def test_main_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec_file = Path(tmp) / "spec.yaml"
            spec_file.write_text(json.dumps(_spec()))
            manifest_file = Path(tmp) / "manifest.json"
            manifest_file.write_text(json.dumps(_manifest()))
            output = Path(tmp) / "out.json"
            stdout = io.StringIO()
            with mock.patch.object(self.module.contract, "DEFAULT_MANIFEST", manifest_file), \
                 mock.patch.object(self.module.contract, "fetch_spec",
                                   side_effect=AssertionError("network")), \
                 mock.patch.object(sys, "argv", ["openapi_surface.py", "--spec-file",
                                                 str(spec_file), "--output", str(output)]), \
                 contextlib.redirect_stdout(stdout):
                self.assertEqual(self.module.main(), 0)
            written = json.loads(output.read_text())
            self.assertEqual(written, self.module.build_surface(spec_file.read_bytes(), _manifest()))
            self.assertEqual(output.with_suffix(".md").read_text(),
                             self.module.render_surface(written))
            self.assertEqual(stdout.getvalue(), "openapi-surface: 4 operations recorded\n")


# --------------------------------------------------------------------------- #
# 3. MCP tool registrations vs the committed MCP surface
# --------------------------------------------------------------------------- #

def _source(module):
    """Source of venice.<module>, wherever PYTHONPATH resolves the package."""
    spec = importlib.util.find_spec("venice")
    path = Path(spec.origin).parent / f"{module}.py"
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _tool_name(decorator, function):
    """Name of an @server.tool(...) registration, or None for other decorators."""
    call = decorator if isinstance(decorator, ast.Call) else None
    target = call.func if call else decorator
    if not (isinstance(target, ast.Attribute) and target.attr == "tool"
            and isinstance(target.value, ast.Name) and target.value.id == "server"):
        return None
    for keyword in (call.keywords if call else []):
        if keyword.arg == "name":
            return keyword.value.value
    return function.name


def _media_store_branch(test):
    """'media'/'none' for `media_store is (not) None`, else None."""
    if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == "media_store" and len(test.ops) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None):
        return "media" if isinstance(test.ops[0], ast.IsNot) else "none"
    return None


def _registrations(body, context="always"):
    """[(tool name, context)] for every registration reachable in ``body``."""
    found = []
    for node in body:
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                name = _tool_name(decorator, node)
                if name is not None:
                    found.append((name, context))
            found.extend(_registrations(node.body, context))
        elif isinstance(node, ast.If):
            branch = _media_store_branch(node.test)
            if branch is None:
                found.extend(_registrations(node.body, context))
                found.extend(_registrations(node.orelse, context))
            else:
                other = "none" if branch == "media" else "media"
                found.extend(_registrations(node.body, branch))
                found.extend(_registrations(node.orelse, other))
        elif isinstance(node, (ast.With, ast.Try)):
            found.extend(_registrations(node.body, context))
    return found


class StaticToolRegistrations(unittest.TestCase):
    """SDK-free drift gate: source registrations == committed profiles."""

    @classmethod
    def setUpClass(cls):
        cls.mcp = _read("mcp-surface.json")
        stdio = _function(_source("_mcp_stdio"), "build_server")
        cls.stdio = _registrations(stdio.body)
        http = _function(_source("mcp_server"), "build_http_server")
        cls.http = _registrations(http.body)

    def profile(self, registrations, store):
        names = [n for n, ctx in registrations if ctx in ("always", store)]
        self.assertEqual(len(names), len(set(names)), f"duplicate registration: {names}")
        return set(names)

    def test_stdio_registrations_match_committed_profile(self):
        self.assertEqual(self.profile(self.stdio, "always"), set(self.mcp["stdio"]))
        self.assertEqual({ctx for _n, ctx in self.stdio}, {"always"})

    def test_http_minimal_registrations_match_committed_profile(self):
        self.assertEqual(self.profile(self.http, "none"), set(self.mcp["http_minimal"]))

    def test_http_media_registrations_match_committed_profile(self):
        self.assertEqual(self.profile(self.http, "media"), set(self.mcp["http_media"]))

    def test_extractor_sees_both_vision_variants(self):
        # Guards the extractor: venice_vision is registered once per branch.
        self.assertEqual(sorted(ctx for n, ctx in self.http if n == "venice_vision"),
                         ["media", "none"])

    def test_extractor_rejects_drift(self):
        tree = ast.parse(
            "def build(server, media_store):\n"
            "    @server.tool(name='a')\n    def f(): pass\n"
            "    if media_store is not None:\n"
            "        @server.tool()\n        def b(): pass\n"
            "    else:\n"
            "        @other.tool()\n        def c(): pass\n"
        )
        found = _registrations(tree.body[0].body)
        self.assertEqual(found, [("a", "always"), ("b", "media")])


@unittest.skipUnless(_HAS_MCP, "mcp SDK not installed (expected on Python 3.9)")
class LiveMcpInventory(unittest.TestCase):
    """Regenerate the inventory through real MCP sessions; compare to the file."""

    @classmethod
    def setUpClass(cls):
        cls.module = _load_script("mcp_surface")

    def test_inventory_equals_committed_snapshot(self):
        self.assertEqual(asyncio.run(self.module.inventory()), _read("mcp-surface.json"))

    def test_main_writes_the_committed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "mcp.json"
            with mock.patch.object(sys, "argv", ["mcp_surface.py", str(output)]):
                self.module.main()
            self.assertEqual(output.read_text(encoding="utf-8"),
                             (CONTRACTS / "mcp-surface.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
