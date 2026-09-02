"""Unit tests for the shared `_models` / `_openai` command helpers.

These were extracted from `chat`/`embed`/`video` (Gitea #19), which exercise
them only end-to-end. This covers the pure logic directly: catalog parsing,
default-trait selection, model validation, and the SDK exception -> exit-code
ladder. No network, no real key, no openai package required.
"""
import ast
import io
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

from venice.client import VeniceAPIError
from venice.commands import _models, _openai


ROOT = Path(__file__).resolve().parents[1]


def _model(mid, traits=None):
    spec = {"traits": traits} if traits is not None else {}
    return {"id": mid, "model_spec": spec}


class _FakeClient:
    """Stands in for the lean urllib client's get_json."""

    def __init__(self, doc=None, raises=None):
        self.doc = doc
        self.raises = raises
        self.calls = []

    def get_json(self, path, params=None):
        self.calls.append((path, params))
        if self.raises is not None:
            raise self.raises
        return self.doc


class TestCatalog(unittest.TestCase):

    def test_hits_models_endpoint_with_the_type_filter(self):
        c = _FakeClient(doc={"data": [_model("m1")]})
        out = _models.catalog(c, "embedding")
        self.assertEqual(c.calls, [("/models", {"type": "embedding"})])
        self.assertEqual(out, [_model("m1")])

    def test_api_error_is_swallowed_to_none(self):
        c = _FakeClient(raises=VeniceAPIError(500, "https://api.example/models", "boom"))
        self.assertIsNone(_models.catalog(c, "text"))

    def test_non_list_data_is_none(self):
        self.assertIsNone(_models.catalog(_FakeClient(doc={"data": "nope"}), "text"))
        self.assertIsNone(_models.catalog(_FakeClient(doc=["not", "a", "dict"]), "text"))


class TestDefaultModel(unittest.TestCase):

    def test_picks_the_first_default_trait_model(self):
        models = [_model("a"), _model("b", ["default"]), _model("c", ["default"])]
        self.assertEqual(_models.default_model(models), "b")

    def test_none_when_no_model_advertises_default(self):
        self.assertIsNone(_models.default_model([_model("a"), _model("b", ["fast"])]))

    def test_tolerates_malformed_entries(self):
        models = ["junk", {"id": "a"}, {"id": "b", "model_spec": None}, _model("c", ["default"])]
        self.assertEqual(_models.default_model(models), "c")


class TestResolveModel(unittest.TestCase):

    def _resolve(self, requested, models, **kw):
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            mid, rc = _models.resolve_model(
                requested, models, label="chat", noun="text model", **kw
            )
        return mid, rc, err.getvalue()

    def test_no_catalog_with_explicit_model_passes_through(self):
        self.assertEqual(self._resolve("some-model", None)[:2], ("some-model", None))

    def test_no_catalog_without_model_exits_2(self):
        mid, rc, err = self._resolve(None, None)
        self.assertEqual((mid, rc), (None, 2))
        self.assertIn("could not fetch the model catalog", err)
        self.assertTrue(err.startswith("chat: "))

    def test_known_model_validates(self):
        models = [_model("a"), _model("b")]
        self.assertEqual(self._resolve("b", models)[:2], ("b", None))

    def test_unknown_model_exits_6_and_lists_available(self):
        mid, rc, err = self._resolve("nope", [_model("a"), _model("b")])
        self.assertEqual((mid, rc), (None, 6))
        self.assertIn("chat: unknown text model 'nope'", err)
        self.assertIn("available: a, b", err)

    def test_default_is_used_when_no_model_requested(self):
        models = [_model("a"), _model("b", ["default"])]
        self.assertEqual(self._resolve(None, models)[:2], ("b", None))

    def test_no_default_advertised_exits_6(self):
        mid, rc, err = self._resolve(None, [_model("a"), _model("b")])
        self.assertEqual((mid, rc), (None, 6))
        self.assertIn("chat: no default text model advertised", err)
        self.assertIn("available: a, b", err)

    # --- #27: the opt-in `defaults.<cmd>.model` hint ---
    #
    # Two separate hint tests on purpose: they pin the two _print_config_hint
    # call sites independently, so deleting either one goes red on its own.

    def test_config_hint_on_no_default(self):
        mid, rc, err = self._resolve(
            None, [_model("a")], config_key="defaults.embed.model"
        )
        self.assertEqual((mid, rc), (None, 6))
        self.assertIn("venice config set defaults.embed.model <id>", err)

    def test_config_hint_on_missing_catalog(self):
        mid, rc, err = self._resolve(None, None, config_key="defaults.embed.model")
        self.assertEqual((mid, rc), (None, 2))
        self.assertIn("venice config set defaults.embed.model <id>", err)

    def test_no_config_hint_when_key_absent(self):
        """The anti-vacuity guard: without it, dropping `if config_key:` and
        printing the bare None would still pass both tests above."""
        self.assertNotIn("venice config set", self._resolve(None, None)[2])
        self.assertNotIn(
            "venice config set", self._resolve(None, [_model("a")])[2]
        )

    def test_unknown_model_gets_no_config_hint(self):
        """Pins the deliberate omission (see resolve_model's docstring): the
        unknown-model message already lists every legal id, and the hint would
        be wrong for a mistyped --model."""
        mid, rc, err = self._resolve(
            "nope", [_model("a")], config_key="defaults.embed.model"
        )
        self.assertEqual((mid, rc), (None, 6))
        self.assertIn("unknown text model 'nope'", err)
        self.assertNotIn("venice config set", err)

    def test_label_and_noun_reach_the_messages(self):
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            _models.resolve_model(
                "nope", [_model("a")], label="video", noun="video model"
            )
        self.assertIn("video: unknown video model 'nope'", err.getvalue())


class TestSupportsCapability(unittest.TestCase):
    """Tri-state capability reader over model_spec.capabilities (#60)."""

    def _caps_model(self, mid, caps):
        return {"id": mid, "model_spec": {"capabilities": caps}}

    def test_true_and_false_read_through(self):
        models = [
            self._caps_model("v", {"supportsVision": True}),
            self._caps_model("t", {"supportsVision": False}),
        ]
        self.assertIs(_models.supports_capability(models, "v", "supportsVision"), True)
        self.assertIs(_models.supports_capability(models, "t", "supportsVision"), False)

    def test_key_matching_ignores_case_and_underscores(self):
        models = [self._caps_model("v", {"supportsVision": True})]
        self.assertIs(
            _models.supports_capability(models, "v", "supports_vision"), True)

    def test_unknown_is_none(self):
        no_caps = [{"id": "m", "model_spec": {"traits": []}}]
        missing_field = [self._caps_model("m", {"supportsWebSearch": True})]
        self.assertIsNone(_models.supports_capability(None, "m", "supportsVision"))
        self.assertIsNone(_models.supports_capability([], "m", "supportsVision"))
        self.assertIsNone(_models.supports_capability(no_caps, "m", "supportsVision"))
        self.assertIsNone(
            _models.supports_capability(missing_field, "m", "supportsVision"))
        self.assertIsNone(
            _models.supports_capability(no_caps, "absent", "supportsVision"))


# --- _openai ---


class TestPromptCacheAffinity(unittest.TestCase):
    def test_mints_an_opaque_key_without_mutating_the_input(self):
        original = {"temperature": 0.2}
        out = _openai.with_prompt_cache_key(original)
        key = _openai.prompt_cache_key(out)
        self.assertIsInstance(key, str)
        self.assertTrue(key.startswith("venice-"))
        self.assertNotIn("extra_body", original)

    def test_explicit_key_preserves_venice_parameters(self):
        original = {"extra_body": {"venice_parameters": {"enable_web_search": True}}}
        out = _openai.with_prompt_cache_key(original, "session-key")
        self.assertEqual(_openai.prompt_cache_key(out), "session-key")
        self.assertEqual(out["extra_body"]["venice_parameters"],
                         {"enable_web_search": True})
        self.assertNotIn("prompt_cache_key", original["extra_body"])

    def test_strip_keeps_other_extra_body_fields_and_does_not_mutate(self):
        original = {"extra_body": {
            "prompt_cache_key": "parent",
            "venice_parameters": {"character_slug": "x"},
        }}
        out = _openai.without_prompt_cache_key(original)
        self.assertIsNone(_openai.prompt_cache_key(out))
        self.assertEqual(out["extra_body"],
                         {"venice_parameters": {"character_slug": "x"}})
        self.assertEqual(_openai.prompt_cache_key(original), "parent")

    def test_strip_removes_empty_extra_body(self):
        out = _openai.without_prompt_cache_key(
            {"temperature": 0.1, "extra_body": {"prompt_cache_key": "x"}})
        self.assertEqual(out, {"temperature": 0.1})


class _StubConnErr(Exception):
    pass


class _StubOpenAI:
    """Minimal stand-in for the openai module's surface used by status_to_exit."""

    APIConnectionError = _StubConnErr

    def __init__(self):
        self.built = None
        self.http_client_built = None

    def DefaultHttpxClient(self, **kwargs):
        self.http_client_built = kwargs
        return "http-client-sentinel"

    def OpenAI(self, **kwargs):
        self.built = kwargs
        return "sdk-client"


class _Status(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class TestImportOpenAI(unittest.TestCase):

    def test_readme_inventory_matches_lazy_import_call_sites(self):
        """Every SDK boundary stays visible in the authoritative docs (#121)."""
        labels = set()
        for path in (ROOT / "src" / "venice").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_openai"
                ):
                    continue
                self.assertTrue(node.args, f"{path}: import_openai needs a label")
                label = node.args[0]
                self.assertIsInstance(
                    label, ast.Constant,
                    f"{path}:{node.lineno}: import_openai label must be literal",
                )
                self.assertIsInstance(
                    label.value, str,
                    f"{path}:{node.lineno}: import_openai label must be a string",
                )
                labels.add(label.value)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        start_marker = "<!-- openai-extra-inventory:start -->"
        end_marker = "<!-- openai-extra-inventory:end -->"
        self.assertEqual(readme.count(start_marker), 1)
        self.assertEqual(readme.count(end_marker), 1)
        inventory = readme.split(start_marker, 1)[1].split(end_marker, 1)[0]
        documented = set(re.findall(r"^- `([^`]+)`:", inventory, re.MULTILINE))
        self.assertEqual(documented, labels)

        # These wrappers delegate to one of the labelled boundaries rather
        # than importing the SDK themselves, so pin their documentation too.
        for feature in (
            "`venice_chat`", "`venice chat --mcp`", "`project_search`", "`reindex`"
        ):
            with self.subTest(feature=feature):
                self.assertIn(feature, inventory)

    def test_returns_the_module_when_present(self):
        self.assertIsNotNone(_openai.import_openai("chat"))

    def test_missing_package_prints_hint_and_returns_none(self):
        err = io.StringIO()
        with mock.patch.dict(sys.modules, {"openai": None}), \
             mock.patch.object(sys, "stderr", err):
            self.assertIsNone(_openai.import_openai("embed"))
        msg = err.getvalue()
        self.assertIn("openai", msg)
        self.assertIn("venice embed", msg)
        self.assertIn("pip install", msg)


class TestBuildOpenAI(unittest.TestCase):

    def test_borrows_key_and_base_url_from_the_lean_client(self):
        stub = _StubOpenAI()
        client = mock.Mock(api_key="k", base_url="https://api.example/v1")
        self.assertEqual(_openai.build_openai(stub, client), "sdk-client")
        self.assertEqual(
            stub.built, {"api_key": "k", "base_url": "https://api.example/v1"}
        )

    def test_base_url_override_uses_given_key(self):
        stub = _StubOpenAI()
        # No lean client needed for an alternate OpenAI-compatible backend.
        _openai.build_openai(stub, base_url="http://localhost:1234/v1", api_key="lk")
        self.assertEqual(
            stub.built, {"api_key": "lk", "base_url": "http://localhost:1234/v1"}
        )

    def test_base_url_override_without_key_uses_placeholder(self):
        stub = _StubOpenAI()
        _openai.build_openai(stub, base_url="http://localhost:1234/v1")
        self.assertEqual(
            stub.built, {"api_key": "not-needed", "base_url": "http://localhost:1234/v1"}
        )

    def test_verify_ca_bundle_builds_sdk_http_client(self):
        stub = _StubOpenAI()
        _openai.build_openai(
            stub, base_url="https://embed.local/v1", verify="/ca.pem"
        )
        self.assertEqual(stub.http_client_built, {"verify": "/ca.pem"})
        self.assertEqual(stub.built["http_client"], "http-client-sentinel")
        self.assertEqual(stub.built["base_url"], "https://embed.local/v1")

    def test_verify_false_disables_verification(self):
        stub = _StubOpenAI()
        _openai.build_openai(
            stub, base_url="https://embed.local/v1", verify=False
        )
        self.assertEqual(stub.http_client_built, {"verify": False})
        self.assertEqual(stub.built["http_client"], "http-client-sentinel")

    def test_verify_none_adds_no_http_client(self):
        stub = _StubOpenAI()
        _openai.build_openai(stub, base_url="http://localhost:1234/v1")
        self.assertIsNone(stub.http_client_built)
        self.assertNotIn("http_client", stub.built)


class TestStatusToExit(unittest.TestCase):

    def _exit(self, exc):
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            rc = _openai.status_to_exit(_StubOpenAI(), exc, "chat")
        return rc, err.getvalue()

    def test_connection_error_is_8(self):
        rc, err = self._exit(_StubConnErr("down"))
        self.assertEqual(rc, 8)
        self.assertIn("chat: connection error", err)

    def test_status_ladder(self):
        for status, expected in ((401, 2), (404, 6), (429, 4), (500, 5), (503, 5), (400, 2), (422, 2)):
            with self.subTest(status=status):
                rc, err = self._exit(_Status(status))
                self.assertEqual(rc, expected)
                self.assertIn("chat: API error", err)

    def test_unknown_status_defaults_to_5(self):
        self.assertEqual(self._exit(_Status(None))[0], 5)
        self.assertEqual(self._exit(_Status("weird"))[0], 5)

    def test_label_reaches_the_message(self):
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            _openai.status_to_exit(_StubOpenAI(), _Status(401), "embed")
        self.assertIn("embed: API error", err.getvalue())


if __name__ == "__main__":
    unittest.main()
