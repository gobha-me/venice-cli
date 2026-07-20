"""Unit tests for the shared `_models` / `_openai` command helpers.

These were extracted from `chat`/`embed`/`video` (Gitea #19), which exercise
them only end-to-end. This covers the pure logic directly: catalog parsing,
default-trait selection, model validation, and the SDK exception -> exit-code
ladder. No network, no real key, no openai package required.
"""
import io
import sys
import unittest
from unittest import mock

from venice.client import VeniceAPIError
from venice.commands import _models, _openai


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

    def _resolve(self, requested, models):
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            mid, rc = _models.resolve_model(
                requested, models, label="chat", noun="text model"
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

    def test_label_and_noun_reach_the_messages(self):
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            _models.resolve_model(
                "nope", [_model("a")], label="video", noun="video model"
            )
        self.assertIn("video: unknown video model 'nope'", err.getvalue())


# --- _openai ---

class _StubConnErr(Exception):
    pass


class _StubOpenAI:
    """Minimal stand-in for the openai module's surface used by status_to_exit."""

    APIConnectionError = _StubConnErr

    def __init__(self):
        self.built = None

    def OpenAI(self, **kwargs):
        self.built = kwargs
        return "sdk-client"


class _Status(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class TestImportOpenAI(unittest.TestCase):

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

    def test_verify_ca_bundle_builds_httpx_client(self):
        stub = _StubOpenAI()
        with mock.patch("httpx.Client") as HttpxClient:
            HttpxClient.return_value = "httpx-sentinel"
            _openai.build_openai(
                stub, base_url="https://embed.local/v1", verify="/ca.pem"
            )
        HttpxClient.assert_called_once_with(verify="/ca.pem")
        self.assertEqual(stub.built["http_client"], "httpx-sentinel")
        self.assertEqual(stub.built["base_url"], "https://embed.local/v1")

    def test_verify_false_disables_verification(self):
        stub = _StubOpenAI()
        with mock.patch("httpx.Client") as HttpxClient:
            HttpxClient.return_value = "httpx-sentinel"
            _openai.build_openai(
                stub, base_url="https://embed.local/v1", verify=False
            )
        HttpxClient.assert_called_once_with(verify=False)
        self.assertEqual(stub.built["http_client"], "httpx-sentinel")

    def test_verify_none_adds_no_http_client(self):
        stub = _StubOpenAI()
        # Default path must not touch httpx or pass http_client at all.
        with mock.patch("httpx.Client") as HttpxClient:
            _openai.build_openai(stub, base_url="http://localhost:1234/v1")
        HttpxClient.assert_not_called()
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
