"""Adversarial regressions for the #247 OpenAPI refresh.

Test author of record: Claude (the production change is Codex's).

Only the HTTP transport is faked -- ``urllib.request.urlopen``, at the module
the unit under test calls it through. Everything above it (argparse, config
layering, request builders, the drift gate's fetch/normalize/compare/render,
the MCP and agent wrappers) runs for real. Keys are ``test-fake-key``; no test
touches the network or the user's config.
"""
import base64
import copy
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlsplit

from tests.test_client import FakeResp
from tests.test_openapi_contract import LOCK_PATH, MANIFEST_PATH, contract
from venice import cli, config
from venice.client import VeniceClient
from venice.commands import _agent, _mcp, _shared, video

FIXTURES = Path(__file__).parent / "fixtures" / "openapi_247"
SPEC_FIXTURE = FIXTURES / "mini-swagger.yaml"
MANIFEST_FIXTURE = FIXTURES / "mini-manifest.json"

_HAS_YAML = importlib.util.find_spec("yaml") is not None
_HAS_MCP = importlib.util.find_spec("mcp") is not None

QUEUE = "/POST /video/queue/request_bodies/application/json/properties"
GENERATE = "/POST /image/generate/request_bodies/application/json/properties"
PNG = b"\x89PNG\r\n\x1a\n#247"
START_FRAME = "data:image/png;base64," + base64.b64encode(PNG).decode()


# --------------------------------------------------------------------------- #
# Offline drift gate (a): the real `scripts/openapi_contract.py` CLI, fed a
# fixture spec through a faked urlopen.
# --------------------------------------------------------------------------- #

class _SpecResponse:
    """What `fetch_spec` reads from `urlopen`: a context manager with a final
    URL, headers, and a bounded `read`."""

    def __init__(self, body, url=None, headers=None):
        self._body = body
        self._url = url or contract.DEFAULT_SOURCE
        self.headers = headers or {}

    def geturl(self):
        return self._url

    def read(self, n=-1):
        return self._body if n is None or n < 0 else self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@unittest.skipUnless(_HAS_YAML, "PyYAML missing: pip install -r scripts/openapi-requirements.txt")
class TestOfflineDriftGate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.lock = self.tmp / "venice.lock.json"
        self.manifest = self.tmp / "implementation.json"
        shutil.copyfile(MANIFEST_FIXTURE, self.manifest)
        self.raw = SPEC_FIXTURE.read_bytes()
        rc, out, err, http = self._cli("refresh", self.raw)
        self.assertEqual(rc, 0, err)
        self.assertIn("wrote", out)
        http.assert_called_once()
        self.baseline_bytes = self.lock.read_bytes()
        self.spec = contract._load_yaml(self.raw)

    def _cli(self, command, served, *extra):
        out, err = io.StringIO(), io.StringIO()
        if isinstance(served, BaseException):
            fake = mock.Mock(side_effect=served)
        else:
            fake = mock.Mock(return_value=served if isinstance(served, _SpecResponse)
                             else _SpecResponse(served))
        with mock.patch.object(contract.urllib.request, "urlopen", fake), \
             redirect_stdout(out), redirect_stderr(err):
            rc = contract.main([
                command, "--lock", str(self.lock),
                "--manifest", str(self.manifest), *extra,
            ])
        for call in fake.call_args_list:
            self.assertEqual(call.args[0].full_url, contract.DEFAULT_SOURCE)
            self.assertEqual(call.args[0].get_method(), "GET")
        return rc, out.getvalue(), err.getvalue(), fake

    def _check_live(self, spec):
        served = spec if isinstance(spec, bytes) else json.dumps(spec).encode()
        rc, out, err, _http = self._cli("check-live", served)
        # check-live is read-only: a red run must never "fix" the baseline.
        self.assertEqual(self.lock.read_bytes(), self.baseline_bytes)
        return rc, out, err

    def _queue_schema(self, spec):
        return spec["components"]["schemas"]["QueueVideoRequest"]

    def test_refreshed_lock_normalizes_refs_allof_and_colon_enums(self):
        lock = json.loads(self.baseline_bytes)
        queue = lock["implemented_operations"]["POST /video/queue"][
            "request_bodies"]["application/json"]
        self.assertEqual(queue["properties"]["aspect_ratio"]["enum"],
                         ["16:9", "2:1", "9:16"])
        self.assertEqual(queue["properties"]["image_strength"],
                         {"maximum": 1, "minimum": 0, "type": "number"})
        generate = lock["implemented_operations"]["POST /image/generate"][
            "request_bodies"]["application/json"]
        self.assertEqual(set(generate["properties"]),
                         {"model", "prompt", "transparent_background"})
        self.assertEqual(generate["required"], ["model", "prompt"])
        self.assertEqual(lock["source"]["version"], "20261023.234127")
        rc, _out, err, http = self._cli("check", b"unused")
        self.assertEqual(rc, 0, err)
        http.assert_not_called()  # offline check never fetches

    def test_identical_upstream_passes(self):
        rc, out, _ = self._check_live(self.raw)
        self.assertEqual(rc, 0, out)
        self.assertIn("No implemented-operation drift detected.", out)
        self.assertNotIn("upstream document changed", out)

    def test_upstream_field_added_fails(self):
        spec = copy.deepcopy(self.spec)
        self._queue_schema(spec)["properties"]["future_optional"] = {"type": "boolean"}
        rc, out, _ = self._check_live(spec)
        self.assertEqual(rc, 1, out)
        self.assertIn(f"- **review** `{QUEUE}/future_optional`", out)

    def test_upstream_field_removed_fails_as_breaking(self):
        spec = copy.deepcopy(self.spec)
        part = spec["paths"]["/image/generate"]["post"]["requestBody"]["content"][
            "application/json"]["schema"]["allOf"][1]
        del part["properties"]["transparent_background"]
        rc, out, _ = self._check_live(spec)
        self.assertEqual(rc, 1, out)
        self.assertIn(f"- **breaking** `{GENERATE}/transparent_background`", out)

    def test_upstream_type_change_fails_as_breaking(self):
        spec = copy.deepcopy(self.spec)
        self._queue_schema(spec)["properties"]["image_strength"]["type"] = "integer"
        rc, out, _ = self._check_live(spec)
        self.assertEqual(rc, 1, out)
        self.assertIn(
            f'- **breaking** `{QUEUE}/image_strength/type`: `"number"` -> `"integer"`',
            out,
        )

    def test_upstream_enum_value_added_fails(self):
        spec = copy.deepcopy(self.spec)
        self._queue_schema(spec)["properties"]["aspect_ratio"]["enum"].append("21:9")
        rc, out, _ = self._check_live(spec)
        self.assertEqual(rc, 1, out)
        self.assertIn(f'- **review** `{QUEUE}/aspect_ratio/enum`: `null` -> `"21:9"`', out)

    def test_upstream_enum_value_removed_fails_as_breaking(self):
        spec = copy.deepcopy(self.spec)
        self._queue_schema(spec)["properties"]["resolution"]["enum"].remove("768p")
        rc, out, _ = self._check_live(spec)
        self.assertEqual(rc, 1, out)
        self.assertIn(f'- **breaking** `{QUEUE}/resolution/enum`: `"768p"` -> `null`', out)

    def test_prompt_limit_moves_in_either_direction_fail(self):
        # #247's root symptom was maxLength 20k -> 32k (a widening, "review")
        # sitting unacknowledged; both directions must turn the gate red.
        for new, severity in ((20000, "breaking"), (40000, "review")):
            with self.subTest(maxLength=new):
                spec = copy.deepcopy(self.spec)
                self._queue_schema(spec)["properties"]["prompt"]["maxLength"] = new
                rc, out, _ = self._check_live(spec)
                self.assertEqual(rc, 1, out)
                self.assertIn(
                    f"- **{severity}** `{QUEUE}/prompt/maxLength`: `32000` -> `{new}`", out
                )

    def test_new_required_field_fails_as_breaking(self):
        spec = copy.deepcopy(self.spec)
        schema = self._queue_schema(spec)
        schema["properties"]["duration"] = {"type": "string"}
        schema["required"].append("duration")
        rc, out, _ = self._check_live(spec)
        self.assertEqual(rc, 1, out)
        self.assertIn(
            "- **breaking** `/POST /video/queue/request_bodies/application/json/required`"
            ': `null` -> `"duration"`',
            out,
        )

    def test_parameter_becoming_required_fails(self):
        spec = copy.deepcopy(self.spec)
        spec["paths"]["/image/generate"]["post"]["parameters"][0]["required"] = True
        rc, out, _ = self._check_live(spec)
        self.assertEqual(rc, 1, out)
        self.assertIn("`/POST /image/generate/parameters/0/required`: `false` -> `true`", out)

    def test_implemented_operation_disappearing_fails_as_breaking(self):
        spec = copy.deepcopy(self.spec)
        del spec["paths"]["/video/queue"]
        rc, out, _ = self._check_live(spec)
        self.assertEqual(rc, 1, out)
        self.assertIn("- **breaking** `/POST /video/queue`", out)

    def test_version_bump_alone_passes_but_is_reported(self):
        spec = copy.deepcopy(self.spec)
        spec["info"]["version"] = "20261102.000000"
        rc, out, _ = self._check_live(spec)
        self.assertEqual(rc, 0, out)
        self.assertIn("Live: version `20261102.000000`", out)
        self.assertIn("upstream document changed", out)

    def test_new_unimplemented_operation_is_informational_not_hidden(self):
        spec = copy.deepcopy(self.spec)
        spec["paths"]["/video/frames"] = {"post": {"operationId": "frames"}}
        rc, out, _ = self._check_live(spec)
        self.assertEqual(rc, 0, out)
        self.assertIn("## New unsupported operations (informational)", out)
        self.assertIn("- `POST /video/frames`", out)

    def test_unfetchable_or_bogus_upstream_never_reads_as_green(self):
        cases = {
            "network": urllib.error.URLError("offline"),
            "empty": _SpecResponse(b""),
            "redirect": _SpecResponse(self.raw, url="https://evil.invalid/swagger.yaml"),
            "oversized": _SpecResponse(self.raw, headers={
                "Content-Length": str(contract.MAX_SPEC_BYTES + 1)}),
            "no-version": json.dumps({"paths": {}}).encode(),
            "not-yaml": b"paths: [unterminated",
        }
        for name, served in cases.items():
            with self.subTest(case=name):
                rc, out, err, _ = self._cli("check-live", served)
                self.assertEqual(rc, 2, out + err)
                self.assertIn("openapi-contract:", err)
                self.assertNotIn("No implemented-operation drift detected.", out)
                self.assertEqual(self.lock.read_bytes(), self.baseline_bytes)

    def test_unapproved_source_is_refused_before_any_request(self):
        rc, _out, err, http = self._cli(
            "check-live", self.raw, "--source", "https://mirror.invalid/swagger.yaml"
        )
        self.assertEqual(rc, 2)
        self.assertIn("refusing unapproved OpenAPI source", err)
        http.assert_not_called()

    def test_manifest_that_does_not_classify_a_locked_field_fails_offline_check(self):
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["operations"]["POST /video/queue"]["request_bodies"][
            "application/json"]["implemented"].remove("image_strength")
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        for command in ("check", "check-live"):
            with self.subTest(command=command):
                rc, _out, err, _ = self._cli(command, self.raw)
                self.assertEqual(rc, 1)
                self.assertIn("unclassified upstream fields: ['image_strength']", err)


# --------------------------------------------------------------------------- #
# The committed contract (a, continued): the refreshed lock really carries the
# #247 deltas, and the code's limits are pinned to it.
# --------------------------------------------------------------------------- #

class TestCommittedLockCarriesIssue247(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.ops = cls.lock["implemented_operations"]

    def _props(self, op, media="application/json"):
        return self.ops[op]["request_bodies"][media]["properties"]

    def _implemented(self, op, media="application/json"):
        return set(self.manifest["operations"][op]["request_bodies"][media]["implemented"])

    def test_transparent_background_is_locked_and_implemented_on_every_image_route(self):
        for op in ("POST /image/generate", "POST /image/edit", "POST /image/multi-edit"):
            with self.subTest(op=op):
                self.assertEqual(self._props(op)["transparent_background"],
                                 {"type": "boolean"})
                self.assertIn("transparent_background", self._implemented(op))

    def test_image_strength_is_queue_only_and_bounds_match_code(self):
        locked = self._props("POST /video/queue")["image_strength"]
        self.assertEqual(locked, {"maximum": 1, "minimum": 0, "type": "number"})
        self.assertIn("image_strength", self._implemented("POST /video/queue"))
        self.assertNotIn("image_strength", self._props("POST /video/quote"))
        agent = _agent._VIDEO_SCHEMA["properties"]["image_strength"]
        self.assertEqual(agent["type"], locked["type"])
        self.assertEqual(agent["minimum"], locked["minimum"])
        self.assertEqual(agent["maximum"], locked["maximum"])

    def test_video_prompt_limits_match_code(self):
        props = self._props("POST /video/queue")
        self.assertEqual(props["prompt"]["maxLength"], 32000)
        self.assertEqual(video.MAX_PROMPT, props["prompt"]["maxLength"])
        self.assertEqual(video.MAX_PROMPT, props["negative_prompt"]["maxLength"])
        agent = _agent._VIDEO_SCHEMA["properties"]
        self.assertEqual(agent["prompt"]["maxLength"], props["prompt"]["maxLength"])
        self.assertEqual(agent["negative_prompt"]["maxLength"],
                         props["negative_prompt"]["maxLength"])

    def test_issue_enum_additions_are_in_the_lock(self):
        props = self._props("POST /video/queue")
        self.assertIn("2:1", props["aspect_ratio"]["enum"])
        self.assertIn("768p", props["resolution"]["enum"])

    def test_pre_247_baseline_would_be_red_against_the_committed_lock(self):
        stale = copy.deepcopy(self.lock)
        stale["source"]["version"] = "20260918.184256"
        for op in ("POST /image/generate", "POST /image/edit", "POST /image/multi-edit"):
            for body in stale["implemented_operations"][op]["request_bodies"].values():
                del body["properties"]["transparent_background"]
        queue = stale["implemented_operations"]["POST /video/queue"][
            "request_bodies"]["application/json"]["properties"]
        del queue["image_strength"]
        queue["prompt"]["maxLength"] = 20000
        queue["aspect_ratio"]["enum"].remove("2:1")

        report = contract.compare_locks(stale, self.lock)
        paths = {path for _sev, path, _old, _new in report["implemented_changes"]}
        for expected in (
            f"{GENERATE}/transparent_background",
            "/POST /image/edit/request_bodies/application/json/properties/transparent_background",
            ("/POST /image/multi-edit/request_bodies/multipart/form-data/properties/"
             "transparent_background"),
            f"{QUEUE}/image_strength",
            f"{QUEUE}/prompt/maxLength",
            f"{QUEUE}/aspect_ratio/enum",
        ):
            self.assertIn(expected, paths)
        self.assertTrue(report["source_changed"])
        self.assertIn("Baseline: version `20260918.184256`",
                      contract.render_report(report, stale, self.lock))
        self.assertEqual(contract.compare_locks(self.lock, copy.deepcopy(self.lock))[
            "implemented_changes"], [])


# --------------------------------------------------------------------------- #
# A fake Venice API at the urlopen boundary, shared by the CLI/MCP/agent tests.
# --------------------------------------------------------------------------- #

def _catalog(model_type):
    if model_type == "image":
        data = [{"id": "venice-sd35", "model_spec": {
            "traits": ["default"], "pricing": {"image": {"usd": 0.01}},
        }}]
    elif model_type == "inpaint":
        data = [{"id": "firered-image-edit", "model_spec": {
            "traits": ["default"],
            "constraints": {"combineImages": True, "maxInputImages": 3,
                            "qualities": ["low", "medium", "high"]},
        }}]
    elif model_type == "video":
        data = [{"id": "seedance-test", "model_spec": {
            "traits": ["default"],
            "constraints": {"durations": ["5s", "10s"],
                            "resolutions": ["720p", "1080p"],
                            "aspect_ratios": ["16:9", "9:16"]},
        }}]
    else:
        raise AssertionError(f"unexpected catalog type {model_type!r}")
    return {"data": data}


class FakeVenice:
    """Records every request; answers like the parts of Venice these paths use."""

    def __init__(self):
        self.requests = []

    def __call__(self, request, timeout=None):
        url = urlsplit(request.full_url)
        path = url.path.replace("/api/v1", "", 1)
        method = request.get_method()
        body = json.loads(request.data.decode("utf-8")) if request.data else None
        self.requests.append((method, path, body))
        if method == "GET" and path == "/models":
            return self._json(_catalog(parse_qs(url.query)["type"][0]))
        if method != "POST":
            raise AssertionError(f"unexpected {method} {path}")
        if path == "/image/generate":
            return self._json({"images": [base64.b64encode(PNG).decode()]})
        if path in ("/image/edit", "/image/multi-edit"):
            return FakeResp(200, PNG, "image/png")
        if path == "/video/quote":
            return self._json({"quote": 0.25})
        if path == "/video/queue":
            return self._json({"queue_id": "job-247"})
        if path == "/video/retrieve":
            return FakeResp(200, b"MP4", "video/mp4")
        if path == "/video/complete":
            return self._json({})
        raise AssertionError(f"unexpected POST {path}")

    @staticmethod
    def _json(doc):
        return FakeResp(200, json.dumps(doc).encode(), "application/json")

    def bodies(self, path):
        return [body for method, p, body in self.requests if method == "POST" and p == path]

    def only(self, path):
        bodies = self.bodies(path)
        assert len(bodies) == 1, (path, self.requests)
        return bodies[0]

    def paths(self):
        return [p for _m, p, _b in self.requests]


class _Hermetic(unittest.TestCase):
    """Temp cwd/HOME, fake key, config doc under test control, faked transport."""

    config_doc = None

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, cwd)
        (self.tmp / "base.png").write_bytes(PNG)
        (self.tmp / "mask.png").write_bytes(PNG + b"mask")
        env = mock.patch.dict(os.environ, {
            "VENICE_API_KEY": "test-fake-key",
            "HOME": str(self.tmp),
            "XDG_CONFIG_HOME": str(self.tmp / ".config"),
            config.ENV_VIDEO_JOBS_FILE: str(self.tmp / "jobs.json"),
        })
        env.start()
        self.addCleanup(env.stop)
        self.set_config({})
        self.api = FakeVenice()
        http = mock.patch("venice.client.urllib.request.urlopen", self.api)
        http.start()
        self.addCleanup(http.stop)

    def set_config(self, defaults):
        self.config_doc = {"version": 1, "mcpServers": {}, "defaults": defaults}
        patcher = mock.patch("venice.userconfig.load_config",
                             lambda *a, **k: self.config_doc)
        patcher.start()
        self.addCleanup(patcher.stop)

    def venice(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                rc = cli.main(list(argv))
            except SystemExit as exc:  # argparse rejections
                rc = exc.code
        self.stderr = err.getvalue()
        return rc

    def client(self):
        return VeniceClient(api_key="test-fake-key")

    def authority(self):
        return _shared.MediaPathAuthority(str(self.tmp))


TB = "transparent_background"
IMAGE_ROUTES = {
    "generate": ("/image/generate", ("image", "a glass bird")),
    "edit": ("/image/edit", ("image-edit", "base.png", "-p", "cut out the bird")),
    "multi-edit": ("/image/multi-edit",
                   ("image-edit", "base.png", "-p", "cut out", "--layer", "mask.png")),
}


# --------------------------------------------------------------------------- #
# (b) transparent_background, CLI
# --------------------------------------------------------------------------- #

class TestTransparentBackgroundCli(_Hermetic):
    def _run(self, route, *flags):
        endpoint, argv = IMAGE_ROUTES[route]
        rc = self.venice(*argv, *flags, "--yes", "--no-balance")
        self.assertEqual(rc, 0, self.stderr)
        other = {e for e, _ in IMAGE_ROUTES.values()} - {endpoint}
        for path in other:
            self.assertEqual(self.api.bodies(path), [], f"{route} leaked to {path}")
        return self.api.only(endpoint)

    def test_true_false_omitted_on_every_route(self):
        for route in IMAGE_ROUTES:
            for flags, expected in (
                (("--transparent-background",), True),
                (("--no-transparent-background",), False),
                ((), None),
            ):
                with self.subTest(route=route, flags=flags):
                    self.api.requests.clear()
                    body = self._run(route, *flags)
                    if expected is None:
                        self.assertNotIn(TB, body)
                    else:
                        self.assertIs(body[TB], expected)

    def test_last_flag_wins(self):
        body = self._run("generate", "--transparent-background",
                         "--no-transparent-background")
        self.assertIs(body[TB], False)

    def test_flag_takes_no_value(self):
        rc = self.venice("image", "x", "--transparent-background=yes", "--yes")
        self.assertEqual(rc, 2)
        self.assertEqual(self.api.requests, [])

    def test_config_default_applies_and_explicit_false_overrides_it(self):
        self.set_config({"image": {TB: True}, "image_edit": {TB: True}})
        for route in IMAGE_ROUTES:
            with self.subTest(route=route, source="config"):
                self.api.requests.clear()
                self.assertIs(self._run(route)[TB], True)
            with self.subTest(route=route, source="explicit --no"):
                self.api.requests.clear()
                self.assertIs(self._run(route, "--no-transparent-background")[TB], False)

    def test_config_false_is_sent_not_dropped(self):
        self.set_config({"image": {TB: False}})
        self.assertIs(self._run("generate")[TB], False)

    def test_save_json_sidecar_round_trips_and_cli_override_beats_sidecar(self):
        self._run("generate", "--transparent-background", "--save-json")
        [sidecar] = list(self.tmp.glob("venice-image-*.json"))
        self.assertIs(json.loads(sidecar.read_text(encoding="utf-8"))[TB], True)
        self.api.requests.clear()
        replay = self.venice("image", "--from-json", str(sidecar), "--yes", "--no-balance")
        self.assertEqual(replay, 0, self.stderr)
        self.assertIs(self.api.only("/image/generate")[TB], True)
        self.api.requests.clear()
        replay = self.venice("image", "--from-json", str(sidecar),
                             "--no-transparent-background", "--yes", "--no-balance")
        self.assertEqual(replay, 0, self.stderr)
        self.assertIs(self.api.only("/image/generate")[TB], False)


# --------------------------------------------------------------------------- #
# (c) image_strength, CLI: queue-only, 0 is a value, out of range never spends
# --------------------------------------------------------------------------- #

class TestImageStrengthCli(_Hermetic):
    def _video(self, *flags):
        return self.venice("video", "a koi pond", "--image", START_FRAME,
                           *flags, "--yes", "--no-balance", "--background")

    def test_zero_low_high_reach_queue_but_never_quote(self):
        for raw, expected in (("0", 0.0), ("0.0", 0.0), ("0.65", 0.65), ("1", 1.0)):
            with self.subTest(image_strength=raw):
                self.api.requests.clear()
                self.assertEqual(self._video("--image-strength", raw), 0, self.stderr)
                queued = self.api.only("/video/queue")
                self.assertIn("image_strength", queued)
                self.assertEqual(queued["image_strength"], expected)
                self.assertIs(type(queued["image_strength"]), float)
                self.assertEqual(queued["image_url"], START_FRAME)
                self.assertNotIn("image_strength", self.api.only("/video/quote"))

    def test_omitted_is_absent_from_quote_and_queue(self):
        self.assertEqual(self._video(), 0, self.stderr)
        self.assertNotIn("image_strength", self.api.only("/video/queue"))
        self.assertNotIn("image_strength", self.api.only("/video/quote"))

    def test_dry_run_quotes_without_strength_and_never_queues(self):
        rc = self.venice("video", "a koi pond", "--image", START_FRAME,
                         "--image-strength", "0.4", "--dry-run", "--no-balance")
        self.assertEqual(rc, 0, self.stderr)
        self.assertNotIn("image_strength", self.api.only("/video/quote"))
        self.assertEqual(self.api.bodies("/video/queue"), [])

    def test_out_of_range_is_rejected_before_any_http(self):
        for raw in ("-0.01", "1.0001", "2", "-1"):
            with self.subTest(image_strength=raw):
                self.api.requests.clear()
                self.assertEqual(self._video(f"--image-strength={raw}"), 2)
                self.assertIn("image_strength must be between 0 and 1", self.stderr)
                self.assertEqual(self.api.requests, [])

    def test_non_numeric_and_non_finite_are_rejected_by_the_parser(self):
        for raw in ("nan", "inf", "-inf", "half", ""):
            with self.subTest(image_strength=raw):
                self.api.requests.clear()
                self.assertEqual(self._video(f"--image-strength={raw}"), 2)
                self.assertEqual(self.api.requests, [])

    def test_config_default_is_validated_like_a_flag(self):
        self.set_config({"video": {"image_strength": 0}})
        self.assertEqual(self._video(), 0, self.stderr)
        self.assertEqual(self.api.only("/video/queue")["image_strength"], 0.0)

        self.api.requests.clear()
        self.set_config({"video": {"image_strength": 1.5}})
        self.assertEqual(self._video(), 2)
        self.assertEqual(self.api.requests, [])

        self.api.requests.clear()
        self.set_config({"video": {"image_strength": 0.9}})
        self.assertEqual(self._video("--image-strength", "0.2"), 0, self.stderr)
        self.assertEqual(self.api.only("/video/queue")["image_strength"], 0.2)

    def test_garbage_config_default_is_ignored_not_sent(self):
        self.set_config({"video": {"image_strength": "strong"}})
        self.assertEqual(self._video(), 0, self.stderr)
        self.assertIn("ignoring invalid config default image_strength", self.stderr)
        self.assertNotIn("image_strength", self.api.only("/video/queue"))


# --------------------------------------------------------------------------- #
# (d) video prompt length boundaries, CLI and builder
# --------------------------------------------------------------------------- #

class TestVideoPromptBoundaryCli(_Hermetic):
    def _video(self, prompt, *flags):
        return self.venice("video", prompt, *flags, "--yes", "--no-balance", "--background")

    def test_exactly_at_limit_is_sent_unchanged(self):
        # Multi-byte so a byte-length check (or a truncation) would show up.
        prompt = "é" * (video.MAX_PROMPT - 1) + "z"
        self.assertEqual(len(prompt), 32000)
        self.assertEqual(self._video(prompt), 0, self.stderr)
        self.assertEqual(self.api.only("/video/queue")["prompt"], prompt)
        self.assertNotIn("prompt", self.api.only("/video/quote"))

    def test_one_over_limit_is_rejected_before_any_http(self):
        self.assertEqual(self._video("p" * 32001), 2)
        self.assertIn("prompt exceeds 32000 chars", self.stderr)
        self.assertEqual(self.api.requests, [])

    def test_empty_and_blank_prompts_are_rejected_before_any_http(self):
        for prompt in ("", "   ", "\n\t"):
            with self.subTest(prompt=prompt):
                self.api.requests.clear()
                self.assertEqual(self._video(prompt), 2)
                self.assertIn("prompt", self.stderr)
                self.assertEqual(self.api.requests, [])

    def test_negative_prompt_shares_the_limit(self):
        ok = "n" * 32000
        self.assertEqual(self._video("koi", "--negative-prompt", ok), 0, self.stderr)
        self.assertEqual(self.api.only("/video/queue")["negative_prompt"], ok)
        self.api.requests.clear()
        self.assertEqual(self._video("koi", "--negative-prompt", ok + "n"), 2)
        self.assertIn("negative_prompt exceeds 32000 chars", self.stderr)
        self.assertEqual(self.api.requests, [])

    def test_builder_refuses_what_the_cli_refuses(self):
        # A future caller that skips the pre-flight must still not build a body.
        self.assertEqual(video.queue_body("m", "p" * 32000, {}, {})["prompt"], "p" * 32000)
        for kwargs, message in (
            ({"prompt": "p" * 32001}, "prompt exceeds"),
            ({"prompt": ""}, "prompt is required"),
            ({"prompt": "p", "negative_prompt": "n" * 32001}, "negative_prompt exceeds"),
            ({"prompt": "p", "image_strength": 1.01}, "between 0 and 1"),
            ({"prompt": "p", "image_strength": True}, "finite number"),
            ({"prompt": "p", "image_strength": float("nan")}, "finite"),
        ):
            with self.subTest(**{k: repr(v)[:20] for k, v in kwargs.items()}):
                prompt = kwargs.pop("prompt")
                with self.assertRaisesRegex(ValueError, message):
                    video.queue_body("m", prompt, {}, {}, **kwargs)


# --------------------------------------------------------------------------- #
# (e) MCP + agent surfaces expose the same fields with the same semantics
# --------------------------------------------------------------------------- #

class TestToolImplSurfaces(_Hermetic):
    """`_mcp.*_tool` is what both the MCP server and the agent loop call."""

    def _image(self, route, **kw):
        if route == "generate":
            res = _mcp.image_tool(self.client(), "a glass bird", output_dir=str(self.tmp),
                                  confirm=True, **kw)
        else:
            layers = [str(self.tmp / "mask.png")] if route == "multi-edit" else None
            res = _mcp.image_edit_tool(
                self.client(), "cut out", input_path=str(self.tmp / "base.png"),
                layer_paths=layers, output_dir=str(self.tmp), confirm=True,
                path_authority=self.authority(), **kw,
            )
        self.assertEqual(res["status"], "ok", res)
        return self.api.only(IMAGE_ROUTES[route][0])

    def test_transparent_background_true_false_omitted(self):
        for route in IMAGE_ROUTES:
            for kw, expected in (({TB: True}, True), ({TB: False}, False),
                                 ({TB: None}, None), ({}, None)):
                with self.subTest(route=route, kw=kw):
                    self.api.requests.clear()
                    body = self._image(route, **kw)
                    if expected is None:
                        self.assertNotIn(TB, body)
                    else:
                        self.assertIs(body[TB], expected)

    def _video(self, **kw):
        return _mcp.video_tool(self.client(), "a koi pond", image_url=START_FRAME,
                               confirm=True, background=True, **kw)

    def test_image_strength_zero_nonzero_omitted(self):
        for kw, expected in (({"image_strength": 0}, 0.0),
                             ({"image_strength": 0.5}, 0.5),
                             ({"image_strength": "0.25"}, 0.25),
                             ({}, None)):
            with self.subTest(kw=kw):
                self.api.requests.clear()
                self.assertEqual(self._video(**kw)["status"], "queued")
                queued = self.api.only("/video/queue")
                self.assertNotIn("image_strength", self.api.only("/video/quote"))
                if expected is None:
                    self.assertNotIn("image_strength", queued)
                else:
                    self.assertEqual(queued["image_strength"], expected)
                    self.assertIs(type(queued["image_strength"]), float)

    def test_bad_image_strength_is_an_error_before_any_http(self):
        for value in (-0.1, 1.1, True, float("nan"), float("inf"), "strong", [0.5]):
            with self.subTest(image_strength=value):
                self.api.requests.clear()
                res = self._video(image_strength=value)
                self.assertEqual(res["status"], "error", res)
                self.assertTrue(res["message"].startswith("video: "), res)
                self.assertEqual(self.api.requests, [])

    def test_prompt_boundaries(self):
        prompt = "q" * 32000
        res = _mcp.video_tool(self.client(), prompt, confirm=True, background=True)
        self.assertEqual(res["status"], "queued", res)
        self.assertEqual(self.api.only("/video/queue")["prompt"], prompt)
        for bad in ("q" * 32001, "", "   "):
            with self.subTest(length=len(bad)):
                self.api.requests.clear()
                res = _mcp.video_tool(self.client(), bad, confirm=True, background=True)
                self.assertEqual(res["status"], "error", res)
                self.assertEqual(self.api.requests, [])
        res = _mcp.video_tool(self.client(), "koi", negative_prompt="n" * 32001,
                              confirm=True, background=True)
        self.assertIn("negative_prompt exceeds 32000", res["message"])

    def test_unconfirmed_call_validates_but_does_not_queue(self):
        res = _mcp.video_tool(self.client(), "koi", image_strength=0.3)
        self.assertNotEqual(res["status"], "queued")
        self.assertEqual(self.api.bodies("/video/queue"), [])


class TestAgentToolSurface(_Hermetic):
    def _tools(self, config_doc=None):
        return {t.name: t for t in _agent.builtin_tools(
            self.client(), output_dir=str(self.tmp), config=config_doc,
            only={"venice_image", "venice_image_edit", "venice_video"},
            media_path_authority=self.authority(),
        )}

    def test_schemas_advertise_the_new_fields_with_impl_matching_types(self):
        tools = self._tools()
        for name in ("venice_image", "venice_image_edit"):
            self.assertEqual(tools[name].parameters["properties"][TB]["type"], "boolean")
        props = tools["venice_video"].parameters["properties"]
        self.assertEqual(props["image_strength"]["type"], "number")
        self.assertEqual((props["image_strength"]["minimum"],
                          props["image_strength"]["maximum"]), (0, 1))
        self.assertEqual(props["prompt"]["maxLength"], video.MAX_PROMPT)
        self.assertEqual(props["negative_prompt"]["maxLength"], video.MAX_PROMPT)
        self.assertNotIn(TB, props)
        for name in ("venice_image", "venice_image_edit"):
            self.assertNotIn("image_strength", tools[name].parameters["properties"])

    def test_invoked_tools_send_the_fields(self):
        tools = self._tools()
        cases = (
            ("venice_image", {"prompt": "bird", TB: False}, "/image/generate"),
            ("venice_image_edit", {"prompt": "cut", "input_path": "base.png", TB: True},
             "/image/edit"),
            ("venice_image_edit", {"prompt": "cut", "input_path": "base.png",
                                   "layer_paths": ["mask.png"], TB: False},
             "/image/multi-edit"),
        )
        for name, args, endpoint in cases:
            with self.subTest(tool=name, endpoint=endpoint):
                self.api.requests.clear()
                res = tools[name].invoke(args, confirm=True)
                self.assertEqual(res["status"], "ok", res)
                self.assertIs(self.api.only(endpoint)[TB], args[TB])

        self.api.requests.clear()
        res = tools["venice_video"].invoke({"prompt": "koi", "image_url": START_FRAME,
                                            "image_strength": 0, "background": True},
                                           confirm=True)
        self.assertEqual(res["status"], "queued", res)
        self.assertEqual(self.api.only("/video/queue")["image_strength"], 0.0)
        self.assertNotIn("image_strength", self.api.only("/video/quote"))

    def test_model_supplied_out_of_range_values_never_spend(self):
        tools = self._tools()
        for args in ({"prompt": "koi", "image_strength": 7},
                     {"prompt": "k" * 32001},
                     {"prompt": "koi", "negative_prompt": "n" * 32001}):
            with self.subTest(args={k: str(v)[:12] for k, v in args.items()}):
                self.api.requests.clear()
                res = tools["venice_video"].invoke(dict(args, background=True),
                                                   confirm=True)
                self.assertEqual(res["status"], "error", res)
                self.assertEqual(self.api.requests, [])

    def test_config_defaults_layer_under_model_arguments(self):
        doc = {"version": 1, "defaults": {
            "image": {TB: True}, "video": {"image_strength": 0.75},
        }}
        tools = self._tools(doc)
        tools["venice_image"].invoke({"prompt": "bird"}, confirm=True)
        self.assertIs(self.api.only("/image/generate")[TB], True)
        self.api.requests.clear()
        tools["venice_image"].invoke({"prompt": "bird", TB: False}, confirm=True)
        self.assertIs(self.api.only("/image/generate")[TB], False)
        self.api.requests.clear()
        tools["venice_video"].invoke({"prompt": "koi", "background": True}, confirm=True)
        self.assertEqual(self.api.only("/video/queue")["image_strength"], 0.75)
        self.api.requests.clear()
        tools["venice_video"].invoke({"prompt": "koi", "background": True,
                                      "image_strength": 0}, confirm=True)
        self.assertEqual(self.api.only("/video/queue")["image_strength"], 0.0)


@unittest.skipUnless(_HAS_MCP, "mcp SDK not installed (expected on Python 3.9)")
class TestMcpServerSurface(_Hermetic):
    def _stdio(self, defaults=None):
        from venice.mcp_server import build_server
        doc = {"version": 1, "defaults": defaults or {}}
        return build_server(self.client(), doc=doc, root=str(self.tmp))

    @staticmethod
    def _schema(server, name):
        return server._tool_manager.get_tool(name).parameters["properties"]

    @staticmethod
    def _accepts(prop, json_type):
        branches = prop.get("anyOf", [prop])
        return any(b.get("type") == json_type for b in branches)

    def test_stdio_schemas_expose_fields_with_cli_types(self):
        server = self._stdio()
        for name in ("venice_image", "venice_image_edit"):
            self.assertTrue(self._accepts(self._schema(server, name)[TB], "boolean"))
        self.assertTrue(self._accepts(
            self._schema(server, "venice_video")["image_strength"], "number"))
        self.assertNotIn(TB, self._schema(server, "venice_video"))

    def test_http_schemas_expose_fields(self):
        from venice.mcp_server import build_http_server

        class Verifier:
            async def verify_token(self, token):
                return None

        server = build_http_server(
            self.client(), doc={}, public_url="https://mcp.example.test/mcp",
            issuer_url="https://auth.example.test",
            jwks_url="https://auth.example.test/jwks", audience="venice-api",
            scopes=["venice:mcp"], token_verifier=Verifier(),
            media_dir=str(self.tmp / "media"),
        )
        for name in ("venice_image", "venice_image_edit"):
            self.assertTrue(self._accepts(self._schema(server, name)[TB], "boolean"))
        self.assertTrue(self._accepts(
            self._schema(server, "venice_video")["image_strength"], "number"))

    def _payload(self, result):
        if isinstance(result, dict):
            return result
        return json.loads(result.content[0].text)

    def test_stdio_tools_forward_fields_to_the_wire(self):
        server = self._stdio()
        image = server._tool_manager.get_tool("venice_image").fn
        for value in (True, False, None):
            with self.subTest(tool="venice_image", value=value):
                self.api.requests.clear()
                res = self._payload(image(prompt="bird", transparent_background=value,
                                          output_dir=str(self.tmp), confirm=True))
                self.assertEqual(res["status"], "ok", res)
                body = self.api.only("/image/generate")
                if value is None:
                    self.assertNotIn(TB, body)
                else:
                    self.assertIs(body[TB], value)

        edit = server._tool_manager.get_tool("venice_image_edit").fn
        self.api.requests.clear()
        res = self._payload(edit(prompt="cut", input_path=str(self.tmp / "base.png"),
                                 layer_paths=[str(self.tmp / "mask.png")],
                                 transparent_background=False,
                                 output_dir=str(self.tmp), confirm=True))
        self.assertEqual(res["status"], "ok", res)
        self.assertIs(self.api.only("/image/multi-edit")[TB], False)

        clip = server._tool_manager.get_tool("venice_video").fn
        self.api.requests.clear()
        res = self._payload(clip(prompt="koi", image_url=START_FRAME, image_strength=0.0,
                                 output_dir=str(self.tmp), confirm=True, max_wait=5))
        self.assertEqual(res["status"], "ok", res)
        self.assertEqual(self.api.only("/video/queue")["image_strength"], 0.0)
        self.assertNotIn("image_strength", self.api.only("/video/quote"))

    def test_stdio_host_false_beats_config_true(self):
        server = self._stdio({"image": {TB: True}})
        image = server._tool_manager.get_tool("venice_image").fn
        image(prompt="bird", output_dir=str(self.tmp), confirm=True)
        self.assertIs(self.api.only("/image/generate")[TB], True)
        self.api.requests.clear()
        image(prompt="bird", transparent_background=False,
              output_dir=str(self.tmp), confirm=True)
        self.assertIs(self.api.only("/image/generate")[TB], False)

    def test_stdio_video_rejects_out_of_range_before_any_http(self):
        clip = self._stdio()._tool_manager.get_tool("venice_video").fn
        res = self._payload(clip(prompt="koi", image_strength=1.5, confirm=True))
        self.assertEqual(res["status"], "error", res)
        self.assertEqual(self.api.requests, [])


if __name__ == "__main__":
    unittest.main()
