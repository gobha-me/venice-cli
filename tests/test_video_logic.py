"""Video behaviour regressions for #255, using fixtures and no live API.

Exercise video functions together; replace only HTTP, terminal input and other
external dependencies. In particular, never replace another video function.
"""

import argparse
import base64
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.error import HTTPError
from urllib.parse import urlsplit

from tests.test_client import FakeResp
from venice import config
from venice.client import VeniceAPIError, VeniceClient
from venice.commands import _video_jobs, video


def parser():
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    video.register(sub)
    video.register_status(sub)
    return result


def catalog():
    return {"data": [{
        "id": "logic-video",
        "model_spec": {
            "traits": ["default"],
            "constraints": {
                "durations": ["5s", "10s"],
                "resolutions": ["720p", "1080p"],
                "aspect_ratios": ["16:9", "9:16"],
            },
        },
    }]}


class VideoParserTests(unittest.TestCase):
    def test_omitted_options_remain_unset_until_config_resolution(self):
        p = parser()
        generate = p.parse_args(["video"])
        self.assertIs(generate.handler, video._run_generate)
        expected = dict.fromkeys([
            "prompt", "model", "duration", "resolution", "aspect_ratio",
            "negative_prompt", "no_audio", "image", "image_strength",
            "end_image", "video", "audio_input", "reference_image",
            "reference_video", "reference_audio", "scene_image",
            "reference_video_duration", "element", "output", "yes",
            "no_cleanup", "max_spend", "no_balance", "poll_interval", "max_wait",
        ])
        self.assertEqual(vars(generate), {
            **expected, "command": "video", "background": False,
            "dry_run": False, "handler": video._run_generate,
        })
        status = p.parse_args(["video-status", "job"])
        self.assertIs(status.handler, video._run_status)
        self.assertEqual(vars(status), {
            **dict.fromkeys([
                "model", "download_url", "output", "no_cleanup",
                "poll_interval", "max_wait",
            ]),
            "command": "video-status", "queue_id": "job",
            "handler": video._run_status,
        })

    def test_each_value_flag_reaches_its_typed_destination(self):
        cases = [
            ("--model", "chosen", "model", "chosen"),
            ("--duration", "10s", "duration", "10s"),
            ("--resolution", "1080p", "resolution", "1080p"),
            ("--aspect-ratio", "9:16", "aspect_ratio", "9:16"),
            ("--negative-prompt", "blur", "negative_prompt", "blur"),
            ("--image", "first.png", "image", "first.png"),
            ("--end-image", "last.png", "end_image", "last.png"),
            ("--video", "clip.mp4", "video", "clip.mp4"),
            ("--audio", "track.wav", "audio_input", "track.wav"),
            ("--image-strength", "0.25", "image_strength", 0.25),
            ("--reference-video-duration", "3.75", "reference_video_duration", 3.75),
            ("--max-spend", "0.75", "max_spend", 0.75),
        ]
        p = parser()
        for flag, raw, dest, expected in cases:
            with self.subTest(flag=flag):
                args = p.parse_args(["video", "prompt", flag, raw])
                self.assertEqual(getattr(args, dest), expected)
                self.assertIs(type(getattr(args, dest)), type(expected))
                self.assertEqual(args.prompt, "prompt")
        for command, positional in (("video", "prompt"), ("video-status", "job")):
            for flag in ("--output", "-o"):
                with self.subTest(command=command, flag=flag):
                    self.assertEqual(p.parse_args([
                        command, positional, flag, "movie.webm",
                    ]).output, Path("movie.webm"))
            args = p.parse_args([
                command, positional, "--poll-interval", "0.25", "--max-wait", "17",
            ])
            self.assertEqual((args.poll_interval, args.max_wait), (0.25, 17.0))
            self.assertIs(type(args.max_wait), float)
        args = p.parse_args([
            "video-status", "job", "--model", "chosen", "--download-url",
            "https://example.test/old.mp4",
        ])
        self.assertEqual((args.model, args.download_url), (
            "chosen", "https://example.test/old.mp4",
        ))

    def test_repeatable_flags_preserve_both_values_and_order(self):
        for flag, dest in (
            ("--reference-image", "reference_image"),
            ("--reference-video", "reference_video"),
            ("--reference-audio", "reference_audio"),
            ("--scene-image", "scene_image"),
            ("--element", "element"),
        ):
            with self.subTest(flag=flag):
                args = parser().parse_args(["video", "prompt", flag, "a", flag, "b"])
                self.assertEqual(getattr(args, dest), ["a", "b"])

    def test_switches_are_boolean_and_opposites_are_exclusive(self):
        p = parser()
        for flag, dest, value in (
            ("--no-audio", "no_audio", True), ("--with-audio", "no_audio", False),
            ("--yes", "yes", True), ("-y", "yes", True),
            ("--background", "background", True), ("--dry-run", "dry_run", True),
        ):
            with self.subTest(flag=flag):
                self.assertIs(getattr(p.parse_args(["video", "p", flag]), dest), value)
        for command in ("video", "video-status"):
            for flag, expected in (("--cleanup", False), ("--no-cleanup", True)):
                self.assertIs(p.parse_args([command, "p", flag]).no_cleanup, expected)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                p.parse_args([command, "p", "--cleanup", "--no-cleanup"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            p.parse_args(["video", "p", "--no-audio", "--with-audio"])

    def test_numeric_flags_reject_nonfinite_and_spend_rejects_negative(self):
        for flag in ("--image-strength", "--reference-video-duration", "--max-spend"):
            for raw in ("nan", "inf", "-inf", "oops"):
                with self.subTest(flag=flag, raw=raw), \
                        contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit) as caught:
                    parser().parse_args(["video", "p", f"{flag}={raw}"])
                self.assertEqual(caught.exception.code, 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(["video", "p", "--max-spend=-0.1"])


class VideoValidationTests(unittest.TestCase):
    def test_queue_validation_messages_and_inclusive_strength_boundaries(self):
        for prompt in (None, 7, "", " \t"):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError) as caught:
                video.validate_queue_options(prompt)
            self.assertEqual(str(caught.exception), "prompt is required")
        for strength in (-0.01, 1.01):
            with self.subTest(strength=strength), self.assertRaises(ValueError) as caught:
                video.validate_queue_options("ok", image_strength=strength)
            self.assertEqual(str(caught.exception), "image_strength must be between 0 and 1")
        for strength in (0, 1, "0.25"):
            self.assertIsNone(video.validate_queue_options("ok", image_strength=strength))
        self.assertEqual(video.queue_body("m", "p", {}, {}, "blur", "0"), {
            "model": "m", "prompt": "p", "negative_prompt": "blur", "image_strength": 0.0,
        })

    def test_model_validation_distinguishes_all_option_fields(self):
        models = catalog()["data"]
        video.validate_model_options([None, {}, *models], "logic-video", duration="10s")
        for field, label in (("duration", "duration"), ("resolution", "resolution"),
                             ("aspect_ratio", "aspect ratio")):
            choices = {"duration": "5s, 10s", "resolution": "720p, 1080p",
                       "aspect_ratio": "16:9, 9:16"}
            for invalid in ("", " ", 1):
                with self.subTest(field=field, invalid=invalid):
                    options = {"duration": "5s", field: invalid}
                    with self.assertRaises(ValueError) as caught:
                        video.validate_model_options(models, "logic-video", **options)
                    self.assertEqual(str(caught.exception), f"{label} must be a non-empty string")
            with self.subTest(field=field), self.assertRaises(ValueError) as caught:
                video.validate_model_options(models, "logic-video", **{
                    "duration": "5s", field: "unsupported",
                })
            self.assertEqual(str(caught.exception),
                             f"{label} 'unsupported' is not supported by 'logic-video'; "
                             f"choose from {choices[field]}")
        with self.assertRaises(ValueError) as caught:
            video.validate_model_options(models, "logic-video", duration=None)
        self.assertEqual(str(caught.exception), "duration must be a non-empty string")

    def test_catalog_missing_model_spec_and_malformed_constraint_entries(self):
        for models, message in (
            (None, "could not fetch the live video catalog"),
            ([], "model 'logic-video' is not in the live video catalog"),
            ([{"id": "logic-video", "model_spec": []}],
             "live video catalog entry for 'logic-video' has no constraints object"),
        ):
            with self.subTest(models=models), self.assertRaises(ValueError) as caught:
                video.validate_model_options(models, "logic-video", duration="5s")
            self.assertEqual(str(caught.exception), message)
        for field in ("durations", "resolutions", "aspect_ratios"):
            for values in (None, "5s", [None], [1], ["ok", " "], ["ok", ""]):
                models = catalog()["data"]
                models[0]["model_spec"]["constraints"][field] = values
                with self.subTest(field=field, values=values), \
                        self.assertRaises(ValueError) as caught:
                    video.validate_model_options(models, "logic-video", duration="5s")
                self.assertEqual(str(caught.exception),
                                 f"live video catalog entry for 'logic-video' has invalid constraints.{field}")


class VideoFixtureCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.root.chmod(0o700)
        self.enterContext(mock.patch.dict(os.environ, {
            "HOME": str(self.root), "XDG_CONFIG_HOME": str(self.root / "config"),
            "VENICE_API_KEY": "test-fake-key",
            config.ENV_VIDEO_JOBS_FILE: str(self.root / "jobs.json"),
        }))
        self.configuration = {"version": 1, "mcpServers": {}, "defaults": {}}
        self.enterContext(mock.patch("venice.userconfig.load_config",
                                     side_effect=lambda: self.configuration))
        # Unexpected HTTP must fail locally, even if a case forgets its fixture.
        self.enterContext(mock.patch("venice.client.urllib.request.urlopen",
                                     side_effect=AssertionError("unexpected HTTP")))

    # unittest.TestCase.enterContext was added in Python 3.11.
    def enterContext(self, manager):
        value = manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)
        return value

    def media_file(self, name, data=b"fixture"):
        path = self.root / name
        path.write_bytes(data)
        return str(path)


class VideoMediaLogicTests(VideoFixtureCase):
    def test_url_schemes_and_unknown_extension_mime_defaults(self):
        for value in ("http://example.test/a", "https://example.test/b", "data:image/png;base64,YQ=="):
            self.assertEqual(video._resolve_media(value, kind="image"), (value, None))
        for value in ("file:///tmp/a", "HTTP://example.test/a", "httpsomething", "a"):
            self.assertFalse(video._is_url(value))
        for kind, mime in (("image", "image/png"), ("audio", "audio/mpeg"), ("video", "video/mp4")):
            with self.subTest(kind=kind):
                path = self.media_file(f"{kind}.unknownsuffix", b"one")
                self.assertEqual(video._resolve_media(path, kind=kind),
                                 (f"data:{mime};base64,b25l", None))

    def test_missing_empty_and_oversize_files_report_to_stderr(self):
        missing = str(self.root / "missing.png")
        empty = self.media_file("empty.png", b"")
        for value, message in (
            (missing, f"video: input file not found: {missing}\n"),
            (empty, f"video: input {empty} is empty\n"),
        ):
            with contextlib.redirect_stderr(io.StringIO()) as err, \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(video._resolve_media(value, kind="image"), (None, 2))
            self.assertEqual(err.getvalue(), message)
            self.assertEqual(out.getvalue(), "")
        for kind, limit in (("image", 25), ("video", 50), ("audio", 15)):
            path = Path(self.media_file(f"large-{kind}.bin"))
            size = limit * 1024 * 1024 + 1
            with path.open("wb") as stream:
                stream.truncate(size)
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(video._resolve_media(str(path), kind=kind), (None, 2))
            self.assertEqual(err.getvalue(),
                             f"video: input {path} is {size} bytes; {kind} inputs must be < {limit} MB\n")

    def test_size_limit_is_inclusive(self):
        path = Path(self.media_file("limit.unknownsuffix"))
        with path.open("wb") as stream:
            stream.truncate(15 * 1024 * 1024)
        url, rc = video._resolve_media(str(path), kind="audio")
        self.assertIsNone(rc)
        self.assertTrue(url.startswith("data:audio/mpeg;base64,"))
        self.assertEqual(len(base64.b64decode(url.split(",", 1)[1])), 15 * 1024 * 1024)

    def test_collection_maps_all_inputs_to_exact_endpoint_fields(self):
        args = SimpleNamespace(
            image=self.media_file("start.unknownsuffix", b"start"),
            end_image="http://example.test/end", audio_input=self.media_file("sound.unknownsuffix", b"sound"),
            video=self.media_file("clip.unknownsuffix", b"clip"),
            reference_image=["data:image/png;base64,YQ=="],
            reference_video=[self.media_file("refvideo.unknownsuffix", b"refv")],
            reference_audio=[self.media_file("refaudio.unknownsuffix", b"refa")],
            scene_image=[self.media_file("scene.unknownsuffix", b"scene")],
            element=[json.dumps({"frontal_image_url": "https://example.test/front", "extra": 7})],
            reference_video_duration=0.0,
        )
        self.assertEqual(video._collect_media(args), (
            {"video_url": "data:video/mp4;base64,Y2xpcA==", "reference_video_total_duration": 0.0},
            {"image_url": "data:image/png;base64,c3RhcnQ=", "end_image_url": "http://example.test/end",
             "audio_url": "data:audio/mpeg;base64,c291bmQ=", "video_url": "data:video/mp4;base64,Y2xpcA==",
             "reference_image_urls": ["data:image/png;base64,YQ=="],
             "reference_video_urls": ["data:video/mp4;base64,cmVmdg=="],
             "reference_audio_urls": ["data:audio/mpeg;base64,cmVmYQ=="],
             "scene_image_urls": ["data:image/png;base64,c2NlbmU="],
             "elements": [{"frontal_image_url": "https://example.test/front", "extra": 7}]},
            None,
        ))
        self.assertEqual(video._collect_media(SimpleNamespace()), ({}, {}, None))

    def test_collection_caps_and_missing_file_fail_before_partial_results(self):
        for attr, cap in (("reference_image", 30), ("reference_video", 10),
                          ("reference_audio", 10), ("scene_image", 4)):
            flag = "--" + attr.replace("_", "-")
            with self.subTest(attr=attr):
                values = ["https://example.test/a"] * cap
                quote, queue, rc = video._collect_media(SimpleNamespace(**{attr: values}))
                self.assertEqual(quote, {})
                self.assertIsNone(rc)
                self.assertEqual(queue[attr + "_urls"], values)
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    self.assertEqual(video._collect_media(SimpleNamespace(**{
                        attr: values + ["https://example.test/b"],
                    })), (None, None, 2))
                self.assertEqual(err.getvalue(), f"video: at most {cap} {flag}\n")
        for attr in ("image", "end_image", "video", "audio_input", "reference_image",
                     "reference_video", "reference_audio", "scene_image"):
            value = str(self.root / "missing")
            if attr.startswith("reference_") or attr == "scene_image":
                value = [value]
            with self.subTest(attr=attr), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(video._collect_media(SimpleNamespace(**{attr: value})), (None, None, 2))

    def test_elements_resolve_nested_media_and_preserve_unknown_fields(self):
        raw = {
            "frontal_image_url": self.media_file("front.unknownsuffix", b"front"),
            "video_url": self.media_file("video.unknownsuffix", b"video"),
            "reference_image_urls": [self.media_file("ref.unknownsuffix", b"ref"),
                                     "https://example.test/one", "https://example.test/two"],
            "future": {"value": True},
        }
        expected = {
            "frontal_image_url": "data:image/png;base64,ZnJvbnQ=",
            "video_url": "data:video/mp4;base64,dmlkZW8=",
            "reference_image_urls": ["data:image/png;base64,cmVm", "https://example.test/one",
                                     "https://example.test/two"],
            "future": {"value": True},
        }
        self.assertEqual(video._resolve_elements([json.dumps(raw)] * 4), ([expected] * 4, None))
        self.assertEqual(video._resolve_elements([json.dumps({"reference_image_urls": []})]),
                         ([{"reference_image_urls": []}], None))

    def test_element_errors_retain_the_specific_diagnostic(self):
        for raw, message in (
            (["{}"] * 5, "video: at most 4 --element\n"),
            (["[]"], "video: --element must be a JSON object\n"),
            ([json.dumps({"reference_image_urls": "url"})],
             "video: element reference_image_urls must be a list\n"),
            ([json.dumps({"reference_image_urls": ["https://example.test/a"] * 4})],
             "video: at most 3 element reference_image_urls\n"),
        ):
            with self.subTest(raw=raw), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(video._resolve_elements(raw), (None, 2))
            self.assertEqual(err.getvalue(), message)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(video._collect_media(SimpleNamespace(element=["{"])), (None, None, 2))
        self.assertTrue(err.getvalue().startswith("video: --element is not valid JSON: "))
        for field in ("frontal_image_url", "video_url", "reference_image_urls"):
            value = str(self.root / "missing")
            if field == "reference_image_urls":
                value = [value]
            with self.subTest(field=field), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(video._resolve_elements([json.dumps({field: value})]), (None, 2))
            self.assertIn("input file not found:", err.getvalue())

    def test_media_context_is_forwarded_for_single_list_and_element_inputs(self):
        path = Path(self.media_file("resolved.unknownsuffix", b"resolved"))
        authority = mock.Mock()
        authority.resolve.return_value = (path, "image/webp")
        args = SimpleNamespace(
            image="single", reference_video=["list"],
            element=[json.dumps({"frontal_image_url": "front", "video_url": "clip",
                                 "reference_image_urls": ["nested"]})],
        )
        encoded = "data:image/webp;base64,cmVzb2x2ZWQ="
        self.assertEqual(video._collect_media(args, path_authority=authority), (
            {}, {"image_url": encoded, "reference_video_urls": [encoded],
                 "elements": [{"frontal_image_url": encoded, "video_url": encoded,
                               "reference_image_urls": [encoded]}]}, None,
        ))
        self.assertEqual(authority.resolve.call_args_list, [
            mock.call("single", kind="image", max_bytes=25 * 1024 * 1024),
            mock.call("list", kind="video", max_bytes=50 * 1024 * 1024),
            mock.call("front", kind="image", max_bytes=25 * 1024 * 1024),
            mock.call("clip", kind="video", max_bytes=50 * 1024 * 1024),
            mock.call("nested", kind="image", max_bytes=25 * 1024 * 1024),
        ])


class VideoHttpLogicTests(VideoFixtureCase):
    def run_http(self, argv, responses, *, stdin=""):
        """Check ordered routes and retain exact wire bodies, without headers."""
        remaining = list(responses)
        seen = []

        def open_fixture(request, timeout=None):
            parsed = urlsplit(request.full_url)
            route = parsed.path.removeprefix("/api/v1")
            body = json.loads(request.data) if request.data is not None else None
            seen.append((request.get_method(), route, parsed.query, body))
            self.assertTrue(remaining, f"unexpected request: {route}")
            expected, response = remaining.pop(0)
            self.assertEqual(route, expected)
            if isinstance(response, Exception):
                raise response
            if isinstance(response, FakeResp):
                return response
            return FakeResp(200, json.dumps(response).encode())

        with mock.patch("venice.client.urllib.request.urlopen", side_effect=open_fixture), \
                mock.patch("sys.stdin", io.StringIO(stdin)), \
                mock.patch("venice.client.time.sleep"), \
                contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()) as err:
            args = parser().parse_args(argv)
            rc = args.handler(args)
        self.assertEqual(remaining, [], "fixture responses were not consumed")
        return rc, out.getvalue(), err.getvalue(), seen

    def generate_routes(self, *, queued=None, quote=0.375):
        return [
            ("/models", catalog()), ("/video/quote", {"quote": quote}),
            ("/video/queue", {"queue_id": "job-logic-123"} if queued is None else queued),
        ]

    def test_background_wire_bodies_stdout_and_hint_are_exact(self):
        rc, out, err, seen = self.run_http([
            "video", "a dancing fox", "--background", "--no-balance",
            "--duration", "10s", "--resolution", "1080p", "--aspect-ratio", "9:16",
            "--negative-prompt", "blur", "--no-audio", "--image-strength", "0.25",
            "--image", "https://example.test/image", "--video", "https://example.test/video",
            "--reference-video-duration", "3.75",
        ], self.generate_routes())
        self.assertEqual((rc, out), (0, "job-logic-123\n"))
        self.assertEqual(err,
                         "Estimated cost: $0.3750 USD (model=logic-video, duration=10s, 1080p)\n"
                         "queued as job-logic-123; fetch with: venice video-status job-logic-123 --model logic-video\n")
        common = {"model": "logic-video", "duration": "10s", "resolution": "1080p",
                  "aspect_ratio": "9:16", "audio": False, "video_url": "https://example.test/video"}
        self.assertEqual(seen, [
            ("GET", "/models", "type=video", None),
            ("POST", "/video/quote", "", {**common, "reference_video_total_duration": 3.75}),
            ("POST", "/video/queue", "", {**common, "prompt": "a dancing fox",
                                          "negative_prompt": "blur", "image_strength": 0.25,
                                          "image_url": "https://example.test/image"}),
        ])

    def test_queue_id_fallback_precedence_and_empty_response(self):
        for queued, expected in (
            ({"id": "legacy"}, "legacy"),
            ({"queue_id": "", "id": "fallback"}, "fallback"),
            ({"queue_id": "primary", "id": "secondary"}, "primary"),
        ):
            with self.subTest(queued=queued):
                rc, out, err, _ = self.run_http(
                    ["video", "p", "--background", "--no-balance"], self.generate_routes(queued=queued))
                self.assertEqual((rc, out), (0, expected + "\n"))
                self.assertIn(f"queued as {expected}; fetch with: venice video-status {expected} --model logic-video\n", err)
        rc, out, err, _ = self.run_http(
            ["video", "p", "--background", "--no-balance"], self.generate_routes(queued={}))
        self.assertEqual((rc, out), (5, ""))
        self.assertIn("queue response missing queue_id: {}\n", err)

    def test_missing_prompt_and_catalog_outage_have_actionable_errors(self):
        for argv in (["video"], ["video", "--background"]):
            rc, out, err, seen = self.run_http(argv, [])
            self.assertEqual((rc, out, seen), (2, "", []))
            self.assertEqual(err, "video: prompt required (or use: venice video-status <id>)\n")
        rc, out, err, _ = self.run_http(["video", "p"], [("/models", {})])
        self.assertEqual((rc, out), (2, ""))
        self.assertEqual(err, "video: could not fetch the live video catalog; cannot validate model-specific options\n")
        doc = catalog()
        doc["data"][0]["model_spec"]["traits"] = []
        rc, _, err, _ = self.run_http(["video", "p"], [("/models", doc)])
        self.assertEqual(rc, 6)
        self.assertIn("video: no default video model", err)
        self.assertIn("venice config set defaults.video.model <id>", err)

    def test_duration_resolution_and_aspect_rejection_stop_before_quote(self):
        for flag in ("--duration", "--resolution", "--aspect-ratio"):
            with self.subTest(flag=flag):
                rc, out, err, seen = self.run_http(["video", "p", flag, "invalid"], [("/models", catalog())])
                self.assertEqual((rc, out), (2, ""))
                self.assertEqual(len(seen), 1)
                self.assertIn("'invalid' is not supported by 'logic-video'", err)

    def test_budget_dry_run_confirmation_and_balance_use_quote_value(self):
        quote_routes = [("/models", catalog()), ("/video/quote", {"quote": 0.375})]
        rc, out, err, _ = self.run_http(
            ["video", "p", "--no-balance", "--background", "--max-spend", "0.374"], quote_routes)
        self.assertEqual((rc, out), (1, ""))
        self.assertIn("video: quote $0.3750 USD exceeds --max-spend $0.3740 USD; aborting\n", err)
        rc, out, _, _ = self.run_http(
            ["video", "p", "--no-balance", "--dry-run", "--max-spend", "0.375"], quote_routes)
        self.assertEqual((rc, out), (0, ""))
        rc, out, err, _ = self.run_http(["video", "p", "--no-balance"], quote_routes)
        self.assertEqual((rc, out), (1, ""))
        self.assertIn("non-interactive; pass --yes to confirm the charge.", err)
        routes = [*quote_routes, ("/api_keys/rate_limits", {"data": {"balances": {"USD": 2.0}}})]
        rc, out, err, _ = self.run_http(["video", "p", "--dry-run"], routes)
        self.assertEqual((rc, out), (0, ""))
        self.assertIn("After charge:   $1.62 USD", err)

    def test_foreground_generates_and_status_downloads_to_requested_path(self):
        for command in ("video", "video-status"):
            output = self.root / (command + ".webm")
            argv = [command, "p" if command == "video" else "saved-job", "--no-cleanup",
                    "--output", str(output), "--poll-interval", "0", "--max-wait", "17"]
            if command == "video":
                argv.extend(["--yes", "--no-balance"])
                routes = self.generate_routes()
                queue_id = "job-logic-123"
            else:
                argv.extend(["--model", "logic-video"])
                routes = []
                queue_id = "saved-job"
            rc, out, err, seen = self.run_http(argv, [
                *routes, ("/video/retrieve", FakeResp(200, b"real-fixture-bytes", "video/webm")),
            ])
            self.assertEqual((rc, out), (0, str(output.resolve()) + "\n"))
            self.assertEqual(output.read_bytes(), b"real-fixture-bytes")
            self.assertIn(f"wrote 18 bytes to {output.resolve()}\n", err)
            self.assertEqual(seen[-1], ("POST", "/video/retrieve", "", {"model": "logic-video", "queue_id": queue_id}))

    def test_bad_quote_and_queue_http_failures_preserve_exit_codes(self):
        rc, out, err, _ = self.run_http(["video", "p", "--no-balance", "--background"], [
            ("/models", catalog()), ("/video/quote", {"quote": "nan"}),
        ])
        self.assertEqual((rc, out), (2, ""))
        self.assertIn("quote response did not contain a finite non-negative cost", err)
        for stage, status, exit_code in (("quote", 422, 3), ("queue", 429, 4)):
            routes = self.generate_routes()[:1 if stage == "quote" else 2]
            routes.append((f"/video/{stage}", HTTPError(
                "https://example.test", status, "fixture", {}, io.BytesIO(b"fixture failure"))))
            with self.subTest(stage=stage):
                rc, out, err, _ = self.run_http(["video", "p", "--background", "--no-balance"], routes)
                self.assertEqual((rc, out), (exit_code, ""))
                self.assertIn("quote rejected:" if stage == "quote" else "queue failed:", err)

    def test_handlers_forward_builtin_configured_and_explicit_polling(self):
        for command in ("video", "video-status"):
            for configured, flags, expected in (
                ({}, [], (5.0, 900.0)),
                ({"poll_interval": 0.25, "max_wait": 17}, [], (0.25, 17.0)),
                ({"poll_interval": 0.25, "max_wait": 17},
                 ["--poll-interval", "0.5", "--max-wait", "23"], (0.5, 23.0)),
                ({}, ["--poll-interval=-1", "--max-wait=-1"], (5.0, 900.0)),
            ):
                self.configuration["defaults"] = {"video": configured}
                argv = [command, "p" if command == "video" else "saved-job",
                        "--model", "logic-video", "--no-cleanup", "--output", str(self.root / "movie.mp4"), *flags]
                routes = []
                if command == "video":
                    argv.extend(["--yes", "--no-balance"])
                    routes = self.generate_routes()
                with self.subTest(command=command, configured=configured, flags=flags), \
                        mock.patch.object(VeniceClient, "poll_retrieve", autospec=True,
                                          return_value=("video/mp4", b"fixture")) as poll:
                    rc, _, err, _ = self.run_http(argv, routes)
                self.assertEqual(rc, 0)
                self.assertEqual((poll.call_args.kwargs["interval"], poll.call_args.kwargs["max_wait"]), expected)
                self.assertEqual(poll.call_args.args[1:], (
                    "/video/retrieve", {"model": "logic-video", "queue_id": "job-logic-123" if command == "video" else "saved-job"},
                ))
                self.assertTrue(callable(poll.call_args.kwargs["on_tick"]))
                if flags and flags[0] == "--poll-interval=-1":
                    self.assertIn(f"{command}: --poll-interval must be >= 0; using 5\n", err)
                    self.assertIn(f"{command}: --max-wait must be >= 0; using 900\n", err)

    def test_status_uses_cached_download_unless_legacy_url_is_explicit(self):
        self.configuration["defaults"] = {"video": {"model": "wrong-model", "no_cleanup": True}}
        _video_jobs.remember("stored-job", "logic-video", "https://example.test/cached")

        def download(client, value, directory, **kwargs):
            self.assertEqual(directory, self.root)
            path = directory / "download.tmp"
            path.write_bytes(b"downloaded-fixture")
            return "video/mp4", path, 18

        for explicit, expected in (([], "https://example.test/cached"),
                                   (["--download-url", "https://example.test/legacy"], "https://example.test/legacy")):
            # A successful save forgets the binding, so restore it for each case.
            _video_jobs.remember("stored-job", "logic-video", "https://example.test/cached")
            with self.subTest(explicit=explicit), \
                    mock.patch.object(VeniceClient, "download_url_to_temp", autospec=True,
                                      side_effect=download) as fetch:
                rc, out, _, seen = self.run_http([
                    "video-status", "stored-job", "--model", "logic-video",
                    "--output", str(self.root / "saved.mp4"), *explicit,
                ], [("/video/retrieve", {"status": "COMPLETED"})])
            self.assertEqual(rc, 0)
            self.assertEqual(Path(out.strip()).read_bytes(), b"downloaded-fixture")
            self.assertEqual(fetch.call_args.args[1:], (expected, self.root))
            self.assertEqual(seen, [("POST", "/video/retrieve", "", {"model": "logic-video", "queue_id": "stored-job"})])

    def test_status_without_model_resolves_catalog_default(self):
        rc, out, _, seen = self.run_http([
            "video-status", "saved-job", "--no-cleanup", "--output", str(self.root / "saved.mp4"),
        ], [("/models", catalog()), ("/video/retrieve", FakeResp(200, b"saved", "video/mp4"))])
        self.assertEqual(rc, 0)
        self.assertEqual(Path(out.strip()).read_bytes(), b"saved")
        self.assertEqual(seen, [("GET", "/models", "type=video", None),
                                ("POST", "/video/retrieve", "", {"model": "logic-video", "queue_id": "saved-job"})])

    def test_background_download_metadata_is_bound_to_returned_id(self):
        rc, out, _, _ = self.run_http(["video", "p", "--background", "--no-balance"],
                                    self.generate_routes(queued={"id": "bg-saved", "download_url": "https://example.test/clip"}))
        self.assertEqual((rc, out), (0, "bg-saved\n"))
        self.assertEqual(_video_jobs.lookup("bg-saved", "logic-video"), "https://example.test/clip")
        self.assertIsNone(_video_jobs.lookup("bg-saved", "other-model"))


class VideoRetrieveLogicTests(VideoFixtureCase):
    def client(self, payload=b"video", ctype="video/mp4"):
        client = mock.create_autospec(VeniceClient, instance=True)
        client.poll_retrieve.return_value = (ctype, payload)
        return client

    def test_retrieve_forwards_every_poll_option_and_preserves_bytes(self):
        tick = mock.Mock()
        for payload in (b"fixture", bytearray(b"fixture"), b""):
            client = self.client(payload, "video/webm")
            result = video.retrieve_bytes(client, "model-7", "job-8", poll_interval=0.125,
                                          max_wait=23.5, on_tick=tick)
            self.assertEqual(result, ("video/webm", bytes(payload), len(payload)))
            self.assertIs(type(result[1]), bytes)
            client.poll_retrieve.assert_called_once_with(
                "/video/retrieve", {"model": "model-7", "queue_id": "job-8"},
                interval=0.125, max_wait=23.5, on_tick=tick, terminal_statuses=("COMPLETED",),
            )
            client.download_url_to_temp.assert_not_called()

    def test_vps_retrieve_requires_url_directory_and_returns_staged_payload(self):
        client = self.client({"status": "COMPLETED"}, "application/json")
        with self.assertRaises(video.NoVideoStream):
            video.retrieve_bytes(client, "m", "q", poll_interval=0, max_wait=1)
        with self.assertRaises(ValueError) as caught:
            video.retrieve_bytes(client, "m", "q", poll_interval=0, max_wait=1,
                                 download_url="https://example.test/clip")
        self.assertEqual(str(caught.exception), "download_dir is required for a VPS video download")
        client.download_url_to_temp.assert_not_called()
        expected = ("video/mp4", self.root / "staged", 37)
        client.download_url_to_temp.return_value = expected
        self.assertEqual(video.retrieve_bytes(
            client, "m", "q", poll_interval=0, max_wait=1,
            download_url="https://example.test/clip", download_dir=str(self.root)), expected)
        client.download_url_to_temp.assert_called_once_with("https://example.test/clip", self.root)

    def save(self, client, *, output=None, no_cleanup=True, download_url=None):
        with contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()) as err:
            rc = video._retrieve_and_save(client, "model-7", "job-12345678", download_url,
                                          self.root if output is None else output, 0.125, 23.5, no_cleanup)
        return rc, out.getvalue(), err.getvalue()

    def test_save_chooses_extension_prints_absolute_path_and_progress(self):
        for ctype, ext, warning in (("video/mp4", ".mp4", ""), ("video/webm", ".webm", ""),
                                    ("video/quicktime", ".mov", ""),
                                    ("application/unknown", ".mp4", "warning: unexpected content-type 'application/unknown'; saving as .mp4\n")):
            client = self.client()

            def poll(*args, response_type=ctype, **kwargs):
                self.assertEqual(args, ("/video/retrieve", {"model": "model-7", "queue_id": "job-12345678"}))
                self.assertEqual(kwargs["interval"], 0.125)
                self.assertEqual(kwargs["max_wait"], 23.5)
                self.assertEqual(kwargs["terminal_statuses"], ("COMPLETED",))
                kwargs["on_tick"]({})
                return response_type, b"fixture-bytes"

            client.poll_retrieve.side_effect = poll
            with self.subTest(ctype=ctype), mock.patch("time.monotonic", return_value=100):
                rc, out, err = self.save(client)
            path = self.root / ("venice-video-job-1234" + ext)
            self.assertEqual((rc, out), (0, str(path.resolve()) + "\n"))
            self.assertEqual(path.read_bytes(), b"fixture-bytes")
            self.assertEqual(err, "\r[wall   0.0s] processing...\n" + warning
                             + f"wrote 13 bytes to {path.resolve()}\n")
            client.post_json.assert_not_called()

    def test_staged_download_uses_output_directory_and_is_moved(self):
        client = self.client({"status": "COMPLETED"}, "application/json")
        staged = self.root / "staged.tmp"
        staged.write_bytes(b"staged-video")
        client.download_url_to_temp.return_value = ("video/webm", staged, 12)
        rc, out, err = self.save(client, download_url="https://example.test/movie")
        path = self.root / "venice-video-job-1234.webm"
        self.assertEqual((rc, out), (0, str(path.resolve()) + "\n"))
        self.assertEqual(path.read_bytes(), b"staged-video")
        self.assertFalse(staged.exists())
        self.assertEqual(err, f"\nwrote 12 bytes to {path.resolve()}\n")
        client.download_url_to_temp.assert_called_once_with("https://example.test/movie", self.root)

    def test_cleanup_and_registry_forget_are_bound_to_the_downloaded_job(self):
        _video_jobs.remember("job-12345678", "model-7", "https://example.test/right")
        _video_jobs.remember("job-12345678", "other-model", "https://example.test/other")
        client = self.client()
        rc, _, _ = self.save(client, no_cleanup=False)
        self.assertEqual(rc, 0)
        self.assertIsNone(_video_jobs.lookup("job-12345678", "model-7"))
        self.assertEqual(_video_jobs.lookup("job-12345678", "other-model"), "https://example.test/other")
        client.post_json.assert_called_once()
        route, body = client.post_json.call_args.args
        self.assertEqual(route, "/video/complete")
        self.assertEqual(body["queue_id"], "job-12345678")
        # Exact cleanup field-set coverage lives in the independent #254 tests.

    def test_cleanup_failure_warns_after_successful_save(self):
        client = self.client()
        client.post_json.side_effect = VeniceAPIError(503, "fixture", "fixture cleanup failure")
        rc, out, err = self.save(client, no_cleanup=False)
        self.assertEqual(rc, 0)
        self.assertEqual(Path(out.strip()).read_bytes(), b"video")
        self.assertIn("warning: cleanup call failed:", err)

    def test_failed_retrieve_reports_retry_and_never_saves_or_cleans(self):
        for failure, rc, text in (
            (VeniceAPIError(429, "fixture", "fixture failure"), 4, "retrieve failed:"),
            (TimeoutError("fixture timeout"), 7, "fixture timeout; check later with:"),
            (video.NoVideoStream("job-12345678"), 2, "video: job completed but returned no video stream"),
        ):
            client = self.client()
            client.poll_retrieve.side_effect = failure
            with self.subTest(failure=type(failure).__name__):
                actual, out, err = self.save(client, no_cleanup=False)
                self.assertEqual((actual, out), (rc, ""))
                self.assertTrue(err.startswith("\n"))
                self.assertIn(text, err)
                self.assertIn("venice video-status job-12345678 --model model-7", err)
                self.assertEqual(list(self.root.glob("venice-video-*")), [])
                client.post_json.assert_not_called()

    def test_write_failure_removes_staged_file_and_keeps_registry(self):
        _video_jobs.remember("job-12345678", "model-7", "https://example.test/right")
        client = self.client({"status": "COMPLETED"}, "application/json")
        staged = self.root / "stage.tmp"
        staged.write_bytes(b"video")
        client.download_url_to_temp.return_value = ("video/mp4", staged, 5)
        rc, out, err = self.save(client, output=self.root / "missing" / "movie.mp4",
                                 no_cleanup=False, download_url="https://example.test/right")
        self.assertEqual((rc, out), (9, ""))
        self.assertFalse(staged.exists())
        self.assertIn("could not write", err)
        self.assertIn("retry with: venice video-status job-12345678 --model model-7", err)
        self.assertEqual(_video_jobs.lookup("job-12345678", "model-7"), "https://example.test/right")
        client.post_json.assert_not_called()
