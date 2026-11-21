"""Behavioral coverage for video argument, request, media, and recovery paths.

The HTTP fixture rejects every unexpected request. Video functions run unchanged;
external dependencies are replaced at their boundaries.
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
from unittest import mock
from urllib.error import HTTPError
from urllib.parse import urlsplit

from tests.test_client import FakeResp
from venice import config
from venice.client import VeniceAPIError
from venice.commands import _shared, _video_jobs, video

MODEL = "studio-video"
JOB = "queued-123456789"
CLIP = b"fixture video bytes"
CATALOG = {
    "data": [
        {
            "id": MODEL,
            "model_spec": {
                "traits": ["default"],
                "constraints": {
                    "durations": ["5s", "12s"],
                    "resolutions": ["720p", "1080p"],
                    "aspect_ratios": ["16:9", "4:3"],
                },
            },
        }
    ],
}


class VideoFixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        self.enterContext(
            mock.patch.dict(
                os.environ,
                {
                    "HOME": str(self.root),
                    "VENICE_API_KEY": "test-fake-key",
                    config.ENV_VIDEO_JOBS_FILE: str(self.root / "jobs.json"),
                },
            )
        )
        self.settings = {"version": 1, "mcpServers": {}, "defaults": {}}
        self.enterContext(
            mock.patch(
                "venice.userconfig.load_config",
                side_effect=lambda: self.settings,
            )
        )
        self.stdout = self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.parser = argparse.ArgumentParser()
        commands = self.parser.add_subparsers(dest="command")
        video.register(commands)
        video.register_status(commands)

    # enterContext was added to TestCase in 3.11; keep the test module usable on 3.9.
    def enterContext(self, manager):
        value = manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)
        return value

    def args(self, *flags):
        return self.parser.parse_args(["video", "clouds over a lake", *flags])

    def local_file(self, name="frame.fixture", data=b"frame"):
        path = self.root / name
        path.write_bytes(data)
        return path

    def http(self, *steps):
        """Each step is (method, path including query, response or exception)."""
        pending = list(steps)
        calls = []

        def respond(request, **_kwargs):
            parsed = urlsplit(request.full_url)
            path = parsed.path.removeprefix("/api/v1")
            if parsed.query:
                path += "?" + parsed.query
            method = request.get_method()
            body = json.loads(request.data) if request.data else None
            calls.append((method, path, body))
            self.assertTrue(pending, f"unexpected request: {method} {path}")
            expected_method, expected_path, response = pending.pop(0)
            self.assertEqual((method, path), (expected_method, expected_path))
            if isinstance(response, Exception):
                raise response
            if isinstance(response, bytes):
                return FakeResp(200, response, "video/mp4")
            return FakeResp(200, json.dumps(response).encode())

        self.enterContext(mock.patch("venice.client.urllib.request.urlopen", respond))
        self.addCleanup(self.assertEqual, pending, [], "unused HTTP fixture steps")
        return calls

    def generate_steps(self, *following, quote=None, queued=None):
        return self.http(
            ("GET", "/models?type=video", CATALOG),
            ("POST", "/video/quote", {"quote": 0.375} if quote is None else quote),
            ("POST", "/video/queue", {"queue_id": JOB} if queued is None else queued),
            *following,
        )


class ParserBehavior(VideoFixture):
    def test_generation_defaults_and_handler(self):
        args = self.args()
        expected_none = (
            "model",
            "duration",
            "resolution",
            "aspect_ratio",
            "negative_prompt",
            "no_audio",
            "image",
            "image_strength",
            "end_image",
            "video",
            "audio_input",
            "reference_image",
            "reference_video",
            "reference_audio",
            "scene_image",
            "reference_video_duration",
            "element",
            "output",
            "yes",
            "no_cleanup",
            "max_spend",
            "no_balance",
            "poll_interval",
            "max_wait",
        )
        for name in expected_none:
            with self.subTest(name=name):
                self.assertIsNone(getattr(args, name))
        self.assertIs(args.handler, video._run_generate)
        self.assertIs(args.background, False)
        self.assertIs(args.dry_run, False)
        self.assertIsNone(self.parser.parse_args(["video"]).prompt)

    def test_scalar_flags_preserve_destinations_and_types(self):
        cases = [
            ("--model", "chosen", "model", "chosen"),
            ("--duration", "12s", "duration", "12s"),
            ("--resolution", "1080p", "resolution", "1080p"),
            ("--aspect-ratio", "4:3", "aspect_ratio", "4:3"),
            ("--negative-prompt", "flicker", "negative_prompt", "flicker"),
            ("--image", "start.png", "image", "start.png"),
            ("--end-image", "end.png", "end_image", "end.png"),
            ("--video", "input.mov", "video", "input.mov"),
            ("--audio", "music.wav", "audio_input", "music.wav"),
            ("--image-strength", "0.625", "image_strength", 0.625),
            ("--reference-video-duration", "2.75", "reference_video_duration", 2.75),
            ("--max-spend", "0.75", "max_spend", 0.75),
            ("--poll-interval", "0.125", "poll_interval", 0.125),
            ("--max-wait", "23.5", "max_wait", 23.5),
            ("--output", "movie.webm", "output", Path("movie.webm")),
            ("-o", "movie.mov", "output", Path("movie.mov")),
        ]
        for flag, value, dest, expected in cases:
            with self.subTest(flag=flag):
                actual = getattr(self.args(flag, value), dest)
                self.assertEqual(actual, expected)
                self.assertIs(type(actual), type(expected))

    def test_boolean_switches_and_aliases(self):
        for flag, dest, value in [
            ("--no-audio", "no_audio", True),
            ("--with-audio", "no_audio", False),
            ("--yes", "yes", True),
            ("-y", "yes", True),
            ("--background", "background", True),
            ("--dry-run", "dry_run", True),
            ("--no-cleanup", "no_cleanup", True),
            ("--cleanup", "no_cleanup", False),
        ]:
            with self.subTest(flag=flag):
                self.assertIs(getattr(self.args(flag), dest), value)
        with self.assertRaises(SystemExit) as raised:
            self.args("--no-audio", "--with-audio")
        self.assertEqual(raised.exception.code, 2)

    def test_repeatable_flags_accumulate_in_order(self):
        for flag, dest in [
            ("--reference-image", "reference_image"),
            ("--reference-video", "reference_video"),
            ("--reference-audio", "reference_audio"),
            ("--scene-image", "scene_image"),
            ("--element", "element"),
        ]:
            with self.subTest(flag=flag):
                self.assertEqual(
                    getattr(self.args(flag, "one", flag, "two"), dest), ["one", "two"]
                )

    def test_invalid_numeric_flags_are_parser_errors(self):
        for flag in ("--image-strength", "--reference-video-duration", "--max-spend"):
            for value in ("nan", "inf", "-inf", "nonsense"):
                with (
                    self.subTest(flag=flag, value=value),
                    self.assertRaises(SystemExit) as raised,
                ):
                    self.args(flag + "=" + value)
                self.assertEqual(raised.exception.code, 2)
        with self.assertRaises(SystemExit) as raised:
            self.args("--max-spend=-0.1")
        self.assertEqual(raised.exception.code, 2)

    def test_status_flags_and_defaults(self):
        args = self.parser.parse_args(["video-status", "job"])
        self.assertIs(args.handler, video._run_status)
        self.assertEqual(args.queue_id, "job")
        for name in (
            "model",
            "download_url",
            "output",
            "no_cleanup",
            "poll_interval",
            "max_wait",
        ):
            self.assertIsNone(getattr(args, name))
        for alias in ("--output", "-o"):
            args = self.parser.parse_args(
                [
                    "video-status",
                    "job",
                    "--model",
                    "other",
                    "--download-url",
                    "https://cdn.test/v",
                    alias,
                    "saved.mp4",
                    "--no-cleanup",
                    "--poll-interval",
                    "0.25",
                    "--max-wait",
                    "19",
                ]
            )
            self.assertEqual(
                (
                    args.model,
                    args.download_url,
                    args.output,
                    args.no_cleanup,
                    args.poll_interval,
                    args.max_wait,
                ),
                ("other", "https://cdn.test/v", Path("saved.mp4"), True, 0.25, 19.0),
            )
        with self.assertRaises(SystemExit) as raised:
            self.parser.parse_args(["video-status"])
        self.assertEqual(raised.exception.code, 2)


class RequestBehavior(VideoFixture):
    def test_shared_fields_are_exact_and_do_not_enable_audio_implicitly(self):
        for flag, expected in [
            ((), {"duration": "12s"}),
            (("--no-audio",), {"duration": "12s", "audio": False}),
            (("--with-audio",), {"duration": "12s"}),
        ]:
            with self.subTest(flag=flag):
                args = self.args("--duration", "12s", *flag)
                self.assertEqual(video._shared_params(args), expected)
        self.assertEqual(
            video._shared_params(
                self.args(
                    "--duration", "5s", "--resolution", "1080p", "--aspect-ratio", "4:3"
                )
            ),
            {"duration": "5s", "resolution": "1080p", "aspect_ratio": "4:3"},
        )

    def test_quote_queue_and_job_bodies_have_exact_fields(self):
        shared = {"duration": "12s", "resolution": "1080p", "audio": False}
        media = {"video_url": "https://media.test/v"}
        self.assertEqual(
            video.quote_body("m", shared, media),
            {
                "model": "m",
                "duration": "12s",
                "resolution": "1080p",
                "audio": False,
                "video_url": "https://media.test/v",
            },
        )
        for strength in (0, 0.625, 1):
            with self.subTest(strength=strength):
                self.assertEqual(
                    video.queue_body("m", "p", shared, media, "noise", strength),
                    {
                        "model": "m",
                        "prompt": "p",
                        "duration": "12s",
                        "resolution": "1080p",
                        "audio": False,
                        "video_url": "https://media.test/v",
                        "negative_prompt": "noise",
                        "image_strength": float(strength),
                    },
                )
        self.assertEqual(
            video.queue_body("m", "p", {}, {}, ""), {"model": "m", "prompt": "p"}
        )
        self.assertEqual(
            shared, {"duration": "12s", "resolution": "1080p", "audio": False}
        )
        self.assertEqual(media, {"video_url": "https://media.test/v"})
        self.assertEqual(video.job_body("m", "q"), {"model": "m", "queue_id": "q"})

    def test_queue_validation_boundaries_and_messages(self):
        video.validate_queue_options("x" * 32000, "y" * 32000, 0)
        video.validate_queue_options("p", "", 1)
        for prompt in (None, 2, "", " \t\n"):
            with (
                self.subTest(prompt=prompt),
                self.assertRaisesRegex(ValueError, "^prompt is required$"),
            ):
                video.validate_queue_options(prompt)
        cases = [
            (("x" * 32001,), "^prompt exceeds 32000 chars$"),
            (("p", 12), "^negative_prompt must be a string$"),
            (("p", "x" * 32001), "^negative_prompt exceeds 32000 chars$"),
            (("p", None, -0.001), "^image_strength must be between 0 and 1$"),
            (("p", None, 1.001), "^image_strength must be between 0 and 1$"),
        ]
        for args, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                video.validate_queue_options(*args)
        for strength in (float("nan"), float("inf"), "not numeric"):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                video.queue_body("m", "p", {}, {}, image_strength=strength)

    def test_constraint_list_shape_and_order(self):
        self.assertEqual(
            video._constraint_values({"durations": ["12s", "5s"]}, "durations", "m"),
            ("12s", "5s"),
        )
        self.assertEqual(
            video._constraint_values({"durations": []}, "durations", "m"), ()
        )
        for values in (None, "5s", ("5s",), ["5s", ""], ["5s", "  "], [3], [None]):
            with (
                self.subTest(values=values),
                self.assertRaisesRegex(
                    ValueError,
                    "^live video catalog entry for 'm' has invalid constraints.durations$",
                ),
            ):
                video._constraint_values({"durations": values}, "durations", "m")

    def test_catalog_validation_requires_selected_entry_and_all_constraints(self):
        selected = CATALOG["data"]
        video.validate_model_options(
            [None, {"id": "other"}, *selected],
            MODEL,
            duration="12s",
            resolution="1080p",
            aspect_ratio="4:3",
        )
        video.validate_model_options(selected, MODEL, duration="5s")
        for models, message in [
            (None, "could not fetch the live video catalog"),
            ([], "is not in the live video catalog"),
            ([{"id": MODEL}], "has no constraints object"),
            ([{"id": MODEL, "model_spec": []}], "has no constraints object"),
            (
                [{"id": MODEL, "model_spec": {"constraints": []}}],
                "has no constraints object",
            ),
        ]:
            with (
                self.subTest(models=models),
                self.assertRaisesRegex(ValueError, message),
            ):
                video.validate_model_options(models, MODEL, duration="5s")
        for field in ("durations", "resolutions", "aspect_ratios"):
            models = json.loads(json.dumps(selected))
            models[0]["model_spec"]["constraints"][field] = ["valid", 1]
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "constraints." + field),
            ):
                video.validate_model_options(models, MODEL, duration="5s")

    def test_catalog_choice_and_missing_value_errors(self):
        for field, label in [
            ("duration", "duration"),
            ("resolution", "resolution"),
            ("aspect_ratio", "aspect ratio"),
        ]:
            for value in ("", " ", 12):
                kwargs = {"duration": "5s", field: value}
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaisesRegex(
                        ValueError,
                        "^" + label + " must be a non-empty string$",
                    ),
                ):
                    video.validate_model_options(CATALOG["data"], MODEL, **kwargs)
            kwargs = {"duration": "5s", field: "unsupported"}
            choices = {
                "duration": "5s, 12s",
                "resolution": "720p, 1080p",
                "aspect_ratio": "16:9, 4:3",
            }
            with self.assertRaises(ValueError) as raised:
                video.validate_model_options(CATALOG["data"], MODEL, **kwargs)
            self.assertEqual(
                str(raised.exception),
                f"{label} 'unsupported' is not supported by '{MODEL}'; choose from {choices[field]}",
            )
        with self.assertRaisesRegex(
            ValueError, "^duration must be a non-empty string$"
        ):
            video.validate_model_options(CATALOG["data"], MODEL, duration=None)
        empty = [
            {
                "id": MODEL,
                "model_spec": {
                    "constraints": {
                        "durations": [],
                        "resolutions": [],
                        "aspect_ratios": [],
                    }
                },
            }
        ]
        with self.assertRaisesRegex(ValueError, "advertises no duration values$"):
            video.validate_model_options(empty, MODEL, duration="5s")


class MediaBehavior(VideoFixture):
    def test_urls_pass_through_and_local_files_use_kind_fallback(self):
        for url in (
            "http://media.test/v",
            "https://media.test/v",
            "data:video/mp4;base64,AA==",
        ):
            with self.subTest(url=url):
                self.assertIs(video._is_url(url), True)
                self.assertEqual(video._resolve_media(url, kind="video"), (url, None))
        for value in ("ftp://media.test/v", "httpsomething", "file.mp4"):
            self.assertIs(video._is_url(value), False)
        for kind, mime in (
            ("image", "image/png"),
            ("video", "video/mp4"),
            ("audio", "audio/mpeg"),
        ):
            with self.subTest(kind=kind):
                path = self.local_file(kind + ".unknown-extension", b"abc")
                self.assertEqual(
                    video._resolve_media(str(path), kind=kind),
                    (f"data:{mime};base64,YWJj", None),
                )

    def test_local_media_missing_empty_and_oversize_fail(self):
        for path in (self.root / "missing", self.root, self.local_file(data=b"")):
            with self.subTest(path=path):
                self.assertEqual(
                    video._resolve_media(str(path), kind="image"), (None, 2)
                )
        self.assertIn("input file not found:", self.stderr.getvalue())
        self.assertIn("is empty", self.stderr.getvalue())
        for kind, cap in (("image", 25), ("video", 50), ("audio", 15)):
            path = self.root / (kind + ".fixture")
            with path.open("wb") as output:
                output.truncate(cap * 1024 * 1024 + 1)
            with self.subTest(kind=kind):
                self.assertEqual(video._resolve_media(str(path), kind=kind), (None, 2))
                self.assertIn(
                    f"{kind} inputs must be < {cap} MB", self.stderr.getvalue()
                )

    def test_media_size_limit_is_inclusive(self):
        # Sparse files exercise each real boundary without storing large fixtures.
        for kind, cap in (("image", 25), ("video", 50), ("audio", 15)):
            path = self.root / (kind + ".boundary")
            with path.open("wb") as output:
                output.truncate(cap * 1024 * 1024)
            with self.subTest(kind=kind):
                encoded, rc = video._resolve_media(str(path), kind=kind)
                self.assertIsNone(rc)
                self.assertEqual(
                    len(base64.b64decode(encoded.split(",", 1)[1])), cap * 1024 * 1024
                )

    def test_path_authority_mime_and_errors_reach_the_caller(self):
        path = self.local_file()
        authority = mock.Mock()
        authority.resolve.return_value = (path, "image/webp")
        self.assertEqual(
            video._resolve_media("relative", kind="image", path_authority=authority),
            ("data:image/webp;base64,ZnJhbWU=", None),
        )
        authority.resolve.assert_called_once_with(
            "relative", kind="image", max_bytes=25 * 1024 * 1024
        )
        authority.resolve.side_effect = _shared.MediaPathError(
            "outside allowed directory"
        )
        self.assertEqual(
            video._resolve_media("relative", kind="image", path_authority=authority),
            (None, 2),
        )
        self.assertIn("video: outside allowed directory", self.stderr.getvalue())

    def test_media_list_empty_order_cap_and_failure(self):
        self.assertEqual(
            video._resolve_media_list(None, kind="image", cap=2, flag="--test"),
            (None, None),
        )
        self.assertEqual(
            video._resolve_media_list([], kind="image", cap=2, flag="--test"),
            ([], None),
        )
        values = ["https://media.test/first", "http://media.test/second"]
        self.assertEqual(
            video._resolve_media_list(values, kind="image", cap=2, flag="--test"),
            (values, None),
        )
        self.assertEqual(
            video._resolve_media_list(values, kind="image", cap=1, flag="--test"),
            (None, 2),
        )
        self.assertIn("video: at most 1 --test", self.stderr.getvalue())
        self.assertEqual(
            video._resolve_media_list(["missing"], kind="image", cap=2, flag="--test"),
            (None, 2),
        )

    def test_all_media_fields_have_correct_endpoint_mapping(self):
        args = self.args()
        for dest in ("image", "end_image", "audio_input", "video"):
            setattr(args, dest, "https://media.test/" + dest)
        for dest in (
            "reference_image",
            "reference_video",
            "reference_audio",
            "scene_image",
        ):
            setattr(args, dest, ["https://media.test/" + dest])
        args.reference_video_duration = 3.25
        args.element = ['{"future": [1, 2]}']
        self.assertEqual(
            video._collect_media(args),
            (
                {
                    "video_url": "https://media.test/video",
                    "reference_video_total_duration": 3.25,
                },
                {
                    "image_url": "https://media.test/image",
                    "end_image_url": "https://media.test/end_image",
                    "audio_url": "https://media.test/audio_input",
                    "video_url": "https://media.test/video",
                    "reference_image_urls": ["https://media.test/reference_image"],
                    "reference_video_urls": ["https://media.test/reference_video"],
                    "reference_audio_urls": ["https://media.test/reference_audio"],
                    "scene_image_urls": ["https://media.test/scene_image"],
                    "elements": [{"future": [1, 2]}],
                },
                None,
            ),
        )
        self.assertEqual(video._collect_media(self.args()), ({}, {}, None))
        args = self.args("--reference-video-duration", "0")
        self.assertEqual(
            video._collect_media(args),
            ({"reference_video_total_duration": 0.0}, {}, None),
        )

    def test_array_caps_and_failure_propagation(self):
        for dest, field, cap in [
            ("reference_image", "reference_image_urls", 30),
            ("reference_video", "reference_video_urls", 10),
            ("reference_audio", "reference_audio_urls", 10),
            ("scene_image", "scene_image_urls", 4),
        ]:
            args = self.args()
            values = [f"https://media.test/{index}" for index in range(cap)]
            setattr(args, dest, values)
            with self.subTest(dest=dest):
                self.assertEqual(
                    video._collect_media(args), ({}, {field: values}, None)
                )
                setattr(args, dest, [*values, "https://media.test/extra"])
                self.assertEqual(video._collect_media(args), (None, None, 2))
                setattr(args, dest, ["missing"])
                self.assertEqual(video._collect_media(args), (None, None, 2))
        for dest in ("image", "end_image", "audio_input", "video"):
            args = self.args()
            setattr(args, dest, "missing")
            with self.subTest(dest=dest):
                self.assertEqual(video._collect_media(args), (None, None, 2))

    def test_elements_resolve_nested_media_and_preserve_unknown_fields(self):
        path = self.local_file("frame.png", b"abc")
        element = {
            "frontal_image_url": str(path),
            "video_url": "https://media.test/v",
            "reference_image_urls": [str(path), "https://media.test/ref"],
            "future": {"v": 3},
        }
        self.assertEqual(
            video._resolve_elements([json.dumps(element)]),
            (
                [
                    {
                        "frontal_image_url": "data:image/png;base64,YWJj",
                        "video_url": "https://media.test/v",
                        "reference_image_urls": [
                            "data:image/png;base64,YWJj",
                            "https://media.test/ref",
                        ],
                        "future": {"v": 3},
                    }
                ],
                None,
            ),
        )
        self.assertEqual(video._resolve_elements(None), (None, None))
        self.assertEqual(video._resolve_elements([]), ([], None))
        self.assertEqual(video._resolve_elements(["{}"] * 4), ([{}] * 4, None))
        self.assertEqual(video._resolve_elements(["{}"] * 5), (None, 2))
        refs = {"reference_image_urls": ["https://media.test/r"] * 3}
        self.assertEqual(video._resolve_elements([json.dumps(refs)]), ([refs], None))

    def test_bad_elements_fail_without_partial_media(self):
        for raw in (
            "{",
            None,
            "[]",
            "null",
            '"text"',
            '{"reference_image_urls":"bad"}',
            '{"frontal_image_url":"missing"}',
            '{"video_url":"missing"}',
            '{"reference_image_urls":["missing"]}',
            json.dumps({"reference_image_urls": ["https://media.test/r"] * 4}),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(video._resolve_elements([raw]), (None, 2))
                args = self.args()
                args.element = [raw]
                self.assertEqual(video._collect_media(args), (None, None, 2))

    def test_authority_reaches_single_array_and_nested_element_inputs(self):
        path = self.local_file(data=b"abc")
        authority = mock.Mock()
        authority.resolve.return_value = (path, "application/fixture")
        args = self.args(
            "--audio",
            "sound",
            "--reference-video",
            "clip",
            "--element",
            '{"frontal_image_url":"face","reference_image_urls":["reference"],"video_url":"motion"}',
        )
        encoded = "data:application/fixture;base64,YWJj"
        self.assertEqual(
            video._collect_media(args, path_authority=authority),
            (
                {},
                {
                    "audio_url": encoded,
                    "reference_video_urls": [encoded],
                    "elements": [
                        {
                            "frontal_image_url": encoded,
                            "reference_image_urls": [encoded],
                            "video_url": encoded,
                        }
                    ],
                },
                None,
            ),
        )
        self.assertEqual(
            authority.resolve.call_args_list,
            [
                mock.call("sound", kind="audio", max_bytes=15 * 1024 * 1024),
                mock.call("clip", kind="video", max_bytes=50 * 1024 * 1024),
                mock.call("face", kind="image", max_bytes=25 * 1024 * 1024),
                mock.call("motion", kind="video", max_bytes=50 * 1024 * 1024),
                mock.call("reference", kind="image", max_bytes=25 * 1024 * 1024),
            ],
        )


class GenerationBehavior(VideoFixture):
    def test_full_request_sequence_and_saved_output(self):
        calls = self.generate_steps(
            ("POST", "/video/retrieve", CLIP),
            ("POST", "/video/complete", {"success": True}),
        )
        args = self.args(
            "--yes",
            "--no-balance",
            "--duration",
            "12s",
            "--resolution",
            "1080p",
            "--aspect-ratio",
            "4:3",
            "--no-audio",
            "--negative-prompt",
            "blur",
            "--image-strength",
            "0",
            "--video",
            "https://media.test/input",
            "--image",
            "https://media.test/start",
            "--reference-video-duration",
            "7.5",
        )
        self.assertEqual(args.handler(args), 0)
        self.assertEqual(
            calls,
            [
                ("GET", "/models?type=video", None),
                (
                    "POST",
                    "/video/quote",
                    {
                        "model": MODEL,
                        "duration": "12s",
                        "resolution": "1080p",
                        "aspect_ratio": "4:3",
                        "audio": False,
                        "video_url": "https://media.test/input",
                        "reference_video_total_duration": 7.5,
                    },
                ),
                (
                    "POST",
                    "/video/queue",
                    {
                        "model": MODEL,
                        "prompt": "clouds over a lake",
                        "duration": "12s",
                        "resolution": "1080p",
                        "aspect_ratio": "4:3",
                        "audio": False,
                        "negative_prompt": "blur",
                        "image_strength": 0.0,
                        "image_url": "https://media.test/start",
                        "video_url": "https://media.test/input",
                    },
                ),
                ("POST", "/video/retrieve", {"model": MODEL, "queue_id": JOB}),
                ("POST", "/video/complete", {"model": MODEL, "queue_id": JOB}),
            ],
        )
        saved = self.root / "venice-video-queued-1.mp4"
        self.assertEqual(saved.read_bytes(), CLIP)
        self.assertEqual(self.stdout.getvalue(), str(saved) + "\n")
        self.assertIn(f"wrote {len(CLIP)} bytes to {saved}", self.stderr.getvalue())
        self.assertIn(f"model={MODEL}, duration=12s, 1080p", self.stderr.getvalue())

    def test_defaults_reach_requests_and_explicit_flags_win(self):
        self.settings["defaults"] = {"video": {"duration": "12s", "no_audio": True}}
        calls = self.generate_steps(queued={"queue_id": JOB})
        self.assertEqual(
            video._run_generate(self.args("--background", "--no-balance")), 0
        )
        self.assertEqual(
            calls[1][2], {"model": MODEL, "duration": "12s", "audio": False}
        )
        self.assertEqual(
            calls[2][2],
            {
                "model": MODEL,
                "prompt": "clouds over a lake",
                "duration": "12s",
                "audio": False,
            },
        )
        calls = self.generate_steps(queued={"queue_id": JOB})
        self.assertEqual(
            video._run_generate(
                self.args(
                    "--background", "--no-balance", "--with-audio", "--duration", "5s"
                )
            ),
            0,
        )
        self.assertEqual(calls[1][2], {"model": MODEL, "duration": "5s"})
        self.assertEqual(
            calls[2][2],
            {"model": MODEL, "prompt": "clouds over a lake", "duration": "5s"},
        )

    def test_dry_run_stops_after_quote_and_displays_estimate(self):
        calls = self.http(
            ("GET", "/models?type=video", CATALOG),
            ("POST", "/video/quote", {"quote": 0.375}),
        )
        self.assertEqual(video._run_generate(self.args("--dry-run", "--no-balance")), 0)
        self.assertEqual(len(calls), 2)
        self.assertIn("Estimated cost:", self.stderr.getvalue())
        self.assertIn("0.375", self.stderr.getvalue())
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertEqual(list(self.root.glob("*.mp4")), [])

    def test_background_falls_back_to_id_and_prints_recovery_command(self):
        calls = self.generate_steps(queued={"queue_id": "", "id": "fallback-job"})
        self.assertEqual(
            video._run_generate(self.args("--background", "--no-balance")), 0
        )
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.stdout.getvalue(), "fallback-job\n")
        self.assertIn(
            f"queued as fallback-job; fetch with: venice video-status fallback-job --model {MODEL}\n",
            self.stderr.getvalue(),
        )

    def test_primary_queue_id_wins_and_missing_ids_fail(self):
        self.generate_steps(queued={"queue_id": "primary", "id": "secondary"})
        self.assertEqual(
            video._run_generate(self.args("--background", "--no-balance")), 0
        )
        self.assertEqual(self.stdout.getvalue(), "primary\n")
        self.generate_steps(queued={})
        self.assertEqual(
            video._run_generate(self.args("--background", "--no-balance")), 5
        )
        self.assertIn("queue response missing queue_id:", self.stderr.getvalue())

    def test_bad_prompt_stops_before_any_http(self):
        calls = self.http()
        for prompt in (None, "", "  ", "x" * 32001):
            args = self.args()
            args.prompt = prompt
            with self.subTest(prompt_length=len(prompt or "")):
                self.assertEqual(video._run_generate(args), 2)
        self.assertEqual(calls, [])

    def test_invalid_media_stops_before_quote(self):
        self.http(("GET", "/models?type=video", CATALOG))
        self.assertEqual(video._run_generate(self.args("--image", "missing")), 2)

    def test_unknown_model_and_bad_catalog_choices_stop_before_quote(self):
        for flags, expected in [
            (("--model", "missing"), 6),
            (("--duration", "99s"), 2),
        ]:
            self.http(("GET", "/models?type=video", CATALOG))
            with self.subTest(flags=flags):
                self.assertEqual(video._run_generate(self.args(*flags)), expected)

    def test_bad_quotes_and_budget_stop_before_queue(self):
        for quote, flags, expected in [
            ({"quote": "invalid"}, (), 2),
            ({"quote": -1}, (), 2),
            ({"quote": 0.375}, ("--max-spend", "0.374"), 1),
        ]:
            self.http(
                ("GET", "/models?type=video", CATALOG), ("POST", "/video/quote", quote)
            )
            with self.subTest(quote=quote, flags=flags):
                self.assertEqual(
                    video._run_generate(self.args("--yes", "--no-balance", *flags)),
                    expected,
                )
        self.generate_steps()
        self.assertEqual(
            video._run_generate(
                self.args("--background", "--no-balance", "--max-spend", "0.375")
            ),
            0,
        )

    def test_confirmation_decline_stops_before_queue(self):
        self.http(
            ("GET", "/models?type=video", CATALOG),
            ("POST", "/video/quote", {"quote": 0.375}),
        )
        self.enterContext(mock.patch("sys.stdin.isatty", return_value=False))
        self.assertEqual(video._run_generate(self.args("--no-balance")), 1)
        self.assertIn("pass --yes", self.stderr.getvalue())

    def test_quote_and_queue_http_errors_have_correct_exit_codes(self):
        for endpoint in ("quote", "queue"):
            for status, expected in ((422, 3), (429, 4), (503, 5), (404, 6), (400, 2)):
                error = HTTPError(
                    "https://api.test/" + endpoint,
                    status,
                    "rejected",
                    {},
                    io.BytesIO(b"bad request"),
                )
                steps = [("GET", "/models?type=video", CATALOG)]
                if endpoint == "queue":
                    steps.append(("POST", "/video/quote", {"quote": 0.375}))
                self.http(*steps, ("POST", "/video/" + endpoint, error))
                with self.subTest(endpoint=endpoint, status=status):
                    self.assertEqual(
                        video._run_generate(self.args("--yes", "--no-balance")),
                        expected,
                    )

    def test_background_download_metadata_is_available_for_status(self):
        url = "https://cdn.test/video?fixture=123"
        self.generate_steps(queued={"queue_id": JOB, "download_url": url})
        self.assertEqual(
            video._run_generate(self.args("--background", "--no-balance")), 0
        )
        self.assertEqual(_video_jobs.lookup(JOB, MODEL), url)
        self.assertEqual(self.stdout.getvalue(), JOB + "\n")
        self.assertNotIn(url, self.stdout.getvalue() + self.stderr.getvalue())

    def test_background_store_failure_reports_queued_job(self):
        # A directory at the configured store path makes persistence fail naturally.
        (self.root / "jobs.json").mkdir()
        self.generate_steps(
            queued={"queue_id": JOB, "download_url": "https://cdn.test/video"}
        )
        self.assertEqual(
            video._run_generate(self.args("--background", "--no-balance")), 9
        )
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertIn(
            f"queued as {JOB}, but could not save private retrieval metadata:",
            self.stderr.getvalue(),
        )
        self.assertIn(
            f"venice video-status {JOB} --model {MODEL}", self.stderr.getvalue()
        )

    def test_balance_display_uses_quote_value(self):
        self.http(
            ("GET", "/models?type=video", CATALOG),
            ("POST", "/video/quote", {"quote": 0.375}),
        )
        # The billing function is an external dependency of the video handler.
        # Capture the requested display decision without replacing video logic.
        with mock.patch(
            "venice.billing.fetch_balance", return_value={"total": 2.0}
        ) as balance:
            self.assertEqual(video._run_generate(self.args("--dry-run")), 0)
        balance.assert_called_once()
        self.assertIn("After charge:", self.stderr.getvalue())
        self.assertIn("After charge:   $1.62 USD\n", self.stderr.getvalue())


class RetrievalBehavior(VideoFixture):
    def client(self, result=("video/mp4", CLIP)):
        client = mock.Mock(spec=["poll_retrieve", "download_url_to_temp", "post_json"])
        client.poll_retrieve.return_value = result
        return client

    def test_retrieve_forwards_identity_cadence_callback_and_terminal_status(self):
        client = self.client(("video/webm", bytearray(CLIP)))
        tick = mock.Mock()
        self.assertEqual(
            video.retrieve_bytes(
                client, "m", "q", poll_interval=0.375, max_wait=17, on_tick=tick
            ),
            ("video/webm", CLIP, len(CLIP)),
        )
        client.poll_retrieve.assert_called_once_with(
            "/video/retrieve",
            {"model": "m", "queue_id": "q"},
            interval=0.375,
            max_wait=17,
            on_tick=tick,
            terminal_statuses=("COMPLETED",),
        )
        client.download_url_to_temp.assert_not_called()

    def test_retrieve_completed_uses_staging_directory_or_reports_missing_metadata(
        self,
    ):
        client = self.client(("application/json", {"status": "COMPLETED"}))
        with self.assertRaises(video.NoVideoStream) as raised:
            video.retrieve_bytes(client, "m", "q", poll_interval=0, max_wait=0)
        self.assertEqual(raised.exception.args, ("q",))
        with self.assertRaisesRegex(
            ValueError, "^download_dir is required for a VPS video download$"
        ):
            video.retrieve_bytes(
                client,
                "m",
                "q",
                poll_interval=0,
                max_wait=0,
                download_url="https://cdn.test/v",
            )
        staged = self.local_file("staging", CLIP)
        client.download_url_to_temp.return_value = ("video/mp4", staged, len(CLIP))
        self.assertEqual(
            video.retrieve_bytes(
                client,
                "m",
                "q",
                poll_interval=0,
                max_wait=0,
                download_url="https://cdn.test/v",
                download_dir=str(self.root),
            ),
            ("video/mp4", staged, len(CLIP)),
        )
        client.download_url_to_temp.assert_called_once_with(
            "https://cdn.test/v", self.root
        )

    def test_save_content_types_default_names_and_cleanup(self):
        for ctype, extension, unknown in [
            ("video/mp4", ".mp4", False),
            ("video/webm", ".webm", False),
            ("video/quicktime", ".mov", False),
            ("other/type", ".mp4", True),
        ]:
            client = self.client((ctype, CLIP))
            with self.subTest(ctype=ctype):
                self.assertEqual(
                    video._retrieve_and_save(
                        client, "m", JOB, None, self.root, 0.25, 9, False
                    ),
                    0,
                )
                self.assertEqual(
                    (self.root / ("venice-video-queued-1" + extension)).read_bytes(),
                    CLIP,
                )
                client.post_json.assert_called_once_with(
                    "/video/complete", {"model": "m", "queue_id": JOB}
                )
                self.assertTrue(self.stderr.getvalue().startswith("\nwrote "))
                if unknown:
                    self.assertIn(
                        "unexpected content-type 'other/type'; saving as .mp4",
                        self.stderr.getvalue(),
                    )

    def test_save_moves_downloaded_file_and_honors_explicit_output_and_no_cleanup(self):
        staged = self.local_file("download.tmp", CLIP)
        client = self.client(("application/json", {"status": "COMPLETED"}))
        client.download_url_to_temp.return_value = ("video/webm", staged, len(CLIP))
        output = self.root / "chosen.bin"
        _video_jobs.remember(JOB, "m", "https://cdn.test/v")
        self.assertEqual(
            video._retrieve_and_save(
                client, "m", JOB, "https://cdn.test/v", output, 0.25, 9, True
            ),
            0,
        )
        self.assertEqual(output.read_bytes(), CLIP)
        self.assertFalse(staged.exists())
        self.assertIsNone(_video_jobs.lookup(JOB, "m"))
        self.assertEqual(self.stdout.getvalue(), str(output) + "\n")
        client.post_json.assert_not_called()

    def test_save_errors_preserve_job_and_remove_staged_file(self):
        for staged_download in (False, True):
            _video_jobs.remember(JOB, "m", "https://cdn.test/v")
            client = self.client()
            if staged_download:
                staged = self.local_file("download.tmp", CLIP)
                client.poll_retrieve.return_value = (
                    "application/json",
                    {"status": "COMPLETED"},
                )
                client.download_url_to_temp.return_value = (
                    "video/mp4",
                    staged,
                    len(CLIP),
                )
            with self.subTest(staged=staged_download):
                self.assertEqual(
                    video._retrieve_and_save(
                        client,
                        "m",
                        JOB,
                        "https://cdn.test/v",
                        self.root / "missing" / "out.mp4",
                        0,
                        1,
                        False,
                    ),
                    9,
                )
                self.assertEqual(_video_jobs.lookup(JOB, "m"), "https://cdn.test/v")
                client.post_json.assert_not_called()
                if staged_download:
                    self.assertFalse(staged.exists())
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertIn("retry with: venice video-status", self.stderr.getvalue())

    def test_retrieve_errors_and_timeouts_preserve_job_and_recovery_hint(self):
        for error, expected in [
            (VeniceAPIError(422, "/video/retrieve", "invalid"), 3),
            (VeniceAPIError(429, "/video/retrieve", "busy"), 4),
            (VeniceAPIError(503, "/video/retrieve", "down"), 5),
            (VeniceAPIError(404, "/video/retrieve", "gone"), 6),
            (VeniceAPIError(0, "/video/retrieve", "broken"), 8),
            (TimeoutError("timed out"), 7),
        ]:
            client = self.client()
            client.poll_retrieve.side_effect = error
            _video_jobs.remember(JOB, "m", "https://cdn.test/v")
            previous_output = len(self.stderr.getvalue())
            with self.subTest(error=type(error).__name__, expected=expected):
                self.assertEqual(
                    video._retrieve_and_save(client, "m", JOB, None, None, 0, 1, False),
                    expected,
                )
                self.assertEqual(_video_jobs.lookup(JOB, "m"), "https://cdn.test/v")
                client.post_json.assert_not_called()
                self.assertTrue(
                    self.stderr.getvalue()[previous_output:].startswith("\n")
                )
        self.assertIn(
            f"check later with: venice video-status {JOB} --model m",
            self.stderr.getvalue(),
        )
        self.assertEqual(self.stdout.getvalue(), "")

    def test_completed_without_stream_gives_recovery_hint(self):
        client = self.client(("application/json", {"status": "COMPLETED"}))
        self.assertEqual(
            video._retrieve_and_save(client, "m", JOB, None, None, 0, 1, False), 2
        )
        self.assertIn("--download-url <url>", self.stderr.getvalue())
        client.post_json.assert_not_called()
        self.assertEqual(self.stdout.getvalue(), "")

    def test_cleanup_failure_warns_after_successful_save(self):
        client = self.client()
        client.post_json.side_effect = VeniceAPIError(
            503, "/video/complete", "unavailable"
        )
        self.assertEqual(
            video._retrieve_and_save(client, "m", JOB, None, None, 0, 1, False), 0
        )
        self.assertEqual((self.root / "venice-video-queued-1.mp4").read_bytes(), CLIP)
        self.assertIn("warning: cleanup call failed:", self.stderr.getvalue())

    def test_progress_callback_uses_elapsed_time(self):
        client = self.client()

        def poll(*_args, **kwargs):
            kwargs["on_tick"](
                {"average_execution_time": 12000, "execution_duration": 4000}
            )
            return "video/mp4", CLIP

        client.poll_retrieve.side_effect = poll
        with mock.patch("venice.commands.video.time.monotonic", side_effect=[100, 103]):
            self.assertEqual(
                video._retrieve_and_save(client, "m", JOB, None, None, 0, 1, True), 0
            )
        self.assertIn(
            "[wall   3.0s | server   4.0s / ~ 12.0s] processing...",
            self.stderr.getvalue(),
        )


class StatusBehavior(VideoFixture):
    def test_explicit_model_avoids_catalog_and_config_model(self):
        self.settings["defaults"] = {"video": {"model": "wrong", "no_cleanup": True}}
        calls = self.http(("POST", "/video/retrieve", CLIP))
        args = self.parser.parse_args(
            ["video-status", JOB, "--model", "queued-model", "-o", "saved.mp4"]
        )
        self.assertEqual(args.handler(args), 0)
        self.assertEqual(
            calls,
            [("POST", "/video/retrieve", {"model": "queued-model", "queue_id": JOB})],
        )
        self.assertEqual((self.root / "saved.mp4").read_bytes(), CLIP)

    def test_status_without_model_uses_catalog_instead_of_config(self):
        self.settings["defaults"] = {"video": {"model": "wrong", "no_cleanup": True}}
        calls = self.http(
            ("GET", "/models?type=video", CATALOG), ("POST", "/video/retrieve", CLIP)
        )
        args = self.parser.parse_args(["video-status", JOB])
        self.assertEqual(video._run_status(args), 0)
        self.assertEqual(calls[1][2], {"model": MODEL, "queue_id": JOB})

    def test_status_unknown_default_catalog_stops_before_retrieve(self):
        self.http(("GET", "/models?type=video", {"data": []}))
        self.assertEqual(
            video._run_status(self.parser.parse_args(["video-status", JOB])), 6
        )

    def test_status_cadence_reaches_real_poll_loop(self):
        self.settings["defaults"] = {"video": {"poll_interval": 2.5, "max_wait": 40}}
        calls = self.http(
            ("POST", "/video/retrieve", {"status": "PROCESSING"}),
            ("POST", "/video/retrieve", CLIP),
        )
        with mock.patch("venice.client.time.sleep") as sleep:
            args = self.parser.parse_args(
                ["video-status", JOB, "--model", MODEL, "--no-cleanup"]
            )
            self.assertEqual(video._run_status(args), 0)
        sleep.assert_called_once_with(2.5)
        self.assertEqual(len(calls), 2)

    def test_status_zero_wait_overrides_config_and_times_out(self):
        self.settings["defaults"] = {"video": {"poll_interval": 2.5, "max_wait": 40}}
        self.http(("POST", "/video/retrieve", {"status": "PROCESSING"}))
        args = self.parser.parse_args(
            ["video-status", JOB, "--model", MODEL, "--max-wait", "0"]
        )
        self.assertEqual(video._run_status(args), 7)
        self.assertIn("not ready after 0.0s", self.stderr.getvalue())

    def test_status_explicit_poll_interval_overrides_config(self):
        self.settings["defaults"] = {"video": {"poll_interval": 2.5}}
        self.http(
            ("POST", "/video/retrieve", {"status": "PROCESSING"}),
            ("POST", "/video/retrieve", CLIP),
        )
        with mock.patch("venice.client.time.sleep") as sleep:
            args = self.parser.parse_args(
                [
                    "video-status",
                    JOB,
                    "--model",
                    MODEL,
                    "--no-cleanup",
                    "--poll-interval",
                    "0.125",
                ]
            )
            self.assertEqual(video._run_status(args), 0)
        sleep.assert_called_once_with(0.125)

    def test_status_uses_registered_download_url_and_clears_saved_metadata(self):
        url = "https://cdn.test/clip"
        _video_jobs.remember(JOB, MODEL, url)
        self.http(("POST", "/video/retrieve", {"status": "COMPLETED"}))
        downloads = []

        class Opener:
            def open(self, request, **_kwargs):
                downloads.append(request.full_url)
                return FakeResp(200, CLIP, "video/webm")

        with mock.patch(
            "venice.client._egress.build_https_opener", return_value=Opener()
        ):
            args = self.parser.parse_args(
                ["video-status", JOB, "--model", MODEL, "--no-cleanup"]
            )
            self.assertEqual(video._run_status(args), 0)
        self.assertEqual(downloads, [url])
        self.assertEqual((self.root / "venice-video-queued-1.webm").read_bytes(), CLIP)
        self.assertIsNone(_video_jobs.lookup(JOB, MODEL))

    def test_status_explicit_download_url_bypasses_bad_registry(self):
        (self.root / "jobs.json").mkdir()
        self.http(("POST", "/video/retrieve", {"status": "COMPLETED"}))
        downloads = []

        class Opener:
            def open(self, request, **_kwargs):
                downloads.append(request.full_url)
                return FakeResp(200, CLIP, "video/mp4")

        with mock.patch(
            "venice.client._egress.build_https_opener", return_value=Opener()
        ):
            args = self.parser.parse_args(
                [
                    "video-status",
                    JOB,
                    "--model",
                    MODEL,
                    "--no-cleanup",
                    "--download-url",
                    "https://cdn.test/explicit",
                ]
            )
            self.assertEqual(video._run_status(args), 0)
        self.assertEqual(downloads, ["https://cdn.test/explicit"])
        self.assertIn(
            "warning: could not clear private video job metadata:",
            self.stderr.getvalue(),
        )

    def test_status_registry_error_stops_before_retrieval(self):
        (self.root / "jobs.json").mkdir()
        calls = self.http()
        args = self.parser.parse_args(["video-status", JOB, "--model", MODEL])
        self.assertEqual(video._run_status(args), 9)
        self.assertEqual(calls, [])
        self.assertIn("cannot read private retrieval metadata:", self.stderr.getvalue())
        self.assertIn(
            f"retry with: venice video-status {JOB} --model {MODEL}",
            self.stderr.getvalue(),
        )
