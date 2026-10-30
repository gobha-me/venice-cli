"""Redaction at the 2 KB error-excerpt boundary, per encoding (r1 follow-up).

Test author of record: Claude. Written fresh for the r1 follow-up; it does not
reuse the ``test_r1_audit`` redaction cases.

``VeniceClient._raise_api_error`` redacts the whole upstream error body and only
then keeps the first 2048 characters. These tests place a synthetic credential
-- raw, JSON-escaped the way ``json.dumps`` writes it inside a body, and
URL-encoded by ``quote`` / ``quote_plus`` -- at every offset from "ends one
character before the cut" to "starts one character after it", so a credential
that straddles the cut is covered at each split point.

Only the Venice HTTP peer is faked (``tests/_r1_loopback.py``, 127.0.0.1).
``VeniceClient``, ``VeniceAPIError``, ``_redact`` and ``_mcp._err`` run
unmodified. Every credential here is synthetic; nothing is read from the real
credentials file.

The ``expected_leak`` cases at the bottom are red-first: they pin redaction
gaps that exist today. src/ is not changed in this branch, so they are
expected to fail, at their leak assertion only, until the code is fixed.
Their preconditions are separate ``*_precondition`` tests that must pass.
"""
import functools
import json
import os
import re
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest import mock

from tests._r1_loopback import LoopbackVenice, Reply
from venice import _redact, auth, config
from venice.client import VeniceAPIError, VeniceClient, build_client_from_auth
from venice.commands import _mcp

CUT = 2048          # VeniceClient error excerpt length
PREVIEW = 500       # VeniceAPIError message body preview length
MARK = "[REDACTED]"

# Synthetic. '/', '+', ' ' and '"' make all four encodings pairwise distinct
# and keep the raw form from being a substring of any encoded one.
SECRET = 'b0und/ry+K "q" 9z'


def encodings(secret):
    """The four forms ``_redact`` promises to remove, derived independently."""
    return {
        "raw": secret,
        "json": json.dumps(secret)[1:-1],
        "quote": urllib.parse.quote(secret, safe=""),
        "quote_plus": urllib.parse.quote_plus(secret, safe=""),
    }


def straddle_offsets(length):
    """Start offsets from 'ends 1 before the cut' to 'starts 1 after it'."""
    return range(CUT - length - 1, CUT + 2)


class _Isolated(unittest.TestCase):
    """Temporary HOME; no ambient VENICE_API_KEY, so only explicit keys count."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        patcher = mock.patch.dict(os.environ, {
            "HOME": str(self.home),
            "XDG_CONFIG_HOME": str(self.home / ".config"),
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(config.ENV_API_KEY, None)
        os.environ.pop(config.ENV_BASE_URL, None)

    def assert_no_piece(self, text, form, width=4):
        """No ``width``-char slice of ``form`` survives in ``text``."""
        for i in range(len(form) - width + 1):
            self.assertNotIn(form[i:i + width], text,
                             f"slice {i}:{i + width} of the credential leaked")

    def errors_for(self, bodies, content_type="text/plain", key=SECRET):
        """Serve each body as a 4xx and return the ``VeniceAPIError`` per body."""
        def respond(seen, _bearer):
            return Reply(400, content_type, bodies[int(seen.query["case"][0])])

        caught = []
        with LoopbackVenice() as api:
            api.on("GET", "/models", respond)
            client = VeniceClient(key, base_url=api.base_url)
            for case in range(len(bodies)):
                try:
                    client.get_json("/models", params={"case": case})
                except VeniceAPIError as e:
                    caught.append(e)
                else:
                    self.fail(f"case {case}: no VeniceAPIError")
        self.assertEqual(api.unexpected, [])
        return caught


# --------------------------------------------------------------------------- #
# 1. Credential straddling the 2048-character excerpt cut
# --------------------------------------------------------------------------- #

class ExcerptBoundaryPerEncoding(_Isolated):
    TAIL = "~" * 40

    def sweep(self, name, wrap=lambda s: s, content_type="text/plain"):
        """Place ``encodings(SECRET)[name]`` at every straddle offset of the
        *served* body; the excerpt must equal redact-then-cut of the body."""
        form = encodings(SECRET)[name]
        head = wrap("\x00")
        prefix_len = head.index("\x00")
        bodies, expected = [], []
        for offset in straddle_offsets(len(form)):
            pad = "." * (offset - prefix_len)
            body = wrap(pad + "\x00").replace("\x00", form) + self.TAIL
            self.assertEqual(body.index(form), offset)
            bodies.append(body)
            expected.append((body.replace(form, MARK))[:CUT])
        errors = self.errors_for(bodies, content_type)
        for offset, error, want in zip(straddle_offsets(len(form)), errors, expected):
            with self.subTest(encoding=name, offset=offset,
                              split=max(0, min(len(form), CUT - offset))):
                self.assertEqual(error.body, want)
                self.assertEqual(len(error.body), CUT)
                self.assert_no_piece(error.body, form)
                self.assert_no_piece(str(error), form)

    def test_raw_credential_at_and_across_the_cut(self):
        self.sweep("raw")

    def test_json_escaped_credential_inside_a_json_body_at_and_across_the_cut(self):
        # The credential sits inside a string value exactly as json.dumps
        # escapes it in a real upstream error document. (The marker is spliced
        # in after dumping because json.dumps would escape it.)
        prefix = json.dumps({"error": {"code": "BAD_KEY", "message": ""}})[:-3]
        self.sweep("json", lambda m: prefix + m + '"}}', "application/json")

    def test_sweep_json_body_is_what_json_dumps_emits(self):
        form = encodings(SECRET)["json"]
        prefix = json.dumps({"error": {"code": "BAD_KEY", "message": ""}})[:-3]
        self.assertEqual(prefix + "." * 3 + form + '"}}',
                         json.dumps({"error": {"code": "BAD_KEY", "message": "..." + SECRET}}))

    def test_json_escaped_credential_body_still_parses_code(self):
        form = encodings(SECRET)["json"]
        doc = json.dumps({"error": {"code": "BAD_KEY", "message": "x" * CUT + SECRET}})
        self.assertIn(form, doc)  # json.dumps really emits the escaped form
        error, = self.errors_for([doc], "application/json")
        self.assertEqual(error.code, "BAD_KEY")
        self.assertNotIn(form, json.dumps(error.__dict__, default=str))

    def test_percent_quoted_credential_at_and_across_the_cut(self):
        self.sweep("quote", lambda m: "GET /v1/cb/" + m)

    def test_quote_plus_credential_at_and_across_the_cut(self):
        self.sweep("quote_plus", lambda m: "rejected form: api_key=" + m)

    def test_credential_as_the_whole_tail_past_the_cut_is_dropped(self):
        # Nothing follows the credential, so the body ends inside it.
        for name, form in encodings(SECRET).items():
            for offset in (CUT - 1, CUT):
                with self.subTest(encoding=name, offset=offset):
                    error, = self.errors_for(["-" * offset + form])
                    self.assertEqual(error.body, ("-" * offset + MARK)[:CUT])

    def test_body_one_short_of_exactly_and_one_over_the_cut(self):
        for size, want in ((CUT - 1, CUT - 1), (CUT, CUT), (CUT + 1, CUT)):
            with self.subTest(size=size):
                error, = self.errors_for(["#" * size])
                self.assertEqual(len(error.body), want)

    def test_message_preview_cut_at_500_never_splits_a_credential(self):
        for name, form in encodings(SECRET).items():
            for offset in (PREVIEW - len(form), PREVIEW - 1, PREVIEW):
                with self.subTest(encoding=name, offset=offset):
                    error, = self.errors_for(["p" * offset + form])
                    preview = str(error).split("\n  body: ", 1)[1]
                    self.assertEqual(preview, ("p" * offset + MARK)[:PREVIEW])
                    self.assert_no_piece(preview, form)


# --------------------------------------------------------------------------- #
# 1b. _redact directly: encoded forms as a producer would emit them
# --------------------------------------------------------------------------- #

class RedactEncodedForms(_Isolated):
    def test_json_dumps_document_round_trips_with_every_value_redacted(self):
        doc = {"error": SECRET, "hint": [SECRET, "ok"], "nested": {"k": SECRET}}
        out = _redact.redact_text(json.dumps(doc), SECRET)
        self.assertEqual(json.loads(out),
                         {"error": MARK, "hint": [MARK, "ok"], "nested": {"k": MARK}})

    def test_urlencoded_query_string_is_redacted(self):
        query = urllib.parse.urlencode({"key": SECRET, "page": "2"})
        self.assertEqual(_redact.redact_text(query, SECRET), f"key={MARK}&page=2")

    def test_quoted_path_segment_is_redacted(self):
        url = "https://api.example/v1/keys/" + urllib.parse.quote(SECRET, safe="") + "/usage"
        self.assertEqual(_redact.redact_text(url, SECRET),
                         f"https://api.example/v1/keys/{MARK}/usage")

    def test_all_forms_side_by_side_leave_only_markers(self):
        forms = encodings(SECRET)
        text = "|".join(forms.values())
        self.assertEqual(_redact.redact_text(text, SECRET), "|".join([MARK] * len(forms)))

    def test_json_form_containing_the_raw_form_is_removed_whole(self):
        # A credential starting with a backslash: its raw form is a substring
        # of its JSON form ('\\tok' contains '\tok'). Removing the raw form
        # first would leave a stray backslash in front of the marker.
        secret = "\\tok3n-synth-r1"
        escaped = json.dumps(secret)[1:-1]
        self.assertIn(secret, escaped)
        self.assertEqual(_redact.redact_text('{"k": "' + escaped + '"}', secret),
                         '{"k": "' + MARK + '"}')

    def test_containing_secret_wins_over_contained_non_prefix_secret(self):
        # "bc" sorts after "abc" lexically but is shorter, so length order
        # (not string order) has to decide which goes first.
        self.assertEqual(_redact.redact_text("abc", "abc", "bc"), MARK)

    def test_text_is_untouched_when_no_credential_is_known(self):
        # No env key and no explicit secret: placeholder-looking text included.
        for text in ("XXXX-XXXX", "", "plain", "%2F \\/ +"):
            with self.subTest(text=text):
                self.assertEqual(_redact.redact_text(text), text)


def _reraise(exc):
    raise exc


def expected_leak(test):
    """Red-first marker: only an AssertionError counts as the expected failure.

    ``unittest.expectedFailure`` alone would also absorb setup errors (e.g. a
    broken ``load_key``). Any other exception is re-raised from a cleanup,
    which unittest runs outside the expected-failure scope, so it is reported
    as an ERROR. A pass is still an unexpected success, which fails the run.
    """

    @functools.wraps(test)
    def wrapper(self):
        try:
            test(self)
        except AssertionError:
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised below as an ERROR
            self.addCleanup(_reraise, exc)

    return unittest.expectedFailure(wrapper)


# --------------------------------------------------------------------------- #
# 2. Red-first: redaction gaps that exist today (src unchanged in this branch)
# --------------------------------------------------------------------------- #

class KnownRedactionGaps(_Isolated):
    # Each red-first test below is marked @expected_leak and holds only the
    # leak assertion and what follows it. The preconditions it relies on
    # live in an unmarked *_precondition test that must pass on its own, so
    # a broken precondition can never be absorbed as an expected failure.

    # Finding F-fu-1: _mcp._err calls
    # _redact.redact_text(message) with no explicit secret, so it can only
    # redact $VENICE_API_KEY. When the key came from the credentials file
    # (no env var), a tool argument that reflects that key into the error
    # envelope is returned to the MCP host unredacted.
    def _write_file_key(self):
        file_key = "r1-fu-file-only-synthetic-Zp8"
        creds = self.home / ".config" / "venice" / "credentials"
        creds.parent.mkdir(parents=True)
        creds.write_text(file_key + "\n", encoding="utf-8")
        creds.chmod(0o600)
        return file_key, creds

    def test_file_loaded_key_reflected_into_mcp_error_envelope_is_redacted_precondition(self):
        file_key, creds = self._write_file_key()
        with LoopbackVenice() as api, mock.patch.object(config, "CREDS_FILE", creds):
            os.environ[config.ENV_BASE_URL] = api.base_url
            self.assertNotIn(config.ENV_API_KEY, os.environ)
            self.assertEqual(auth.load_key(), file_key)
            client = build_client_from_auth()
            result = _mcp.job_status_tool(client, queue_id="q", type=file_key, model="m")
        self.assertEqual(api.seen, [])
        self.assertEqual(result["status"], "error")

    @expected_leak
    def test_file_loaded_key_reflected_into_mcp_error_envelope_is_redacted(self):
        file_key, creds = self._write_file_key()
        with LoopbackVenice() as api, mock.patch.object(config, "CREDS_FILE", creds):
            os.environ[config.ENV_BASE_URL] = api.base_url
            client = build_client_from_auth()
            result = _mcp.job_status_tool(client, queue_id="q", type=file_key, model="m")
        self.assertNotIn(file_key, json.dumps(result))

    # Finding F-fu-2: percent-escapes are case-insensitive (RFC 3986 2.1), but
    # _redact only removes urllib's uppercase form, so '%2f' / '%2b' echoes of
    # the credential survive into VeniceAPIError.body.
    @staticmethod
    def _lowercase_escaped():
        return re.sub(r"%[0-9A-F]{2}", lambda m: m.group(0).lower(),
                      urllib.parse.quote(SECRET, safe=""))

    def test_lowercase_percent_escapes_are_redacted_precondition(self):
        lower = self._lowercase_escaped()
        self.assertIn("%2f", lower)
        # errors_for's own checks (a VeniceAPIError per body, no unexpected
        # request) run here, outside the expected-failure scope.
        _error, = self.errors_for(["callback /cb/" + lower])

    @expected_leak
    def test_lowercase_percent_escapes_are_redacted(self):
        lower = self._lowercase_escaped()
        error, = self.errors_for(["callback /cb/" + lower])
        self.assertNotIn(lower, error.body)
        self.assertIn(MARK, error.body)

    # Finding F-fu-3: JSON allows (and PHP / some Java encoders emit) '\/' for
    # '/'. _redact only knows json.dumps's form, which leaves '/' bare, so an
    # upstream that escapes slashes echoes the credential past redaction.
    @staticmethod
    def _solidus_escaped():
        slashed = encodings(SECRET)["json"].replace("/", "\\/")
        body = '{"error": {"message": "bad key ' + slashed + '"}}'
        return slashed, body

    def test_json_solidus_escaped_credential_is_redacted_precondition(self):
        slashed, body = self._solidus_escaped()
        self.assertNotEqual(slashed, encodings(SECRET)["json"])
        self.assertEqual(json.loads(body)["error"]["message"], "bad key " + SECRET)
        # errors_for's own checks run here, outside the expected-failure scope.
        _error, = self.errors_for([body], "application/json")

    @expected_leak
    def test_json_solidus_escaped_credential_is_redacted(self):
        slashed, body = self._solidus_escaped()
        error, = self.errors_for([body], "application/json")
        self.assertNotIn(slashed, error.body)
        self.assertIn(MARK, error.body)

if __name__ == "__main__":
    unittest.main()
