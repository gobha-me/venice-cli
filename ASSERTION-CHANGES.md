# Assertion changes: e4c243b -> this re-make

Each assertion in e4c243b is listed with its fate in the replacement. The
replacement was written from scratch. e4c243b was used only as an inventory of
behaviours to cover.

Verdicts: **kept** (equivalent), **strengthened**, **changed** (same intent,
different mechanism or oracle), **dropped**.

Global change: e4c243b mocked `urlopen` with a raised `HTTPError`, used
`mock.Mock()` clients for `complete_job`, and patched
`mcp_server.get_access_token` for the remote profile. The re-make mocks
**only the Venice HTTP peer**, using a loopback server (`tests/_r1_loopback.py`).
Remote auth goes through the server's real ASGI app with an injected token
verifier, which is a supported constructor parameter. `urlopen` is patched in
just one test, to raise `URLError`/`OSError`, the one transport exception a
loopback socket cannot produce with chosen text.

## tests/test_mcp_harness.py

| # | e4c243b assertion | verdict | replacement / reason |
|---|---|---|---|
| H1 | `assertFalse(result.is_error, name)` per call | kept | `_Harness.ok()` |
| H2 | `payload["status"] == expected` | kept | `_Harness.ok()` |
| H3 | local `listed == LOCAL_TOOLS` (hard-coded set) | changed + strengthened | `listed == set(SURFACE["stdio"])`, the committed mcp-surface.json, so the live list, the committed snapshot, and the SDK-free AST check (`StaticToolRegistrations`) must all agree. A hard-coded set could drift from the snapshot unnoticed. |
| H4 | each returned path `is_file()` | strengthened | also requires every paid local tool to return at least one path, and every path to resolve inside the requested output dir |
| H5 | `set(args) == listed` (called == listed) | kept | asserted before calling, in `exercise_local` |
| H6 | in-process `api.unexpected == []` | kept | `assert_cleanup_contract` |
| H7 | `assertTrue(cleanups)` (some `/video/complete` happened) | strengthened | `/video/complete` bodies must equal `[{"queue_id": q}]` for exactly the queue ids that were retrieved, in order |
| H8 | every cleanup body has keys == `{"queue_id"}` | strengthened | exact list equality as in H7, plus every `/video/retrieve` body has keys `{"model","queue_id"}`, and every `/audio/complete` carries a non-empty `model` |
| H9 | stdio subprocess `api.unexpected == []` | strengthened | now also runs the full cleanup contract (H7/H8) on the subprocess's wire traffic |
| H10 | remote `listed == REMOTE_TOOLS` | changed + strengthened | `== set(SURFACE["http_media"])` (see H3) |
| H11 | stored tools: `result["media"]` truthy | strengthened | plus `path`/`paths` must be absent from remote results (no server paths leak) |
| H12 | sfx/music/video status `"queued"` | strengthened | plus `queue_id` (backend id) must be absent from the queued handle |
| H13 | `status["media"] == result["media"]` | kept | |
| H14 | `result["media"]` truthy | kept | |
| H15 | remote `called == listed` | kept | |
| H16 | remote `api.unexpected == []` | strengthened | plus the cleanup contract, and exactly one `/image/generate` (the unconfirmed call spent nothing) |
| H17 | minimal `listed == {"venice_chat","venice_vision"}` | changed | `== set(SURFACE["http_minimal"])`. `CommittedMcpSurface.test_profile_relationships` separately pins that profile to exactly those two names. |
| H18 | minimal chat and vision both `ok` | kept | |
| H19 | minimal `api.unexpected == []` | strengthened | plus exactly two `/chat/completions` requests |
| new | -- | added | unknown tool, missing required arg, and `scale=3` are refused; an unconfirmed dynamic-price call returns `confirmation_required`; remote vision with both `media_uri` and `image_url` errors; an unknown job id errors; a second `media_delete` errors; minimal vision rejects credentialed, ftp, local-path, and non-base64 data URLs; a hostile upstream echoing the bearer never reaches a stdio tool result |

**Caveat:** this whole file skips without the `mcp`/`openai` extras, which are
absent in this environment. None of H1-H19 or the new assertions has been
executed here. See BROKEN-COPY-RESULTS.md, "Skipped".

## tests/test_openapi_surface.py

| # | e4c243b assertion | verdict | replacement / reason |
|---|---|---|---|
| S-1 | `surface["source"] == lock["source"]` | kept | `test_header_matches_lock` also pins `format == 1` and the top-level key set |
| S-2 | surface op set == lock `all_operations` keys | strengthened | plus the lock list has no duplicate keys |
| S-3 | `len(operations) == 55` | kept | deliberate snapshot pin, documented in-test |
| S-4 | every operation's `responses` truthy | strengthened | each response is exactly `{"content": dict, "headers": dict}` with string status keys |
| S-5 | implemented op: coverage == manifest entry | kept | plus coverage must not contain `unsupported` |
| S-6 | implemented op: `request_bodies` == lock | strengthened | `parameters` == lock too, and `request_body_required` is a bool |
| S-7 | unimplemented op: `coverage["unsupported"]` truthy | strengthened | exact `{"unsupported": <canonical text>}`; unsupported count == total - implemented, and > 0 |
| S-8 | nested response field `== {"type":"string","enum":["a","b"]}` | kept | `test_nested_response_fields_of_unsupported_operations_are_retained` |
| S-9 | `"unsupported" in coverage` (built) | strengthened | exact equality with the canonical text, for both an unsupported op and an op with no request body |
| S-10 | `"## \`GET /unsupported\`" in render` | strengthened | section order, header line, trailing newline, consumers, required/optional parameters, implemented / omitted / unsupported field lines, declared-unsupported body text, "No named schema properties", and the Responses lines |
| new | -- | added | committed `surface.md` equals `render_surface(surface.json)` re-ordered by the lock; `operation_id` matches upstream; manifest ⊆ surface and == lock-implemented; `$ref` request bodies, responses, and headers resolve; YAML integer status codes become strings; a manifest op missing upstream raises `ContractError`; inputs are not mutated; `main()` writes JSON + `.md` without touching the network; committed mcp-surface.json profile structure; SDK-free AST check that source registrations equal each committed profile; SDK-gated byte-for-byte regeneration of mcp-surface.json |

## tests/test_r1_audit.py

| # | e4c243b assertion | verdict | replacement / reason |
|---|---|---|---|
| A-1 | `complete_body("job") == {"queue_id": "job"}` | strengthened | five ids (empty, unicode, 4 KB, slash/space); a fresh dict per call; `complete_body("m","q")` raises `TypeError` |
| A-2 | `job_body("model","job") == {"model","queue_id"}` | kept | `test_retrieve_body_still_carries_model` |
| A-3 | mock client `post_json` called with `("/video/complete", {"queue_id": "job"})` | changed | real `VeniceClient` over loopback; the body is asserted **as received on the wire**, and also through the CLI subprocess (`video-status`), `video_tool`, and `job_result_tool` |
| A-4 | mock `jobs.forget` called with `("job","model")` | changed + strengthened | real `_video_jobs` registry in a temp file: `remember`, then `complete_job`, then `lookup` is `None`. Also forgets when the backend rejects the cleanup. |
| A-5 | audio complete body `{"model","queue_id"}` (mock) | changed + strengthened | on the wire, for `sfx`, `music`, and `Music`; also via `job_result_tool`; unknown types send nothing |
| A-6 | env secret: raw, JSON-escaped, %-encoded forms all redact to `[REDACTED]` | strengthened | four forms (raw, JSON, `quote`, `quote_plus`) for both explicit and env secrets, standalone and embedded; `/`, space, `+`, quote, unicode, and control characters each exercised |
| A-7 | `redact_text("hello credential","credential") == "hello [REDACTED]"` | kept | `test_environment_and_explicit_secrets_both_apply` and related tests; plus every occurrence, longest-first, empty/None/non-str secrets, non-str values |
| A-8 | HTTPError excerpt crossing 2 KB: secret not in `str(e)`, `secret[:10]` not in body, `[REDACTED]` in body | strengthened | real 401 over loopback. Ten offsets around the 2048 boundary, each asserting the **exact** expected excerpt, length ≤ 2048, and that no 6-char window of any encoded form appears in `str/body/url/code`. Exactly 2047/2048/2049-byte bodies are covered. The message preview is capped at 500 after redaction. |
| A-9 | `_raise_api_error` with JSON `code` == key: not in `str`, not in `code` | strengthened | via a real response; `code == "[REDACTED]"` for both `error.code` and top-level `code`; malformed JSON, HTML, invalid UTF-8, empty body, and key in the URL path also covered |
| A-10 | URLError(secret): not in `str(e)` | strengthened | URLError **and** OSError; `status == 0`; `"connection error: "` prefix retained; `[REDACTED]` present |
| A-11 | `_safe_api_error(SDKError) == "rejected [REDACTED]"` | strengthened | `Bearer`/`bearer`/`BEARER`; encoded echo; None/{}/bytes/empty-bearer headers fall back to env; empty bearer leaves text intact; redaction precedes the 1000-char truncation (exact output); HTML collapses to title; `status_to_exit` connection path |
| A-12 | env secret in `VeniceAPIError`: not in `str`, not in `body` | strengthened | exact body; `secret=` kwarg redacts url/body/code/message; falsy and non-str `code` rendering |
| new | -- | added | MCP `_err` envelope redacts every form; a key reflected through `job_status`'s `type` argument is redacted; upstream echoes through `video_tool` (quote rejected, no queue/spend) and `job_status` are redacted; CLI cleanup failure warns without echoing the key |

## tests/test_openapi_contract.py -- edit to a pre-existing assertion (flagged)

| e4c243b change | verdict | reason |
|---|---|---|
| `"POST /video/complete": set(video.job_body("m","q"))` -> `set(video.complete_body("q"))` | kept (identical edit) | **Pre-existing (pre-1a844f2) assertion input, edited because it is unavoidable.** 1a844f2 removed `model` from the `POST /video/complete` entry in `contracts/openapi/implementation.json` and switched every production cleanup call to `complete_body`. The old line compared the manifest against the *retrieve* builder, and it fails on 1a844f2 (`'model'` in the first set but not the second; observed before the edit). The new line compares against the builder production actually uses. Breaks V1 and V4 prove this line can still fail. |

No other pre-existing test file was modified. `pyproject.toml` gained a
`[tool.mutmut]` section for scoping only.

## Tests written then removed in this re-make (unfalsifiable)

* `redact_text` returns `str`: no realistic product break fails it.
* a real refused connection carries no key: the reason text never contains a
  key, so only `status == 0` could fail.
* the four test variants are pairwise distinct: a guard on the test's own
  fixture, not on product code.
