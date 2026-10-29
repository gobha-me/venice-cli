# R1 Venice API and MCP audit

Date: 2026-10-29. Branch: `venice-cli/r1-audit`. Local work only.

## Result and scope

The implemented request surface is synchronized with the current public Swagger.
Video cleanup now sends only `queue_id`; retrieval still sends `model` and
`queue_id`. Known credentials are redacted in HTTP and SDK diagnostics before
truncation. Module boundaries and documentation were reviewed for all 70 runtime
Python modules and all four maintenance scripts. Every runtime module now has a
module docstring and its own README section.

This is a module responsibility/import/docstring audit with contract and protocol
verification, not a claim of a line-by-line security review of every legacy
function. The CLI deliberately implements a curated API surface: 18 operations
have dedicated consumers and 37 existing operations remain unsupported by a
dedicated command/tool. Their availability is documented, not advertised as
implemented parity.

## Public specification and drift

Source: <https://docs.venice.ai/swagger.yaml>, fetched without authentication.

| Snapshot | Version | SHA-256 |
| --- | --- | --- |
| Previous committed request lock | `20261026.193006` | `11e9fead672ea14e0f3e878fd48c62d72bc7aa2fd583e66048a2d03649d76d9b` |
| Fetched and reviewed | `20261027.204242` | `ef6561d9ff9f98cff00404e502dea8eb41a69520d9bbed98108214b41daf1d3b` |

There are 55 operations in both endpoint inventories: **zero endpoints added,
zero removed**, one implemented request contract changed. The exhaustive
normalized change list against the previous request lock is:

| Operation / field | Before | Current | Implementation change |
| --- | --- | --- | --- |
| `POST /video/complete` / `model.deprecated` | Absent | `true` | Stop emitting deprecated `model` during cleanup. |
| `POST /video/complete` / required `model` | Required | Optional | New `video.complete_body(queue_id)`; use it from CLI cleanup, synchronous MCP video generation, and shared background-job completion. |

No request property names were added or removed from the 18 implemented
contracts. No other normalized implemented request constraints changed. Audio
completion still needs its model. Video retrieve still needs its model. Public
MCP input schemas are unchanged by the cleanup fix because generation/model
selection and job identity remain distinct from the backend cleanup request.

The previous lock retained request schemas only for the implemented operations.
It did **not** retain responses or schemas for the 37 unsupported operations.
Historical field drift for those areas therefore cannot be reconstructed from
that baseline; it is not reported as zero. This audit establishes a complete
current field baseline in [surface.json](contracts/openapi/surface.json), covering
all nested named request/response properties, unions, enums, normalized bounds,
and response headers. It normalizes the same schema keywords as the existing
contract checker, rather than preserving prose, examples, or every OpenAPI
annotation. Missing upstream request schemas are left empty rather than invented.

Every current endpoint and top-level request field is listed and classified in
[surface.md](contracts/openapi/surface.md). Existing intentional field omissions
are copied from [implementation.json](contracts/openapi/implementation.json),
including omitted optional chat sampling controls, audio options, advanced video
controls, tracking identifiers, and multipart image input. The new cleanup-model
omission has a specific deprecation reason. The generic HTTP client can construct
requests to other routes, but this does not constitute a dedicated supported
command or MCP tool.

### Existing unsupported endpoint inventory

These are surface gaps, not upstream endpoint additions in this revision. Their
parameters and every published request field are in the linked full inventory.

- `DELETE /api_keys`
- `GET /api_keys`
- `GET /api_keys/generate_web3_key`
- `GET /api_keys/rate_limits/log`
- `GET /api_keys/{id}`
- `GET /billing/balance`
- `GET /billing/usage`
- `GET /billing/usage-analytics`
- `GET /billing/usage-history`
- `GET /characters`
- `GET /characters/{slug}`
- `GET /characters/{slug}/reviews`
- `GET /crypto/rpc/networks`
- `GET /image/styles`
- `GET /models/compatibility_mapping`
- `GET /models/traits`
- `GET /x402/balance/{walletAddress}`
- `GET /x402/transactions/{walletAddress}`
- `PATCH /api_keys`
- `POST /api_keys`
- `POST /api_keys/generate_web3_key`
- `POST /audio/transcriptions`
- `POST /audio/voice-changer/complete`
- `POST /audio/voice-changer/queue`
- `POST /audio/voice-changer/quote`
- `POST /audio/voice-changer/retrieve`
- `POST /audio/voices`
- `POST /augment/scrape`
- `POST /augment/search`
- `POST /augment/text-parser`
- `POST /crypto/rpc/{network}`
- `POST /decisions`
- `POST /images/generations`
- `POST /responses`
- `POST /systemone`
- `POST /video/transcriptions`
- `POST /x402/top-up`

### MCP surface comparison and harness

[mcp-surface.json](contracts/openapi/mcp-surface.json) records schemas negotiated
with the SDK for all three profiles. Local stdio advertises 10 tools; remote
HTTP without storage advertises chat and vision; remote HTTP with storage
advertises 14 tools. Agent-only tools such as model lookup, local memory,
semantic search, and filesystem operations are not additional advertised MCP
server tools. Their catalog schemas were preserved during registry extraction.

The following table lists every advertised MCP tool and all its input fields.
The local and remote profiles intentionally differ on paths, storage URIs, jobs,
and operator spend authority. Exact types, required fields, and optional output
schemas are in the JSON snapshot.

| Profile | Tool | Input fields |
| --- | --- | --- |
| `http_media` | `venice_bg_remove` | `confirm`, `media_uri` |
| `http_media` | `venice_chat` | `character`, `max_tokens`, `message`, `model`, `system`, `temperature`, `web_search` |
| `http_media` | `venice_image` | `aspect_ratio`, `confirm`, `format`, `height`, `model`, `negative_prompt`, `prompt`, `resolution`, `seed`, `style_references`, `transparent_background`, `variants`, `width` |
| `http_media` | `venice_image_edit` | `aspect_ratio`, `confirm`, `layer_media_uris`, `media_uri`, `model`, `output_format`, `prompt`, `quality`, `resolution`, `transparent_background` |
| `http_media` | `venice_job_result` | `job_id` |
| `http_media` | `venice_job_status` | `job_id` |
| `http_media` | `venice_media_delete` | `media_uri` |
| `http_media` | `venice_media_import` | `url` |
| `http_media` | `venice_music` | `confirm`, `duration`, `instrumental`, `lyrics`, `model`, `prompt`, `speed` |
| `http_media` | `venice_sfx` | `confirm`, `duration`, `model`, `prompt` |
| `http_media` | `venice_tts` | `confirm`, `format`, `model`, `speed`, `text`, `voice` |
| `http_media` | `venice_upscale` | `confirm`, `creativity`, `media_uri`, `scale` |
| `http_media` | `venice_video` | `aspect_ratio`, `audio_media_uri`, `confirm`, `duration`, `end_image_media_uri`, `image_media_uri`, `image_strength`, `model`, `negative_prompt`, `no_audio`, `prompt`, `reference_audio_media_uris`, `reference_image_media_uris`, `reference_video_media_uris`, `resolution`, `scene_image_media_uris`, `video_media_uri` |
| `http_media` | `venice_vision` | `image_url`, `max_tokens`, `media_uri`, `model`, `prompt` |
| `http_minimal` | `venice_chat` | `character`, `max_tokens`, `message`, `model`, `system`, `temperature`, `web_search` |
| `http_minimal` | `venice_vision` | `image_url`, `max_tokens`, `model`, `prompt` |
| `stdio` | `venice_bg_remove` | `confirm`, `image_url`, `input_path`, `max_spend`, `output_dir` |
| `stdio` | `venice_chat` | `character`, `max_tokens`, `message`, `model`, `system`, `temperature`, `web_search` |
| `stdio` | `venice_image` | `aspect_ratio`, `cfg_scale`, `confirm`, `disable_prompt_optimization_thinking`, `embed_exif_metadata`, `enable_web_search`, `enhance_prompt`, `format`, `height`, `hide_watermark`, `lora_strength`, `max_spend`, `model`, `negative_prompt`, `output_dir`, `prompt`, `quality`, `resolution`, `safe_mode`, `seed`, `steps`, `style_preset`, `style_references`, `transparent_background`, `variants`, `width` |
| `stdio` | `venice_image_edit` | `aspect_ratio`, `confirm`, `disable_prompt_optimization_thinking`, `enhance_prompt`, `image_url`, `input_path`, `layer_paths`, `max_spend`, `model`, `output_dir`, `output_format`, `prompt`, `quality`, `resolution`, `safe_mode`, `transparent_background` |
| `stdio` | `venice_music` | `confirm`, `duration`, `instrumental`, `lyrics`, `max_spend`, `model`, `output_dir`, `prompt`, `speed` |
| `stdio` | `venice_sfx` | `confirm`, `duration`, `max_spend`, `model`, `output_dir`, `prompt` |
| `stdio` | `venice_tts` | `confirm`, `format`, `max_spend`, `model`, `output_dir`, `speed`, `text`, `voice` |
| `stdio` | `venice_upscale` | `confirm`, `creativity`, `input_path`, `max_spend`, `output_dir`, `scale` |
| `stdio` | `venice_video` | `aspect_ratio`, `audio_url`, `confirm`, `duration`, `end_image_url`, `image_strength`, `image_url`, `max_spend`, `max_wait`, `model`, `negative_prompt`, `no_audio`, `output_dir`, `prompt`, `reference_audio_urls`, `reference_image_urls`, `reference_video_duration`, `reference_video_urls`, `resolution`, `scene_image_urls`, `video_url` |
| `stdio` | `venice_vision` | `image_url`, `input_path`, `max_tokens`, `mode`, `model`, `prompt` |

`tests/test_mcp_harness.py` attaches the actual server to SDK sessions and a
loopback REST fixture; tool implementations are not mocked. The real stdio
subprocess exercises all 10 local tools, including saved output files. A second
in-process local session calls all 10 tools. The full remote-media session calls
all 14 tools, imports/deletes an image, generates synthetic outputs, queues each
of sfx/music/video, retrieves and completes each job, and checks stored-resource
results. The minimal remote profile calls both tools. Set equality between
listed and exercised tools detects future unexercised additions.

The remote-media harness supplies synthetic authenticated context in-process;
it does not claim to test OAuth over a public network. Existing HTTP transport,
JWT signature, audience, principal isolation, and storage tests also pass in the
focused suite. Existing native-vision tests cover the operator declaration and
MCP image-content path. All REST responses are synthetic, all API URLs are
loopback, subprocess keys are environment-only placeholders, and HOME/output/job
storage are temporary. No fake handler records authorization values.

`VENICE_API_KEY` was absent when checked at the start (only presence was
reported). **Authenticated live calls were skipped.** No real model listings,
generation, bulk operations, or other authenticated Venice calls were made. The
only Venice network requests were unauthenticated public Swagger fetches. No
credential paths were searched or read, and no SSH or forge operations were used.

## Module sizing and documentation

Two responsibility splits were warranted and implemented:

- `mcp_server.py`: 1,629 to 1,036 lines. Local registration moved to
  `_mcp_stdio.py`; authentication, principal checks, and HTTP range helpers moved
  to `_mcp_auth.py`; common result/precedence helpers moved to `_mcp_common.py`.
  Public `build_server`, `serve`, and `JWKSJWTVerifier` imports remain available.
- `commands/_agent.py`: 4,574 to 3911 lines. Static schemas,
  `ToolSpec`, capability metadata, and selectors moved to `_tool_registry.py`.
  Existing schema/registry aliases remain available. The schema module imports
  neither the MCP SDK nor execution-loop state.

The remaining large modules were assessed by responsibility, shared state,
imports, and largest functions/classes. The 1,531-line `CostLedger` remains a
follow-up candidate for an accounting-specific extraction; splitting its mutable
budget/usage behavior during this contract change adds risk without changing the
reviewed API. Remote tool registration still has a large closure (848 lines)
whose handlers share principal-bound storage, quotas, and job publication.
`_mcp` keeps common spend and media-authority enforcement; `_code` separates file
policy from `_exec`; REPL state stays separate from one-shot CLI parsing. No
merges were justified: the small adapters preserve command/SDK/I/O boundaries,
and shared helpers prevent duplicated enforcement. These retained sizes are
maintenance findings, not a claim that all modules are permanently ideal.

Module-level docstrings were missing in `venice.__init__` and `venice.__main__`;
both were added. New helper modules and public cleanup/redaction/inventory
functions have docstrings. The README now has a separate section for each of
these modules and each maintenance script, describing purpose and boundaries.
The table records the per-module sizing disposition; detailed retained boundary
reasoning appears in the corresponding README section.

| Module | Final lines | Disposition |
| --- | ---: | --- |
| `venice.__init__` | 2 | Retain; add missing module docstring |
| `venice.__main__` | 7 | Retain; add missing module docstring |
| `venice._egress` | 467 | Retain; boundary documented |
| `venice._mcp_auth` | 162 | Extracted cohesive helper |
| `venice._mcp_common` | 24 | Extracted cohesive helper |
| `venice._mcp_stdio` | 442 | Extracted cohesive helper |
| `venice._numeric` | 30 | Retain; boundary documented |
| `venice._redact` | 27 | New shared diagnostic guard |
| `venice.audio_player` | 59 | Retain; boundary documented |
| `venice.audio_post` | 343 | Retain; boundary documented |
| `venice.auth` | 259 | Retain; boundary documented |
| `venice.billing` | 129 | Retain; boundary documented |
| `venice.cli` | 56 | Retain; boundary documented |
| `venice.client` | 416 | Retain; boundary documented |
| `venice.commands.__init__` | 34 | Retain; boundary documented |
| `venice.commands._agent` | 3911 | Split responsibilities; retain public aliases |
| `venice.commands._audio` | 132 | Retain; boundary documented |
| `venice.commands._browser` | 290 | Retain; boundary documented |
| `venice.commands._browser_proxy` | 194 | Retain; boundary documented |
| `venice.commands._code` | 1568 | Retain; boundary documented |
| `venice.commands._compact` | 679 | Retain; boundary documented |
| `venice.commands._context_archive` | 728 | Retain; boundary documented |
| `venice.commands._exec` | 563 | Retain; boundary documented |
| `venice.commands._index` | 798 | Retain; boundary documented |
| `venice.commands._mailbox` | 147 | Retain; boundary documented |
| `venice.commands._mcp` | 1700 | Retain; boundary documented |
| `venice.commands._mcp_client` | 478 | Retain; boundary documented |
| `venice.commands._memory` | 439 | Retain; boundary documented |
| `venice.commands._models` | 151 | Retain; boundary documented |
| `venice.commands._openai` | 158 | Retain; boundary documented |
| `venice.commands._persona` | 126 | Retain; boundary documented |
| `venice.commands._queue` | 87 | Retain; boundary documented |
| `venice.commands._repl` | 1014 | Retain; boundary documented |
| `venice.commands._review` | 915 | Retain; boundary documented |
| `venice.commands._session` | 446 | Retain; boundary documented |
| `venice.commands._shared` | 445 | Retain; boundary documented |
| `venice.commands._steer` | 188 | Retain; boundary documented |
| `venice.commands._tool_registry` | 716 | Extracted cohesive helper |
| `venice.commands._video_jobs` | 171 | Retain; boundary documented |
| `venice.commands.balance` | 132 | Retain; boundary documented |
| `venice.commands.bg_remove` | 121 | Retain; boundary documented |
| `venice.commands.cache_probe` | 423 | Retain; boundary documented |
| `venice.commands.chat` | 865 | Retain; boundary documented |
| `venice.commands.code` | 1301 | Retain; boundary documented |
| `venice.commands.completion` | 304 | Retain; boundary documented |
| `venice.commands.config` | 290 | Retain; boundary documented |
| `venice.commands.contact_sheet` | 99 | Retain; boundary documented |
| `venice.commands.embed` | 268 | Retain; boundary documented |
| `venice.commands.image` | 1035 | Retain; boundary documented |
| `venice.commands.image_edit` | 320 | Retain; boundary documented |
| `venice.commands.index` | 130 | Retain; boundary documented |
| `venice.commands.login` | 39 | Retain; boundary documented |
| `venice.commands.master` | 53 | Retain; boundary documented |
| `venice.commands.mcp_serve` | 387 | Retain; boundary documented |
| `venice.commands.memory` | 138 | Retain; boundary documented |
| `venice.commands.models` | 232 | Retain; boundary documented |
| `venice.commands.music` | 413 | Retain; boundary documented |
| `venice.commands.review` | 273 | Retain; boundary documented |
| `venice.commands.search` | 100 | Retain; boundary documented |
| `venice.commands.secret` | 93 | Retain; boundary documented |
| `venice.commands.sessions` | 234 | Retain; boundary documented |
| `venice.commands.sfx` | 264 | Retain; boundary documented |
| `venice.commands.tts` | 440 | Retain; boundary documented |
| `venice.commands.upscale` | 197 | Retain; boundary documented |
| `venice.commands.video` | 817 | Retain; boundary documented |
| `venice.config` | 98 | Retain; boundary documented |
| `venice.image_montage` | 241 | Retain; boundary documented |
| `venice.mcp_server` | 1036 | Split responsibilities; retain public aliases |
| `venice.remote_media` | 739 | Retain; boundary documented |
| `venice.userconfig` | 918 | Retain; boundary documented |
| `scripts.openapi_contract` | 724 | Retain; boundary documented |
| `scripts.openapi_surface` | 113 | New full-surface inventory |
| `scripts.mcp_surface` | 49 | New full-surface inventory |
| `scripts.verify_release` | 166 | Retain; boundary documented |

## Credential diagnostics

`_redact.redact_text` removes known complete credentials and JSON/URL encodings
before diagnostic truncation. It never loads files. The HTTP error path passes
the active client credential explicitly, so the guard also covers file-loaded
credentials without the audit reading a file. Error-code fields and connection
reasons receive the same treatment. SDK diagnostic handling obtains the bearer
value from the failed request, and MCP chat/vision errors reuse that bounded,
redacted rendering. The shared MCP error envelope also removes the environment
credential. Existing login/named-secret UX stays hidden-input only; no token
argument was added.

Regressions use synthetic credentials, including one crossing the 2 KB excerpt
boundary, and verify that neither exception text nor stored diagnostic fields
retain them. This is diagnostic redaction, not a claim that arbitrary generated
content or third-party debug logging has been globally filtered.

## Validation results

Environment: Python 3.12; isolated local venv with `mcp==2.3.0`,
`openai==3.41.0`, `pexpect==4.9.0`, `PyYAML==6.0.2`, and `ruff==0.17.3`.
Every suite used a temporary HOME and `VENICE_API_KEY=test-fake-key`.

| Check | Actual result |
| --- | --- |
| First full `make test` during implementation | **FAIL**: 2,048 tests, 1 failure, 1 skip, 155.596 s. Cleanup contract test still used retrieve builder after manifest update. This run overlapped implementation and is not described as a pristine baseline. |
| First focused MCP/client run | **FAIL**: 116 tests, 47 errors, 7.168 s. Missing `Annotated`/`Field` imports after extraction; fixed without changing assertions. |
| Initial registry import smoke | **FAIL**: extraction omitted `MODEL_TYPES`; corrected the import. |
| Intermediate focused run | **FAIL**: 432 tests, 1 error, 4.850 s. Existing test accessed `_agent._video`; restored its compatibility alias. |
| Final focused regressions, contract, MCP, client, agent, and harness | **PASS**: 438 tests, 14.763 s, no failures/errors/skips. |
| Standalone every-tool MCP harness | **PASS**: all 4 protocol tests, 4.861 s. Also included in focused/full suites. |
| Final full `make test` | **PASS**: 2,062 tests, zero failures/errors, 1 expected skip, 164.327 s (2,061 passed). |
| `make lint` | **PASS**: repository syntax lint remained enabled throughout. |
| `make scan` | **PASS**: tracked source invisible-character/identifier scan clean, including staged additions. |
| `make openapi-check` | **PASS**: 18 implemented operations fully classified. |
| `make openapi-live` | **PASS**: second unauthenticated public-spec fetch matches reviewed lock; no implemented request drift. |
| Ruff fatal-error rules across `src scripts tests` | **PASS**: `--isolated --select E9,F63,F7,F82`. |
| Ruff all Pyflakes rules on new/changed implementation and new test files | **PASS**: `--isolated --select F` on the affected file list. |
| Broad inherited Ruff profile, initial | **FAIL**: 1,518 diagnostics before source edits. |
| Broad inherited Ruff profile, final | **FAIL**: 1,574 diagnostics; this optional broad profile is not green. |
| `git diff --check` / staged equivalent | **PASS** after correcting four trailing blank lines in extracted modules. |

The broad inherited Ruff configuration includes modernization/style rules beyond
the repository's syntax-lint target. Extracted MCP wrappers intentionally retain
concrete annotations and no postponed-annotation import for SDK schema building;
this conflicts with its `FA100` suggestions. Other diagnostics include typing,
import layout, mutable class attributes, and style. The increase is not reported
as pre-existing-only debt or as a pass. No lint configuration, assertions, hooks,
or verification gates were disabled to get a green result on a narrower check.

The SDK emits a deprecation warning about unspecified
`AuthSettings.validate_token_resource`; the existing verifier validates its
configured audience itself. No SDK-minimum compatibility change was made during
this audit. The dedicated real-browser containment test is explicitly skipped by
the normal suite and belongs to its separate browser CI job; that job was not run.
No tests were run on other Python versions in this local audit.

### Existing assertion change and test authorship

Only `tests/test_openapi_contract.py` changed among existing test modules. Its
`POST /video/complete` entry now calls `video.complete_body("q")` instead of
`video.job_body("m", "q")`, because cleanup and retrieve no longer have the same
wire fields. The exact set-equality assertion against the implementation
manifest is unchanged; no assertion was weakened or removed. New tests check the
exact cleanup body, preserved audio/retrieve fields, redaction, full inventory
coverage, and every advertised MCP tool through protocol sessions.

All new/changed tests are isolated in commit `e4c243b`, explicitly marked
`Agent-Role: tests (Codex, needs non-author re-make)`. They are author-authored
regressions and still require the requested independent non-author re-make; this
audit does not claim independent review or independent test authorship.

## Local commits

- `1a844f2`: API/MCP inventory, cleanup sync, diagnostic redaction, module splits,
  and README module documentation. Trailer: `Agent-Role: change`.
- `e4c243b`: test changes only. Trailer:
  `Agent-Role: tests (Codex, needs non-author re-make)`.
- Final report commit: documentation only, `Agent-Role: change`.

Every commit uses `Projects Lead (agent) <agents@noreply.invalid>` and
`Co-authored-by: Codex (gpt-6.1-sol)`. Nothing was pushed; no PR, forge command,
SSH access, history rewrite, force operation, or verification bypass was used.
