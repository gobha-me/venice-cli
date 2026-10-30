# Reviewed contract updates

The lock records upstream request schemas, including optional fields beyond the
curated CLI surface. `implementation.json` classifies top-level request fields;
it does not imply that every nested message option has a CLI flag. See
`CONTRIBUTING.md` for the refresh and drift-check procedure.

## 2026-10-27 (#247)

Reviewed public Swagger version `20261026.193006`, SHA-256
`11e9fead672ea14e0f3e878fd48c62d72bc7aa2fd583e66048a2d03649d76d9b`,
against the previous lock at `20260918.184256`. All 37 normalized changes were
review-required additions or relaxed limits.

- Image generation, edit, and multi-edit now expose optional
  `transparent_background` through CLI flags, config, MCP, and agent tools.
  Omitted values remain omitted; explicit `false` is preserved. Image generation
  sidecars also retain it for replay. Support depends on the selected model.
- Video queue exposes optional `image_strength` from 0 to 1 on those same
  surfaces. It is absent from quote requests because the quote schema does not
  accept it. Video prompts and negative prompts have a 32,000-character limit,
  validated before catalog/quote requests on CLI and tool paths.
- Video aspect ratio `2:1` and resolution `768p` use the existing live model
  catalog constraints. There is no static allowlist to expand: values are
  accepted only when the selected model advertises them.
- TTS adds `tts-heygen-voice` and `tts-heygen-voice-turbo`. Edit model enums add
  `flux-3-image-edit`, `ideogram-v4-5-edit`, `nano-banana-2-1-edit`,
  `qwen-image-2-1-pro-edit`, and `qwen-image-2-1-turbo-edit`. Existing model IDs
  are free-form at the parser; TTS and multi-edit validate against the catalog.
  No new hardcoded model lists or changes to defaults are needed.
- Chat `messages[].content[].image_url.detail` (`auto`, `high`, `low`,
  `original`) is intentionally not exposed by the vision tools: they construct
  only `image_url.url`, leaving detail to the server default. Message-level
  `cache_control` for system/assistant messages is also not exposed by chat
  flags or constructed by the CLI. Provider cache metrics and cache-probe do
  not imply an explicit cache-control request surface. These optional nested
  additions are recorded in the lock without inventing new chat controls.
- Multipart image requests remain intentionally unsupported: the client uses
  JSON image inputs, as recorded in the manifest.

The drift checker and scheduled workflow retain their existing behavior: every
implemented-operation change, including additive review-required changes,
fails `check-live` until reviewed and refreshed. Only the public Swagger was
fetched for this update; no authenticated Venice endpoints were called.

## 2026-10-29 (local R1 audit)

Reviewed public Swagger version `20261027.204242`, SHA-256
`ef6561d9ff9f98cff00404e502dea8eb41a69520d9bbed98108214b41daf1d3b`.
The operation inventory remains at 55 endpoints; the 18 implemented request
contracts differ in two places: `POST /video/complete.model` is now deprecated
and no longer required. Cleanup now emits only `queue_id`; retrieve continues
to send both `model` and `queue_id`.

[surface.md](surface.md) lists every current endpoint and top-level request field,
with support/omission classifications. [surface.json](surface.json) also records
nested request and response schemas and response headers for all 55 operations.
This establishes a response/unsupported-field baseline; the previous lock did
not retain those schemas, so historical changes to those fields cannot be
reconstructed from it. [mcp-surface.json](mcp-surface.json) records the actual
negotiated schemas for all three MCP profiles.

Regenerate the full inventories with `scripts/openapi_surface.py` and
`scripts/mcp_surface.py` as described in the README module guide. They complement
the existing curated request drift gate and do not change its fail-on-drift policy.
See [issue #252](https://github.com/gobha-me/venice-cli/issues/252) for the audit findings and limitations.
