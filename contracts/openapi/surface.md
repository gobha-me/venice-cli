# Public API surface inventory

Source: https://docs.venice.ai/swagger.yaml (version `20261027.204242`).

Generated with `scripts/openapi_surface.py`. Full nested request and response
fields, unions, enums, bounds, and response headers are in [surface.json](surface.json).
Coverage describes the union of CLI, internal, agent, and MCP consumers;
it does not claim that every field has a CLI flag or MCP argument.
Unimplemented operations are existing surface gaps, not newly added endpoints.

## `GET /api_keys`

No dedicated client command or MCP tool; generic HTTP transport only.

Responses: `200`, `401`, `403`, `500`.

## `POST /api_keys`

No dedicated client command or MCP tool; generic HTTP transport only.

Request `application/json`:

- `apiKeyType`: unsupported
- `consumptionLimit`: unsupported
- `description`: unsupported
- `expiresAt`: unsupported
- `ipAllowlist`: unsupported
- `limitPeriod`: unsupported
- `modelPrivacy`: unsupported

Responses: `200`, `400`, `401`, `403`, `429`, `500`.

## `PATCH /api_keys`

No dedicated client command or MCP tool; generic HTTP transport only.

Request `application/json`:

- `consumptionLimit`: unsupported
- `description`: unsupported
- `expiresAt`: unsupported
- `id`: unsupported
- `ipAllowlist`: unsupported
- `limitPeriod`: unsupported
- `modelPrivacy`: unsupported

Responses: `200`, `400`, `401`, `403`, `500`.

## `DELETE /api_keys`

No dedicated client command or MCP tool; generic HTTP transport only.

Parameters: `query:id`.

Responses: `200`, `400`, `401`, `403`, `500`.

## `GET /api_keys/generate_web3_key`

No dedicated client command or MCP tool; generic HTTP transport only.

Responses: `200`.

## `POST /api_keys/generate_web3_key`

No dedicated client command or MCP tool; generic HTTP transport only.

Request `application/json`:

- `address`: unsupported
- `apiKeyType`: unsupported
- `consumptionLimit`: unsupported
- `description`: unsupported
- `expiresAt`: unsupported
- `ipAllowlist`: unsupported
- `limitPeriod`: unsupported
- `modelPrivacy`: unsupported
- `signature`: unsupported
- `token`: unsupported

Responses: `200`.

## `GET /api_keys/rate_limits`

Consumers: `cli:balance`, `internal:spend_display`.

Responses: `200`, `401`, `403`, `500`.

## `GET /api_keys/rate_limits/log`

No dedicated client command or MCP tool; generic HTTP transport only.

Responses: `200`, `401`, `403`, `500`.

## `GET /api_keys/{id}`

No dedicated client command or MCP tool; generic HTTP transport only.

Parameters: `path:id` (required).

Responses: `200`, `401`, `403`, `500`.

## `POST /audio/complete`

Consumers: `cli:sfx_status`, `cli:music_status`, `mcp:job_result`, `agent:job_result`.

Request `application/json`:

- `model`: implemented
- `queue_id`: implemented

Responses: `200`, `400`, `401`, `402`, `403`, `500`.

## `POST /audio/queue`

Consumers: `cli:sfx`, `cli:music`, `mcp:sfx_music`, `agent:sfx_music`.

Request `application/json`:

- `duration_seconds`: implemented
- `force_instrumental`: implemented
- `language_code`: The curated music surface does not expose language selection.
- `loop`: Loop generation is not exposed by the current audio tools.
- `lyrics_optimizer`: Lyrics optimization is not exposed by the current audio tools.
- `lyrics_prompt`: implemented
- `model`: implemented
- `prompt`: implemented
- `speed`: implemented
- `voice`: Voice selection is not exposed for queued audio generation.

Responses: `200`, `400`, `401`, `402`, `403`, `422`, `429`, `500`, `503`.

## `POST /audio/quote`

Consumers: `cli:sfx`, `cli:music`, `mcp:sfx_music`, `agent:sfx_music`.

Request `application/json`:

- `character_count`: TTS uses catalog pricing locally and does not call the quote endpoint.
- `duration_seconds`: implemented
- `model`: implemented

Responses: `200`, `400`, `403`, `500`.

## `POST /audio/retrieve`

Consumers: `cli:sfx_status`, `cli:music_status`, `mcp:job_status_result`, `agent:job_status_result`.

Request `application/json`:

- `delete_media_on_completion`: Cleanup is a separate explicit complete call after local persistence succeeds.
- `model`: implemented
- `queue_id`: implemented

Responses: `200`, `400`, `401`, `402`, `403`, `404`, `422`, `429`, `500`, `503`.

## `POST /audio/speech`

Consumers: `cli:tts`, `mcp:tts`, `agent:tts`.

Request `application/json`:

- `input`: implemented
- `language`: Language hints are not exposed by the current TTS surface.
- `model`: implemented
- `prompt`: Voice-conditioning prompts are not exposed by the current TTS surface.
- `response_format`: implemented
- `speed`: implemented
- `streaming`: The CLI requests one bounded response and persists it atomically.
- `temperature`: Sampling controls are intentionally omitted from the curated TTS surface.
- `top_p`: Sampling controls are intentionally omitted from the curated TTS surface.
- `voice`: implemented

Responses: `200`, `400`, `401`, `402`, `403`, `415`, `429`, `500`, `503`.

## `POST /audio/transcriptions`

No dedicated client command or MCP tool; generic HTTP transport only.

Request `multipart/form-data`:

- `file`: unsupported
- `language`: unsupported
- `model`: unsupported
- `response_format`: unsupported
- `timestamps`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `413`, `415`, `422`, `429`, `500`, `503`.

## `POST /audio/voice-changer/complete`

No dedicated client command or MCP tool; generic HTTP transport only.

Request `application/json`:

- `model`: unsupported
- `queue_id`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `429`, `500`.

## `POST /audio/voice-changer/queue`

No dedicated client command or MCP tool; generic HTTP transport only.

Request `application/json`:

- `audio_url`: unsupported
- `file`: unsupported
- `model`: unsupported
- `remove_background_noise`: unsupported
- `seed`: unsupported
- `voice`: unsupported

Request `multipart/form-data`:

- `audio_url`: unsupported
- `file`: unsupported
- `model`: unsupported
- `remove_background_noise`: unsupported
- `seed`: unsupported
- `voice`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `413`, `422`, `429`, `500`, `503`.

## `POST /audio/voice-changer/quote`

No dedicated client command or MCP tool; generic HTTP transport only.

Request `application/json`:

- `duration_seconds`: unsupported
- `model`: unsupported

Responses: `200`, `400`, `403`, `500`.

## `POST /audio/voice-changer/retrieve`

No dedicated client command or MCP tool; generic HTTP transport only.

Request `application/json`:

- `delete_media_on_completion`: unsupported
- `model`: unsupported
- `queue_id`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `404`, `422`, `429`, `500`, `503`, `504`.

## `POST /audio/voices`

No dedicated client command or MCP tool; generic HTTP transport only.

Request `multipart/form-data`:

- `file`: unsupported
- `model`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `413`, `415`, `429`, `500`, `503`.

## `POST /augment/scrape`

No dedicated client command or MCP tool; generic HTTP transport only.

Request `application/json`:

- `url`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `429`, `500`, `502`.

## `POST /augment/search`

No dedicated client command or MCP tool; generic HTTP transport only.

Request `application/json`:

- `limit`: unsupported
- `query`: unsupported
- `search_provider`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `429`, `500`.

## `POST /augment/text-parser`

No dedicated client command or MCP tool; generic HTTP transport only.

Request `multipart/form-data`:

- `file`: unsupported
- `response_format`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `429`, `500`.

## `GET /billing/balance`

No dedicated client command or MCP tool; generic HTTP transport only.

Responses: `200`, `401`, `403`, `500`.

## `GET /billing/usage`

No dedicated client command or MCP tool; generic HTTP transport only.

Responses: `410`.

## `GET /billing/usage-analytics`

No dedicated client command or MCP tool; generic HTTP transport only.

Parameters: `query:endDate`, `query:lookback`, `query:startDate`.

Responses: `200`, `400`, `401`, `403`, `500`, `504`.

## `GET /billing/usage-history`

No dedicated client command or MCP tool; generic HTTP transport only.

Parameters: `query:currency`, `query:cursor`, `query:endTimestamp`, `query:pageSize`, `query:startTimestamp`.

Responses: `200`, `400`, `401`, `403`, `500`.

## `GET /characters`

No dedicated client command or MCP tool; generic HTTP transport only.

Parameters: `query:categories`, `query:isAdult`, `query:isPro`, `query:isWebEnabled`, `query:limit`, `query:modelId`, `query:offset`, `query:search`, `query:sortBy`, `query:sortOrder`, `query:tags`.

Responses: `200`, `401`, `403`, `500`.

## `GET /characters/{slug}`

No dedicated client command or MCP tool; generic HTTP transport only.

Parameters: `path:slug` (required).

Responses: `200`, `401`, `403`, `404`, `500`.

## `GET /characters/{slug}/reviews`

No dedicated client command or MCP tool; generic HTTP transport only.

Parameters: `path:slug` (required), `query:page`, `query:pageSize`.

Responses: `200`, `400`, `401`, `403`, `404`, `500`.

## `POST /chat/completions`

Consumers: `cli:cache-probe`, `cli:chat`, `cli:code`, `cli:review`, `mcp:chat_vision`, `agent:completion_loop`.

Parameters: `header:Accept-Encoding`.

Request `application/json`:

- `anon_user_id`: End-user tracking identifiers are not collected or sent.
- `fallbacks`: Fallback routing is not exposed by the curated completion surfaces.
- `frequency_penalty`: Advanced sampling controls are not exposed.
- `include`: Responses-style include controls are not used by chat completions.
- `logprobs`: Log-probability output is not exposed.
- `max_completion_tokens`: The compatibility surface currently uses max_tokens.
- `max_temp`: Dynamic-temperature controls are not exposed.
- `max_tokens`: implemented
- `messages`: implemented
- `metadata`: Server-side request metadata is not exposed.
- `min_p`: Advanced sampling controls are not exposed.
- `min_temp`: Dynamic-temperature controls are not exposed.
- `model`: implemented
- `n`: The CLI requests one completion per turn.
- `parallel_tool_calls`: Tool concurrency is controlled by the local dispatcher, not the upstream flag.
- `presence_penalty`: Advanced sampling controls are not exposed.
- `prompt_cache_key`: Prompt-cache affinity is server-managed for the current surface.
- `prompt_cache_retention`: Prompt-cache retention is not exposed.
- `reasoning`: Responses-style reasoning configuration is not exposed here.
- `reasoning_effort`: Reasoning effort is not exposed by the current CLI.
- `repetition_penalty`: Advanced sampling controls are not exposed.
- `response_format`: Structured response formatting is not exposed.
- `seed`: Completion seeding is not exposed.
- `stop`: Stop sequences are not exposed.
- `stop_token_ids`: Provider-specific stop token IDs are not exposed.
- `store`: Server-side completion storage is not enabled by the CLI.
- `stream`: implemented
- `stream_options`: implemented
- `temperature`: implemented
- `text`: Responses-style text configuration is not used by chat completions.
- `tool_choice`: implemented
- `tools`: implemented
- `top_k`: Advanced sampling controls are not exposed.
- `top_logprobs`: Log-probability output is not exposed.
- `top_p`: Advanced sampling controls are not exposed.
- `user`: End-user tracking identifiers are not collected or sent.
- `venice_parameters`: implemented
- `verbosity`: Verbosity controls are not exposed.

Responses: `200`, `400`, `401`, `402`, `403`, `415`, `429`, `500`, `503`, `504`.

## `GET /crypto/rpc/networks`

No dedicated client command or MCP tool; generic HTTP transport only.

Responses: `200`.

## `POST /crypto/rpc/{network}`

No dedicated client command or MCP tool; generic HTTP transport only.

Parameters: `header:Idempotency-Key`, `path:network` (required).

Request `application/json`:

- `id`: unsupported
- `jsonrpc`: unsupported
- `method`: unsupported
- `params`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `429`, `500`.

## `POST /decisions`

No dedicated client command or MCP tool; generic HTTP transport only.

Parameters: `header:Accept-Encoding`.

Request `application/json`:

- `images`: unsupported
- `model`: unsupported
- `questions`: unsupported
- `state`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `415`, `429`, `500`, `503`.

## `POST /embeddings`

Consumers: `cli:embed`, `cli:index`, `mcp:semantic_search`, `agent:semantic_search`.

Parameters: `header:Accept-Encoding`.

Request `application/json`:

- `dimensions`: implemented
- `encoding_format`: implemented
- `input`: implemented
- `model`: implemented
- `user`: End-user tracking identifiers are not collected or sent.

Responses: `200`, `400`, `401`, `402`, `403`, `415`, `429`, `500`, `503`.

## `POST /image/background-remove`

Consumers: `cli:bg_remove`, `mcp:bg_remove`, `agent:bg_remove`.

Request `application/json`:

- `anon_user_id`: End-user tracking identifiers are not collected or sent.
- `image`: implemented
- `image_url`: implemented

Request `multipart/form-data`:

The stdlib client uses bounded JSON base64 or URL input.

- `anon_user_id`: unsupported
- `image`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `415`, `429`, `500`, `503`.

## `POST /image/edit`

Consumers: `cli:image_edit`, `mcp:image_edit`, `agent:image_edit`.

Request `application/json`:

- `anon_user_id`: End-user tracking identifiers are not collected or sent.
- `aspect_ratio`: implemented
- `disable_prompt_optimization_thinking`: implemented
- `enhance_prompt`: implemented
- `image`: implemented
- `model`: implemented
- `modelId`: The deprecated modelId alias is intentionally not emitted for single-image edit.
- `output_format`: implemented
- `prompt`: implemented
- `resolution`: implemented
- `safe_mode`: implemented
- `transparent_background`: implemented

Request `multipart/form-data`:

The stdlib client uses bounded JSON base64 or URL input.

- `anon_user_id`: unsupported
- `aspect_ratio`: unsupported
- `disable_prompt_optimization_thinking`: unsupported
- `enhance_prompt`: unsupported
- `image`: unsupported
- `model`: unsupported
- `modelId`: unsupported
- `output_format`: unsupported
- `prompt`: unsupported
- `resolution`: unsupported
- `safe_mode`: unsupported
- `transparent_background`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `415`, `429`, `500`, `503`.

## `POST /image/generate`

Consumers: `cli:image`, `mcp:image`, `agent:image`.

Parameters: `header:Accept-Encoding`.

Request `application/json`:

- `anon_user_id`: End-user tracking identifiers are not collected or sent.
- `aspect_ratio`: implemented
- `cfg_scale`: implemented
- `disable_prompt_optimization_thinking`: implemented
- `embed_exif_metadata`: implemented
- `enable_web_search`: implemented
- `enhance_prompt`: implemented
- `format`: implemented
- `height`: implemented
- `hide_watermark`: implemented
- `inpaint`: Image editing is exposed through the dedicated edit operations.
- `lora_strength`: implemented
- `model`: implemented
- `negative_prompt`: implemented
- `prompt`: implemented
- `quality`: implemented
- `resolution`: implemented
- `return_binary`: The existing image decoder consumes the documented JSON image envelope.
- `safe_mode`: implemented
- `seed`: implemented
- `steps`: implemented
- `style_preset`: implemented
- `style_references`: implemented
- `transparent_background`: implemented
- `variants`: implemented
- `width`: implemented

Responses: `200`, `400`, `401`, `402`, `403`, `415`, `422`, `429`, `500`, `503`.

## `POST /image/multi-edit`

Consumers: `cli:image_edit_layers`, `mcp:image_edit_layers`, `agent:image_edit_layers`.

Request `application/json`:

- `anon_user_id`: End-user tracking identifiers are not collected or sent.
- `aspect_ratio`: implemented
- `disable_prompt_optimization_thinking`: implemented
- `enhance_prompt`: implemented
- `images`: implemented
- `modelId`: implemented
- `output_format`: implemented
- `prompt`: implemented
- `quality`: implemented
- `resolution`: implemented
- `safe_mode`: implemented
- `transparent_background`: implemented

Request `multipart/form-data`:

The stdlib client uses bounded JSON base64 inputs.

- `anon_user_id`: unsupported
- `aspect_ratio`: unsupported
- `disable_prompt_optimization_thinking`: unsupported
- `enhance_prompt`: unsupported
- `images`: unsupported
- `modelId`: unsupported
- `output_format`: unsupported
- `prompt`: unsupported
- `quality`: unsupported
- `resolution`: unsupported
- `safe_mode`: unsupported
- `transparent_background`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `415`, `429`, `500`, `503`.

## `GET /image/styles`

No dedicated client command or MCP tool; generic HTTP transport only.

Responses: `200`, `401`, `500`.

## `POST /image/upscale`

Consumers: `cli:upscale`, `mcp:upscale`, `agent:upscale`.

Request `application/json`:

- `creativity`: implemented
- `image`: implemented
- `scale`: implemented

Request `multipart/form-data`:

The stdlib client uses bounded JSON base64 input.

- `creativity`: unsupported
- `image`: unsupported
- `scale`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `415`, `429`, `500`, `503`.

## `POST /images/generations`

No dedicated client command or MCP tool; generic HTTP transport only.

Parameters: `header:Accept-Encoding`.

Request `application/json`:

- `anon_user_id`: unsupported
- `background`: unsupported
- `model`: unsupported
- `moderation`: unsupported
- `n`: unsupported
- `output_compression`: unsupported
- `output_format`: unsupported
- `prompt`: unsupported
- `quality`: unsupported
- `response_format`: unsupported
- `size`: unsupported
- `style`: unsupported
- `user`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `415`, `422`, `429`, `500`, `503`.

## `GET /models`

Consumers: `cli:models`, `cli:cache-probe`, `internal:model_validation`, `mcp:model_backed_tools`, `agent:model_backed_tools`.

Parameters: `query:type`.

Responses: `200`, `403`, `500`.

## `GET /models/compatibility_mapping`

No dedicated client command or MCP tool; generic HTTP transport only.

Parameters: `query:type`.

Responses: `200`, `401`, `403`, `500`.

## `GET /models/traits`

No dedicated client command or MCP tool; generic HTTP transport only.

Parameters: `query:type`.

Responses: `200`, `401`, `403`, `500`.

## `POST /responses`

No dedicated client command or MCP tool; generic HTTP transport only.

Request `application/json`:

- `anon_user_id`: unsupported
- `fallbacks`: unsupported
- `include`: unsupported
- `input`: unsupported
- `instructions`: unsupported
- `max_output_tokens`: unsupported
- `model`: unsupported
- `parallel_tool_calls`: unsupported
- `prompt_cache_retention`: unsupported
- `reasoning`: unsupported
- `stream`: unsupported
- `temperature`: unsupported
- `tool_choice`: unsupported
- `tools`: unsupported
- `top_p`: unsupported
- `venice_parameters`: unsupported
- `web_search`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `429`, `500`.

## `POST /systemone`

No dedicated client command or MCP tool; generic HTTP transport only.

Parameters: `header:Accept-Encoding`.

Request `application/json`:

- `images`: unsupported
- `model`: unsupported
- `questions`: unsupported
- `state`: unsupported

Responses: `200`, `400`, `401`, `402`, `403`, `415`, `429`, `500`, `503`.

## `POST /video/complete`

Consumers: `cli:video_status`, `mcp:job_result`, `agent:job_result`.

Request `application/json`:

- `model`: Deprecated and no longer required; video cleanup is identified by queue_id.
- `queue_id`: implemented

Responses: `200`, `400`, `401`, `402`, `403`, `409`, `500`.

## `POST /video/queue`

Consumers: `cli:video`, `mcp:video`, `agent:video`.

Request `application/json`:

- `aspect_ratio`: implemented
- `audio`: implemented
- `audio_url`: implemented
- `camera_trajectory`: Advanced camera-path control is outside the curated video surface.
- `compression`: Enhancement-only controls are outside the curated video surface.
- `consents`: Provider-specific consent envelopes are not exposed.
- `creativity`: Enhancement-only controls are outside the curated video surface.
- `duration`: implemented
- `elements`: implemented
- `end_image_url`: implemented
- `enhancement_model`: Enhancement-only controls are outside the curated video surface.
- `grain`: Enhancement-only controls are outside the curated video surface.
- `h264_output`: Enhancement-only controls are outside the curated video surface.
- `halo`: Enhancement-only controls are outside the curated video surface.
- `image_strength`: implemented
- `image_url`: implemented
- `keyframes`: Advanced keyframe control is not exposed.
- `model`: implemented
- `negative_prompt`: implemented
- `noise`: Enhancement-only controls are outside the curated video surface.
- `omni_reference_task_type`: Provider-specific Omni reference modes are not exposed.
- `output_format`: The current surface persists the server-selected video format.
- `prompt`: implemented
- `realism`: Enhancement-only controls are outside the curated video surface.
- `recover_detail`: Enhancement-only controls are outside the curated video surface.
- `reference_audio_urls`: implemented
- `reference_document_urls`: implemented
- `reference_image_urls`: implemented
- `reference_video_urls`: implemented
- `resolution`: implemented
- `scene_image_urls`: implemented
- `sharp`: Enhancement-only controls are outside the curated video surface.
- `slowdown_factor`: Enhancement-only controls are outside the curated video surface.
- `softness`: Enhancement-only controls are outside the curated video surface.
- `target_fps`: Enhancement-only controls are outside the curated video surface.
- `upscale_factor`: Enhancement-only controls are outside the curated video surface.
- `video_url`: implemented

Responses: `200`, `400`, `401`, `402`, `403`, `409`, `413`, `422`, `500`.

## `POST /video/quote`

Consumers: `cli:video`, `mcp:video`, `agent:video`.

Request `application/json`:

- `aspect_ratio`: implemented
- `audio`: implemented
- `duration`: implemented
- `enhancement_model`: Enhancement-only controls are outside the curated video surface.
- `model`: implemented
- `reference_video_total_duration`: implemented
- `resolution`: implemented
- `slowdown_factor`: Enhancement-only controls are outside the curated video surface.
- `target_fps`: Enhancement-only controls are outside the curated video surface.
- `upscale_factor`: Enhancement-only controls are outside the curated video surface.
- `video_url`: implemented

Responses: `200`, `400`, `403`.

## `POST /video/retrieve`

Consumers: `cli:video_status`, `mcp:job_status_result`, `agent:job_status_result`.

Request `application/json`:

- `delete_media_on_completion`: Cleanup is a separate explicit complete call after local persistence succeeds.
- `model`: implemented
- `queue_id`: implemented

Responses: `200`, `400`, `401`, `402`, `403`, `404`, `422`, `500`, `503`.

## `POST /video/transcriptions`

No dedicated client command or MCP tool; generic HTTP transport only.

Responses: `410`.

## `GET /x402/balance/{walletAddress}`

No dedicated client command or MCP tool; generic HTTP transport only.

Parameters: `header:SIGN-IN-WITH-X` (required), `path:walletAddress` (required).

Responses: `200`, `400`, `401`, `403`, `429`, `500`.

## `POST /x402/top-up`

No dedicated client command or MCP tool; generic HTTP transport only.

Parameters: `header:PAYMENT-SIGNATURE`.

Request `application/json`:

No named schema properties published.

Responses: `200`, `400`, `402`, `429`, `500`.

## `GET /x402/transactions/{walletAddress}`

No dedicated client command or MCP tool; generic HTTP transport only.

Parameters: `header:SIGN-IN-WITH-X` (required), `path:walletAddress` (required), `query:limit`, `query:offset`.

Responses: `200`, `400`, `401`, `403`, `429`, `500`.
