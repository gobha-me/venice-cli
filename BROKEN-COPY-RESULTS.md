# Broken-copy results (r1 test re-make)

Each break: a copy of `src/`, `scripts/`, `tests/`, `contracts/` was made into
`runs/test-r1/broken/`, exactly **one** edit was applied (the driver asserts the
anchor text occurs exactly once, so a stale anchor cannot silently become a
no-op), and the r1 suites were run from inside the copy with
`PYTHONPATH=<copy>/src` (so path-loaded `scripts/` and `contracts/` were the
broken ones too):

```
python -m unittest tests.test_r1_audit tests.test_openapi_surface \
                   tests.test_openapi_contract tests.test_mcp_harness
```

Environment: Python 3.12, PyYAML present, **`mcp` and `openai` SDKs absent** --
the 7 SDK-gated tests skip here (listed at the end). No real key, no network
beyond 127.0.0.1; `VENICE_API_KEY` was stripped from the child environment.

**Result: baseline copy 101 run / 0 failed / 7 skipped. 65/65 breaks caught.**
After the last round, every non-skipped test I wrote has failed under at least
one break (see "Never-failed" below).

How it was driven: the session's command policy refused `cp -r` and a non-`src`
`PYTHONPATH`, so the driver (`runs/test-r1/r1break/driver.py`, kept outside the
repo) was launched through a temporary, **uncommitted** unittest shim
(`tests/_r1_breaks.py`, deleted afterwards). The driver only writes under
`runs/test-r1/`.

## Breaks and the tests that caught them

Test ids are `module.Class.test`. `(E)` = error rather than assertion failure.

### Video cleanup contract

| id | break | failing tests |
|----|-------|---------------|
| V1 | `complete_body` also emits `model` | test_openapi_contract.CommittedInventoryTests.test_actual_request_builders_match_manifest_field_sets; test_r1_audit.CliVideoStatusCleanup.{test_cleanup_posts_only_queue_id_after_saving, test_failed_cleanup_warns_without_echoing_key}; CompleteBodyContract.{test_complete_body_is_exactly_queue_id, test_complete_body_returns_independent_dicts}; McpCleanupDelegation.{test_complete_job_video_forgets_even_when_backend_rejects, test_complete_job_video_sends_queue_id_and_forgets_private_metadata, test_job_result_video_completes_with_queue_id_only, test_video_tool_end_to_end_cleanup_body, test_video_tool_survives_cleanup_failure} |
| V2 | CLI `video-status` cleanup sends `job_body(model, q)` | CliVideoStatusCleanup.{test_cleanup_posts_only_queue_id_after_saving, test_failed_cleanup_warns_without_echoing_key} |
| V3 | CLI ignores `--no-cleanup` | CliVideoStatusCleanup.test_no_cleanup_flag_skips_complete |
| V4 | retrieve body drops `model` | test_openapi_contract...test_actual_request_builders_match_manifest_field_sets; CliVideoStatusCleanup.test_cleanup_posts_only_queue_id_after_saving; CompleteBodyContract.test_retrieve_body_still_carries_model; McpCleanupDelegation.{test_job_result_video_completes_with_queue_id_only, test_video_tool_end_to_end_cleanup_body} |
| V5 | `complete_body` grows an unused `model` parameter | CompleteBodyContract.test_complete_body_cannot_be_given_a_model |
| M1 | MCP `video_tool` cleanup sends `job_body` | McpCleanupDelegation.{test_video_tool_end_to_end_cleanup_body, test_video_tool_survives_cleanup_failure} |
| M2 | MCP `complete_job` (video) sends `job_body` | McpCleanupDelegation.{test_complete_job_video_forgets_even_when_backend_rejects, test_complete_job_video_sends_queue_id_and_forgets_private_metadata, test_job_result_video_completes_with_queue_id_only} |
| M3 | MCP `complete_job` (audio) drops `model` | McpCleanupDelegation.{test_complete_job_audio_types_keep_model, test_job_result_audio_completes_with_model} |
| M4 | `complete_job` does not forget private video metadata | McpCleanupDelegation.{test_complete_job_video_forgets_even_when_backend_rejects, test_complete_job_video_sends_queue_id_and_forgets_private_metadata} |
| M5 | unknown job types fall back to the video route | McpCleanupDelegation.test_complete_job_unknown_type_sends_nothing; McpErrorEnvelope.test_reflected_argument_is_redacted_in_tool_error |

### `_redact.redact_text`

| id | break | failing tests |
|----|-------|---------------|
| R1 | skip URL-encoding (`quote` and `quote_plus`) | ClientErrorRedaction.{test_encoded_echoes_are_redacted, test_key_in_request_path_is_redacted_from_url}; McpErrorEnvelope.test_err_redacts_environment_key; RedactText.{test_encoded_form_containing_the_raw_form_is_removed_whole, test_environment_secret_is_removed_without_being_passed, test_every_encoded_variant_of_an_explicit_secret_is_removed, test_slash_is_percent_encoded_in_the_quote_variant}; SdkDiagnostics.test_encoded_bearer_echo_is_redacted |
| R2 | skip `quote()` only | same as R1 minus RedactText.test_encoded_form_containing_the_raw_form_is_removed_whole |
| R3 | skip `quote_plus()` only | ClientErrorRedaction.test_encoded_echoes_are_redacted; McpErrorEnvelope.test_err_redacts_environment_key; RedactText.{test_environment_secret_is_removed_without_being_passed, test_every_encoded_variant_of_an_explicit_secret_is_removed, test_slash_is_percent_encoded_in_the_quote_variant} |
| R4 | `quote()` keeps `/` (default `safe`) | ClientErrorRedaction.{test_encoded_echoes_are_redacted, test_key_in_request_path_is_redacted_from_url}; McpErrorEnvelope.test_err_redacts_environment_key; RedactText.{test_environment_secret_is_removed_without_being_passed, test_every_encoded_variant_of_an_explicit_secret_is_removed}; SdkDiagnostics.test_encoded_bearer_echo_is_redacted |
| R5 | skip JSON-escaped form | ClientErrorRedaction.test_encoded_echoes_are_redacted; McpErrorEnvelope.test_err_redacts_environment_key; RedactText.{test_environment_secret_is_removed_without_being_passed, test_every_encoded_variant_of_an_explicit_secret_is_removed, test_json_escapes_unicode_and_controls} |
| R6 | no `VENICE_API_KEY` fallback | ClientErrorRedaction.test_file_loaded_key_is_redacted_when_environment_holds_another; McpErrorEnvelope.{test_err_redacts_environment_key, test_reflected_argument_is_redacted_in_tool_error}; RedactText.{test_environment_and_explicit_secrets_both_apply, test_environment_secret_is_removed_without_being_passed}; SdkDiagnostics.test_missing_or_odd_headers_fall_back_to_environment; VeniceAPIErrorDirect.test_environment_key_redacted_without_secret_keyword |
| R7 | shortest-first replacement order | RedactText.test_longer_secret_wins_over_its_prefix |
| R8 | empty-secret guard removed (`""` is "replaced" between every character) | 32 tests -- ClientErrorRedaction.{test_connection_error_reason_is_redacted, test_echoed_bearer_is_redacted_from_every_field, test_empty_error_body, test_encoded_echoes_are_redacted, test_excerpt_boundary_never_leaks_a_partial_key, test_excerpt_length_at_exactly_2kb_and_neighbours, test_file_loaded_key_is_redacted_without_any_environment_key, test_key_in_request_path_is_redacted_from_url, test_malformed_json_error_body_is_still_redacted, test_message_body_preview_is_capped_at_500_after_redaction, test_top_level_code_field_is_redacted}; McpErrorEnvelope.{test_upstream_echo_through_job_status_is_redacted, test_upstream_echo_through_video_tool_is_redacted}; RedactText.{test_empty_and_non_string_secrets_are_ignored, test_encoded_form_containing_the_raw_form_is_removed_whole, test_every_encoded_variant_of_an_explicit_secret_is_removed, test_every_occurrence_is_removed, test_json_escapes_unicode_and_controls, test_longer_secret_wins_over_its_prefix, test_non_string_values_are_stringified, test_slash_is_percent_encoded_in_the_quote_variant}; SdkDiagnostics.{test_bearer_from_request_is_redacted_without_environment, test_empty_bearer_does_not_mangle_text, test_encoded_bearer_echo_is_redacted, test_html_bodies_collapse_to_title, test_missing_or_odd_headers_fall_back_to_environment, test_redaction_happens_before_truncation, test_status_to_exit_connection_error_is_redacted}; VeniceAPIErrorDirect.{test_environment_key_redacted_without_secret_keyword, test_falsy_code_stays_none_and_is_not_rendered, test_non_string_code_is_rendered_as_text, test_secret_keyword_redacts_url_body_code_and_message} |
| R9 | non-string guard removed | RedactText.test_empty_and_non_string_secrets_are_ignored (E) |

### Client error path (`VeniceClient` / `VeniceAPIError`)

| id | break | failing tests |
|----|-------|---------------|
| C1 | excerpt truncated **before** redaction | ClientErrorRedaction.test_excerpt_boundary_never_leaks_a_partial_key |
| C2 | excerpt redaction omits the client key | ClientErrorRedaction.test_excerpt_boundary_never_leaks_a_partial_key |
| C3 | `_raise_api_error` drops `secret=` | ClientErrorRedaction.{test_echoed_bearer_is_redacted_from_every_field, test_key_in_request_path_is_redacted_from_url, test_top_level_code_field_is_redacted}; McpErrorEnvelope.test_upstream_echo_through_video_tool_is_redacted |
| C4 | URLError reason not redacted | ClientErrorRedaction.test_connection_error_reason_is_redacted |
| C5 | OSError text not redacted | ClientErrorRedaction.test_connection_error_reason_is_redacted |
| C6 | `code` not redacted | CliVideoStatusCleanup.test_failed_cleanup_warns_without_echoing_key; ClientErrorRedaction.{test_echoed_bearer_is_redacted_from_every_field, test_top_level_code_field_is_redacted}; McpErrorEnvelope.test_upstream_echo_through_video_tool_is_redacted; VeniceAPIErrorDirect.{test_non_string_code_is_rendered_as_text, test_secret_keyword_redacts_url_body_code_and_message} |
| C7 | `url` not redacted | ClientErrorRedaction.test_key_in_request_path_is_redacted_from_url; VeniceAPIErrorDirect.test_secret_keyword_redacts_url_body_code_and_message |
| C8 | `body` not redacted | VeniceAPIErrorDirect.{test_environment_key_redacted_without_secret_keyword, test_secret_keyword_redacts_url_body_code_and_message} |
| C9 | body decoded with `errors="ignore"` | ClientErrorRedaction.test_html_and_invalid_utf8_bodies_are_redacted |
| C10 | excerpt cap 2049 (off by one) | ClientErrorRedaction.{test_excerpt_boundary_never_leaks_a_partial_key, test_excerpt_length_at_exactly_2kb_and_neighbours} |
| C11 | message preview 501 (off by one) | ClientErrorRedaction.test_message_body_preview_is_capped_at_500_after_redaction |

### SDK diagnostics and the MCP envelope

| id | break | failing tests |
|----|-------|---------------|
| O1 | `_safe_api_error` ignores request bearer | SdkDiagnostics.{test_bearer_from_request_is_redacted_without_environment, test_encoded_bearer_echo_is_redacted, test_redaction_happens_before_truncation, test_status_to_exit_connection_error_is_redacted} |
| O2 | bearer prefix case-sensitive | SdkDiagnostics.test_bearer_from_request_is_redacted_without_environment |
| O3 | `status_to_exit` prints raw connection error | SdkDiagnostics.test_status_to_exit_connection_error_is_redacted |
| O4 | truncate before redacting | SdkDiagnostics.test_redaction_happens_before_truncation |
| E1 | MCP `_err` not redacted | McpErrorEnvelope.{test_err_redacts_environment_key, test_reflected_argument_is_redacted_in_tool_error} |

### MCP tool-set drift

| id | break | failing tests |
|----|-------|---------------|
| T1 | drop `venice_music` from stdio server | StaticToolRegistrations.test_stdio_registrations_match_committed_profile |
| T2 | add an extra stdio tool | StaticToolRegistrations.test_stdio_registrations_match_committed_profile |
| T3 | drop `venice_media_delete` from http_media | StaticToolRegistrations.test_http_media_registrations_match_committed_profile |
| T4 | register `venice_media_import` unconditionally | StaticToolRegistrations.{test_http_media_registrations_match_committed_profile, test_http_minimal_registrations_match_committed_profile} |
| T5 | committed mcp-surface.json: http_minimal tool renamed | CommittedMcpSurface.{test_entries_are_complete_tool_descriptors, test_profile_relationships}; StaticToolRegistrations.test_http_minimal_registrations_match_committed_profile |
| T6 | committed mcp-surface.json: `stdio` profile renamed | CommittedMcpSurface.{test_entries_are_complete_tool_descriptors, test_profile_relationships (E), test_profiles}; StaticToolRegistrations.test_stdio_registrations_match_committed_profile (E) |
| T7 | drop minimal-profile `venice_vision` | StaticToolRegistrations.{test_extractor_sees_both_vision_variants, test_http_minimal_registrations_match_committed_profile} |

With the SDK installed, T1-T4 and T7 would additionally fail
`test_mcp_harness.*` (listed == committed profile, called == listed) and
`LiveMcpInventory.test_inventory_equals_committed_snapshot`. **Not observed
here; see "Skipped" below.**

### `scripts/openapi_surface.py` and committed snapshots

| id | break | failing tests |
|----|-------|---------------|
| S1 | `request_body_required` ignores `$ref` bodies | BuildSurface.test_request_body_required_follows_refs_and_defaults_false |
| S2 | status keys not stringified | BuildSurface.test_unquoted_yaml_status_codes_become_string_keys |
| S3 | unsupported coverage text changed | BuildSurface.{test_coverage_comes_from_manifest_or_is_marked_unsupported, test_render} |
| S4 | response `$ref` not resolved | BuildSurface.test_responses_resolve_refs_headers_and_stringify_status (E) |
| S5 | header `$ref` not resolved | BuildSurface.test_responses_resolve_refs_headers_and_stringify_status |
| S6 | render drops `(required)` | BuildSurface.test_render; CommittedSurface.test_committed_markdown_is_the_rendering_of_committed_json |
| S7 | render drops "No named schema properties" | BuildSurface.test_render; CommittedSurface.test_committed_markdown_is_the_rendering_of_committed_json |
| S8 | render ignores intentional omissions | BuildSurface.test_render; CommittedSurface.test_committed_markdown_is_the_rendering_of_committed_json |
| S9 | `main` writes `.markdown` | BuildSurface.test_main_writes_json_and_markdown (E) |
| S10 | committed surface.md edited by hand | CommittedSurface.test_committed_markdown_is_the_rendering_of_committed_json |
| S11 | committed surface.json drops an unsupported op | CommittedSurface.{test_committed_markdown_is_the_rendering_of_committed_json (E), test_every_upstream_operation_is_inventoried_exactly_once, test_operation_ids_match_upstream (E)} |
| S12 | committed surface.json source version drifts | CommittedSurface.{test_committed_markdown_is_the_rendering_of_committed_json, test_header_matches_lock} |
| S13 | committed coverage drifts from manifest | CommittedSurface.{test_committed_markdown_is_the_rendering_of_committed_json, test_coverage_and_request_shapes} |
| S14 | committed response lacks `headers` | CommittedSurface.test_every_operation_declares_responses_with_content_and_headers |
| S15 | `build_surface` emits format 2 | BuildSurface.test_inventory_covers_all_operations_and_source |
| S16 | response schemas lose nested `properties` | BuildSurface.{test_nested_response_fields_of_unsupported_operations_are_retained (E), test_responses_resolve_refs_headers_and_stringify_status (E)} |
| S17 | missing-upstream manifest ops silently skipped | BuildSurface.test_manifest_operation_missing_upstream_is_an_error |
| S18 | coverage default written into the manifest | BuildSurface.test_inputs_are_not_mutated |
| S19 | undeclared fields rendered "omitted" | BuildSurface.test_render_marks_undeclared_fields_unsupported_and_optional_params; CommittedSurface.test_committed_markdown_is_the_rendering_of_committed_json |
| S20 | every parameter rendered `(required)` | same two as S19 |
| S21 | committed surface.json renames an implemented op | CommittedSurface.{test_committed_markdown... (E), test_coverage_and_request_shapes, test_every_implemented_operation_is_in_the_surface, test_every_upstream_operation_is_inventoried_exactly_once, test_operation_ids_match_upstream (E)} |
| X1 | `openapi_contract` ignores path-level parameters | BuildSurface.{test_parameters_are_inherited_from_the_path_item, test_render, test_render_marks_undeclared_fields_unsupported_and_optional_params} |
| Z1 | **test-oracle** break: AST extractor ignores `name=` | StaticToolRegistrations.{test_extractor_rejects_drift, test_extractor_sees_both_vision_variants, test_http_media_registrations_match_committed_profile, test_http_minimal_registrations_match_committed_profile} |

Z1 breaks the test's own helper, not product code: `test_extractor_rejects_drift`
is a self-check of the AST oracle, and no product change can fail it by design.

## Gaps found and closed during the run

* **S2** was initially *not* caught: the synthetic spec was serialized as JSON,
  which forces every key to a string before YAML parsing, so an unstringified
  `200` could never appear. Closed with
  `test_unquoted_yaml_status_codes_become_string_keys` (raw YAML with bare `200:`).
* A first pass left 6 r1-audit and 13 surface tests that no break failed. Each
  got a targeted break (V5, M5, C9-C11, T6, T7, S12-S21, X1, Z1). The three
  tests with no product-level failure mode were **removed** rather than kept as
  unfalsifiable: a `str` return-type check, a refused-connection check
  that only asserted `status == 0`, and a variants-are-distinct guard on the
  test's own fixture.

## Never-failed (by design or out of scope)

13 tests never failed. All are pre-existing `tests/test_openapi_contract.py`
tests (only included because of the one-line edit) and are not part of this
re-make: `CommittedInventoryTests.{test_background_remove_bodies_match_exactly_one_live_union_branch,
test_contract_files_are_canonical_json, test_manifest_completely_classifies_the_locked_contract,
test_mcp_and_agent_share_every_model_facing_media_contract,
test_model_specific_constraints_point_back_to_models,
test_required_request_fields_cannot_be_marked_unsupported,
test_source_endpoint_literals_are_declared}` and
`DriftClassificationTests.{test_added_required_field_is_breaking, test_enum_narrowing_is_breaking,
test_optional_field_addition_requires_review, test_unresolved_reference_fails_closed,
test_unsupported_addition_is_informational, test_yaml_loader_preserves_unquoted_aspect_ratios}`.

## Skipped here: NOT proven (SDK absent)

* `test_mcp_harness.LocalProfile.{test_every_stdio_tool_over_a_real_subprocess, test_every_stdio_tool_in_process, test_upstream_credential_echo_never_reaches_the_tool_result}`
* `test_mcp_harness.RemoteProfiles.{test_every_media_tool_and_job_lifecycle, test_minimal_profile_lists_and_calls_both_tools}`
* `test_openapi_surface.LiveMcpInventory.{test_inventory_equals_committed_snapshot, test_main_writes_the_committed_bytes}`

These have **never been executed**: neither passing nor failing has been
observed. They must be run (and broken-copy checked) in an environment with
`pip install -e ".[all,test]"` before this branch is trusted. The SDK-free
`StaticToolRegistrations` and `CommittedMcpSurface` classes provide the
tool-drift guarantee that *was* proven here (T1-T7).
