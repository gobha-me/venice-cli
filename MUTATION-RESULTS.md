# Mutation results (r1 follow-up)

## Status: run attempted, failed in mutmut's own harness. No score produced. 70% floor NOT demonstrated.

Command, exactly as the brief allows:

```
/config/agent-lanes/venice-cli/runs/lead-bin/mut-test-r1.sh
```

Mutant generation completed. The **stats phase** (mutmut's first pytest pass,
which maps tests to mutated functions) aborted at the first trampoline hit:

```
File ".../site-packages/mutmut/runners/harness.py", line 157, in pytest_runtest_teardown
    state().tests_by_mangled_function_name[function].add(strip_prefix(item._nodeid, prefix="mutants/"))
AttributeError: 'TestCaseFunction' object has no attribute '_nodeid'. Did you mean: 'nodeid'?
ERROR tests/test_r1_audit.py::CompleteBodyContract::test_complete_body_cannot_be_given_a_model
1 passed, 1 error in 2.13s
failed to collect stats. runner returned 1
```

Cause: the follow-up venv (`runs/test-r1-fu/venv`) has **pytest 9.2.0**
(`_pytest/_version.py`), and `_pytest/nodes.py` there has no `_nodeid`.
mutmut 3.8.0's harness reads `item._nodeid`. The lead's baseline used
pytest 9.1.1 (LEAD-RESULTS.md), where the attribute still exists. The failing
test is a 7de7a07 `test_r1_audit` test the lead's run also executed, so this
branch's changes did not cause it. It is an environment incompatibility.

I did not pin or downgrade pytest, patch mutmut, or run mutmut another way.
Each of those would route around the wrapper (see brief). To fix it, a lead
should run the wrapper with `pytest<9.2` in the venv, or with a mutmut release
that uses `item.nodeid`. `mutants/` was removed afterwards (`rm -rf mutants`).

| module | killed | survived | no tests | timeout | score |
|---|---|---|---|---|---|
| venice._redact | -- | -- | -- | -- | not produced (harness error) |
| venice.commands.video | -- | -- | -- | -- | not produced |
| scripts.openapi_surface | -- | -- | -- | -- | not produced |

Lead baseline at 7de7a07 for reference: _redact 35/43 = 81.4%; video 137/237
= 57.8%; openapi_surface 355 "no tests"; total 61.4% of tested.
**Whether this branch reaches 70% is unknown. It has not been measured.**

## Changes made toward the score (in this commit, unmeasured)

1. **scripts mapping fixed.** `tests/test_openapi_surface.py::_load_script`
   now names the module `scripts.<name>` instead of `r1_<name>`. mutmut keys
   mutants by the path-derived dotted name (`get_mutant_name` ->
   `scripts.openapi_surface.x_...`), and its trampoline records
   `orig.__module__`, so the two now match. Nothing is weakened: the same
   file is executed the same way with `scripts/` on `sys.path`. Only the
   module's `__name__` changed (verified: `tests.test_openapi_surface` 34/34
   with the SDKs, plus `test_mcp_harness`). Expect the 355 "no tests" to become
   killed or survived, which may move the total in **either** direction.
2. **Test selection extended** with existing modules that exercise
   `video.py`: `tests/test_video.py`, `tests/test_video_jobs.py`,
   `tests/test_openapi_drift_regressions.py`, plus the new
   `tests/test_r1_redact_boundary.py`. `only_mutate` is unchanged; no module
   was dropped.

## The 8 _redact survivors from the lead baseline

Analysis by reading the code. These kills have **not** been confirmed by mutmut.

| # | mutant | verdict | reason / killing test |
|---|---|---|---|
| 1 | env default `""` -> `None` | **equivalent** | `known.add(None)` is skipped by `isinstance(secret, str) and secret`, the same as `""` |
| 2 | env default omitted (`get("VENICE_API_KEY", )`) | **equivalent** | `.get` returns `None`; same as #1 |
| 3 | env default `""` -> `"XXXX"` | killable, should die | `RedactEncodedForms.test_text_is_untouched_when_no_credential_is_known`: with no env key, the mutant treats `"XXXX"` as a secret, and `"XXXX-XXXX"` becomes `[REDACTED]-[REDACTED]` |
| 4 | `quote(secret, safe="")` -> `safe="XXXX"` | **equivalent** | `X` is an unreserved character that `quote` never escapes, and `safe="XXXX"` still drops the default `safe="/"`. Output is identical for every input |
| 5 | `quote_plus(secret, )` (safe omitted) | **equivalent** | `quote_plus`'s default `safe` is already `""` |
| 6 | `quote_plus(secret, safe="XXXX")` | **equivalent** | as #4 |
| 7 | `sorted(..., key=None, reverse=True)` | killable, should die | `test_json_form_containing_the_raw_form_is_removed_whole` (secret `\tok3n...`: the raw form sorts lexically *after* its JSON form and is a substring of it, so string order leaves `\[REDACTED]`), and `test_containing_secret_wins_over_contained_non_prefix_secret` (`"abc"`, `"bc"`: the mutant yields `a[REDACTED]`) |
| 8 | `sorted(variants, reverse=True)` (key omitted) | killable, should die | same tests as #7 |

So 5 are equivalent and 3 should now be killed. If so, the best possible
`_redact` score is 38/38 non-equivalent, or 38/43 = 88.4% raw.

## Surviving video.py mutants

Not listed. No run in this session produced a survivor list. The lead's
baseline puts them mostly in `_collect_media` (59), `validate_model_options`
(18) and `retrieve_bytes` (8). The added `test_video*` modules target those
functions, but their effect is unmeasured.
