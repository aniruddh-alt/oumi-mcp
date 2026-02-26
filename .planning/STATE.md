# Project State

## Project Reference
**Oumi MCP Server — Issue #2 Fixes + Registry Refactor**
Making launcher.status() the source of truth for job state.

## Current Position
- Phase: 3 of 4 — Guidance Content
- Plan: 1 of 1 complete (phase 3 done)
- Status: Phase 3 complete, ready for Phase 4

## Progress
[██████░░░░] 75% — Phase 3 of 4 complete (12 tasks done)

## Recent Decisions
- Cohesive refactor (not surgical fixes) for cancel/runtime
- By-layer execution: job_service → server → guidance → tests
- Include tests alongside code fixes
- No co-author lines in commits
- JobRecord is identity mapping only — no cached status field
- Registry pruning is age-based only (7 days); no more status-based eviction
- Legacy 'status' field in persisted JSON is silently popped on load for backward compat
- _list_job_summaries queries launcher.status() for cloud jobs directly, not registry
- cancel_job polls live status before cancelling to avoid redundant cloud API calls
- validate_datasets accepts client_cwd param, resolves relative paths there
- not_found_warning from validate_paths surfaced as user-facing warning
- suggested_configs uses task_type inferred from config keys, falls back to sft
- _check_env_overrides deleted: function body was empty (_ENV_WARNINGS = {}), always returned []
- Version compat note placed early in CLOUD_LAUNCH_RESOURCE so users see it before encountering issues
- Eval + infer caveats use XML <caveats> tags consistent with resource format
- Operational warnings (ephemeral storage, sky exec) use Markdown ## Important: headers

## Pending Todos
None

## Blockers/Concerns
None

## Session Continuity
Last session: 2026-02-26 — Phase 3 guidance content executed (2 tasks, all tests passing)
Stopped at: Phase 3 Plan 1 complete — ready for Phase 4 (tests)
