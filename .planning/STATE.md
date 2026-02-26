# Project State

## Project Reference
**Oumi MCP Server — Issue #2 Fixes + Registry Refactor**
Making launcher.status() the source of truth for job state.

## Current Position
- Phase: 1 of 4 — Registry Refactor
- Plan: 1 of 1 complete (phase 1 done)
- Status: Phase 1 complete, ready for Phase 2

## Progress
[██░░░░░░░░] 25% — Phase 1 of 4 complete (8 tasks done)

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

## Pending Todos
None

## Blockers/Concerns
None

## Session Continuity
Last session: 2026-02-26 — Phase 1 registry refactor executed (8 tasks, all tests passing)
Stopped at: Phase 1 Plan 1 complete — ready for Phase 2 (server-level bug fixes)
