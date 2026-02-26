# Oumi MCP Server — Issue #2 Fixes + Registry Refactor

## What This Is

A quality pass on the Oumi MCP server (`feature/cloud-runs` branch) that addresses user-reported friction (GitHub Issue #2) and code bugs found during review. The core architectural change makes `launcher.status()` the single source of truth for job state, reducing `JobRegistry` to a thin identity mapping.

## Core Value

Reliable job lifecycle management — users can trust that job status, cancellation, and listing reflect reality, not stale cached state.

## Requirements

### Validated
- launcher.status() is source of truth for all job state queries
- JobRegistry stores identity mappings only (no status field)
- _runtimes evicted after cloud launch completes
- Cancel uses "cancelled" status, race condition fixed
- validate_datasets resolves paths against client_cwd, not MCP CWD
- Dead code removed (_check_env_overrides, import re, env_warnings)
- Guidance content covers ephemeral storage, sky exec, version compat, inference output, eval caveats

### Out of Scope
- Cloud file resolution pre-flight (separate design doc exists)
- New MCP tools or endpoints
- Breaking changes to MCP tool API signatures

## Key Decisions

1. **Cohesive refactor** over surgical fixes for cancel/runtime lifecycle
2. **By-layer execution order**: job_service → server → guidance → tests
3. **Include tests alongside code fixes** (Phase 4)
4. **Remove _check_env_overrides entirely** (replaced by startup stripping)
5. **No co-author lines** in git commits (user preference)

## Constraints

- Must not break existing MCP tool API contracts
- Tests must pass after each phase
- Branch: `feature/cloud-runs`
