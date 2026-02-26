---
phase: 03-guidance-content
plan: 01
subsystem: documentation
tags: [mle_prompt, cloud-launch, eval, inference, skypilot, guidance]

# Dependency graph
requires:
  - phase: 02-server-bug-fixes
    provides: Fixed server + job service used by the resources being documented
provides:
  - Ephemeral storage warning in CLOUD_LAUNCH_RESOURCE
  - Sky exec / file sync warning in CLOUD_LAUNCH_RESOURCE
  - Version compatibility note in CLOUD_LAUNCH_RESOURCE
  - Inference output schema (predictions.jsonl format) in INFER_COMMAND_RESOURCE
  - lm_harness version caveat in EVAL_COMMAND_RESOURCE
  - Cross-version config template note in EVAL and INFER resources
affects: [users reading guidance://cloud-launch, guidance://eval, guidance://infer resources]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "MCP guidance resources use XML tags for caveats/output_format sections"
    - "Markdown used for operational warnings (ephemeral storage, sky exec)"

key-files:
  created: []
  modified:
    - src/oumi_mcp_server/prompts/mle_prompt.py

key-decisions:
  - "Version compat note placed after Why-You-Need-a-Job-Config section (user sees it early)"
  - "Eval + infer caveats use XML <caveats> tag consistent with resource format"
  - "Ephemeral storage and sky exec warnings use Markdown headers (consistent with surrounding text)"

patterns-established:
  - "Operational warnings (storage, file sync) use ## Important: prefix headers"
  - "Version/compat caveats use <caveats><item> XML tags within resource blocks"

requirements-completed: []

# Metrics
duration: 8min
completed: 2026-02-26
---

# Phase 3 Plan 1: Guidance Content Summary

**Added ephemeral storage + sky exec warnings, version compat note, inference output schema, and eval/infer caveats to MCP resource strings in mle_prompt.py**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-26T23:14:00Z
- **Completed:** 2026-02-26T23:22:59Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- CLOUD_LAUNCH_RESOURCE: Added "Important: Ephemeral Storage" section warning that local disk is not preserved across cluster stops/recreations
- CLOUD_LAUNCH_RESOURCE: Added "Important: Existing Clusters and File Sync" section warning that `sky exec` does not re-sync changed local files
- CLOUD_LAUNCH_RESOURCE: Added "Version Compatibility" section documenting 0.7 vs 0.1.x API field name differences (e.g. `evaluation_backend` vs `evaluation_platform`)
- EVAL_COMMAND_RESOURCE: Added `<caveats>` block with lm_harness version-compat warning and cross-family config template note
- INFER_COMMAND_RESOURCE: Added `<output_format>` block documenting predictions.jsonl schema (full message history, not raw prediction) and `<caveats>` block with cross-version config template note

## Task Commits

Each task was committed atomically:

1. **Task 11: Add ephemeral storage + sky exec warnings** - `aa30117` (docs)
2. **Task 12: Add version compat, inference output schema, eval caveats** - `8a59d59` (docs)

## Files Created/Modified
- `src/oumi_mcp_server/prompts/mle_prompt.py` - Added 56 lines of guidance content across CLOUD_LAUNCH_RESOURCE, EVAL_COMMAND_RESOURCE, and INFER_COMMAND_RESOURCE

## Decisions Made
- Version compat note placed early in CLOUD_LAUNCH_RESOURCE (after "Why You Need a Job Config") so users see it before they encounter problems
- Used `<caveats>` XML tags for eval/infer caveats to match the existing XML resource format
- Used Markdown headers (`## Important: ...`) for operational warnings to match the Markdown style of surrounding CLOUD_LAUNCH_RESOURCE content

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 3 guidance content complete
- Ready for Phase 4 (Tests)
- All 72 existing tests continue to pass

---
*Phase: 03-guidance-content*
*Completed: 2026-02-26*
