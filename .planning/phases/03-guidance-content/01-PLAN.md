# Phase 3: Guidance Content — Plan

**Goal:** Address user friction points from Issue #2 with better documentation in MCP resources.

**Source:** `docs/plans/2026-02-26-issue2-implementation-plan.md` (Tasks 11-12)

## Tasks

### Task 11: Add ephemeral storage + sky exec warnings to CLOUD_LAUNCH_RESOURCE
- File: `src/oumi_mcp_server/prompts/mle_prompt.py:560-573`
- Add "Important: Ephemeral Storage" section
- Add "Important: Existing Clusters and File Sync" section

### Task 12: Add version compat, inference output schema, eval caveats
- File: `src/oumi_mcp_server/prompts/mle_prompt.py` (multiple locations)
- Add lm_harness caveat to EVAL_COMMAND_RESOURCE
- Add output schema + cross-version note to INFER_COMMAND_RESOURCE
- Add version compatibility note to CLOUD_LAUNCH_RESOURCE
