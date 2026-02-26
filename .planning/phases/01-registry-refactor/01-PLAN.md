# Phase 1: Registry Refactor — Plan

**Goal:** Make launcher.status() the source of truth. Reduce JobRegistry to identity mapping. Fix cancel race. Add _runtimes eviction.

**Source:** `docs/plans/2026-02-26-issue2-implementation-plan.md` (Tasks 1-8)

## Tasks

### Task 1: Remove `status` field from JobRecord
- File: `src/oumi_mcp_server/job_service.py:48-60`
- Remove status from dataclass, run tests to see breakage

### Task 2: Simplify JobRegistry — age-based pruning only
- File: `src/oumi_mcp_server/job_service.py:90-202`
- Remove _TERMINAL_STATUSES, rewrite _prune() to age-based, add legacy status pop in _load()

### Task 3: Add _runtimes eviction
- File: `src/oumi_mcp_server/job_service.py:204-221`
- Add evict_runtime(), cleanup_stale_runtimes(), hook into get_registry()

### Task 4: Rewrite cancel() — fix race condition
- File: `src/oumi_mcp_server/job_service.py:914-990`
- No registry status writes, add cancel_requested race guard in _launch_cloud, remove all reg.update(..., status=...) calls

### Task 5: Fix _get_cloud_logs thread safety
- File: `src/oumi_mcp_server/job_service.py:1157-1167`
- Remove cross-thread stream.close() on timeout

### Task 6: Simplify poll_status — remove registry writes
- File: `src/oumi_mcp_server/job_service.py:837-911`
- Update runtime only, keep reg.update for identity fields (oumi_job_id, cluster_name) only

### Task 7: Update server.py — remove status from JobRecord creation/helpers
- File: `src/oumi_mcp_server/server.py` (multiple locations)
- Remove status= from JobRecord construction, rewrite _job_status_str and _is_job_done, rewrite _list_job_summaries to use launcher.status()

### Task 8: Update cancel_job in server — check live status before cancelling
- File: `src/oumi_mcp_server/server.py:2520-2595`
- Poll live status before calling cancel, reject if already done

## Execution Notes
- Tasks 1-6 are sequential changes to job_service.py (each builds on previous)
- Task 7 is the server-side wiring
- Task 8 completes the cancel flow
- Commit after each task
