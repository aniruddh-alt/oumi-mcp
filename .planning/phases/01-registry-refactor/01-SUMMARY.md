---
phase: 01-registry-refactor
plan: 01
subsystem: api
tags: [python, asyncio, job-registry, skypilot, oumi-launcher]

# Dependency graph
requires: []
provides:
  - JobRegistry as identity-only mapping (no cached status)
  - launcher.status() as authoritative source of job state
  - _runtimes eviction via evict_runtime() and cleanup_stale_runtimes()
  - cancel race guard in _launch_cloud
  - thread-safe _get_cloud_logs timeout handling
affects: [02-server-fixes, 03-guidance-content, 04-tests]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Status never stored in registry — always derived from launcher.status() or rt.process.poll()"
    - "evict_runtime() called after cloud launch completes so _runtimes stays scoped to in-flight only"
    - "cancel_requested flag on JobRuntime is the race guard for pre-launch cancels"

key-files:
  created: []
  modified:
    - src/oumi_mcp_server/job_service.py
    - src/oumi_mcp_server/server.py
    - tests/test_job_registry.py
    - tests/test_job_recovery_and_control.py
    - tests/test_client_cwd.py

key-decisions:
  - "JobRecord is identity mapping only — job_id, cloud, cluster_name, oumi_job_id, model_name, submit_time, output_dir"
  - "Registry pruning is age-based only (7 days); no more status-based eviction"
  - "Legacy 'status' field in persisted JSON is silently popped on load for backward compat"
  - "_list_job_summaries queries launcher.status() for cloud jobs directly, not registry"
  - "cancel_job polls live status before cancelling to avoid redundant cloud API calls"

patterns-established:
  - "Status derivation: rt.oumi_status.status for cloud, rt.process.poll() for local"
  - "update() in JobRegistry is now safe — warns and skips if job_id not found"

requirements-completed: []

# Metrics
duration: 45min
completed: 2026-02-26
---

# Phase 1 Plan 1: Registry Refactor Summary

**JobRegistry reduced to thin identity mapping with launcher.status() as source of truth, cancel race fixed, and _runtimes eviction added**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-02-26T00:00:00Z
- **Completed:** 2026-02-26
- **Tasks:** 8 of 8
- **Files modified:** 5

## Accomplishments
- Removed `status` field from `JobRecord` — registry is now an identity-only mapping
- Rewrote `JobRegistry._prune()` to age-based eviction (7 days), removing status dependency
- Added `evict_runtime()` and `cleanup_stale_runtimes()` to scope `_runtimes` to in-flight jobs
- Fixed cancel race: `_launch_cloud` checks `cancel_requested` after `launcher.up` returns and immediately cancels
- Fixed `_get_cloud_logs` thread safety: removed cross-thread `stream.close()` on timeout
- `poll_status` now only updates identity fields (oumi_job_id, cluster_name), no status writes
- `_job_status_str` and `_is_job_done` derive state from `rt.process`/`rt.oumi_status`, not `record.status`
- `_list_job_summaries` queries `launcher.status()` directly for cloud jobs
- `cancel_job` polls live status before attempting cancellation to reject already-done jobs

## Task Commits

Each task was committed atomically:

1. **Task 1: Remove status field from JobRecord** - `99198c9` (refactor)
2. **Task 2: Simplify JobRegistry — age-based pruning** - `4deac4b` (refactor)
3. **Task 3: Add _runtimes eviction** - `9b59f02` (feat)
4. **Task 4: Rewrite cancel() + fix race in _launch_cloud** - `bd573f5` (refactor)
5. **Task 5: Fix _get_cloud_logs thread safety** - `30a5dc6` (fix)
6. **Task 6: Simplify poll_status — remove registry writes** - `3c00bd1` (refactor)
7. **Task 7: Update server.py — no record.status** - `cf65f06` (refactor)
8. **Task 8: cancel_job checks live status** - `130107b` (fix)

## Files Created/Modified
- `src/oumi_mcp_server/job_service.py` - JobRecord, JobRegistry, _runtimes, cancel, poll_status, _launch_cloud, wait_local_completion
- `src/oumi_mcp_server/server.py` - _job_status_str, _is_job_done, _list_job_summaries, cancel_job, run_oumi_job, get_job_logs
- `tests/test_job_registry.py` - Updated: removed status= from all JobRecord fixtures
- `tests/test_job_recovery_and_control.py` - Updated: removed status= from all JobRecord fixtures
- `tests/test_client_cwd.py` - Updated: removed status= from all JobRecord fixtures

## Decisions Made
- Registry pruning is age-based (7 days) rather than status-based — cleaner since status is no longer stored
- `evict_runtime()` is called after cloud launch succeeds, scoping _runtimes to truly in-flight jobs only
- Legacy JSON files with `status` field are handled by popping it on `_load()` — no migration needed
- `_list_job_summaries` now calls `launcher.status()` directly and cross-references registry for MCP job IDs

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test fixtures referencing removed status= field**
- **Found during:** Task 1 (Remove status field from JobRecord)
- **Issue:** 3 test files had JobRecord() constructor calls with `status=` kwarg that no longer exists
- **Fix:** Removed `status=` from all test fixtures in test_job_registry.py, test_job_recovery_and_control.py, test_client_cwd.py
- **Files modified:** tests/test_job_registry.py, tests/test_job_recovery_and_control.py, tests/test_client_cwd.py
- **Verification:** All 72 tests pass after fix
- **Committed in:** `99198c9` (Task 1 commit)

**2. [Rule 3 - Blocking] Fixed server.py ephemeral JobRecord in get_job_logs**
- **Found during:** Task 4 (testing after cancel rewrite)
- **Issue:** `get_job_logs` in server.py created an ephemeral `JobRecord` with `status="unknown"` — blocked test_get_job_logs_by_direct_identity test
- **Fix:** Removed `status=` from ephemeral JobRecord in server.py get_job_logs
- **Files modified:** src/oumi_mcp_server/server.py
- **Verification:** test_get_job_logs_by_direct_identity_tries_cloud_retrieval passes
- **Committed in:** `bd573f5` (Task 4 commit)

---

**Total deviations:** 2 auto-fixed (1 test fixture update, 1 blocking server.py fix)
**Impact on plan:** Both fixes required for correctness. No scope creep.

## Issues Encountered
- test_job_registry.py `test_update` test was asserting `reg.get("j1").status` which no longer exists — updated to check `oumi_job_id` instead

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 1 registry refactor complete — job_service.py and server.py now consistent with status-free architecture
- Ready for Phase 2: Server-level bug fixes (validate_datasets CWD, not_found_warning, dead code removal)
- All 72 tests passing

---
*Phase: 01-registry-refactor*
*Completed: 2026-02-26*
