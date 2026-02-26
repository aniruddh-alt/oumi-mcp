# Roadmap — Issue #2 Fixes + Registry Refactor

## Milestone 1: Issue #2 Fixes

### Phase 1: Registry Refactor (job_service.py) ✓ COMPLETE
**Goal:** Make launcher.status() the source of truth. Reduce JobRegistry to identity mapping. Fix cancel race. Add _runtimes eviction.

**Tasks:** 8/8 complete
1. ✓ Remove `status` field from JobRecord
2. ✓ Simplify JobRegistry — age-based pruning only
3. ✓ Add `_runtimes` eviction (evict_runtime, cleanup_stale_runtimes)
4. ✓ Rewrite cancel() — "cancelled" status, fix race condition
5. ✓ Fix `_get_cloud_logs` thread safety (remove cross-thread stream.close)
6. ✓ Simplify poll_status — remove registry status writes
7. ✓ Update server.py — remove status from JobRecord creation/helpers
8. ✓ Update cancel_job in server — check live status before cancelling

**Summary:** `.planning/phases/01-registry-refactor/01-SUMMARY.md`
**Dependencies:** None (foundation layer)

### Phase 2: Server Bug Fixes (server.py) ✓ COMPLETE
**Goal:** Fix validate_datasets CWD bug, surface not_found_warning, fix suggested_configs, remove dead code.

**Tasks:** 2/2 complete
9. ✓ Fix validate_datasets CWD (add client_cwd parameter)
10. ✓ Surface not_found_warning + fix suggested_configs + remove dead code

**Summary:** `.planning/phases/02-server-bug-fixes/02-01-SUMMARY.md`
**Dependencies:** Phase 1 (server changes build on new registry)

### Phase 3: Guidance Content (mle_prompt.py) ✓ COMPLETE
**Goal:** Address user friction points from Issue #2 with better documentation.

**Tasks:** 2/2 complete
11. ✓ Add ephemeral storage + sky exec warnings to CLOUD_LAUNCH_RESOURCE
12. ✓ Add version compat, inference output schema, eval caveats

**Summary:** `.planning/phases/03-guidance-content/03-01-SUMMARY.md`
**Dependencies:** None (independent of code changes, but sequenced after Phase 1 for clean commits)

### Phase 4: Tests
**Goal:** Rewrite tests for new architecture, add coverage for new features.

**Tasks:** 8
13. Rewrite test_job_registry.py for new registry
14. Add validate_datasets client_cwd test
15. Fix test_relative_path_exists_locally_still_unreachable
16. Add cancel_job end-to-end + cancel timeout tests
17. Add _launch_cloud with client_cwd test
18. Add list_jobs via launcher.status() test
19. Final integration pass — run all tests, fix breakage
20. Final commit — update design doc

**Dependencies:** Phases 1 + 2 (tests validate the refactored code)
