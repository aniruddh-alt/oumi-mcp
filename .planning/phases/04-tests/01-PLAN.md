# Phase 4: Tests — Plan

**Goal:** Rewrite tests for new architecture, add coverage for new features, ensure all tests pass.

**Source:** `docs/plans/2026-02-26-issue2-implementation-plan.md` (Tasks 13-20)

## Tasks

### Task 13: Rewrite test_job_registry.py for new registry
- File: `tests/test_job_registry.py`
- Tests: add_and_get, persists_to_disk, prune_old, find_by_cloud_identity, update_persists, update_missing_noop, legacy_records_with_status

### Task 14: Add validate_datasets client_cwd test
- File: `tests/test_cloud_file_checks.py`
- Test relative ds_path resolved against client_cwd

### Task 15: Fix test_relative_path_exists_locally_still_unreachable
- File: `tests/test_cloud_file_checks.py:108-116`
- Change training.output_dir to dataset path

### Task 16: Add cancel_job end-to-end + cancel timeout tests
- File: `tests/test_job_recovery_and_control.py`
- Test cancel delegates to launcher.cancel, test timeout returns structured error

### Task 17: Add _launch_cloud with client_cwd test
- File: `tests/test_job_recovery_and_control.py`
- Verify client_cwd sets working_dir in _build_cloud_job_config

### Task 18: Add list_jobs via launcher.status() test
- File: `tests/test_job_recovery_and_control.py`
- Test launcher.status() called, results enriched with MCP IDs

### Task 19: Final integration pass — run all tests, fix breakage
- Run full test suite, fix any remaining failures from status removal

### Task 20: Final commit — update design doc as completed
- Verify all tests pass, commit design docs
