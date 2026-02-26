# Phase 2: Server Bug Fixes — Plan

**Goal:** Fix validate_datasets CWD bug, surface not_found_warning, fix suggested_configs, remove dead code.

**Source:** `docs/plans/2026-02-26-issue2-implementation-plan.md` (Tasks 9-10)

## Tasks

### Task 9: Fix validate_datasets CWD
- File: `src/oumi_mcp_server/server.py:861-924`
- Add client_cwd parameter, resolve relative paths against it
- Update call site in _pre_flight_check

### Task 10: Surface not_found_warning + fix suggested_configs + remove dead code
- File: `src/oumi_mcp_server/server.py` (multiple locations) + `models.py`
- Add not_found_warning branch in pre-flight
- Fix suggested_configs to use task type instead of cloud name
- Delete _check_env_overrides, _ENV_WARNINGS, env_warnings key
- Remove import re
- Remove env_warnings from PreFlightCheckResponse in models.py
