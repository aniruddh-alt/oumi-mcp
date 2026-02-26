# Design: Issue #2 Fixes + Code Quality Pass

Date: 2026-02-26
Branch: feature/cloud-runs
Issue: https://github.com/aniruddh-alt/oumi-mcp/issues/2

## Problem Statement

Two categories of work:

1. **User-facing friction** (Issue #2): version mismatch confusion, ephemeral storage data loss, undocumented inference output format, `sky exec` re-sync surprise, `lm_harness` version issues.

2. **Code bugs** (from code review): `_runtimes` memory leak, cancel race conditions, `validate_datasets` using MCP CWD instead of `client_cwd`, dead code, test gaps.

3. **Architectural gap**: The local `JobRegistry` maintains its own status as a middleman, but `oumi.launcher.status()` queries SkyPilot directly and is the actual source of truth. The registry can miss jobs launched outside the MCP, show stale statuses, and accumulate zombie records.

## Design

### Phase 1: Registry Refactor — launcher.status() as Source of Truth

**Goal:** Reduce `JobRegistry` to a thin ID mapping. All status queries go through `launcher.status()`.

#### 1.1 New JobRegistry Role

Current `JobRecord` stores: `job_id, command, config_path, cloud, cluster_name, oumi_job_id, model_name, status, submit_time, output_dir`

New `JobRecord` stores only the **identity mapping** (no `status` field):
```
job_id          # MCP-generated human-friendly ID
command         # oumi subcommand (train, eval, etc.)
config_path     # original config path
cloud           # cloud provider name
cluster_name    # SkyPilot cluster name
oumi_job_id     # SkyPilot job ID (set after launcher.up returns)
model_name      # extracted from config (display only)
submit_time     # ISO 8601 (for pruning old entries)
output_dir      # extracted from config (display only)
```

`status` is removed. The registry answers "which cloud identity does this MCP job ID map to?" — nothing more.

#### 1.2 Status Resolution

`get_job_status(job_id)` flow:
1. Look up `JobRecord` from registry by `mcp_job_id`
2. If record has `oumi_job_id` + `cloud` + `cluster_name`: call `launcher.status(cloud=cloud, cluster=cluster, id=oumi_job_id)` → get live `JobStatus`
3. If record exists but `oumi_job_id` is empty (job still launching): check `_runtimes[job_id]` for in-flight state
4. If no record found: return not-found response

`get_job_status(oumi_job_id, cloud, cluster_name)` flow (direct cloud identity):
1. Call `launcher.status(cloud=cloud, cluster=cluster, id=oumi_job_id)` directly
2. Enrich with MCP job ID if a mapping exists

#### 1.3 list_jobs() via launcher.status()

`list_jobs()` flow:
1. Call `launcher.status()` with no filters → get all jobs across all enabled clouds
2. For each returned `JobStatus`, look up MCP job ID from registry (if exists)
3. Return merged list: every SkyPilot job shows up, MCP-submitted ones are enriched with `mcp_job_id`

This means jobs launched via `sky` CLI directly will appear in `list_jobs()` output.

#### 1.4 _runtimes Scoping

`_runtimes` is only needed for:
- **Local jobs**: `Popen` handle, file handles, process monitoring
- **Cloud jobs during launch**: `runner_task` (the async task running `launcher.up`), `cancel_requested` flag

Once a cloud job has `oumi_job_id` set (launch completed), the runtime entry can be evicted — all further state comes from `launcher.status()`.

Add `evict_runtime(job_id)`:
- Calls `rt.close_log_files()`
- Cancels `rt.runner_task` if still running
- Deletes from `_runtimes`

Hook into: (a) after successful cloud launch, (b) registry pruning.

#### 1.5 Cancel Flow

**Pre-launch cancel** (no `oumi_job_id` yet):
- Set `rt.cancel_requested = True`
- Cancel `rt.runner_task` if present
- Registry record stays (no status to update — it never had one)
- When `_launch_cloud` checks `rt.cancel_requested` after `launcher.up` returns, it calls `launcher.cancel()` immediately

**Post-launch cancel** (has `oumi_job_id`):
- Call `launcher.cancel(oumi_job_id, cloud, cluster_name)` → returns `JobStatus`
- No registry update needed — next `get_job_status()` will get the cancelled state from launcher

**Local job cancel**:
- Send `SIGTERM` (or `SIGKILL` if `force=True`) to `rt.process`
- `wait_local_completion` handles cleanup

#### 1.6 Pruning

Registry pruning simplifies: just remove entries older than 7 days. No status-based logic needed (we don't store status). Evict corresponding `_runtimes` entries during prune.

---

### Phase 2: Server-level Bug Fixes

#### 2.1 validate_datasets CWD fix

Add `client_cwd: str = ""` parameter to `validate_datasets()`. Resolve relative dataset paths with `_resolve_path(ds_path, Path(client_cwd))`. Update call site in `_pre_flight_check()`.

#### 2.2 Surface not_found_warning in pre-flight

Add `elif path_status == "not_found_warning"` branch in `_pre_flight_check()` that appends a warning to the `warnings` list.

#### 2.3 Fix suggested_configs query

Extract task type from the config and use that as the search query instead of the cloud provider name.

#### 2.4 Remove dead code

- Delete `_check_env_overrides()`, `_ENV_WARNINGS`, and `env_warnings` response key
- Remove unused `import re`

---

### Phase 3: Guidance Content

#### 3.1 CLOUD_LAUNCH_RESOURCE additions

- **Ephemeral storage warning**: "Training outputs are ephemeral. The cluster's local disk is not preserved across stops or recreations. Use `file_mounts` or post-training sync to copy artifacts to S3/GCS before stopping."
- **sky exec vs sky launch**: "When submitting to an existing cluster, SkyPilot uses `sky exec` which skips file re-sync. Local edits won't be reflected. Use `sky launch` with the same cluster name to force re-sync, or use explicit `file_mounts`."

#### 3.2 Version compatibility note

Add to `get_started()` or `CLOUD_LAUNCH_RESOURCE`:
"The MCP documents Oumi 0.7 APIs. If using an older version (e.g. 0.1.x), some field names may differ. Check `get_docs()` for the installed version's API, or pin `oumi>=0.7` in your setup script."

#### 3.3 INFER_COMMAND_RESOURCE — output schema

Add example showing `predictions.jsonl` contains full conversation history with model response appended as assistant turn.

#### 3.4 EVAL_COMMAND_RESOURCE — lm_harness caveat

Add: "lm_harness tasks may have version-specific compatibility with Oumi. If evaluation fails with dtype/model constructor errors, verify the installed oumi version or fall back to a direct evaluation script."

#### 3.5 Inference config cross-version note

Add: "Config structure is consistent across model versions within a family. If search_configs() doesn't return your exact version, use a config from the same family as a template."

---

### Phase 4: Tests

#### 4.1 _launch_cloud with client_cwd

Test that passing `client_cwd="/some/project"` to `_launch_cloud` sets `job_config.working_dir == "/some/project"`.

#### 4.2 cancel_job end-to-end

Test `server.cancel_job(job_id=...)` where job IS in registry, verify delegation to `launcher.cancel()` and correct response.

#### 4.3 registry.update() persistence (becomes: registry identity mapping persistence)

New registry has no `update()` for status. Test that `add()` persists and can be read by a fresh registry instance.

#### 4.4 Fix test_relative_path_exists_locally_still_unreachable

Use a dataset path instead of `output_dir`.

#### 4.5 Cancel timeout test

Mock `launcher.cancel` to exceed timeout, verify structured error response.

#### 4.6 validate_datasets with client_cwd

Test with a dataset file at `{tmpdir}/data/train.jsonl`, verify `validate_datasets(client_cwd=tmpdir)` returns `"ok_local"`.

#### 4.7 list_jobs via launcher.status()

Test that `list_jobs()` calls `launcher.status()` and merges with MCP job ID mappings.

#### 4.8 get_job_status via launcher.status()

Test that `get_job_status(job_id)` resolves the MCP ID to cloud identity and queries `launcher.status()`.

---

## File Impact

| File | Changes |
|------|---------|
| `job_service.py` | Major: JobRecord loses `status`, registry becomes ID mapping, `_runtimes` eviction, cancel rewrite, `_get_cloud_logs` thread fix |
| `server.py` | Medium: `get_job_status`/`list_jobs`/`cancel_job` rewired to use launcher, `validate_datasets` CWD fix, dead code removal, pre-flight warning surfacing |
| `models.py` | Small: Remove `status` from relevant TypedDicts, remove `env_warnings` from `PreFlightCheckResponse` |
| `mle_prompt.py` | Medium: 5 guidance content additions |
| `constants.py` | Minimal: remove any status-related constants if present |
| `test_job_registry.py` | Rewrite: test identity mapping, not status tracking |
| `test_job_recovery_and_control.py` | Major: rewrite status/cancel tests against new launcher-backed flow |
| `test_cloud_file_checks.py` | Small: fix dataset path test |

## Execution Order

Phase 1 → Phase 2 → Phase 3 (independent) → Phase 4 (depends on 1+2)

Phases 2 and 3 are independent of each other and can run in parallel after Phase 1.
