# Issue #2 Fixes + Registry Refactor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `launcher.status()` the source of truth for job state, fix code bugs from review, and improve guidance content to address user friction (Issue #2).

**Architecture:** Reduce `JobRegistry` to a thin identity mapping (MCP job ID → cloud coordinates). All status resolution goes through `oumi.launcher.status()` for cloud jobs or `rt.process.poll()` for local jobs. `_runtimes` is scoped to in-flight jobs only, with explicit eviction.

**Tech Stack:** Python 3.13, MCP SDK, oumi.launcher, SkyPilot, asyncio, pytest

---

## Phase 1: Registry Refactor (job_service.py)

### Task 1: Remove `status` field from JobRecord

**Files:**
- Modify: `src/oumi_mcp_server/job_service.py:48-60`

**Step 1: Remove the `status` field from the dataclass**

```python
@dataclass
class JobRecord:
    """Persisted job metadata — identity mapping only.

    All fields are strings for simple JSON serde.
    The registry does NOT store job status; status is always
    queried live from ``oumi.launcher.status()`` (cloud) or
    ``rt.process.poll()`` (local).
    """

    job_id: str
    command: str
    config_path: str
    cloud: str
    cluster_name: str
    oumi_job_id: str
    model_name: str
    submit_time: str  # ISO 8601
    output_dir: str = ""
```

**Step 2: Run tests to see what breaks**

Run: `cd /Users/aniruddhanramesh/dev/oumi/projects/oumi-mcp && python -m pytest tests/ -x --tb=short 2>&1 | head -80`
Expected: Multiple failures — anything referencing `record.status` or `JobRecord(..., status=...)`.

**Step 3: Commit**

```bash
git add src/oumi_mcp_server/job_service.py
git commit -m "refactor: remove status field from JobRecord"
```

---

### Task 2: Simplify JobRegistry — remove status-based pruning

**Files:**
- Modify: `src/oumi_mcp_server/job_service.py:90-202`

**Step 1: Remove `_TERMINAL_STATUSES` and rewrite `_prune()` to age-based only**

```python
_MAX_REGISTRY_AGE_DAYS = 7
_MAX_REGISTRY_SIZE = 200
_CLOUD_LOG_TIMEOUT = 30.0


class JobRegistry:
    """Single-file JSON registry mapping MCP job IDs to cloud identities.

    Evicts entries older than ``_MAX_REGISTRY_AGE_DAYS`` on load.
    Caps total records at ``_MAX_REGISTRY_SIZE``, dropping oldest first.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._jobs: dict[str, JobRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for entry in data:
                # Handle legacy records that still have 'status' field
                entry.pop("status", None)
                r = JobRecord(**entry)
                self._jobs[r.job_id] = r
        except Exception:
            logger.warning("Could not load %s, starting fresh", self._path)
        pruned = self._prune()
        if pruned:
            logger.info("Pruned %d stale job records from registry", pruned)
            self._save()

    def _prune(self) -> int:
        """Remove entries older than the age cutoff, then cap total size."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=_MAX_REGISTRY_AGE_DAYS)
        to_remove: list[str] = []
        for jid, rec in self._jobs.items():
            try:
                ts = datetime.fromisoformat(rec.submit_time)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    to_remove.append(jid)
            except (ValueError, TypeError):
                to_remove.append(jid)
        for jid in to_remove:
            del self._jobs[jid]
        removed = len(to_remove)

        # Cap total size — drop oldest first
        if len(self._jobs) > _MAX_REGISTRY_SIZE:
            by_time = sorted(self._jobs.items(), key=lambda x: x[1].submit_time)
            while len(self._jobs) > _MAX_REGISTRY_SIZE and by_time:
                jid, _ = by_time.pop(0)
                del self._jobs[jid]
                removed += 1

        return removed

    def _save(self) -> None:
        records = [dataclasses.asdict(r) for r in self._jobs.values()]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
        tmp.rename(self._path)

    def add(self, record: JobRecord) -> None:
        self._jobs[record.job_id] = record
        self._save()

    def update(self, job_id: str, **fields: Any) -> None:
        record = self._jobs.get(job_id)
        if record is None:
            logger.warning("Registry.update: job_id %s not found, skipping", job_id)
            return
        for k, v in fields.items():
            setattr(record, k, v)
        self._save()

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def find_by_cloud_identity(self, cloud: str, oumi_job_id: str) -> JobRecord | None:
        for r in self._jobs.values():
            if r.cloud == cloud and r.oumi_job_id == oumi_job_id:
                return r
        return None

    def all(self) -> list[JobRecord]:
        return list(self._jobs.values())

    def remove(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)
        self._save()
```

**Step 2: Commit**

```bash
git add src/oumi_mcp_server/job_service.py
git commit -m "refactor: simplify JobRegistry to age-based pruning, guard update()"
```

---

### Task 3: Add `_runtimes` eviction

**Files:**
- Modify: `src/oumi_mcp_server/job_service.py:204-221`

**Step 1: Add `evict_runtime()` and hook into pruning**

Replace the `_runtimes` section (lines 204-221) with:

```python
_runtimes: dict[str, JobRuntime] = {}


def get_runtime(job_id: str) -> JobRuntime:
    if job_id not in _runtimes:
        _runtimes[job_id] = JobRuntime()
    return _runtimes[job_id]


def evict_runtime(job_id: str) -> None:
    """Remove a runtime entry, closing any open handles."""
    rt = _runtimes.pop(job_id, None)
    if rt is None:
        return
    rt.close_log_files()
    if rt.runner_task and not rt.runner_task.done():
        rt.runner_task.cancel()


def cleanup_stale_runtimes() -> None:
    """Remove runtime entries whose job_id is no longer in the registry."""
    reg = get_registry()
    stale = [jid for jid in _runtimes if reg.get(jid) is None]
    for jid in stale:
        evict_runtime(jid)
    if stale:
        logger.info("Evicted %d stale runtime entries", len(stale))


_registry: JobRegistry | None = None


def get_registry(path: Path | None = None) -> JobRegistry:
    """Return the global ``JobRegistry``, creating it on first access."""
    global _registry
    if _registry is None:
        _registry = JobRegistry(path or DEFAULT_JOBS_FILE)
    return _registry
```

**Step 2: Commit**

```bash
git add src/oumi_mcp_server/job_service.py
git commit -m "feat: add _runtimes eviction and stale cleanup"
```

---

### Task 4: Rewrite cancel() — use "cancelled" status, fix race

**Files:**
- Modify: `src/oumi_mcp_server/job_service.py:914-990`

**Step 1: Rewrite the cancel function**

```python
async def cancel(
    record: JobRecord, rt: JobRuntime, *, force: bool = False
) -> JobCancelResponse:
    """Cancel a job.

    For **local** jobs, sends SIGTERM (or SIGKILL if *force* is True).
    For **cloud** jobs, delegates to ``oumi.launcher.cancel()``.
    """
    # Pre-launch cancel: job hasn't reached the cloud yet
    if not record.oumi_job_id and rt.process is None:
        rt.cancel_requested = True
        rt.error_message = "Cancellation requested while launch is pending."
        if rt.runner_task and not rt.runner_task.done():
            rt.runner_task.cancel()
        return {
            "success": True,
            "message": (
                f"Cancellation requested for {record.job_id}. "
                "If the cloud launch completes, the MCP will attempt "
                "best-effort cancellation."
            ),
        }

    # Local job cancel
    if record.cloud == "local" and rt.process is not None:
        try:
            if force:
                rt.process.kill()
                action = "killed (SIGKILL)"
            else:
                rt.process.terminate()
                action = "terminated (SIGTERM)"
            rt.cancel_requested = True
            rt.error_message = f"Cancelled by user ({action})"
            logger.info("Local job %s %s", record.job_id, action)
            return {
                "success": True,
                "message": f"Job {record.job_id} {action}.",
            }
        except OSError as exc:
            return {
                "success": False,
                "error": f"Failed to cancel local job {record.job_id}: {exc}",
            }

    # Cloud job cancel — delegate to launcher
    try:
        result_status = await asyncio.to_thread(
            launcher.cancel,
            record.oumi_job_id,
            record.cloud,
            record.cluster_name,
        )
        rt.cancel_requested = True
        rt.oumi_status = result_status
        return {
            "success": True,
            "message": (
                f"Job {record.job_id} cancel requested on "
                f"{record.cloud}/{record.cluster_name}."
            ),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to cancel job {record.job_id}: {exc}",
        }
```

Key changes:
- No `reg.update(record.job_id, status=...)` calls — registry has no status field
- Removed the `record.status in {"completed", "failed"}` early return — status is not stored; the caller (server) checks live status before calling cancel
- Cloud cancel just delegates to launcher; next status poll will reflect cancellation

**Step 2: Fix the cancel race in `_launch_cloud` — check `cancel_requested` BEFORE writing status**

In `_launch_cloud()` (line 756-796), after `launcher.up` returns, add the cancel check before updating the registry:

```python
        cluster, status = await asyncio.to_thread(
            launcher.up,
            job_config,
            record.cluster_name or None,
        )
        rt.cluster_obj = cluster
        oumi_job_id = status.id if status else ""
        rt.oumi_status = status
        cluster_name = status.cluster if status else record.cluster_name

        # Update registry with cloud identity (no status field)
        reg.update(
            record.job_id,
            oumi_job_id=oumi_job_id,
            cluster_name=cluster_name,
        )
        record = reg.get(record.job_id) or record
        logger.info(
            "Cloud job %s launched on %s (oumi_id=%s)",
            record.job_id,
            record.cloud,
            record.oumi_job_id,
        )

        # Race guard: if cancel was requested while launcher.up was in-flight,
        # immediately cancel the cloud job now that we have an oumi_job_id.
        if rt.cancel_requested and record.oumi_job_id:
            try:
                result_status = await asyncio.to_thread(
                    launcher.cancel,
                    record.oumi_job_id,
                    record.cloud,
                    record.cluster_name,
                )
                rt.oumi_status = result_status
            except Exception as cancel_exc:
                rt.error_message = (
                    "Cancellation was requested during launch, but automatic "
                    f"cloud cancellation failed: {cancel_exc}"
                )
            # Evict cloud runtime after reconciliation — launcher is source of truth now
            evict_runtime(record.job_id)
            return

        # Cloud launch succeeded — evict runtime (launcher.status is source of truth)
        evict_runtime(record.job_id)
```

**Step 3: Remove all `reg.update(..., status=...)` calls throughout job_service.py**

Search for every `reg.update(record.job_id, status=` and remove those calls. The registry no longer has a `status` field. Specifically:
- Line 670: `get_registry().update(record.job_id, status="failed")` in `wait_local_completion` → remove
- Line 709: `reg.update(record.job_id, status="launching")` in `_launch_cloud` → remove
- Line 795: `reg.update(record.job_id, status="failed")` in `_launch_cloud` exception → remove
- Lines 870-875: `reg.update(...)` in `poll_status` → remove the entire `reg.update` call (keep the `rt.oumi_status = status` assignment)
- Lines 897-902: same in fallback `launcher.status` path of `poll_status` → remove

**Step 4: Commit**

```bash
git add src/oumi_mcp_server/job_service.py
git commit -m "refactor: rewrite cancel flow, remove status from registry updates, fix race"
```

---

### Task 5: Fix `_get_cloud_logs` thread safety

**Files:**
- Modify: `src/oumi_mcp_server/job_service.py:1157-1167`

**Step 1: Remove the cross-thread `stream.close()` on timeout**

Replace the timeout handler (lines 1157-1176) with:

```python
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(_read_stream),
            timeout=_CLOUD_LOG_TIMEOUT,
        )
    except asyncio.TimeoutError:
        # Don't cross-thread close the stream — just return partial output.
        # The worker thread will finish when the stream yields EOF or errors.
        raw = "".join(chunks)
        if raw:
            logger.debug(
                "Cloud log read timed out for job %s after %.0fs, "
                "returning %d partial lines",
                record.job_id,
                _CLOUD_LOG_TIMEOUT,
                raw.count("\n"),
            )
```

**Step 2: Commit**

```bash
git add src/oumi_mcp_server/job_service.py
git commit -m "fix: remove cross-thread stream.close() in _get_cloud_logs"
```

---

### Task 6: Simplify `poll_status` — remove registry writes

**Files:**
- Modify: `src/oumi_mcp_server/job_service.py:837-911`

**Step 1: Rewrite poll_status to only update runtime state, not registry**

```python
async def poll_status(record: JobRecord, rt: JobRuntime) -> OumiJobStatus | None:
    """Fetch the latest status for a job from the launcher.

    For **local** jobs, returns None (status derived from ``rt.process``).
    For **cloud** jobs, queries the cluster or launcher and updates ``rt.oumi_status``.
    """
    if record.cloud == "local":
        return None

    if rt.error_message and rt.cluster_obj is None:
        return rt.oumi_status

    # Try cluster.get_job first (fastest path)
    if rt.cluster_obj and record.oumi_job_id:
        try:
            status = await asyncio.to_thread(
                rt.cluster_obj.get_job, record.oumi_job_id
            )
            if status:
                rt.oumi_status = status
                # Update registry with cloud identity if it changed
                reg = get_registry()
                reg.update(
                    record.job_id,
                    oumi_job_id=status.id or record.oumi_job_id,
                    cluster_name=status.cluster or record.cluster_name,
                )
                return status
        except Exception:
            logger.warning(
                "cluster.get_job failed for %s; falling back to launcher.status",
                record.job_id,
                exc_info=True,
            )

    # Fallback: launcher.status (works even without a cluster object)
    try:
        if not record.oumi_job_id:
            return rt.oumi_status
        all_statuses = await asyncio.to_thread(
            launcher.status,
            cloud=record.cloud,
            cluster=record.cluster_name or None,
            id=record.oumi_job_id,
        )
        for _, jobs in all_statuses.items():
            for s in jobs:
                if s.id == record.oumi_job_id:
                    rt.oumi_status = s
                    reg = get_registry()
                    reg.update(
                        record.job_id,
                        oumi_job_id=s.id or record.oumi_job_id,
                        cluster_name=s.cluster or record.cluster_name,
                    )
                    return s
    except Exception:
        logger.warning(
            "launcher.status failed for %s; returning stale status",
            record.job_id,
            exc_info=True,
        )

    return rt.oumi_status
```

**Step 2: Commit**

```bash
git add src/oumi_mcp_server/job_service.py
git commit -m "refactor: poll_status updates runtime only, no registry status writes"
```

---

### Task 7: Update server.py — remove `status` from JobRecord creation and status helpers

**Files:**
- Modify: `src/oumi_mcp_server/server.py:1978-1988, 2096-2130, 2329-2355, 2392-2402`

**Step 1: Remove `status=` from JobRecord construction (line 1978-1988)**

```python
    record = JobRecord(
        job_id=job_id,
        command=command,
        config_path=abs_config,
        cloud=cloud,
        cluster_name=cluster_name,
        oumi_job_id="",
        model_name=model_name,
        submit_time=submit_time,
        output_dir=output_dir,
    )
```

Note: `output_dir` needs to be extracted — check `extract_job_metadata` is called earlier in `run_oumi_job`. It is — `model_name` is already extracted at line ~1830. Add `output_dir` extraction there too.

**Step 2: Rewrite `_job_status_str` to not reference `record.status`**

```python
def _job_status_str(record: JobRecord, rt: JobRuntime) -> str:
    """Derive a human-readable status string for any job (local or cloud)."""
    if rt.cancel_requested:
        return "cancelled"
    is_local = record.cloud == "local"
    if is_local:
        proc = rt.process
        if proc is None:
            if rt.error_message:
                return "failed"
            # Still launching (runner_task exists but process not spawned yet)
            if rt.runner_task and not rt.runner_task.done():
                return "launching"
            return "unknown"
        rc = proc.poll()
        if rc is None:
            return "running"
        return "completed" if rc == 0 else "failed"
    # Cloud job — use launcher status
    if rt.oumi_status:
        return rt.oumi_status.status
    if rt.error_message:
        return "failed"
    # No oumi_job_id yet — still launching
    if not record.oumi_job_id:
        return "launching"
    return "unknown"
```

**Step 3: Rewrite `_is_job_done` to not reference `record.status`**

```python
def _is_job_done(record: JobRecord, rt: JobRuntime) -> bool:
    """Return True if the job is in a terminal state."""
    is_local = record.cloud == "local"
    if is_local and rt.process is not None:
        return rt.process.poll() is not None
    if rt.oumi_status and rt.oumi_status.done:
        return True
    if rt.error_message and not rt.runner_task:
        return True
    if rt.cancel_requested:
        return True
    return False
```

**Step 4: Remove `status=` from ephemeral JobRecord in `get_job_logs` (line 2392-2402)**

```python
            ephemeral = JobRecord(
                job_id=oumi_job_id,
                command="",
                config_path="",
                cloud=cloud,
                cluster_name=cluster_name,
                oumi_job_id=oumi_job_id,
                model_name="",
                submit_time="",
            )
```

**Step 5: Remove `reg.update(record.job_id, status="failed")` from the local job error handler (line 2002)**

The `start_local_job` exception handler at line 2000-2008 currently calls `reg.update(record.job_id, status="failed")`. Remove that line — `rt.error_message` is already set, which is how status is derived now.

**Step 6: Rewrite `_list_job_summaries` to use `launcher.status()` for cloud jobs**

```python
async def _list_job_summaries(status_filter: str = "all") -> list[JobSummary]:
    """Build job summaries from launcher (cloud) and registry (local)."""
    reg = get_registry()
    summaries: list[JobSummary] = []

    # Cloud jobs: query launcher for live state
    try:
        all_statuses = await asyncio.to_thread(launcher.status)
        for cloud_name, jobs in all_statuses.items():
            for job_status in jobs:
                # Try to find MCP job ID from registry
                mapping = reg.find_by_cloud_identity(cloud_name, job_status.id)
                mcp_id = mapping.job_id if mapping else ""
                model = mapping.model_name if mapping else ""
                cmd = mapping.command if mapping else ""

                is_done = bool(job_status.done)
                if status_filter == "running" and is_done:
                    continue
                if status_filter == "completed" and not is_done:
                    continue

                summaries.append({
                    "job_id": mcp_id or job_status.id,
                    "command": cmd,
                    "status": job_status.status,
                    "cloud": cloud_name,
                    "cluster": job_status.cluster,
                    "model_name": model,
                    "is_done": is_done,
                })
    except Exception:
        logger.warning("launcher.status failed; falling back to registry only", exc_info=True)

    # Local jobs: check from registry + runtime
    for record in reg.all():
        if record.cloud != "local":
            continue
        rt = get_runtime(record.job_id)
        status_str = _job_status_str(record, rt)
        is_done = _is_job_done(record, rt)
        if status_filter == "running" and is_done:
            continue
        if status_filter == "completed" and not is_done:
            continue
        summaries.append({
            "job_id": record.job_id,
            "command": record.command,
            "status": status_str,
            "cloud": "local",
            "cluster": "local",
            "model_name": record.model_name,
            "is_done": is_done,
        })

    return summaries
```

**Step 7: Commit**

```bash
git add src/oumi_mcp_server/server.py
git commit -m "refactor: server uses launcher.status() as source of truth, no record.status"
```

---

### Task 8: Update cancel_job in server to check live status before cancelling

**Files:**
- Modify: `src/oumi_mcp_server/server.py:2520-2595`

**Step 1: Add a live-status check before delegating to cancel**

In `cancel_job`, after resolving the record (line 2594), check if the job is already done by polling live status before calling cancel:

```python
    rt = get_runtime(record.job_id)

    # For cloud jobs, check live status first — the job may already be done
    if record.cloud != "local" and record.oumi_job_id:
        live = await poll_status(record, rt)
        if live and live.done:
            return {
                "success": False,
                "error": (
                    f"Job {record.job_id} is already finished "
                    f"(status: {live.status})"
                ),
            }

    return await cancel(record, rt, force=force)
```

**Step 2: Commit**

```bash
git add src/oumi_mcp_server/server.py
git commit -m "fix: cancel_job checks live status before attempting cancellation"
```

---

## Phase 2: Server-level Bug Fixes (server.py)

### Task 9: Fix validate_datasets CWD

**Files:**
- Modify: `src/oumi_mcp_server/server.py:861-924`

**Step 1: Add `client_cwd` parameter and use it for path resolution**

```python
def validate_datasets(cfg: dict, client_cwd: str = "") -> dict[str, str]:
    """Validate dataset accessibility for each dataset in the config."""
    data = cfg.get("data") or {}
    results: dict[str, str] = {}
    base_dir = Path(client_cwd) if client_cwd else Path.cwd()

    for split in ("train", "eval", "validation", "test"):
        split_cfg = data.get(split) or {}
        for ds in split_cfg.get("datasets") or []:
            ds_name = ds.get("dataset_name", "")
            ds_path = ds.get("dataset_path", "")

            if not ds_name and not ds_path:
                continue

            key = ds_name or ds_path

            if key in results:
                continue

            if ds_name:
                try:
                    from oumi.core.registry import REGISTRY

                    reg_result = REGISTRY.get_dataset(ds_name)
                    if reg_result is not None:
                        results[key] = "ok_registry"
                        continue
                except Exception:
                    pass

            if ds_path:
                p = Path(ds_path).expanduser()
                if not p.is_absolute():
                    p = (base_dir / p).resolve()
                if p.exists():
                    results[key] = "ok_local"
                    continue

            if ds_name:
                try:
                    import datasets

                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(datasets.load_dataset_builder, ds_name)
                        future.result(timeout=5)
                    results[key] = "ok_hub"
                    continue
                except TimeoutError:
                    results[key] = "warning_timeout"
                    continue
                except Exception:
                    pass

            results[key] = "not_found"

    return results
```

**Step 2: Update the call site in `_pre_flight_check`**

Find line ~688 where `validate_datasets(cfg)` is called and change to:
```python
    dataset_checks = validate_datasets(cfg, client_cwd=client_cwd)
```

**Step 3: Commit**

```bash
git add src/oumi_mcp_server/server.py
git commit -m "fix: validate_datasets resolves paths against client_cwd, not MCP CWD"
```

---

### Task 10: Surface not_found_warning + fix suggested_configs + remove dead code

**Files:**
- Modify: `src/oumi_mcp_server/server.py:704-778, 927-944, 24, 149-151`
- Modify: `src/oumi_mcp_server/models.py:149-151`

**Step 1: Add `not_found_warning` handling in pre-flight (after line 711)**

```python
    for path_key, path_status in path_results.items():
        if path_status == "local_machine_path_error":
            errors.append(
                f"Local machine path '{path_key}' will not exist on the remote VM. "
                "Use a repo-relative path (e.g., 'data/...') that resolves from "
                "your working_dir."
            )
        elif path_status == "not_found_warning":
            warnings.append(
                f"Path '{path_key}' not found locally. "
                "Verify it will be available on the VM via file_mounts, "
                "working_dir, or setup_script."
            )
```

**Step 2: Fix suggested_configs to use task type instead of cloud name (line 775-778)**

```python
    if target_cloud:
        all_cfgs = get_all_configs()
        # Use task type from config for relevant suggestions, not cloud name
        task_type = cfg.get("task_type", "") or ""
        if not task_type:
            # Infer from config keys
            if cfg.get("training"):
                task_type = "sft"
            elif cfg.get("evaluation") or cfg.get("tasks"):
                task_type = "eval"
            elif cfg.get("generation") and cfg.get("input_path"):
                task_type = "infer"
        suggested = search_configs_service(
            all_cfgs, task=task_type or "sft", limit=5
        )
        result["suggested_configs"] = [c["path"] for c in suggested]
```

**Step 3: Delete `_check_env_overrides` function (lines 927-944)**

Delete the entire function.

**Step 4: Remove its call site (line 701) and the `env_warnings` response key (line 770-771)**

Remove:
```python
    env_warnings = _check_env_overrides(target_cloud)
    warnings.extend(env_warnings)
```

And remove:
```python
    if env_warnings:
        result["env_warnings"] = env_warnings
```

**Step 5: Remove `env_warnings` from `PreFlightCheckResponse` in models.py (line 170)**

Delete:
```python
    env_warnings: NotRequired[list[str]]
```

**Step 6: Remove `import re` from server.py (line 24)**

**Step 7: Commit**

```bash
git add src/oumi_mcp_server/server.py src/oumi_mcp_server/models.py
git commit -m "fix: surface not_found_warning, fix suggested_configs, remove dead code"
```

---

## Phase 3: Guidance Content (mle_prompt.py)

### Task 11: Add ephemeral storage + sky exec warnings to CLOUD_LAUNCH_RESOURCE

**Files:**
- Modify: `src/oumi_mcp_server/prompts/mle_prompt.py:560-573`

**Step 1: Add two new sections before the closing `"""`**

Insert before the closing triple-quote of `CLOUD_LAUNCH_RESOURCE` (after line 573):

```python
## Important: Ephemeral Storage

Training outputs on the cluster's local disk are **not preserved** across cluster stops
or recreations. If the cluster is stopped, restarted, or torn down, all local files
(checkpoints, adapters, logs) are lost.

**Before stopping or deleting a cluster:**
1. Use `sky rsync-down <cluster> ~/sky_workdir/<output_dir> ./local_output/` to download artifacts
2. Or configure your training config to save checkpoints to a cloud bucket (S3/GCS) via `storage_mounts`

## Important: Existing Clusters and File Sync

When submitting a job to an **existing** cluster, SkyPilot uses `sky exec` instead of
`sky launch`. This means:
- Local file changes are **NOT re-synced** to the VM
- The VM still has the files from the original `sky launch`

**If you edited files locally and need them on the VM:**
- Use `sky launch` with the same `cluster_name` to force a full re-sync
- Or use explicit `file_mounts` for files that change between submissions
```

**Step 2: Commit**

```bash
git add src/oumi_mcp_server/prompts/mle_prompt.py
git commit -m "docs: add ephemeral storage and sky exec warnings to cloud launch guide"
```

---

### Task 12: Add version compat note, inference output schema, eval caveat

**Files:**
- Modify: `src/oumi_mcp_server/prompts/mle_prompt.py:295-335, 676-716`

**Step 1: Add lm_harness caveat to EVAL_COMMAND_RESOURCE (before closing `</resource>` at line 334)**

Insert before `</resource>`:
```xml
<caveats>
<item>lm_harness evaluation tasks may have version-specific compatibility with Oumi.
If evaluation fails with dtype or model constructor errors, verify the installed oumi
version matches task expectations, or fall back to a direct evaluation script.</item>
<item>Config structure is consistent across model versions within a family (e.g. Qwen2.5
and Qwen3 use the same config shape). If search_configs() doesn't return your exact model
version, use a config from the same family as a template.</item>
</caveats>
```

**Step 2: Add output schema and cross-version note to INFER_COMMAND_RESOURCE (before closing `</resource>` at line 715)**

Insert before `</resource>`:
```xml
<output_format>
<description>Each line in predictions.jsonl contains the full conversation history
with the model's response appended as an assistant turn:</description>
<example>{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 equals 4."}]}</example>
<note>The output is NOT just the raw prediction — it includes the full message history.
Parse the last assistant message to extract the model's response.</note>
</output_format>

<caveats>
<item>Config structure is consistent across model versions within a family. If
search_configs() doesn't return your exact model version (e.g. Qwen2.5), use a config
from the same family (e.g. Qwen3) as a template — the structure is identical.</item>
</caveats>
```

**Step 3: Add version compatibility note to CLOUD_LAUNCH_RESOURCE**

Add near the top of the `CLOUD_LAUNCH_RESOURCE`, after the "Why You Need a Job Config" section:

```markdown
## Version Compatibility

The MCP tools document Oumi 0.7 APIs. If the cloud VM installs a different version
(e.g. `pip install oumi[gpu]` installs 0.1.x), some field names may differ:
- `evaluation_backend` (0.7) → `evaluation_platform` (0.1.x)

**To avoid mismatches:** pin the version in your setup script:
```bash
pip install 'oumi[gpu]>=0.7'
```
Or use `get_docs()` to check the installed version's API.
```

**Step 4: Commit**

```bash
git add src/oumi_mcp_server/prompts/mle_prompt.py
git commit -m "docs: add version compat, inference output schema, eval caveats"
```

---

## Phase 4: Tests

### Task 13: Rewrite test_job_registry.py for the new registry

**Files:**
- Modify: `tests/test_job_registry.py`

**Step 1: Read the existing test file to understand current structure**

Run: `cat tests/test_job_registry.py`

**Step 2: Rewrite tests for the new schema (no `status` field)**

Key tests:
- `test_add_and_get`: Create a JobRecord without `status`, add to registry, get it back
- `test_persists_to_disk`: Add a record, create a new registry from same path, verify it loads
- `test_prune_removes_old_entries`: Add a record with old `submit_time`, verify it's pruned
- `test_find_by_cloud_identity`: Add record with cloud+oumi_job_id, find it
- `test_update_persists`: Call `update(job_id, oumi_job_id="new")`, reload from disk, verify
- `test_update_missing_key_is_noop`: Call `update("nonexistent", ...)`, verify no crash
- `test_legacy_records_with_status_field`: Write a JSON file with `"status": "running"`, load registry, verify it works (backwards compat)

**Step 3: Run tests**

Run: `python -m pytest tests/test_job_registry.py -v`

**Step 4: Commit**

```bash
git add tests/test_job_registry.py
git commit -m "test: rewrite registry tests for status-free identity mapping"
```

---

### Task 14: Add validate_datasets client_cwd test

**Files:**
- Modify: `tests/test_cloud_file_checks.py` (or create `tests/test_validate_datasets.py` if more appropriate)

**Step 1: Write the test**

```python
class TestValidateDatasetsCwd(unittest.TestCase):
    def test_relative_dataset_path_resolved_against_client_cwd(self):
        """validate_datasets resolves relative ds_path against client_cwd."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            (data_dir / "train.jsonl").write_text('{"text": "hello"}')

            cfg = {
                "data": {
                    "train": {
                        "datasets": [{"dataset_path": "data/train.jsonl"}]
                    }
                }
            }
            result = validate_datasets(cfg, client_cwd=tmp)
            assert result.get("data/train.jsonl") == "ok_local"

    def test_relative_dataset_path_without_client_cwd_not_found(self):
        """Without client_cwd, relative paths resolve against CWD (likely wrong)."""
        cfg = {
            "data": {
                "train": {
                    "datasets": [{"dataset_path": "nonexistent/data.jsonl"}]
                }
            }
        }
        result = validate_datasets(cfg)
        assert result.get("nonexistent/data.jsonl") == "not_found"
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_cloud_file_checks.py::TestValidateDatasetsCwd -v`

**Step 3: Commit**

```bash
git add tests/test_cloud_file_checks.py
git commit -m "test: validate_datasets resolves paths against client_cwd"
```

---

### Task 15: Fix test_relative_path_exists_locally_still_unreachable

**Files:**
- Modify: `tests/test_cloud_file_checks.py:108-116`

**Step 1: Read the existing test**

Run: `cat tests/test_cloud_file_checks.py`

**Step 2: Change `training.output_dir` to a dataset path**

Replace:
```python
cfg = {
    "model": {"model_name": "meta-llama/Llama-3.1-8B"},
    "training": {"output_dir": "data/train.jsonl"},
}
```

With:
```python
cfg = {
    "model": {"model_name": "meta-llama/Llama-3.1-8B"},
    "data": {
        "train": {
            "datasets": [{"dataset_path": "data/train.jsonl"}]
        }
    },
}
```

(Adjust assertions if `_check_cloud_files` scans different keys for datasets vs training paths.)

**Step 3: Run tests**

Run: `python -m pytest tests/test_cloud_file_checks.py -v`

**Step 4: Commit**

```bash
git add tests/test_cloud_file_checks.py
git commit -m "test: fix relative path test to use dataset path instead of output_dir"
```

---

### Task 16: Add cancel_job end-to-end + cancel timeout tests

**Files:**
- Modify: `tests/test_job_recovery_and_control.py`

**Step 1: Add server-level cancel test (record found path)**

```python
async def test_cancel_job_delegates_to_job_service_cancel(self):
    """cancel_job with a registry record delegates to job_service.cancel."""
    record = JobRecord(
        job_id="cancel-e2e-test",
        command="train",
        config_path="/tmp/config.yaml",
        cloud="gcp",
        cluster_name="my-cluster",
        oumi_job_id="sky-42",
        model_name="test-model",
        submit_time=datetime.now(tz=timezone.utc).isoformat(),
    )
    mock_reg = MagicMock()
    mock_reg.get.return_value = record
    mock_reg.find_by_cloud_identity.return_value = record

    mock_cancel_result = OumiJobStatus(
        name="cancel-e2e-test", id="sky-42",
        status="CANCELLED", cluster="my-cluster",
        metadata="", done=True, state=JobState.CANCELLED,
    )

    with (
        patch("oumi_mcp_server.server.get_registry", return_value=mock_reg),
        patch("oumi_mcp_server.server.get_runtime") as mock_get_rt,
        patch("oumi_mcp_server.job_service.get_registry", return_value=mock_reg),
        patch("oumi_mcp_server.server.poll_status", return_value=None),
        patch("oumi_mcp_server.job_service.launcher") as mock_launcher,
    ):
        mock_launcher.cancel = MagicMock(return_value=mock_cancel_result)
        rt = JobRuntime()
        mock_get_rt.return_value = rt

        response = await server.cancel_job(job_id="cancel-e2e-test")
        self.assertTrue(response["success"])
        mock_launcher.cancel.assert_called_once_with("sky-42", "gcp", "my-cluster")
```

**Step 2: Add cancel timeout test**

```python
async def test_cancel_job_direct_cloud_timeout(self):
    """cancel_job returns structured error on launcher.cancel timeout."""
    with (
        patch("oumi_mcp_server.server._resolve_job_record", return_value=None),
        patch("oumi_mcp_server.server.launcher") as mock_launcher,
    ):
        async def slow_cancel(*args):
            await asyncio.sleep(60)

        mock_launcher.cancel = slow_cancel

        response = await server.cancel_job(
            oumi_job_id="sky-timeout",
            cloud="gcp",
            cluster_name="slow-cluster",
        )
        self.assertFalse(response["success"])
        self.assertIn("timed out", response.get("error", ""))
```

**Step 3: Run tests**

Run: `python -m pytest tests/test_job_recovery_and_control.py -v`

**Step 4: Commit**

```bash
git add tests/test_job_recovery_and_control.py
git commit -m "test: add cancel_job end-to-end and timeout tests"
```

---

### Task 17: Add _launch_cloud with client_cwd test

**Files:**
- Modify: `tests/test_job_recovery_and_control.py`

**Step 1: Add test that verifies client_cwd sets working_dir**

```python
async def test_cloud_launch_uses_client_cwd_as_working_dir(self):
    """_launch_cloud sets working_dir to client_cwd for training configs."""
    record = JobRecord(
        job_id="cwd-test",
        command="train",
        config_path="/tmp/train_config.yaml",
        cloud="gcp",
        cluster_name="",
        oumi_job_id="",
        model_name="test-model",
        submit_time=datetime.now(tz=timezone.utc).isoformat(),
    )
    rt = JobRuntime()
    rt.run_dir = Path("/tmp/oumi-mcp-test-run")
    rt.run_dir.mkdir(parents=True, exist_ok=True)

    captured_job_config = {}

    def mock_up(job_config, cluster_name):
        captured_job_config["working_dir"] = job_config.working_dir
        status = OumiJobStatus(
            name="cwd-test", id="sky-cwd",
            status="PENDING", cluster="auto-cluster",
            metadata="", done=False, state=JobState.PENDING,
        )
        return MagicMock(), status

    with (
        patch("oumi_mcp_server.job_service.get_registry") as mock_reg,
        patch("oumi_mcp_server.job_service.launcher") as mock_launcher,
        patch("oumi_mcp_server.job_service._is_job_config", return_value=False),
        patch("oumi_mcp_server.job_service._stage_cloud_config", return_value="config.yaml"),
        patch("oumi_mcp_server.job_service._build_cloud_job_config") as mock_build,
    ):
        mock_reg.return_value.update = MagicMock()
        mock_reg.return_value.get = MagicMock(return_value=record)
        mock_launcher.up = mock_up

        # Mock _build_cloud_job_config to capture working_dir
        mock_jc = MagicMock()
        mock_jc.working_dir = None
        mock_jc.name = ""
        mock_build.return_value = mock_jc

        await job_service._launch_cloud(
            record, rt,
            client_cwd="/users/me/project",
            accelerators="A100:1",
        )

        # Verify _build_cloud_job_config received client_cwd as working_dir
        call_kwargs = mock_build.call_args
        assert call_kwargs.kwargs.get("working_dir") == "/users/me/project" or \
               call_kwargs[1].get("working_dir") == "/users/me/project"
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_job_recovery_and_control.py::TestCloudLaunchClientCwd -v`

**Step 3: Commit**

```bash
git add tests/test_job_recovery_and_control.py
git commit -m "test: verify _launch_cloud passes client_cwd as working_dir"
```

---

### Task 18: Add list_jobs via launcher.status() test

**Files:**
- Modify: `tests/test_job_recovery_and_control.py`

**Step 1: Write the test**

```python
async def test_list_jobs_queries_launcher_status(self):
    """list_jobs calls launcher.status() and enriches with MCP job IDs."""
    mock_status = OumiJobStatus(
        name="sky-train", id="42",
        status="RUNNING", cluster="gpu-cluster",
        metadata="", done=False, state=JobState.RUNNING,
    )
    mapping_record = JobRecord(
        job_id="train_20260226_abc",
        command="train",
        config_path="/tmp/c.yaml",
        cloud="gcp",
        cluster_name="gpu-cluster",
        oumi_job_id="42",
        model_name="llama-8b",
        submit_time=datetime.now(tz=timezone.utc).isoformat(),
    )

    mock_reg = MagicMock()
    mock_reg.all.return_value = []  # no local jobs
    mock_reg.find_by_cloud_identity.return_value = mapping_record

    with (
        patch("oumi_mcp_server.server.get_registry", return_value=mock_reg),
        patch("oumi_mcp_server.server.launcher") as mock_launcher,
    ):
        mock_launcher.status = MagicMock(return_value={"gcp": [mock_status]})
        result = await server.list_jobs()

    assert len(result) == 1
    assert result[0]["job_id"] == "train_20260226_abc"  # enriched with MCP ID
    assert result[0]["status"] == "RUNNING"
    assert result[0]["cloud"] == "gcp"
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_job_recovery_and_control.py -v`

**Step 3: Commit**

```bash
git add tests/test_job_recovery_and_control.py
git commit -m "test: list_jobs queries launcher.status and enriches with MCP IDs"
```

---

### Task 19: Final integration pass — run all tests, fix breakage

**Step 1: Run full test suite**

Run: `cd /Users/aniruddhanramesh/dev/oumi/projects/oumi-mcp && python -m pytest tests/ -v --tb=short 2>&1 | tail -60`

**Step 2: Fix any remaining test failures**

Likely fixes needed:
- Any test that constructs `JobRecord(..., status="running")` needs `status=` removed
- Any test that asserts `record.status == "..."` needs to check live status instead
- Import changes if `_TERMINAL_STATUSES` was referenced in tests

**Step 3: Commit all fixes**

```bash
git add -u
git commit -m "fix: update remaining tests for status-free JobRecord"
```

---

### Task 20: Final commit — update design doc as completed

**Step 1: Verify all tests pass**

Run: `python -m pytest tests/ -v`

**Step 2: Commit design doc**

```bash
git add docs/plans/
git commit -m "docs: add design and implementation plan for issue #2 fixes"
```
