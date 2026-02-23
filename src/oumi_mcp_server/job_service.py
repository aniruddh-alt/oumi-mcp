"""Job management service for Oumi MCP execution tools.

Provides job submission, status polling, cancellation, and log streaming
for both local and cloud execution.

Design:
    - **Local jobs** (``cloud == "local"``): spawned directly via
      ``subprocess.Popen`` running the Oumi CLI (e.g. ``oumi train -c …``).
      This avoids the ``oumi.launcher.LocalCluster`` requirement for a
      ``working_dir``, which the MCP server cannot reliably provide.
    - **Cloud jobs**: delegated to ``oumi.launcher.up()`` which handles
      SkyPilot cluster lifecycle, multi-cloud routing, etc.
    - A thin ``JobRegistry`` maps MCP job IDs to ``JobRecord`` objects.
    - ``tail_log_file()`` provides async log tailing for streaming to
      the MCP client via ``ctx.info()``.
"""

import asyncio
import io
import json
import logging
import os
import shlex
import shutil
import subprocess
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import oumi.launcher as launcher
from oumi.core.launcher.base_cluster import BaseCluster
from oumi.core.launcher.base_cluster import JobStatus as OumiJobStatus

from oumi_mcp_server.config_service import parse_yaml
from oumi_mcp_server.constants import (
    JOB_LOGS_DIR,
    JOB_RUNS_DIR,
    JOB_STATE_DIR,
    LOG_TAIL_INTERVAL_SECONDS,
    MAX_COMPLETED_JOBS,
)
from oumi_mcp_server.models import JobCancelResponse

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    """Tracks a single Oumi job submitted via the MCP server.

    Attributes:
        job_id: MCP-generated unique identifier.
        command: Oumi CLI subcommand (train, evaluate, etc.).
        config_path: Absolute path to the YAML config.
        cloud: Cloud provider name (e.g. "local", "gcp", "aws").
        cluster_name: Cluster name used with the launcher.
        oumi_job_id: The job ID returned by ``oumi.launcher`` (on the cluster).
        model_name: Model name extracted from config for display.
        output_dir: Output directory extracted from config for display.
        log_dir: Directory where stdout/stderr log files are written.
        process: The ``subprocess.Popen`` handle for local jobs.
        cluster_obj: The ``BaseCluster`` returned by ``launcher.up()``
            (cloud jobs only).
        submit_time: UTC timestamp when the job was submitted.
        oumi_status: Latest ``JobStatus`` snapshot from the launcher
            (cloud jobs only).
        error_message: Error string if submission itself failed.
        runner_task: The background asyncio task running the job.
    """

    job_id: str
    command: str
    config_path: str
    cloud: str = "local"
    cluster_name: str = ""
    oumi_job_id: str = ""
    model_name: str = ""
    output_dir: str = ""
    log_dir: Path = field(init=False)
    process: subprocess.Popen | None = field(default=None, repr=False)  # type: ignore[type-arg]
    cluster_obj: BaseCluster | None = field(default=None, repr=False)
    submit_time: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    oumi_status: OumiJobStatus | None = field(default=None, repr=False)
    error_message: str | None = None
    runner_task: asyncio.Task[None] | None = field(default=None, repr=False)
    run_dir: Path = field(init=False)
    staged_config_path: str = ""
    launch_state: str = "pending"
    cancel_requested: bool = False
    launch_attempts: int = 0
    launch_started_at: str = ""
    launch_finished_at: str = ""
    launcher_error_type: str = ""
    idempotency_key: str = ""
    _stdout_f: Any = field(default=None, repr=False)
    _stderr_f: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Set log_dir to a per-job subdirectory under JOB_LOGS_DIR."""
        self.log_dir = JOB_LOGS_DIR / self.job_id
        self.run_dir = JOB_RUNS_DIR / self.job_id

    def close_log_files(self) -> None:
        """Safely close stdout/stderr file handles if open."""
        for f in (self._stdout_f, self._stderr_f):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        self._stdout_f = None
        self._stderr_f = None

    @property
    def is_local(self) -> bool:
        """True if this is a local job (not cloud)."""
        return self.cloud == "local"

    @property
    def is_done(self) -> bool:
        """True if the job has finished (locally or on a cloud cluster)."""
        if self.launch_state in {"completed", "failed"}:
            return True
        if self.process is not None:
            return self.process.poll() is not None
        if self.oumi_status is not None:
            return self.oumi_status.done
        return self.error_message is not None


class JobRegistry:
    """Async-safe in-memory registry of ``JobRecord`` instances.

    Evicts the oldest finished jobs when the count exceeds
    ``MAX_COMPLETED_JOBS``.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._idempotency_map: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._state_dir = JOB_STATE_DIR
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._load_persisted_jobs()

    def _job_state_path(self, job_id: str) -> Path:
        return self._state_dir / f"{job_id}.json"

    def _record_to_payload(self, record: JobRecord) -> dict[str, Any]:
        return {
            "job_id": record.job_id,
            "command": record.command,
            "config_path": record.config_path,
            "cloud": record.cloud,
            "cluster_name": record.cluster_name,
            "oumi_job_id": record.oumi_job_id,
            "model_name": record.model_name,
            "output_dir": record.output_dir,
            "submit_time": record.submit_time.isoformat(),
            "error_message": record.error_message,
            "staged_config_path": record.staged_config_path,
            "launch_state": record.launch_state,
            "cancel_requested": record.cancel_requested,
            "launch_attempts": record.launch_attempts,
            "launch_started_at": record.launch_started_at,
            "launch_finished_at": record.launch_finished_at,
            "launcher_error_type": record.launcher_error_type,
            "idempotency_key": record.idempotency_key,
        }

    def _payload_to_record(self, payload: dict[str, Any]) -> JobRecord:
        submit_time_raw = payload.get("submit_time")
        submit_time = datetime.now(tz=timezone.utc)
        if isinstance(submit_time_raw, str):
            try:
                submit_time = datetime.fromisoformat(submit_time_raw)
            except ValueError:
                pass
        return JobRecord(
            job_id=str(payload.get("job_id", "")),
            command=str(payload.get("command", "")),
            config_path=str(payload.get("config_path", "")),
            cloud=str(payload.get("cloud", "local") or "local"),
            cluster_name=str(payload.get("cluster_name", "")),
            oumi_job_id=str(payload.get("oumi_job_id", "")),
            model_name=str(payload.get("model_name", "")),
            output_dir=str(payload.get("output_dir", "")),
            submit_time=submit_time,
            error_message=payload.get("error_message"),
            staged_config_path=str(payload.get("staged_config_path", "")),
            launch_state=str(payload.get("launch_state", "pending") or "pending"),
            cancel_requested=bool(payload.get("cancel_requested", False)),
            launch_attempts=int(payload.get("launch_attempts", 0) or 0),
            launch_started_at=str(payload.get("launch_started_at", "")),
            launch_finished_at=str(payload.get("launch_finished_at", "")),
            launcher_error_type=str(payload.get("launcher_error_type", "")),
            idempotency_key=str(payload.get("idempotency_key", "")),
        )

    def _persist_unlocked(self, record: JobRecord) -> None:
        payload = self._record_to_payload(record)
        path = self._job_state_path(record.job_id)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _delete_persisted_unlocked(self, job_id: str) -> None:
        path = self._job_state_path(job_id)
        try:
            if path.exists():
                path.unlink()
        except OSError:
            logger.debug(
                "Failed deleting persisted state for %s", job_id, exc_info=True
            )

    def _load_persisted_jobs(self) -> None:
        for state_path in sorted(self._state_dir.glob("*.json")):
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                record = self._payload_to_record(payload)
                if record.job_id:
                    self._jobs[record.job_id] = record
                    if record.idempotency_key:
                        self._idempotency_map[record.idempotency_key] = record.job_id
            except Exception:
                logger.debug("Failed loading persisted job state %s", state_path)

    async def register(self, record: JobRecord) -> None:
        async with self._lock:
            if record.job_id in self._jobs:
                raise ValueError(f"Job ID already exists: {record.job_id}")
            self._jobs[record.job_id] = record
            if record.idempotency_key:
                self._idempotency_map[record.idempotency_key] = record.job_id
            self._persist_unlocked(record)
            self._evict_finished_unlocked()

    async def get_by_idempotency_key(self, key: str) -> JobRecord | None:
        """Return the existing job for *key*, or None if not found."""
        async with self._lock:
            job_id = self._idempotency_map.get(key)
            if job_id:
                return self._jobs.get(job_id)
            return None

    def _evict_finished_unlocked(self) -> None:
        finished = [r for r in self._jobs.values() if r.is_done]
        if len(finished) <= MAX_COMPLETED_JOBS:
            return
        finished.sort(key=lambda r: r.submit_time)
        for r in finished[: len(finished) - MAX_COMPLETED_JOBS]:
            del self._jobs[r.job_id]
            if r.idempotency_key:
                self._idempotency_map.pop(r.idempotency_key, None)
            self._delete_persisted_unlocked(r.job_id)
            logger.debug("Evicted finished job %s from registry", r.job_id)

    async def persist(self, record: JobRecord) -> None:
        async with self._lock:
            self._jobs[record.job_id] = record
            self._persist_unlocked(record)

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def get_by_cloud_identity(
        self,
        *,
        cloud: str,
        cluster_name: str = "",
        oumi_job_id: str,
    ) -> JobRecord | None:
        async with self._lock:
            for record in self._jobs.values():
                if record.oumi_job_id != oumi_job_id or record.cloud != cloud:
                    continue
                if cluster_name and record.cluster_name != cluster_name:
                    continue
                return record
            return None

    async def all_jobs(self) -> list[JobRecord]:
        async with self._lock:
            return list(self._jobs.values())

    async def running(self) -> list[JobRecord]:
        async with self._lock:
            return [r for r in self._jobs.values() if not r.is_done]

    async def completed(self) -> list[JobRecord]:
        async with self._lock:
            return [r for r in self._jobs.values() if r.is_done]


_registry: JobRegistry | None = None


def get_registry() -> JobRegistry:
    """Return the global ``JobRegistry``, creating it on first access."""
    global _registry
    if _registry is None:
        _registry = JobRegistry()
    return _registry


def make_job_id(command: str, job_name: str | None = None) -> str:
    """Generate a human-friendly job ID.

    Format: ``{command}_{YYYYMMDD_HHMMSS}_{6-hex}`` or the caller-supplied
    *job_name* if provided.
    """
    if job_name:
        return job_name
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{command}_{ts}_{short}"


def extract_job_metadata(config_path: str) -> dict[str, Any]:
    """Extract model_name and output_dir from an Oumi YAML config.

    Returns a dict with ``model_name`` and ``output_dir`` keys.
    Missing values default to ``"unknown"`` / ``"./output"``.
    """
    config = parse_yaml(config_path)
    model_name = config.get("model", {}).get("model_name", "unknown") or "unknown"

    output_dir = (
        config.get("training", {}).get("output_dir")
        or config.get("output_dir")
        or "./output"
    )
    return {"model_name": model_name, "output_dir": output_dir}


_COMMAND_MAP: dict[str, str] = {
    "train": "oumi train",
    "evaluate": "oumi evaluate",
    "eval": "oumi evaluate",
    "infer": "oumi infer",
    "synth": "oumi synthesize",
    "analyze": "oumi analyze",
    "tune": "oumi tune",
    "quantize": "oumi quantize",
}

_DEFAULT_CLOUD_SETUP_SCRIPT = """set -e
python3 -m pip install --upgrade pip uv
uv pip install --system "oumi[gpu]>=0.7,<0.9" || uv pip install --system "oumi>=0.7,<0.9"
oumi --version || { echo "ERROR: oumi installation failed"; exit 1; }
"""

_DEFAULT_CREDENTIAL_FILES: list[str] = [
    "~/.cache/huggingface/token",
    "~/.netrc",
]


def _is_job_config(config_path: Path) -> bool:
    """Return True if *config_path* is a launcher job config (not a training config).

    A job config has top-level ``resources``, ``setup``, or ``run`` keys rather
    than training-specific keys like ``model`` or ``training``.
    """
    try:
        data = parse_yaml(str(config_path))
        if not isinstance(data, dict):
            return False
        job_config_keys = {"resources", "setup", "run"}
        return bool(job_config_keys.intersection(data.keys()))
    except Exception:
        return False


def _parse_gpu_count(accelerators: str | None) -> int:
    """Parse the number of GPUs from an accelerator spec string.

    Handles formats like ``"A100:8"`` (→ 8), ``"A100"`` (→ 1),
    and ``None`` (→ 0).
    """
    if not accelerators:
        return 0
    parts = accelerators.split(":")
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return 1


def _default_file_mounts() -> dict[str, str]:
    """Return file mounts for common credential files that exist locally.

    Includes ``~/.cache/huggingface/token`` (HuggingFace auth) and
    ``~/.netrc`` (WandB / general HTTP credentials) when they exist.
    """
    mounts: dict[str, str] = {}
    for cred_path in _DEFAULT_CREDENTIAL_FILES:
        local_path = Path(cred_path).expanduser()
        if local_path.exists():
            mounts[cred_path] = cred_path
    return mounts


def _build_local_command(config_path: str, command: str) -> list[str]:
    """Build an argv list for a local Oumi CLI invocation (no shell)."""
    oumi_cmd = _COMMAND_MAP.get(command, f"oumi {command}")
    parts = oumi_cmd.split()  # e.g. ["oumi", "train"]
    return [*parts, "-c", config_path]


def _build_shell_command(
    config_path: str,
    command: str,
    *,
    num_nodes: int = 1,
    accelerators: str | None = None,
) -> str:
    """Build the shell command string for an Oumi CLI invocation (cloud runs).

    SkyPilot copies ``working_dir`` to the remote and executes this script
    from within it, so *config_path* is a relative filename (e.g. ``config.yaml``).

    Extends PATH to cover common uv/pip install locations before executing so
    the oumi binary is reachable even when the run step's shell differs from
    the setup step's shell.  Verifies oumi is found before running.
    """
    num_gpus = _parse_gpu_count(accelerators)
    if num_gpus > 1 or num_nodes > 1:
        oumi_cmd = f"oumi distributed torchrun -m oumi {command}"
    else:
        oumi_cmd = _COMMAND_MAP.get(command, f"oumi {command}")

    path_preamble = (
        'export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"\n'
        'command -v oumi || { echo "ERROR: oumi not found on PATH after setup"; exit 1; }\n'
    )
    return f"set -e\n{path_preamble}{oumi_cmd} -c {shlex.quote(config_path)}"


def _stage_cloud_config(record: JobRecord, *, working_dir: str | None = None) -> str:
    """Copy config (and optionally a working directory) into a per-job run directory.

    For training-config wrapping mode, only the config file is copied.
    For job-config passthrough mode, pass *working_dir* to copy the entire
    source directory tree so relative references inside the config are preserved.

    Returns the staged config filename (relative to the run directory).
    """
    record.run_dir.mkdir(parents=True, exist_ok=True)

    if working_dir:
        src = Path(working_dir).expanduser()
        if src.is_dir() and src != record.run_dir:
            shutil.copytree(src, record.run_dir, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, record.run_dir / src.name)

    staged_config = record.run_dir / "config.yaml"
    shutil.copy2(record.config_path, staged_config)
    record.staged_config_path = str(staged_config)
    return staged_config.name


def _build_cloud_job_config(
    config_path: str,
    command: str,
    *,
    cloud: str,
    working_dir: str,
    accelerators: str | None = None,
    job_name: str | None = None,
    envs: dict[str, str] | None = None,
    file_mounts: dict[str, str] | None = None,
    disk_size: int | None = None,
    use_spot: bool = False,
    num_nodes: int = 1,
    setup_script: str | None = None,
) -> launcher.JobConfig:
    """Build an ``oumi.launcher.JobConfig`` for **cloud** execution.

    For cloud jobs the launcher handles cluster lifecycle via SkyPilot.
    ``working_dir`` is a per-job staging directory copied by the launcher.
    The run script references the staged config path relative to that directory.

    Automatically selects ``oumi distributed torchrun`` when multiple GPUs or
    nodes are requested.  Default file mounts include common credential files
    (``~/.cache/huggingface/token``, ``~/.netrc``) when they exist locally;
    *file_mounts* entries take precedence and can override them.
    """
    run_script = _build_shell_command(
        config_path, command, num_nodes=num_nodes, accelerators=accelerators
    )

    resources = launcher.JobResources(cloud=cloud)
    if accelerators:
        resources.accelerators = accelerators
    if disk_size is not None:
        resources.disk_size = disk_size
    if use_spot:
        resources.use_spot = True

    # Merge default credential mounts with any caller-supplied mounts.
    # Caller-supplied entries take precedence (update overwrites defaults).
    effective_mounts = _default_file_mounts()
    if file_mounts:
        effective_mounts.update(file_mounts)

    return launcher.JobConfig(
        name=job_name,
        num_nodes=num_nodes,
        resources=resources,
        working_dir=working_dir,
        setup=setup_script or _DEFAULT_CLOUD_SETUP_SCRIPT,
        run=run_script,
        envs=envs or {},
        file_mounts=effective_mounts,
    )


def start_local_job(record: JobRecord) -> None:
    """Start a local job by spawning the Oumi CLI directly.

    Creates the log directory, starts the subprocess via ``Popen``, and
    sets ``record.process`` and ``record.oumi_job_id``. Raises on failure
    (e.g. command not found, permission denied).

    Bypasses ``oumi.launcher.LocalCluster`` (which requires a
    ``working_dir``) and instead runs ``oumi <cmd> -c <config>``
    via ``subprocess.Popen`` with an argv list (no shell).

    Stdout and stderr are written to files in ``record.log_dir`` so
    that ``tail_log_file()`` can stream them to the MCP client.
    """
    cmd_argv = _build_local_command(record.config_path, record.command)

    record.launch_state = "launching"
    record.launch_attempts += 1
    record.launch_started_at = datetime.now(tz=timezone.utc).isoformat()
    record.log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
    stdout_path = record.log_dir / f"{ts}_{record.job_id}.stdout"
    stderr_path = record.log_dir / f"{ts}_{record.job_id}.stderr"

    env = os.environ.copy()
    env["OUMI_LOGGING_DIR"] = str(record.log_dir)

    stdout_f = open(stdout_path, "w")
    stderr_f = open(stderr_path, "w")

    try:
        proc = subprocess.Popen(
            cmd_argv,
            env=env,
            stdout=stdout_f,
            stderr=stderr_f,
        )
    except Exception:
        stdout_f.close()
        stderr_f.close()
        raise

    record.process = proc
    record.oumi_job_id = str(proc.pid)
    record.launch_finished_at = datetime.now(tz=timezone.utc).isoformat()
    record.launch_state = "running"
    record._stdout_f = stdout_f
    record._stderr_f = stderr_f
    logger.info(
        "Local job %s started (pid=%s): %s",
        record.job_id,
        proc.pid,
        " ".join(cmd_argv),
    )


async def wait_local_completion(record: JobRecord) -> None:
    """Await completion of a local job subprocess.

    Waits for the process to exit (in a thread) and sets
    ``record.error_message`` on non-zero exit code.  This is slow
    (minutes/hours) and should be run as a background task.
    """
    proc = record.process
    if proc is None:
        return

    stderr_path = None
    if record._stderr_f is not None:
        try:
            stderr_path = record._stderr_f.name
        except Exception:
            pass

    try:
        returncode = await asyncio.to_thread(proc.wait)

        if returncode != 0:
            record.error_message = f"Process exited with code {returncode}." + (
                f" See stderr: {stderr_path}" if stderr_path else ""
            )
            record.launch_state = "failed"
            logger.warning(
                "Local job %s exited with code %d", record.job_id, returncode
            )
        else:
            record.launch_state = "completed"
            logger.info("Local job %s completed successfully", record.job_id)
    except Exception as exc:
        record.error_message = str(exc)
        record.launch_state = "failed"
        logger.exception("Failed to run local job %s", record.job_id)
    finally:
        record.close_log_files()
        await get_registry().persist(record)


async def _launch_cloud(
    record: JobRecord,
    *,
    accelerators: str | None = None,
    envs: dict[str, str] | None = None,
    file_mounts: dict[str, str] | None = None,
    disk_size: int | None = None,
    use_spot: bool = False,
    num_nodes: int = 1,
    setup_script: str | None = None,
) -> None:
    """Launch a cloud job via ``oumi.launcher.up()``.

    Supports two modes:

    * **Job-config passthrough** (when the config file itself is a launcher job
      config with ``resources``/``setup``/``run`` keys): loads the config directly
      via ``launcher.JobConfig.from_yaml()`` so all cloud-specific fields
      (``envs``, ``file_mounts``, ``storage_mounts``, ``disk_size``, etc.) are
      preserved as written.

    * **Training-config wrapping**: wraps a training YAML in a minimal
      ``launcher.JobConfig``, applying any caller-supplied *envs*, *file_mounts*,
      *disk_size*, *use_spot*, *num_nodes*, and *setup_script* overrides on top of
      sensible defaults (pinned oumi version, auto-mounted credential files).

    Updates *record* in-place with the cluster object, oumi job ID, and
    initial status.  On failure, sets ``record.error_message``.
    """
    record.launch_state = "launching"
    record.launch_attempts += 1
    record.launch_started_at = datetime.now(tz=timezone.utc).isoformat()
    await get_registry().persist(record)

    try:
        config_path = Path(record.config_path)
        job_config_mode = _is_job_config(config_path)

        if job_config_mode:
            # Passthrough: load the job config YAML directly.
            # Stage the config file into the run dir; the job config's own
            # working_dir (if any) is preserved inside the YAML.
            config_parent = str(Path(record.config_path).expanduser().resolve().parent)
            _stage_cloud_config(record, working_dir=config_parent)
            job_config = launcher.JobConfig.from_yaml(record.staged_config_path)
            # Allow the job name to be overridden with the MCP job ID.
            if not job_config.name:
                job_config.name = record.job_id
            # If caller provided extra overrides, layer them on top.
            if envs:
                merged = dict(job_config.envs or {})
                merged.update(envs)
                job_config.envs = merged
            if file_mounts:
                merged_mounts = dict(job_config.file_mounts or {})
                merged_mounts.update(file_mounts)
                job_config.file_mounts = merged_mounts
        else:
            staged_config_name = _stage_cloud_config(record)
            job_config = _build_cloud_job_config(
                staged_config_name,
                record.command,
                cloud=record.cloud,
                working_dir=str(record.run_dir),
                accelerators=accelerators,
                job_name=record.job_id,
                envs=envs,
                file_mounts=file_mounts,
                disk_size=disk_size,
                use_spot=use_spot,
                num_nodes=num_nodes,
                setup_script=setup_script,
            )
        cluster, status = await asyncio.to_thread(
            launcher.up,
            job_config,
            record.cluster_name or None,
        )
        record.cluster_obj = cluster
        record.oumi_job_id = status.id if status else ""
        record.oumi_status = status
        record.cluster_name = status.cluster if status else record.cluster_name
        record.launch_finished_at = datetime.now(tz=timezone.utc).isoformat()
        record.launch_state = "submitted"
        logger.info(
            "Cloud job %s launched on %s (oumi_id=%s)",
            record.job_id,
            record.cloud,
            record.oumi_job_id,
        )
        if record.cancel_requested and record.oumi_job_id:
            try:
                result_status = await asyncio.to_thread(
                    launcher.cancel,
                    record.oumi_job_id,
                    record.cloud,
                    record.cluster_name,
                )
                record.oumi_status = result_status
                record.launch_state = "cancel_requested"
            except Exception as cancel_exc:
                record.error_message = (
                    "Cancellation was requested during launch, but automatic "
                    f"cloud cancellation failed: {cancel_exc}"
                )
                record.launcher_error_type = type(cancel_exc).__name__
        await get_registry().persist(record)
    except Exception as exc:
        record.error_message = str(exc)
        record.launcher_error_type = type(exc).__name__
        record.launch_finished_at = datetime.now(tz=timezone.utc).isoformat()
        record.launch_state = "failed"
        logger.exception("Failed to launch cloud job %s", record.job_id)
        await get_registry().persist(record)


async def launch_job(
    record: JobRecord,
    *,
    accelerators: str | None = None,
    envs: dict[str, str] | None = None,
    file_mounts: dict[str, str] | None = None,
    disk_size: int | None = None,
    use_spot: bool = False,
    num_nodes: int = 1,
    setup_script: str | None = None,
) -> None:
    """Launch a job -- local or cloud -- in a background thread.

    For local jobs, spawns the Oumi CLI directly via subprocess.
    For cloud jobs, delegates to ``oumi.launcher.up()``.
    """
    if record.is_local:
        start_local_job(record)
        await wait_local_completion(record)
    else:
        await _launch_cloud(
            record,
            accelerators=accelerators,
            envs=envs,
            file_mounts=file_mounts,
            disk_size=disk_size,
            use_spot=use_spot,
            num_nodes=num_nodes,
            setup_script=setup_script,
        )


async def poll_status(record: JobRecord) -> OumiJobStatus | None:
    """Fetch the latest status for a job.

    For **local** jobs, checks the subprocess return code directly.
    For **cloud** jobs, tries ``cluster.get_job()`` first, then falls
    back to ``launcher.status()``.

    Returns an ``OumiJobStatus`` for cloud jobs, or ``None`` for local
    jobs (status is derived from ``record.process`` instead).
    """
    # ---- Local jobs: derive status from the subprocess ----
    if record.is_local:
        # Nothing to poll -- the process handle *is* the status.
        return None

    # ---- Cloud jobs: delegate to oumi.launcher ----
    if record.error_message and record.cluster_obj is None:
        return record.oumi_status

    def _derive_launch_state(status: OumiJobStatus) -> str:
        if not status.done:
            return "running"
        status_str = (status.status or "").lower()
        if "fail" in status_str:
            return "failed"
        if "cancel" in status_str:
            return "cancel_requested"
        return "completed"

    if record.cluster_obj and record.oumi_job_id:
        try:
            status = await asyncio.to_thread(
                record.cluster_obj.get_job, record.oumi_job_id
            )
            if status:
                record.oumi_status = status
                record.oumi_job_id = status.id or record.oumi_job_id
                record.cluster_name = status.cluster or record.cluster_name
                record.launch_state = _derive_launch_state(status)
                await get_registry().persist(record)
                return status
        except Exception:
            logger.warning(
                "launcher.status failed for %s; returning stale status",
                record.job_id,
                exc_info=True,
            )

    try:
        if not record.oumi_job_id:
            return record.oumi_status
        all_statuses = await asyncio.to_thread(
            launcher.status,
            cloud=record.cloud,
            cluster=record.cluster_name or None,
            id=record.oumi_job_id,
        )
        for _, jobs in all_statuses.items():
            for s in jobs:
                if s.id == record.oumi_job_id:
                    record.oumi_status = s
                    record.oumi_job_id = s.id or record.oumi_job_id
                    record.cluster_name = s.cluster or record.cluster_name
                    record.launch_state = _derive_launch_state(s)
                    await get_registry().persist(record)
                    return s
    except Exception:
        logger.warning(
            "launcher.status failed for %s; returning stale status",
            record.job_id,
            exc_info=True,
        )

    return record.oumi_status


async def cancel(record: JobRecord, *, force: bool = False) -> JobCancelResponse:
    """Cancel a job.

    For **local** jobs, sends SIGTERM (or SIGKILL if *force* is True)
    to the subprocess.  For **cloud** jobs, delegates to
    ``oumi.launcher.cancel()``.

    Returns a dict with ``success`` (bool) and ``message`` or ``error``.
    """
    if record.is_done:
        return {
            "success": False,
            "error": (
                f"Job {record.job_id} is already finished "
                f"(status: {record.oumi_status.status if record.oumi_status else 'unknown'})"
            ),
        }

    if not record.oumi_job_id and record.process is None:
        record.cancel_requested = True
        record.launch_state = "cancel_requested"
        record.error_message = "Cancellation requested while launch is pending."
        await get_registry().persist(record)
        return {
            "success": True,
            "message": (
                f"Cancellation requested for {record.job_id}. "
                "If the cloud launch completes, the MCP will attempt best-effort cancellation."
            ),
        }

    if record.is_local and record.process is not None:
        try:
            if force:
                record.process.kill()  # SIGKILL
                action = "killed (SIGKILL)"
            else:
                record.process.terminate()  # SIGTERM
                action = "terminated (SIGTERM)"
            record.cancel_requested = True
            record.launch_state = "cancel_requested"
            record.error_message = f"Cancelled by user ({action})"
            logger.info("Local job %s %s", record.job_id, action)
            await get_registry().persist(record)
            return {
                "success": True,
                "message": f"Job {record.job_id} {action}.",
            }
        except OSError as exc:
            return {
                "success": False,
                "error": f"Failed to cancel local job {record.job_id}: {exc}",
            }

    try:
        result_status = await asyncio.to_thread(
            launcher.cancel,
            record.oumi_job_id,
            record.cloud,
            record.cluster_name,
        )
        record.cancel_requested = True
        record.launch_state = "cancel_requested"
        record.oumi_status = result_status
        await get_registry().persist(record)
        return {
            "success": True,
            "message": f"Job {record.job_id} cancel requested on {record.cloud}/{record.cluster_name}.",
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to cancel job {record.job_id}: {exc}",
        }


def get_log_paths(record: JobRecord) -> dict[str, Path | None]:
    """Return paths to the stdout and stderr log files for a job.

    For **local** jobs spawned by ``_launch_local()``, files are named
    ``{timestamp}_{job_id}.stdout`` / ``.stderr``.

    For **cloud** jobs that went through ``oumi.launcher``, the
    ``LocalClient`` names them ``{timestamp}_{oumi_job_id}.stdout`` /
    ``.stderr``.

    Since the timestamp prefix varies, we glob for the job/oumi ID
    suffix first, then fall back to any matching extension.

    Returns a dict with ``"stdout"`` and ``"stderr"`` keys, each
    mapping to a ``Path`` or ``None`` if the file doesn't exist yet.
    """
    result: dict[str, Path | None] = {"stdout": None, "stderr": None}
    log_dir = record.log_dir
    if not log_dir.is_dir():
        return result

    id_candidates = [record.job_id]
    if record.oumi_job_id and record.oumi_job_id != record.job_id:
        id_candidates.append(record.oumi_job_id)

    for suffix in ("stdout", "stderr"):
        for candidate_id in id_candidates:
            matches = sorted(log_dir.glob(f"*_{candidate_id}.{suffix}"))
            if matches:
                result[suffix] = matches[-1]
                break
        else:
            matches = sorted(log_dir.glob(f"*.{suffix}"))
            if matches:
                result[suffix] = matches[-1]

    return result


async def tail_log_file(
    path: Path,
    done_event: asyncio.Event,
    poll_interval: float = LOG_TAIL_INTERVAL_SECONDS,
) -> AsyncIterator[str]:
    """Async generator that yields new lines from *path* as they appear.

    Behaves like ``tail -f``: opens the file, seeks to the current end,
    then yields new complete lines as they are written.  Stops when
    *done_event* is set **and** no more data is available.

    If the file does not exist yet, waits up to ``poll_interval`` between
    checks until it appears or *done_event* fires.
    """
    while not path.exists():
        if done_event.is_set():
            return
        await asyncio.sleep(poll_interval)

    position = 0
    partial = ""

    while True:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0

        if size > position:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(position)
                    chunk = f.read()
                    position = f.tell()
            except OSError:
                chunk = ""

            if chunk:
                partial += chunk
                while "\n" in partial:
                    line, partial = partial.split("\n", 1)
                    yield line

        if done_event.is_set():
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(position)
                    remaining = f.read()
            except OSError:
                remaining = ""
            if remaining:
                partial += remaining
            if partial:
                yield partial
            return

        await asyncio.sleep(poll_interval)


async def stream_cloud_logs(
    record: JobRecord,
    done_event: asyncio.Event,
) -> AsyncIterator[str]:
    """Yield log lines from ``cluster.get_logs_stream()`` for cloud jobs.

    Falls back silently (returns without yielding) if the cluster does not
    support log streaming (raises ``NotImplementedError``).
    """
    cluster = record.cluster_obj
    if cluster is None:
        return

    try:
        stream: io.TextIOBase = await asyncio.to_thread(
            cluster.get_logs_stream,
            record.cluster_name,
            record.oumi_job_id or None,
        )
    except NotImplementedError:
        logger.debug(
            "Cloud %s does not support get_logs_stream for job %s",
            record.cloud,
            record.job_id,
        )
        return
    except Exception:
        logger.debug(
            "get_logs_stream failed for job %s",
            record.job_id,
            exc_info=True,
        )
        return

    def _read_lines() -> list[str]:
        """Read available lines from the stream (blocking)."""
        lines: list[str] = []
        try:
            while True:
                line = stream.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
        except Exception:
            pass
        return lines

    while not done_event.is_set():
        lines = await asyncio.to_thread(_read_lines)
        for line in lines:
            yield line
        if not lines:
            await asyncio.sleep(LOG_TAIL_INTERVAL_SECONDS)
