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
import dataclasses
import io
import json
import logging
import os
import shlex
import shutil
import subprocess
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import oumi.launcher as launcher
from oumi.core.launcher.base_cluster import BaseCluster
from oumi.core.launcher.base_cluster import JobStatus as OumiJobStatus

from oumi_mcp_server.config_service import parse_yaml
from oumi_mcp_server.constants import (
    DEFAULT_JOBS_FILE,
    LOG_TAIL_INTERVAL_SECONDS,
)
from oumi_mcp_server.models import JobCancelResponse

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    """Persisted job metadata. All fields are strings for simple JSON serde."""

    job_id: str
    command: str
    config_path: str
    cloud: str
    cluster_name: str
    oumi_job_id: str
    model_name: str
    status: str  # "running", "completed", "failed"
    submit_time: str  # ISO 8601


@dataclass
class JobRuntime:
    """Ephemeral per-job state -- lives only in memory, never persisted."""

    process: subprocess.Popen | None = None  # type: ignore[type-arg]
    cluster_obj: BaseCluster | None = None
    runner_task: asyncio.Task[None] | None = None
    oumi_status: OumiJobStatus | None = None
    stdout_f: Any = None
    stderr_f: Any = None
    log_dir: Path | None = None
    run_dir: Path | None = None
    staged_config_path: str = ""
    cancel_requested: bool = False
    error_message: str | None = None

    def close_log_files(self) -> None:
        for f in (self.stdout_f, self.stderr_f):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        self.stdout_f = None
        self.stderr_f = None


class JobRegistry:
    """Single-file JSON registry of job records."""

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
                r = JobRecord(**entry)
                self._jobs[r.job_id] = r
        except Exception:
            logger.warning("Could not load %s, starting fresh", self._path)

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
        record = self._jobs[job_id]
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


_runtimes: dict[str, JobRuntime] = {}


def get_runtime(job_id: str) -> JobRuntime:
    if job_id not in _runtimes:
        _runtimes[job_id] = JobRuntime()
    return _runtimes[job_id]


_registry: JobRegistry | None = None


def get_registry(path: Path | None = None) -> JobRegistry:
    """Return the global ``JobRegistry``, creating it on first access."""
    global _registry
    if _registry is None:
        _registry = JobRegistry(path or DEFAULT_JOBS_FILE)
    return _registry


def make_job_id(command: str, job_name: str | None = None) -> str:
    """Generate a human-friendly job ID.

    Format: ``{command}_{YYYYMMDD_HHMMSS}_{6-hex}`` or the caller-supplied
    *job_name* if provided (sanitized to prevent path traversal).
    """
    if job_name:
        sanitized = job_name.replace("..", "_").replace("/", "_").replace("\\", "_")
        sanitized = sanitized.strip("._- ")
        if not sanitized:
            raise ValueError(f"Invalid job_name after sanitization: {job_name!r}")
        return sanitized
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
pip install uv
uv pip install --system oumi[gpu]
command -v oumi || { echo "ERROR: oumi not found after install"; exit 1; }
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


def _stage_cloud_config(
    record: JobRecord, rt: JobRuntime, *, working_dir: str | None = None
) -> str:
    """Copy config (and optionally a working directory) into a per-job run directory.

    For training-config wrapping mode, only the config file is copied.
    For job-config passthrough mode, pass *working_dir* to copy the entire
    source directory tree so relative references inside the config are preserved.

    Returns the staged config filename (relative to the run directory).
    """
    assert rt.run_dir is not None
    rt.run_dir.mkdir(parents=True, exist_ok=True)

    if working_dir:
        src = Path(working_dir).expanduser()
        if src.is_dir() and src != rt.run_dir:
            shutil.copytree(src, rt.run_dir, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, rt.run_dir / src.name)

    staged_config = rt.run_dir / "config.yaml"
    shutil.copy2(record.config_path, staged_config)
    rt.staged_config_path = str(staged_config)
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
    effective_mounts = _default_file_mounts()
    if file_mounts:
        effective_mounts.update(file_mounts)

    return launcher.JobConfig(
        name=job_name,
        num_nodes=num_nodes,
        resources=resources,
        working_dir=working_dir,
        setup=_DEFAULT_CLOUD_SETUP_SCRIPT,
        run=run_script,
        envs=envs or {},
        file_mounts=effective_mounts,
    )


def start_local_job(record: JobRecord, rt: JobRuntime) -> None:
    """Start a local job by spawning the Oumi CLI directly.

    Creates the log directory, starts the subprocess via ``Popen``, and
    sets ``rt.process`` and ``record.oumi_job_id``. Raises on failure
    (e.g. command not found, permission denied).

    Stdout and stderr are written to files in ``rt.log_dir`` so
    that ``tail_log_file()`` can stream them to the MCP client.
    """
    cmd_argv = _build_local_command(record.config_path, record.command)

    assert rt.log_dir is not None
    rt.log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
    stdout_path = rt.log_dir / f"{ts}_{record.job_id}.stdout"
    stderr_path = rt.log_dir / f"{ts}_{record.job_id}.stderr"

    env = os.environ.copy()
    env["OUMI_LOGGING_DIR"] = str(rt.log_dir)

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

    rt.process = proc
    record.oumi_job_id = str(proc.pid)
    get_registry().update(record.job_id, oumi_job_id=record.oumi_job_id, status="running")
    rt.stdout_f = stdout_f
    rt.stderr_f = stderr_f
    logger.info(
        "Local job %s started (pid=%s): %s",
        record.job_id,
        proc.pid,
        " ".join(cmd_argv),
    )


async def wait_local_completion(record: JobRecord, rt: JobRuntime) -> None:
    """Await completion of a local job subprocess.

    Waits for the process to exit (in a thread) and updates
    ``record.status`` via the registry on completion.
    """
    proc = rt.process
    if proc is None:
        return

    stderr_path = None
    if rt.stderr_f is not None:
        try:
            stderr_path = rt.stderr_f.name
        except Exception:
            pass

    try:
        returncode = await asyncio.to_thread(proc.wait)

        if returncode != 0:
            rt.error_message = f"Process exited with code {returncode}." + (
                f" See stderr: {stderr_path}" if stderr_path else ""
            )
            get_registry().update(record.job_id, status="failed")
            logger.warning(
                "Local job %s exited with code %d", record.job_id, returncode
            )
        else:
            get_registry().update(record.job_id, status="completed")
            logger.info("Local job %s completed successfully", record.job_id)
    except Exception as exc:
        rt.error_message = str(exc)
        get_registry().update(record.job_id, status="failed")
        logger.exception("Failed to run local job %s", record.job_id)
    finally:
        rt.close_log_files()


async def _launch_cloud(
    record: JobRecord,
    rt: JobRuntime,
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
      sensible defaults (oumi with GPU extras, auto-mounted credential files).

    Updates *record* and *rt* in-place with the cluster object, oumi job ID, and
    initial status.  On failure, sets ``rt.error_message``.
    """
    reg = get_registry()
    reg.update(record.job_id, status="launching")

    try:
        config_path = Path(record.config_path)
        job_config_mode = _is_job_config(config_path)

        if job_config_mode:
            config_parent = str(Path(record.config_path).expanduser().resolve().parent)
            _stage_cloud_config(record, rt, working_dir=config_parent)
            job_config = launcher.JobConfig.from_yaml(rt.staged_config_path)
            if not job_config.name:
                job_config.name = record.job_id
            if envs:
                merged = dict(job_config.envs or {})
                merged.update(envs)
                job_config.envs = merged
            if file_mounts:
                merged_mounts = dict(job_config.file_mounts or {})
                merged_mounts.update(file_mounts)
                job_config.file_mounts = merged_mounts
        else:
            staged_config_name = _stage_cloud_config(record, rt)
            job_config = _build_cloud_job_config(
                staged_config_name,
                record.command,
                cloud=record.cloud,
                working_dir=str(rt.run_dir),
                accelerators=accelerators,
                job_name=record.job_id,
                envs=envs,
                file_mounts=file_mounts,
                disk_size=disk_size,
                use_spot=use_spot,
                num_nodes=num_nodes,
            )
        cluster, status = await asyncio.to_thread(
            launcher.up,
            job_config,
            record.cluster_name or None,
        )
        rt.cluster_obj = cluster
        oumi_job_id = status.id if status else ""
        rt.oumi_status = status
        cluster_name = status.cluster if status else record.cluster_name
        reg.update(
            record.job_id,
            oumi_job_id=oumi_job_id,
            cluster_name=cluster_name,
            status="running",
        )
        # Refresh the local record reference
        record = reg.get(record.job_id) or record
        logger.info(
            "Cloud job %s launched on %s (oumi_id=%s)",
            record.job_id,
            record.cloud,
            record.oumi_job_id,
        )
        if rt.cancel_requested and record.oumi_job_id:
            try:
                result_status = await asyncio.to_thread(
                    launcher.cancel,
                    record.oumi_job_id,
                    record.cloud,
                    record.cluster_name,
                )
                rt.oumi_status = result_status
                reg.update(record.job_id, status="failed")
            except Exception as cancel_exc:
                rt.error_message = (
                    "Cancellation was requested during launch, but automatic "
                    f"cloud cancellation failed: {cancel_exc}"
                )
    except Exception as exc:
        rt.error_message = str(exc)
        reg.update(record.job_id, status="failed")
        logger.exception("Failed to launch cloud job %s", record.job_id)


async def launch_job(
    record: JobRecord,
    rt: JobRuntime,
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
    if record.cloud == "local":
        start_local_job(record, rt)
        await wait_local_completion(record, rt)
    else:
        await _launch_cloud(
            record,
            rt,
            accelerators=accelerators,
            envs=envs,
            file_mounts=file_mounts,
            disk_size=disk_size,
            use_spot=use_spot,
            num_nodes=num_nodes,
            setup_script=setup_script,
        )


async def poll_status(record: JobRecord, rt: JobRuntime) -> OumiJobStatus | None:
    """Fetch the latest status for a job.

    For **local** jobs, checks the subprocess return code directly.
    For **cloud** jobs, tries ``cluster.get_job()`` first, then falls
    back to ``launcher.status()``.

    Returns an ``OumiJobStatus`` for cloud jobs, or ``None`` for local
    jobs (status is derived from ``rt.process`` instead).
    """
    if record.cloud == "local":
        return None

    reg = get_registry()

    if rt.error_message and rt.cluster_obj is None:
        return rt.oumi_status

    def _derive_status(status: OumiJobStatus) -> str:
        if not status.done:
            return "running"
        status_str = (status.status or "").lower()
        if "fail" in status_str:
            return "failed"
        if "cancel" in status_str:
            return "failed"
        return "completed"

    if rt.cluster_obj and record.oumi_job_id:
        try:
            status = await asyncio.to_thread(
                rt.cluster_obj.get_job, record.oumi_job_id
            )
            if status:
                rt.oumi_status = status
                reg.update(
                    record.job_id,
                    oumi_job_id=status.id or record.oumi_job_id,
                    cluster_name=status.cluster or record.cluster_name,
                    status=_derive_status(status),
                )
                return status
        except Exception:
            logger.warning(
                "cluster.get_job failed for %s; falling back to launcher.status",
                record.job_id,
                exc_info=True,
            )

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
                    reg.update(
                        record.job_id,
                        oumi_job_id=s.id or record.oumi_job_id,
                        cluster_name=s.cluster or record.cluster_name,
                        status=_derive_status(s),
                    )
                    return s
    except Exception:
        logger.warning(
            "launcher.status failed for %s; returning stale status",
            record.job_id,
            exc_info=True,
        )

    return rt.oumi_status


async def cancel(
    record: JobRecord, rt: JobRuntime, *, force: bool = False
) -> JobCancelResponse:
    """Cancel a job.

    For **local** jobs, sends SIGTERM (or SIGKILL if *force* is True)
    to the subprocess.  For **cloud** jobs, delegates to
    ``oumi.launcher.cancel()``.

    Returns a dict with ``success`` (bool) and ``message`` or ``error``.
    """
    reg = get_registry()

    if record.status in {"completed", "failed"}:
        return {
            "success": False,
            "error": (
                f"Job {record.job_id} is already finished "
                f"(status: {rt.oumi_status.status if rt.oumi_status else record.status})"
            ),
        }

    if not record.oumi_job_id and rt.process is None:
        rt.cancel_requested = True
        rt.error_message = "Cancellation requested while launch is pending."
        reg.update(record.job_id, status="failed")
        if rt.runner_task and not rt.runner_task.done():
            rt.runner_task.cancel()
        return {
            "success": True,
            "message": (
                f"Cancellation requested for {record.job_id}. "
                "If the cloud launch completes, the MCP will attempt best-effort cancellation."
            ),
        }

    if record.cloud == "local" and rt.process is not None:
        try:
            if force:
                rt.process.kill()  # SIGKILL
                action = "killed (SIGKILL)"
            else:
                rt.process.terminate()  # SIGTERM
                action = "terminated (SIGTERM)"
            rt.cancel_requested = True
            rt.error_message = f"Cancelled by user ({action})"
            reg.update(record.job_id, status="failed")
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

    try:
        result_status = await asyncio.to_thread(
            launcher.cancel,
            record.oumi_job_id,
            record.cloud,
            record.cluster_name,
        )
        rt.cancel_requested = True
        rt.oumi_status = result_status
        reg.update(record.job_id, status="failed")
        return {
            "success": True,
            "message": f"Job {record.job_id} cancel requested on {record.cloud}/{record.cluster_name}.",
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to cancel job {record.job_id}: {exc}",
        }


def get_log_paths(record: JobRecord, rt: JobRuntime) -> dict[str, Path | None]:
    """Return paths to the stdout and stderr log files for a job.

    Returns a dict with ``"stdout"`` and ``"stderr"`` keys, each
    mapping to a ``Path`` or ``None`` if the file doesn't exist yet.
    """
    result: dict[str, Path | None] = {"stdout": None, "stderr": None}
    log_dir = rt.log_dir
    if log_dir is None or not log_dir.is_dir():
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
    rt: JobRuntime,
    done_event: asyncio.Event,
) -> AsyncIterator[str]:
    """Yield log lines from ``cluster.get_logs_stream()`` for cloud jobs.

    Falls back silently (returns without yielding) if the cluster does not
    support log streaming (raises ``NotImplementedError``).
    """
    cluster = rt.cluster_obj
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

    try:
        while not done_event.is_set():
            lines = await asyncio.to_thread(_read_lines)
            for line in lines:
                yield line
            if not lines:
                await asyncio.sleep(LOG_TAIL_INTERVAL_SECONDS)
    finally:
        try:
            stream.close()
        except Exception:
            pass
