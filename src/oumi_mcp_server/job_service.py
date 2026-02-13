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
import logging
import os
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
    log_dir: Path = field(default_factory=lambda: JOB_LOGS_DIR)
    process: subprocess.Popen | None = field(default=None, repr=False)  # type: ignore[type-arg]
    cluster_obj: BaseCluster | None = field(default=None, repr=False)
    submit_time: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    oumi_status: OumiJobStatus | None = field(default=None, repr=False)
    error_message: str | None = None
    runner_task: asyncio.Task[None] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Set log_dir to a per-job subdirectory under JOB_LOGS_DIR."""
        self.log_dir = JOB_LOGS_DIR / self.job_id

    @property
    def is_local(self) -> bool:
        """True if this is a local job (not cloud)."""
        return self.cloud == "local"

    @property
    def is_done(self) -> bool:
        """True if the job has finished (locally or on a cloud cluster)."""
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
        self._lock = asyncio.Lock()

    async def register(self, record: JobRecord) -> None:
        async with self._lock:
            if record.job_id in self._jobs:
                raise ValueError(f"Job ID already exists: {record.job_id}")
            self._jobs[record.job_id] = record
            self._evict_finished_unlocked()

    def _evict_finished_unlocked(self) -> None:
        finished = [r for r in self._jobs.values() if r.is_done]
        if len(finished) <= MAX_COMPLETED_JOBS:
            return
        finished.sort(key=lambda r: r.submit_time)
        for r in finished[: len(finished) - MAX_COMPLETED_JOBS]:
            del self._jobs[r.job_id]
            logger.debug("Evicted finished job %s from registry", r.job_id)

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def all_jobs(self) -> list[JobRecord]:
        async with self._lock:
            return list(self._jobs.values())

    async def running(self) -> list[JobRecord]:
        async with self._lock:
            return [r for r in self._jobs.values() if not r.is_done]

    async def completed(self) -> list[JobRecord]:
        async with self._lock:
            return [r for r in self._jobs.values() if r.is_done]


registry = JobRegistry()


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


def _build_shell_command(config_path: str, command: str) -> str:
    """Build the shell command string for an Oumi CLI invocation."""
    oumi_cmd = _COMMAND_MAP.get(command, f"oumi {command}")
    return f"set -e\n{oumi_cmd} -c {config_path}"


def _build_cloud_job_config(
    config_path: str,
    command: str,
    *,
    cloud: str,
    cluster_name: str | None = None,
    accelerators: str | None = None,
    job_name: str | None = None,
) -> launcher.JobConfig:
    """Build an ``oumi.launcher.JobConfig`` for **cloud** execution.

    For cloud jobs the launcher handles cluster lifecycle via SkyPilot.
    ``working_dir`` is set to a temporary directory since the launcher
    copies it to the remote node -- the actual execution uses the
    ``run`` script which references the config by absolute path.
    """
    run_script = _build_shell_command(config_path, command)

    resources = launcher.JobResources(cloud=cloud)
    if accelerators:
        resources.accelerators = accelerators

    return launcher.JobConfig(
        name=job_name,
        resources=resources,
        working_dir=".",
        run=run_script,
    )


def start_local_job(record: JobRecord) -> None:
    """Start a local job by spawning the Oumi CLI directly.

    Creates the log directory, starts the subprocess via ``Popen``, and
    sets ``record.process`` and ``record.oumi_job_id``. Raises on failure (e.g. command not found, permission denied).

    Bypasses ``oumi.launcher.LocalCluster`` (which requires a
    ``working_dir``) and instead runs ``oumi <cmd> -c <config>``
    via ``subprocess.Popen``.

    Stdout and stderr are written to files in ``record.log_dir`` so
    that ``tail_log_file()`` can stream them to the MCP client.
    """
    run_script = _build_shell_command(record.config_path, record.command)

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
            run_script,
            shell=True,
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
    record._stdout_f = stdout_f  # type: ignore[attr-defined]
    record._stderr_f = stderr_f  # type: ignore[attr-defined]
    logger.info(
        "Local job %s started (pid=%s): %s",
        record.job_id,
        proc.pid,
        run_script.replace("\n", " && "),
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

    stdout_f = getattr(record, "_stdout_f", None)
    stderr_f = getattr(record, "_stderr_f", None)
    stderr_path = None
    if stderr_f is not None:
        try:
            stderr_path = stderr_f.name
        except Exception:
            pass

    try:
        returncode = await asyncio.to_thread(proc.wait)

        if returncode != 0:
            record.error_message = f"Process exited with code {returncode}." + (
                f" See stderr: {stderr_path}" if stderr_path else ""
            )
            logger.warning(
                "Local job %s exited with code %d", record.job_id, returncode
            )
        else:
            logger.info("Local job %s completed successfully", record.job_id)
    except Exception as exc:
        record.error_message = str(exc)
        logger.exception("Failed to run local job %s", record.job_id)
    finally:
        if stdout_f is not None:
            stdout_f.close()
        if stderr_f is not None:
            stderr_f.close()


async def _launch_cloud(
    record: JobRecord,
    *,
    accelerators: str | None = None,
) -> None:
    """Launch a cloud job via ``oumi.launcher.up()``.

    Updates *record* in-place with the cluster object, oumi job ID, and
    initial status.  On failure, sets ``record.error_message``.
    """
    job_config = _build_cloud_job_config(
        record.config_path,
        record.command,
        cloud=record.cloud,
        cluster_name=record.cluster_name or None,
        accelerators=accelerators,
        job_name=record.job_id,
    )

    try:
        cluster, status = await asyncio.to_thread(
            launcher.up,
            job_config,
            record.cluster_name or None,
        )
        record.cluster_obj = cluster
        record.oumi_job_id = status.id if status else ""
        record.oumi_status = status
        record.cluster_name = status.cluster if status else record.cluster_name
        logger.info(
            "Cloud job %s launched on %s (oumi_id=%s)",
            record.job_id,
            record.cloud,
            record.oumi_job_id,
        )
    except Exception as exc:
        record.error_message = str(exc)
        logger.exception("Failed to launch cloud job %s", record.job_id)


async def launch_job(
    record: JobRecord,
    *,
    accelerators: str | None = None,
) -> None:
    """Launch a job -- local or cloud -- in a background thread.

    For local jobs, spawns the Oumi CLI directly via subprocess.
    For cloud jobs, delegates to ``oumi.launcher.up()``.
    """
    if record.is_local:
        start_local_job(record)
        await wait_local_completion(record)
    else:
        await _launch_cloud(record, accelerators=accelerators)


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

    if record.cluster_obj and record.oumi_job_id:
        try:
            status = await asyncio.to_thread(
                record.cluster_obj.get_job, record.oumi_job_id
            )
            if status:
                record.oumi_status = status
                return status
        except Exception:
            logger.debug(
                "cluster.get_job failed for %s, falling back to launcher.status",
                record.job_id,
            )

    try:
        all_statuses = await asyncio.to_thread(
            launcher.status,
            cloud=record.cloud,
            cluster=record.cluster_name or None,
            id=record.oumi_job_id or None,
        )
        for _cloud, jobs in all_statuses.items():
            for s in jobs:
                if s.id == record.oumi_job_id:
                    record.oumi_status = s
                    return s
    except Exception:
        logger.debug("launcher.status failed for %s", record.job_id)

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
        if record.runner_task and not record.runner_task.done():
            record.runner_task.cancel()
        record.error_message = "Cancelled before launch"
        return {
            "success": True,
            "message": f"Job {record.job_id} cancelled before it was launched.",
        }

    if record.is_local and record.process is not None:
        try:
            if force:
                record.process.kill()  # SIGKILL
                action = "killed (SIGKILL)"
            else:
                record.process.terminate()  # SIGTERM
                action = "terminated (SIGTERM)"
            record.error_message = f"Cancelled by user ({action})"
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
        record.oumi_status = result_status
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
