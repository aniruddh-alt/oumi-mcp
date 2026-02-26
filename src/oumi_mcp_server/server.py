"""Oumi MCP Server - ML Training Config Discovery and Execution.

~500 ready-to-use YAML configs for fine-tuning LLMs (Llama, Qwen, Phi, etc.).
Local execution via subprocess, cloud execution via oumi.launcher.

IMPORTANT — ALWAYS call get_started() FIRST before using any other tool.
get_started() returns the full tool catalog, resource list, and recommended
workflow. Without it you will miss critical path-resolution rules and the
correct order of operations.

Path rules:
- All path-sensitive tools require client_cwd (the user's project root).
- Config file path: absolute OR relative to client_cwd.
- Local jobs: subprocess runs from client_cwd; paths inside YAML resolve there.
- Cloud jobs: client_cwd becomes working_dir on remote VM;
  use repo-relative paths inside YAML. NEVER use local machine paths.
"""

import asyncio
import dataclasses
import json
import logging
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, Iterable, cast
from zipfile import ZipFile

import httpx
import oumi.launcher as launcher
import yaml
from fastmcp import FastMCP
from huggingface_hub import auth_check, whoami
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    RepositoryNotFoundError,
)
from oumi.core.configs import (
    AnalyzeConfig,
    AsyncEvaluationConfig,
    EvaluationConfig,
    InferenceConfig,
    JobConfig,
    JudgeConfig,
    QuantizationConfig,
    SynthesisConfig,
    TrainingConfig,
    TuningConfig,
)
from packaging.version import Version

from oumi_mcp_server.config_service import (
    clear_config_caches,
    extract_key_settings,
    find_config_match,
    get_all_configs,
    get_bundled_configs_dir,
    get_cache_dir,
    get_categories,
    get_configs_dir,
    parse_yaml,
)
from oumi_mcp_server.config_service import (
    search_configs as search_configs_service,
)
from oumi_mcp_server.constants import (
    BUNDLED_OUMI_VERSION,
    CONFIG_SYNC_TIMEOUT_SECONDS,
    CONFIGS_SYNC_INTERVAL_HOURS,
    CONFIGS_SYNC_MARKER,
    CONFIGS_VERSION_MARKER,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_STREAM_LINES,
    GITHUB_CONFIGS_ZIP_URL,
    GITHUB_REPO_URL,
    GITHUB_ZIP_PREFIX,
    HARDWARE_PACKAGES,
    HF_API_TIMEOUT_SECONDS,
    JOB_LOGS_DIR,
    JOB_RUNS_DIR,
    MIN_CC_BF16,
    MIN_CC_FLASH_ATTN,
    MIN_TORCH_VERSION_COMPILE,
    MIN_TORCH_VERSION_SDPA,
    VALID_OUMI_COMMANDS,
    ValidatorTaskType,
)
from oumi_mcp_server.docs_service import (
    get_module_list,
    search_docs,
    start_background_indexing,
)
from oumi_mcp_server.job_service import (
    JobRecord,
    JobRuntime,
    _generate_job_config_template,
    _get_cloud_logs,
    _is_job_config,
    _parse_gpu_count,
    cancel,
    get_log_paths,
    get_registry,
    get_runtime,
    launch_job,
    make_job_id,
    poll_status,
    start_local_job,
    wait_local_completion,
)
from oumi_mcp_server.models import (
    CategoriesResponse,
    CloudReadiness,
    ClusterLifecycleResponse,
    ConfigDetail,
    ConfigMetadata,
    DocsSearchResponse,
    HardwareInfo,
    JobCancelResponse,
    JobLogsResponse,
    JobStatusResponse,
    JobSubmissionResponse,
    JobSummary,
    ListModulesResponse,
    PreFlightCheckResponse,
    ValidateConfigResponse,
)
from oumi_mcp_server.prompts.mle_prompt import (
    ANALYZE_COMMAND_RESOURCE,
    CLOUD_LAUNCH_RESOURCE,
    EVAL_COMMAND_RESOURCE,
    INFER_COMMAND_RESOURCE,
    MLE_WORKFLOW_RESOURCE,
    POST_TRAINING_RESOURCE,
    SYNTH_COMMAND_RESOURCE,
    TRAIN_COMMAND_RESOURCE,
)

try:
    import torch as _torch
except Exception:
    _torch = None
torch = _torch

try:
    import sky as _sky
    from sky import check as _sky_check
    from sky.clouds.cloud import CloudCapability as _CloudCapability
except Exception:
    _sky = None
    _sky_check = None
    _CloudCapability = None
sky = _sky
sky_check = _sky_check
CloudCapability = _CloudCapability

_CLOUD_ENV_VAR_HINTS: dict[str, str] = {
    "WANDB_API_KEY": "Weights & Biases logging",
    "WANDB_PROJECT": "Weights & Biases project name",
    "HF_TOKEN": "HuggingFace token (alternative to ~/.cache/huggingface/token)",
    "COMET_API_KEY": "Comet ML logging",
}


def _build_missing_env_warning(envs: dict[str, str] | None) -> str:
    """Return a warning string listing local env vars that won't reach the remote VM."""
    missing = []
    for var, description in _CLOUD_ENV_VAR_HINTS.items():
        if os.environ.get(var) and (not envs or var not in envs):
            missing.append(f"  - {var} ({description})")
    if not missing:
        return ""
    return (
        "\n\nWARNING: These env vars exist locally but won't be set on the remote VM:\n"
        + "\n".join(missing)
        + '\n  Pass them via the `envs` parameter: envs={"WANDB_API_KEY": "..."}'
    )


def get_package_version(package_name: str) -> str | None:
    """Return the installed version string for *package_name*, or None."""
    try:
        return _pkg_version(package_name)
    except PackageNotFoundError:
        return None


def get_oumi_version() -> str:
    """Return the installed oumi version, or "unknown"."""
    return get_package_version("oumi") or "unknown"


def is_oumi_dev_build(version: str) -> bool:
    """Return True if *version* looks like a setuptools_scm dev build."""
    return ".dev" in version or "+" in version


def get_oumi_git_tag() -> str | None:
    """Map the installed oumi version to the corresponding Git tag.

    Dev builds (e.g. ``0.8.dev35+ge2b81b3fe``) have no matching tag.
    Release versions (e.g. ``0.7``) map to ``v0.7``.
    """
    version = get_package_version("oumi")
    if not version or is_oumi_dev_build(version):
        return None
    return f"v{version}"


def get_configs_zip_url(tag: str | None = None) -> tuple[str, str]:
    """Return ``(zip_url, zip_prefix)`` for downloading configs.

    If *tag* is provided, returns the tagged archive URL; otherwise falls back
    to the main branch.
    """
    if tag:
        url = f"{GITHUB_REPO_URL}/archive/refs/tags/{tag}.zip"
        prefix = f"oumi-{tag.lstrip('v')}/configs/"
        return url, prefix
    return GITHUB_CONFIGS_ZIP_URL, GITHUB_ZIP_PREFIX


def get_gpu_info() -> dict[str, Any]:
    """Return a dict describing available GPU/accelerator hardware."""
    info: dict[str, Any] = {
        "accelerator_type": "none",
        "accelerator_count": 0,
        "accelerators": [],
        "gpu_name": None,
        "gpu_memory_bytes": None,
    }
    try:
        if torch is not None and torch.cuda.is_available():
            info["accelerator_type"] = "cuda"
            count = torch.cuda.device_count()
            info["accelerator_count"] = count
            if count > 0:
                props = torch.cuda.get_device_properties(0)
                info["gpu_name"] = props.name
                info["gpu_memory_bytes"] = props.total_mem
                info["accelerators"] = [
                    {
                        "name": props.name,
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                ]
        elif (
            torch is not None
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            info["accelerator_type"] = "mps"
            info["accelerator_count"] = 1
    except Exception:
        pass
    return info


logger = logging.getLogger(__name__)

mcp = FastMCP(
    "Oumi Config Server",
    instructions=(
        "IMPORTANT: Always call get_started() FIRST before using any other tool. "
        "It returns the full tool catalog, path rules, and recommended workflow."
    ),
)


def _configure_logging() -> None:
    """Reduce noisy third-party INFO logs on stderr in MCP clients."""
    logger.setLevel(logging.INFO)
    for noisy_logger in (
        "mcp.server.lowlevel.server",
        "mcp.server.lowlevel",
        "mcp.shared.session",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


_OUMI_ENV_OVERRIDES = ("OUMI_USE_SPOT_VM", "OUMI_FORCE_EDITABLE_INSTALL")


def _strip_oumi_env_overrides() -> None:
    """Remove oumi env vars that silently override launcher config values.

    These are CLI convenience toggles (e.g. "always use spot") that make
    sense for interactive ``oumi launch up`` but break programmatic callers
    like this MCP server — the tool's explicit parameters should be the
    sole source of truth.
    """
    for var in _OUMI_ENV_OVERRIDES:
        val = os.environ.pop(var, None)
        if val:
            logger.info("Stripped inherited env var %s=%r from MCP process", var, val)


def _resolve_path(raw: str, client_cwd: Path) -> Path:
    """Resolve a path string against the client's working directory.

    Absolute paths (after ``~`` expansion) are returned as-is.
    Relative paths are resolved against *client_cwd*.
    """
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (client_cwd / expanded).resolve()


def _resolve_config_path(config: str, client_cwd: str) -> tuple[Path, str | None]:
    """Resolve and validate a config file path against the client's CWD.

    Requires *client_cwd* to be absolute. Relative *config* paths are
    resolved against it. Returns ``(resolved_path, None)`` on success,
    or ``(Path(), error_message)`` on failure.
    """
    cwd = Path(client_cwd).expanduser()
    if not cwd.is_absolute():
        return Path(), (
            f"client_cwd must be absolute, got: '{client_cwd}'. "
            "Provide the full path to your project directory."
        )
    p = _resolve_path(config, cwd)
    if not p.exists():
        return Path(), f"Config file not found: {p}"
    if not p.is_file():
        return Path(), f"Config path is not a file: {p}"
    return p, None


def _load_yaml_strict(config_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load YAML config and return a user-facing error when invalid."""
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        return None, f"Invalid YAML config: {exc}"
    if cfg is None:
        return None, "Config file is empty."
    if not isinstance(cfg, dict):
        return None, "Config root must be a mapping/object."
    return cfg, None


def _extract_job_metadata_from_cfg(cfg: dict[str, Any]) -> tuple[str, str]:
    """Extract model name and output dir from parsed config."""
    model_name = (
        (cfg.get("model") or {}).get("model_name", "unknown")
        if isinstance(cfg.get("model"), dict)
        else "unknown"
    )
    if not model_name:
        model_name = "unknown"
    raw_training = cfg.get("training")
    training = raw_training if isinstance(raw_training, dict) else {}
    output_dir = training.get("output_dir") or cfg.get("output_dir") or "./output"
    return str(model_name), str(output_dir)


@mcp.resource("guidance://mle-workflow")
async def get_mle_workflow_guidance() -> str:
    """ML engineering workflow guidance for Oumi.

    This resource provides a full ML workflow and tool usage guidance for
    training LLMs with Oumi. Cursor may choose to fetch this resource and
    include it as context when working with Oumi MCP tools.
    """
    return MLE_WORKFLOW_RESOURCE


@mcp.resource("guidance://mle-train")
async def get_train_command_guidance() -> str:
    """MLE guidance for oumi train."""
    return TRAIN_COMMAND_RESOURCE


@mcp.resource("guidance://mle-synth")
async def get_synth_command_guidance() -> str:
    """MLE guidance for oumi synth."""
    return SYNTH_COMMAND_RESOURCE


@mcp.resource("guidance://mle-analyze")
async def get_analyze_command_guidance() -> str:
    """MLE guidance for oumi analyze."""
    return ANALYZE_COMMAND_RESOURCE


@mcp.resource("guidance://mle-eval")
async def get_eval_command_guidance() -> str:
    """MLE guidance for oumi evaluate/eval."""
    return EVAL_COMMAND_RESOURCE


@mcp.resource("guidance://mle-infer")
async def get_infer_command_guidance() -> str:
    """MLE guidance for oumi infer."""
    return INFER_COMMAND_RESOURCE


@mcp.resource("guidance://cloud-launch")
async def get_cloud_launch_guidance() -> str:
    """Cloud job launch guidance — job config anatomy, setup patterns, examples.

    Read this resource when planning a cloud training run. Explains what a job
    config is, the key fields to customize, common setup patterns (dataset
    downloads, extra packages), and how ``run_oumi_job`` works with both
    training configs and job configs.
    """
    return CLOUD_LAUNCH_RESOURCE


@mcp.resource("guidance://post-training")
async def get_post_training_guidance() -> str:
    """Post-training guidance — downloading weights, evaluation, teardown, merging.

    Read this resource after a cloud training job succeeds. Covers the full
    post-training lifecycle: downloading model weights via SkyPilot CLI,
    running evaluation on the live cluster, tearing down to stop billing,
    merging LoRA adapters locally, and pushing to HuggingFace Hub.
    """
    return POST_TRAINING_RESOURCE


@mcp.tool()
def search_configs(
    query: str = "",
    task: str = "",
    model: str = "",
    keyword: str | list[str] = "",
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[ConfigMetadata]:
    """Search the Oumi config library (~500 configs for LLM fine-tuning).

    All filters are case-insensitive substring matches. Combine to narrow.

    Args:
        query: Path substring — size ("8b"), variant ("instruct"), or
            technique ("lora"). Space-separated words use AND logic.
        task: Task type: sft, dpo, kto, grpo, eval, infer, pretrain.
        model: Model family: llama3_1, qwen3, phi4, gemma3, deepseek_r1, etc.
        keyword: Content substring match. List = AND logic.
        limit: Max results (default 20).
    """
    configs = get_all_configs()
    return search_configs_service(configs, query, task, model, keyword, limit)


@mcp.tool()
def get_config(path: str, include_content: bool = False) -> ConfigDetail:
    """Get details about a specific Oumi config file.

    Use the returned config as a REFERENCE to understand structure, field names,
    and reasonable defaults — do NOT copy it verbatim. Build the user's config
    from scratch, adapting only the relevant settings (model, dataset, training
    params, PEFT) to match their specific requirements, hardware, and data.

    Args:
        path: Config path from search_configs(), or a partial path
            (e.g. "llama3_1/sft/8b_lora" will match).
        include_content: Include full YAML content (default False).
    """
    configs = get_all_configs()
    match = find_config_match(path, configs)

    if match is None:
        return {
            "path": "",
            "description": "",
            "model_name": "",
            "task_type": "",
            "datasets": [],
            "reward_functions": [],
            "peft_type": "",
            "key_settings": {},
            "content": "",
            "error": f"Config not found: {path}",
        }

    configs_dir = get_configs_dir()
    config_path = configs_dir / match["path"]
    config = parse_yaml(str(config_path))

    content = ""
    if include_content:
        try:
            content = config_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read config content: {e}")
            content = f"Error reading file: {e}"

    return {
        "path": match["path"],
        "description": match["description"],
        "model_name": match["model_name"],
        "task_type": match["task_type"],
        "datasets": match["datasets"] or [],
        "reward_functions": match["reward_functions"] or [],
        "peft_type": match["peft_type"] or "",
        "key_settings": extract_key_settings(config),
        "content": content,
        "error": "",
    }


def _build_version_warning() -> str:
    """Return a warning string if configs may not match the installed oumi."""
    source = get_configs_source()
    oumi_ver = get_oumi_version()

    if oumi_ver == "unknown":
        return ""

    if source.startswith("cache:main") and not is_oumi_dev_build(oumi_ver):
        return (
            f"Configs were synced from the main branch but oumi {oumi_ver} "
            "is a release build. Config fields may not match the installed "
            "library. Run config_sync(force=True) after upgrading oumi."
        )

    if source.startswith("bundled:"):
        bundled_ver = source.split(":", 1)[1]
        if bundled_ver != oumi_ver and not is_oumi_dev_build(oumi_ver):
            return (
                f"Using bundled configs from oumi {bundled_ver} but oumi "
                f"{oumi_ver} is installed. Some configs may reference fields "
                "not present in (or removed from) the installed library."
            )

    return ""


@mcp.tool()
def list_categories() -> CategoriesResponse:
    """List available config categories, model families, and API providers."""
    configs_dir = get_configs_dir()
    configs = get_all_configs()
    return get_categories(
        configs_dir,
        len(configs),
        oumi_version=get_oumi_version(),
        configs_source=get_configs_source(),
        version_warning=_build_version_warning(),
    )


TASK_MAPPING = {
    "analyze": AnalyzeConfig,
    "async_evaluation": AsyncEvaluationConfig,
    "evaluation": EvaluationConfig,
    "inference": InferenceConfig,
    "job": JobConfig,
    "judge": JudgeConfig,
    "quantization": QuantizationConfig,
    "synthesis": SynthesisConfig,
    "training": TrainingConfig,
    "tuning": TuningConfig,
}


@mcp.tool()
def pre_flight_check(
    config: str, client_cwd: str, cloud: str = ""
) -> PreFlightCheckResponse:
    """Run pre-flight checks to catch issues before launching.

    Validates: HF auth & gated repo access, hardware/packages, local paths,
    and cloud credentials (with actual API calls, not just file checks).

    When ``blocking=True`` in the response, there are hard blockers that
    WILL prevent the run from succeeding — surface these as showstoppers.

    Args:
        config: Absolute path, or relative to client_cwd, to the YAML config file.
        client_cwd: REQUIRED. Absolute path to the client's working directory
            (project root). Resolves relative config paths and sets the execution
            context for local and cloud jobs.
        cloud: Target cloud provider (e.g. "gcp", "aws"). Validates
            credentials and returns ``suggested_configs`` for that cloud.
            Leave empty for local runs.
    """
    return _pre_flight_check(config, client_cwd=client_cwd, cloud=cloud)


def _pre_flight_check(
    config: str, client_cwd: str, cloud: str = ""
) -> PreFlightCheckResponse:
    """Run pre-flight checks (internal implementation)."""
    errors: list[str] = []
    warnings: list[str] = []
    repo_access: dict[str, str] = {}

    config_path, path_error = _resolve_config_path(config, client_cwd)
    if path_error:
        errors.append(path_error)
        return {
            "blocking": True,
            "summary": f"BLOCKED: {path_error}",
            "hf_authenticated": False,
            "repo_access": {},
            "hardware": _empty_hardware(),
            "cloud_readiness": _empty_cloud_readiness(),
            "errors": errors,
            "warnings": [],
            "paths": {},
        }

    cfg, load_error = _load_yaml_strict(config_path)
    if load_error:
        errors.append(load_error)
        return {
            "blocking": True,
            "summary": f"BLOCKED: {load_error}",
            "hf_authenticated": False,
            "repo_access": {},
            "hardware": _empty_hardware(),
            "cloud_readiness": _empty_cloud_readiness(),
            "errors": errors,
            "warnings": [],
            "paths": {},
        }
    assert cfg is not None
    hf_authenticated = False
    hf_token: bool | None = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            future = executor.submit(whoami)
            future.result(timeout=HF_API_TIMEOUT_SECONDS)
            hf_authenticated = True
            hf_token = True
        except HFValidationError:
            errors.append("Invalid HF token")
        except TimeoutError:
            warnings.append(f"HF auth check timed out after {HF_API_TIMEOUT_SECONDS}s")
            hf_token = None
        except Exception as e:
            warnings.append(f"HF auth check failed (may be transient): {e}")
            hf_token = None

    for repo_id, repo_types in get_repos(cfg).items():
        if repo_id in repo_access:
            continue
        for repo_type in repo_types:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        auth_check, repo_id, repo_type=repo_type, token=hf_token
                    )
                    future.result(timeout=HF_API_TIMEOUT_SECONDS)
                repo_access[repo_id] = "ok"
            except GatedRepoError:
                repo_access[repo_id] = "gated"
                errors.append(f"Gated {repo_type} requires access grant: {repo_id}")
            except RepositoryNotFoundError:
                repo_access[repo_id] = "not_found"
                errors.append(f"{repo_type.title()} not found: {repo_id}")
            except TimeoutError:
                repo_access[repo_id] = "error"
                warnings.append(
                    f"Timed out checking {repo_id} after {HF_API_TIMEOUT_SECONDS}s"
                )
            except HfHubHTTPError as e:
                repo_access[repo_id] = "error"
                errors.append(f"HF Hub error for {repo_id}: {str(e)}")
            except Exception as e:
                repo_access[repo_id] = "error"
                errors.append(f"Error checking {repo_id}: {str(e)}")
            break

    hw_errors, hw_warnings, hardware = check_hardware(cfg)
    errors.extend(hw_errors)
    warnings.extend(hw_warnings)

    target_cloud = cloud if cloud and cloud != "local" else ""
    cloud_errors, cloud_warnings, cloud_readiness = check_cloud_readiness(
        target_cloud=target_cloud,
    )
    errors.extend(cloud_errors)
    warnings.extend(cloud_warnings)

    dataset_checks = validate_datasets(cfg)
    for ds_key, ds_status in dataset_checks.items():
        if ds_status == "not_found":
            errors.append(
                f"Dataset '{ds_key}' is not a registered Oumi dataset, not found on "
                "HuggingFace Hub, and no local dataset_path provided. Use a full HF ID "
                "(e.g., 'yahma/alpaca-cleaned'), a registered name (e.g., "
                "'text_sft_jsonl'), or set dataset_path."
            )
        elif ds_status == "warning_timeout":
            warnings.append(f"HF Hub probe for dataset '{ds_key}' timed out (5s)")

    env_warnings = _check_env_overrides(target_cloud)
    warnings.extend(env_warnings)

    path_results = validate_paths(cfg, Path(client_cwd), cloud=target_cloud)
    for path_key, path_status in path_results.items():
        if path_status == "local_machine_path_error":
            errors.append(
                f"Local machine path '{path_key}' will not exist on the remote VM. "
                "Use a repo-relative path (e.g., 'data/...') that resolves from "
                "your working_dir."
            )

    if target_cloud:
        skyignore_warnings = _check_skyignore(config_path.parent, path_results)
        warnings.extend(skyignore_warnings)

    cloud_file_checks: dict[str, str] = {}
    if target_cloud:
        cloud_file_checks = _check_cloud_files(cfg, config_path, target_cloud)
        for path_key, check_status in cloud_file_checks.items():
            if check_status == "missing_local_source":
                errors.append(
                    f"file_mounts source '{path_key}' does not exist locally. "
                    "The file won't be copied to the remote VM."
                )
            elif check_status == "not_reachable_on_vm":
                errors.append(
                    f"Path '{path_key}' has no delivery mechanism to the remote VM. "
                    "Use a job config with file_mounts, add it to working_dir, "
                    "or download it in setup_script."
                )
            elif check_status == "working_dir_suspicious":
                warnings.append(
                    f"'{path_key}' does not exist locally. Use 'working_dir: .' "
                    "(resolved to client_cwd at launch) or verify the path."
                )
            elif check_status == "unverifiable_remote":
                warnings.append(
                    f"Remote path '{path_key}' can't be validated locally. "
                    "Ensure it exists on the VM via setup_script or storage_mounts."
                )

    is_blocking = len(errors) > 0
    if is_blocking:
        summary = (
            f"BLOCKED: {len(errors)} issue(s) must be resolved before running. "
            f"First: {errors[0]}"
        )
    elif warnings:
        summary = (
            f"Ready with {len(warnings)} warning(s) (may be fine for remote clusters)"
        )
    else:
        summary = "Ready: all checks passed"

    result: PreFlightCheckResponse = {
        "blocking": is_blocking,
        "summary": summary,
        "hf_authenticated": hf_authenticated,
        "repo_access": repo_access,
        "hardware": hardware,
        "cloud_readiness": cloud_readiness,
        "errors": errors,
        "warnings": warnings,
        "paths": path_results,
    }

    if dataset_checks:
        result["dataset_checks"] = dataset_checks
    if env_warnings:
        result["env_warnings"] = env_warnings
    if cloud_file_checks:
        result["cloud_file_checks"] = cloud_file_checks

    if target_cloud:
        all_cfgs = get_all_configs()
        suggested = search_configs_service(all_cfgs, query=target_cloud, limit=5)
        result["suggested_configs"] = [c["path"] for c in suggested]

    return result


def _looks_like_hf_repo(val: str) -> bool:
    """Return True if *val* looks like an HF repo ID (org/name)."""
    return bool(val) and val.count("/") == 1 and not val.startswith(("/", ".", "~"))


def _is_local_machine_path(path_str: str) -> bool:
    """Return True if *path_str* is a local machine absolute path.

    Detects paths rooted at /Users/, /home/<local-user>/, or matching
    Path.home(). Remote absolute paths (e.g. /home/ubuntu/...) are NOT
    considered local.
    """
    p = Path(path_str)
    if not p.is_absolute():
        return False
    home = Path.home()
    if p == home or str(p).startswith(str(home) + "/"):
        return True
    if path_str.startswith("/Users/"):
        return True
    return False


def validate_paths(cfg: dict, base_dir: Path, *, cloud: str = "") -> dict[str, str]:
    """Extract all local paths from config and validate they exist.

    For **local** jobs (``cloud == ""``): resolves relative paths against
    *base_dir* and reports ``"ok"`` or ``"not_found"``.

    For **cloud** jobs (``cloud != ""``): applies cloud path sanitization —
    blocks local machine paths, accepts repo-relative and remote absolute paths.
    Return values: ``"ok"``, ``"ok_remote"``, ``"not_found_warning"``,
    ``"local_machine_path_error"``.
    """
    is_cloud = bool(cloud)
    paths: dict[str, str] = {}

    def _extract(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if isinstance(val, str) and key.endswith(
                    ("_dir", "_path", "_file", "_folder")
                ):
                    if _looks_like_hf_repo(val):
                        continue
                    if is_cloud:
                        _check_cloud_path(val)
                    else:
                        _check_local_path(val)
                else:
                    _extract(val)
        elif isinstance(obj, list):
            for item in obj:
                _extract(item)

    def _check_local_path(val: str) -> None:
        p = Path(val).expanduser()
        if not p.is_absolute():
            p = base_dir / p
            paths[f"{val} (resolved to {p})"] = "ok" if p.exists() else "not_found"
        else:
            paths[val] = "ok" if p.exists() else "not_found"

    def _check_cloud_path(val: str) -> None:
        if _is_local_machine_path(val):
            paths[val] = "local_machine_path_error"
        elif Path(val).is_absolute():
            # Remote absolute (e.g. /home/ubuntu/output/) — can't validate
            paths[val] = "ok_remote"
        else:
            # Relative — check existence under project root (will be synced)
            resolved = base_dir / val
            paths[val] = "ok" if resolved.exists() else "not_found_warning"

    _extract(cfg)
    return paths


def validate_datasets(cfg: dict) -> dict[str, str]:
    """Validate dataset accessibility for each dataset in the config.

    Mirrors Oumi's dataset resolution chain:
    1. REGISTRY.get_dataset(name) → found in registry
    2. dataset_path set → check local existence
    3. datasets.load_dataset_builder(name) → HF Hub metadata probe (no download)

    Returns a dict mapping dataset identifiers to status strings:
    ``"ok_registry"``, ``"ok_local"``, ``"ok_hub"``, ``"not_found"``,
    ``"warning_timeout"``.
    """
    data = cfg.get("data") or {}
    results: dict[str, str] = {}

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


def _check_env_overrides(cloud: str) -> list[str]:
    """Check for environment variables that silently override config behavior.

    Only relevant for cloud jobs. ``OUMI_USE_SPOT_VM`` and
    ``OUMI_FORCE_EDITABLE_INSTALL`` are stripped at startup by
    ``_strip_oumi_env_overrides()`` so they won't appear here.
    """
    if not cloud or cloud == "local":
        return []

    warnings: list[str] = []
    _ENV_WARNINGS: dict[str, str] = {}
    for var, msg in _ENV_WARNINGS.items():
        val = os.environ.get(var)
        if val:
            warnings.append(f"Env var {var}={val!r} is set: {msg}")

    return warnings


def _check_skyignore(config_dir: Path, path_results: dict[str, str]) -> list[str]:
    """Check if a .skyignore file might exclude files needed by the config.

    Walks up from *config_dir* looking for ``.skyignore``. If found, parses
    its patterns and warns if any config paths appear to match.
    """
    warnings: list[str] = []

    skyignore_path: Path | None = None
    search = config_dir.resolve()
    for _ in range(10):  # limit depth
        candidate = search / ".skyignore"
        if candidate.is_file():
            skyignore_path = candidate
            break
        parent = search.parent
        if parent == search:
            break
        search = parent

    if skyignore_path is None:
        return warnings

    try:
        patterns = [
            line.strip()
            for line in skyignore_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    except OSError:
        return warnings

    if not patterns:
        return warnings

    warnings.append(
        f"Found .skyignore at {skyignore_path} — verify it doesn't exclude "
        "files needed on the remote VM."
    )

    for path_key in path_results:
        raw_path = path_key.split(" (resolved to")[0]
        for pattern in patterns:
            bare = pattern.rstrip("/")
            if raw_path == bare or raw_path.startswith(bare + "/"):
                warnings.append(
                    f"Config path '{raw_path}' may be excluded by "
                    f".skyignore pattern '{pattern}'"
                )
                break

    return warnings


def _check_cloud_files(cfg: dict, config_path: Path, cloud: str) -> dict[str, str]:
    """Validate that files referenced in config will be available on the remote VM.

    For **job-config passthrough** (has ``resources``/``setup``/``run`` keys):
    validates ``file_mounts`` local sources exist and ``working_dir`` resolves
    correctly.

    For **training-config wrapping**: scans ``_dir``/``_path``/``_file`` values
    and flags paths that have no delivery mechanism to the VM.

    Returns a dict mapping paths to status strings.
    """
    if not cloud or cloud == "local":
        return {}

    results: dict[str, str] = {}
    job_keys = {"resources", "setup", "run"}
    is_job_cfg = bool(job_keys.intersection(cfg.keys()))

    if is_job_cfg:
        _check_job_config_files(cfg, config_path, results)
    else:
        _check_training_config_files(cfg, results)

    return results


def _check_job_config_files(
    cfg: dict, config_path: Path, results: dict[str, str]
) -> None:
    """Validate file_mounts sources and working_dir for job-config passthrough."""
    file_mounts = cfg.get("file_mounts") or {}
    for remote_dest, local_src in file_mounts.items():
        if not isinstance(local_src, str):
            continue
        expanded = Path(local_src).expanduser()
        if expanded.exists():
            results[local_src] = "ok"
        else:
            results[local_src] = "missing_local_source"

    working_dir = cfg.get("working_dir")
    if working_dir is not None:
        wd_str = str(working_dir)
        # working_dir: . is the correct portable default — client_cwd resolves it
        # at launch time. Only flag truly broken values (nonexistent absolute paths).
        if wd_str != ".":
            wd_path = Path(wd_str).expanduser()
            if not wd_path.is_absolute():
                wd_path = config_path.parent / wd_path
            if not wd_path.exists():
                results[f"working_dir: {wd_str}"] = "working_dir_suspicious"


def _check_training_config_files(cfg: dict, results: dict[str, str]) -> None:
    """Flag training config paths that won't be delivered to the VM.

    In wrapping mode, only config.yaml is staged. Any relative or local-absolute
    path referencing data files will not exist on the remote VM.
    """

    def _extract_paths(obj: object) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if isinstance(val, str) and key.endswith(
                    ("_dir", "_path", "_file", "_folder")
                ):
                    _classify(val)
                else:
                    _extract_paths(val)
        elif isinstance(obj, list):
            for item in obj:
                _extract_paths(item)

    def _classify(val: str) -> None:
        if not val or val.isspace():
            return
        if _looks_like_hf_repo(val) and "." not in val.split("/")[-1]:
            return

        p = Path(val)
        if p.is_absolute():
            if _is_local_machine_path(val):
                results[val] = "not_reachable_on_vm"
            else:
                results[val] = "unverifiable_remote"
        else:
            results[val] = "not_reachable_on_vm"

    _extract_paths(cfg)


def get_repos(cfg: dict) -> dict[str, set[str]]:
    """Extract all HF repo IDs with their repo types from a parsed config."""
    repos: dict[str, set[str]] = {}

    def add(repo_id: str, repo_type: str) -> None:
        if _looks_like_hf_repo(repo_id):
            repos.setdefault(repo_id, set()).add(repo_type)

    model = cfg.get("model") or {}
    add(model.get("model_name", ""), "model")
    add(model.get("tokenizer_name", ""), "model")

    data = cfg.get("data") or {}
    for split in ("train", "eval", "validation", "test"):
        split_cfg = data.get(split) or {}
        for ds in split_cfg.get("datasets") or []:
            add(ds.get("dataset_name", ""), "dataset")
            ds_kwargs = ds.get("dataset_kwargs") or {}
            add(ds_kwargs.get("hf_dataset_path", "") or "", "dataset")

    training = cfg.get("training") or {}
    add(training.get("teacher_model_name_or_path", ""), "model")

    gold = training.get("gold") or {}
    add(gold.get("teacher_model_name_or_path", ""), "model")

    return repos


def _empty_hardware() -> HardwareInfo:
    """Return a default HardwareInfo with no accelerator detected."""
    return {
        "accelerator_type": "none",
        "accelerator_count": 0,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "compute_capability": None,
        "cuda_version": None,
        "packages": {},
    }


def _empty_cloud_readiness() -> CloudReadiness:
    """Return a default CloudReadiness with nothing checked."""
    return {
        "sky_installed": False,
        "enabled_clouds": [],
        "target_cloud_ready": None,
        "target_cloud": "",
    }


def _skypilot_version_label() -> str:
    """Return a short, user-facing SkyPilot version label."""
    version = get_package_version("skypilot") or get_package_version("sky")
    return f"skypilot={version}" if version else "skypilot=unknown"


def _compat_warning(message: str) -> str:
    """Format a version-aware SkyPilot compatibility warning."""
    return f"SkyPilot API compatibility issue ({_skypilot_version_label()}): {message}"


def _compat_error(message: str) -> str:
    """Format a version-aware SkyPilot compatibility error."""
    return f"SkyPilot API compatibility error ({_skypilot_version_label()}): {message}"


def _cloud_names(values: list[Any]) -> list[str]:
    """Normalize SkyPilot cloud objects/strings to canonical cloud names."""
    names: list[str] = []
    for value in values:
        name = str(value).strip()
        if name:
            names.append(name.upper())
    return sorted(set(names))


def _get_compute_capability(sky: Any) -> Any:
    """Return SkyPilot compute capability enum value."""
    try:
        if CloudCapability is None:
            raise RuntimeError("CloudCapability is unavailable.")
        return CloudCapability.COMPUTE
    except Exception as exc:
        raise RuntimeError(
            "Could not resolve CloudCapability.COMPUTE from SkyPilot."
        ) from exc


def _get_enabled_clouds(sky: Any, sky_check: Any) -> list[Any]:
    """Return enabled clouds using SkyPilot's old/new check API variants."""
    get_enabled = getattr(sky_check, "get_cached_enabled_clouds_or_refresh", None)
    if not callable(get_enabled):
        raise RuntimeError(
            "sky.check.get_cached_enabled_clouds_or_refresh() is unavailable."
        )
    try:
        return list(cast(Iterable[Any], get_enabled()))
    except TypeError as exc:
        if "capability" not in str(exc):
            raise
        capability = _get_compute_capability(sky)
        return list(cast(Iterable[Any], get_enabled(capability)))


def _target_cloud_ready(
    sky: Any,
    sky_check: Any,
    *,
    target_cloud: str,
    enabled_clouds: list[str],
) -> bool:
    """Check if a target cloud is ready across SkyPilot API versions."""
    target_name = target_cloud.upper()
    if target_name in enabled_clouds:
        return True

    check_capability = getattr(sky_check, "check_capability", None)
    if callable(check_capability):
        capability = _get_compute_capability(sky)
        try:
            status = check_capability(capability, quiet=True, clouds=[target_cloud])
        except TypeError:
            status = check_capability(capability, clouds=[target_cloud])
        if isinstance(status, dict):
            ready_clouds: list[str] = []
            for cloud_list in status.values():
                if isinstance(cloud_list, list):
                    ready_clouds.extend(str(cloud).upper() for cloud in cloud_list)
            return target_name in ready_clouds
        return False

    # Legacy fallback for older SkyPilot APIs.
    check_one_cloud = getattr(sky_check, "check_one_cloud", None)
    if callable(check_one_cloud):
        cloud_obj = sky.CLOUD_REGISTRY.from_str(target_cloud)  # type: ignore[attr-defined]
        return bool(check_one_cloud(cloud_obj))

    raise RuntimeError(
        "No supported targeted cloud check API found "
        "(expected sky.check.check_capability or sky.check.check_one_cloud)."
    )


def check_cloud_readiness(
    target_cloud: str = "",
) -> tuple[list[str], list[str], CloudReadiness]:
    """Check SkyPilot cloud credentials and readiness.

    Uses SkyPilot's cloud-check APIs to discover which clouds have valid
    credentials (uses cache, refreshes if needed).

    If *target_cloud* is provided (e.g. ``"gcp"``), additionally validates
    that specific cloud via supported SkyPilot targeted check APIs and reports
    a blocking error if credentials are invalid.

    Returns ``(errors, warnings, cloud_readiness)``.
    """
    errors: list[str] = []
    warnings: list[str] = []
    result = _empty_cloud_readiness()

    if sky is None or sky_check is None:
        result["sky_installed"] = False
        if target_cloud:
            errors.append(
                "SkyPilot (sky) is not installed. "
                "Install it with: pip install 'skypilot-nightly[all]'"
            )
        return errors, warnings, result

    result["sky_installed"] = True

    # Broad check: get all enabled clouds (uses cache, refreshes if empty).
    try:
        enabled = _get_enabled_clouds(sky, sky_check)
    except RuntimeError as exc:
        # SkyPilot may raise RuntimeError when no clouds are enabled. Treat
        # those as non-blocking for broad checks; treat other runtime failures
        # as compatibility errors.
        msg = str(exc).lower()
        if "no cloud access" in msg or "no enabled cloud" in msg:
            enabled = []
        else:
            if target_cloud:
                errors.append(_compat_error(f"Target cloud check failed: {exc}"))
                result["target_cloud"] = target_cloud
                result["target_cloud_ready"] = False
            else:
                warnings.append(
                    _compat_warning(f"Failed to check cloud credentials: {exc}")
                )
            return errors, warnings, result
    except Exception as exc:
        message = _compat_warning(f"Failed to check cloud credentials: {exc}")
        if target_cloud:
            errors.append(_compat_error(f"Target cloud check failed: {exc}"))
            result["target_cloud"] = target_cloud
            result["target_cloud_ready"] = False
        else:
            warnings.append(message)
        return errors, warnings, result

    enabled_names = _cloud_names(enabled)
    result["enabled_clouds"] = enabled_names

    # Targeted check: validate the specific cloud the user wants to use
    if target_cloud:
        result["target_cloud"] = target_cloud
        try:
            ok = _target_cloud_ready(
                sky,
                sky_check,
                target_cloud=target_cloud,
                enabled_clouds=enabled_names,
            )
        except Exception as exc:
            result["target_cloud_ready"] = False
            errors.append(_compat_error(f"Target cloud check failed: {exc}"))
            return errors, warnings, result

        if ok:
            result["target_cloud_ready"] = True
        else:
            result["target_cloud_ready"] = False
            errors.append(
                f"Cloud '{target_cloud}' is not ready. "
                f"Enabled clouds: {enabled_names}. "
                "Run 'sky check' for setup instructions."
            )

    if not enabled_names:
        warnings.append(
            "No cloud providers have valid credentials. "
            "Run 'sky check' to configure cloud access."
        )

    return errors, warnings, result


def check_hardware(cfg: dict) -> tuple[list[str], list[str], HardwareInfo]:
    """Detect local hardware and check compatibility with config requirements."""

    errors: list[str] = []
    warnings: list[str] = []

    gpu_info = get_gpu_info()
    accel_type = gpu_info.get("accelerator_type", "none")
    has_gpu = accel_type in ("cuda", "mps")
    cc_str: str | None = None
    cc: float = 0.0

    if accel_type == "cuda" and gpu_info.get("accelerators"):
        cc_str = gpu_info["accelerators"][0].get("compute_capability")
        if cc_str:
            try:
                cc = float(cc_str)
            except ValueError:
                pass

    packages: dict[str, str] = {}
    for pkg in HARDWARE_PACKAGES:
        ver = get_package_version(pkg)
        if ver:
            packages[pkg] = ver
    torch_ver = packages.get("torch")

    hardware = _empty_hardware()
    hardware["accelerator_type"] = accel_type
    hardware["accelerator_count"] = gpu_info.get("accelerator_count", 0)
    hardware["gpu_name"] = gpu_info.get("gpu_name")
    hardware["gpu_memory_gb"] = (
        round(gpu_info["gpu_memory_bytes"] / (1024**3), 1)
        if gpu_info.get("gpu_memory_bytes")
        else None
    )
    hardware["compute_capability"] = cc_str
    hardware["cuda_version"] = gpu_info.get("cuda_version")
    hardware["packages"] = packages

    model_cfg = cfg.get("model") or {}
    attn_impl = model_cfg.get("attn_implementation", "")
    peft_cfg = cfg.get("peft") or {}
    training_cfg = cfg.get("training") or {}
    ds_cfg = cfg.get("deepspeed") or {}
    fsdp = training_cfg.get("fsdp") or ""
    dtype = model_cfg.get("torch_dtype_str", "") or training_cfg.get("dtype", "")
    uses_bf16 = "bf16" in str(dtype).lower()

    if attn_impl == "flash_attention_2" and "flash-attn" not in packages:
        errors.append(
            "Config requires flash_attention_2 but 'flash-attn' is not installed"
        )

    if peft_cfg.get("q_lora") and "bitsandbytes" not in packages:
        errors.append("Config requires QLoRA but 'bitsandbytes' is not installed")

    if ds_cfg.get("enable_deepspeed") and "deepspeed" not in packages:
        errors.append("Config enables DeepSpeed but 'deepspeed' is not installed")

    if torch_ver:
        tv = Version(torch_ver)
        if attn_impl == "sdpa" and tv < Version(MIN_TORCH_VERSION_SDPA):
            errors.append(
                f"Config requires SDPA but torch {torch_ver} < {MIN_TORCH_VERSION_SDPA}"
            )
        if training_cfg.get("compile") and tv < Version(MIN_TORCH_VERSION_COMPILE):
            errors.append(
                f"Config requires torch.compile but torch {torch_ver} "
                f"< {MIN_TORCH_VERSION_COMPILE}"
            )

    if (fsdp or ds_cfg.get("enable_deepspeed")) and accel_type == "none":
        warnings.append(
            "Config uses FSDP/DeepSpeed but no GPU detected locally. "
            "This is fine if targeting a remote cluster."
        )

    if fsdp and accel_type == "mps":
        warnings.append("FSDP is not supported on MPS (Apple Silicon)")

    if attn_impl == "flash_attention_2" and accel_type == "mps":
        warnings.append("flash_attention_2 is not supported on MPS (Apple Silicon)")

    if training_cfg.get("fused_optimizer") and not has_gpu:
        warnings.append("Config uses fused optimizer but no GPU detected locally")

    if uses_bf16 and not has_gpu:
        warnings.append(
            "Config uses bf16 but no GPU detected locally. "
            "This is fine if targeting a remote cluster."
        )

    if accel_type == "cuda" and cc > 0:
        if uses_bf16 and cc < MIN_CC_BF16:
            warnings.append(
                f"Config uses bf16 but GPU compute capability {cc_str} "
                f"< {MIN_CC_BF16} (Ampere). bf16 may not be natively supported."
            )
        if attn_impl == "flash_attention_2" and cc < MIN_CC_FLASH_ATTN:
            warnings.append(
                f"Config uses flash_attention_2 but GPU compute capability "
                f"{cc_str} < {MIN_CC_FLASH_ATTN} (Ampere). "
                f"Flash attention 2 requires Ampere or newer."
            )

    return errors, warnings, hardware


@mcp.tool()
def validate_config(
    config: str, task_type: ValidatorTaskType, client_cwd: str
) -> ValidateConfigResponse:
    """Validate an Oumi YAML config against its schema.

    Args:
        config: Absolute path, or relative to client_cwd, to the YAML config file.
        task_type: Config type: training, evaluation, inference, tuning,
            synthesis, quantization, job, judge, analyze, async_evaluation.
        client_cwd: REQUIRED. Absolute path to the client's working directory
            (project root). Resolves relative config paths.
    """
    config_path, path_error = _resolve_config_path(config, client_cwd)
    if path_error:
        return {"ok": False, "error": path_error}
    try:
        cfg = TASK_MAPPING[task_type].from_yaml(config_path)
        cfg.finalize_and_validate()
        return {"ok": True, "error": None}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@mcp.tool()
def get_started() -> str:
    """Overview of all tools, resources, and recommended workflow.

    CALL THIS FIRST — before using any other Oumi MCP tool.
    Returns the full tool catalog, resource list, path-resolution rules,
    and the correct order of operations for both local and cloud workflows.
    """
    return """# Oumi MCP - ML Training Config Server

> **You called `get_started()` — good.** This is the required first step
> before using any other Oumi MCP tool. Read through the workflow below
> to understand tool ordering, path rules, and config usage guidelines.

## Available Tools

### Discovery
| Tool | Purpose | Example |
|------|---------|---------|
| `list_categories()` | See available models & config types | Start here |
| `search_configs(query, task, model, keyword)` | Find training configs | `search_configs(model="llama3_1", task="sft")` |
| `get_config(path, include_content)` | Get a **reference** config (see usage note below) | `get_config("llama3_1/sft/8b_lora", include_content=True)` |
| `validate_config(config, task_type, client_cwd)` | Validate before training | `validate_config("configs/train.yaml", "training", client_cwd="/home/user/project")` |
| `pre_flight_check(config, client_cwd, cloud)` | Catch issues before launch | `pre_flight_check("configs/train.yaml", client_cwd="/home/user/project", cloud="gcp")` |
| `get_docs(query, module, kind)` | Search Oumi Python API docs | `get_docs(["TrainingConfig"])` |
| `list_modules()` | List indexed API modules | `list_modules()` |

### Execution
| Tool | Purpose | Example |
|------|---------|---------|
| `run_oumi_job(config, cmd, client_cwd)` | Execute Oumi command (dry-run by default) | `run_oumi_job("configs/train.yaml", "train", client_cwd="/home/user/project")` |
| `get_job_status(job_id)` | Status snapshot (no streaming) | `get_job_status("train_20260206_...")` |
| `get_job_logs(job_id, lines)` | Tail log snapshot | `get_job_logs("train_20260206_...", lines=200)` |
| `cancel_job(job_id)` | Cancel a running job | `cancel_job("train_20260206_...")` |
| `list_jobs()` | List running and completed jobs | `list_jobs(status="running")` |
| `stop_cluster(cloud, cluster_name)` | Stop cluster (preserves infra, reduces compute cost) | `stop_cluster("gcp", "sky-xxxx")` |
| `down_cluster(cloud, cluster_name, confirm, user_confirmation)` | Delete cluster entirely — irreversible | `down_cluster("gcp", "sky-xxxx", confirm=True, user_confirmation="DOWN")` |

### ⚠️  How to use `get_config` correctly

Configs returned by `get_config(path, include_content=True)` are **reference
recipes** — they show you the correct YAML structure, field names, and sensible
defaults for a given model/task combination. **Do NOT copy them verbatim.**

Instead:
1. Study the structure and note which fields are relevant to the user's task.
2. Build a NEW config from scratch, adapting only the settings that apply
   (model name, dataset, training params, PEFT config, output dir).
3. Customize values for the user's specific hardware, data, and goals.
4. Omit sections that don't apply (e.g., drop `peft:` for full fine-tuning).

Copying a reference config wholesale leads to wrong datasets, wrong output
paths, unnecessary settings, and confused users.

## MCP Resources

### Guidance
| Resource | What it contains | Best time to use |
|----------|------------------|-----------------|
| `guidance://mle-workflow` | End-to-end MLE workflow, decision checkpoints | Full playbook, new project |
| `guidance://mle-train` | Training command usage, sizing heuristics | Planning or running training |
| `guidance://mle-synth` | Synthetic data generation flow | Generating synthetic datasets |
| `guidance://mle-analyze` | Dataset analysis, bias/quality checks | Before training, data audit |
| `guidance://mle-eval` | Evaluation strategies, benchmarks | Benchmarking or comparing runs |
| `guidance://mle-infer` | Inference best practices, latency tuning | Running inference or sanity checks |
| `guidance://cloud-launch` | Cloud job config anatomy, setup patterns | Before launching a cloud training run |
| `guidance://post-training` | Download weights, evaluate, teardown, merge LoRA | After cloud training succeeds |

### Jobs
| Resource | What it contains |
|----------|-----------------|
| `jobs://running` | Currently running jobs (JSON array) |
| `jobs://completed` | Recently finished jobs (JSON array) |
| `jobs://{job_id}/logs` | Full log output for a specific job |

## ⚠️  CRITICAL: Working Directory and Paths

**The MCP server runs in a DIFFERENT directory than the user's project.**
You MUST pass `client_cwd` (absolute path to the project root) to all path-sensitive tools.

**What `client_cwd` does:**
- Resolves relative config paths (e.g. `configs/train.yaml` → `/home/user/project/configs/train.yaml`)
- Sets the subprocess working directory for local jobs
- Becomes the `working_dir` synced to the remote VM for cloud jobs

**Example:** If the user's project is at `/home/user/my-project`:
```
validate_config("configs/train.yaml", "training", client_cwd="/home/user/my-project")
run_oumi_job("configs/train.yaml", "train", client_cwd="/home/user/my-project")
```

**Paths inside configs:**
- **Local jobs**: absolute or relative to `client_cwd` (resolved at runtime)
- **Cloud jobs**: repo-relative paths only (resolve from `working_dir` on the remote VM)
  - ❌ BAD: `/Users/you/data/train.jsonl` — doesn't exist on the VM
  - ✅ GOOD: `data/train.jsonl` — resolves from synced `working_dir`

## ☁️  Cloud Job Workflow (REQUIRED — follow this order)

**When a user asks to run a cloud training job, ALWAYS follow these steps:**

```
CWD = "/home/user/my-project"  # user's project root — pass as client_cwd everywhere

Step 1: pre_flight_check("configs/train.yaml", client_cwd=CWD, cloud="gcp")
        # → check credentials, then use suggested_configs paths with get_config() for reference YAMLs
Step 2: run_oumi_job("configs/train.yaml", "train", client_cwd=CWD, cloud="gcp")    # dry_run (default)
        # → returns a complete job config YAML template with TODO markers
Step 3: Save the template as job.yaml in the project, customize TODO sections
        (setup, storage_mounts, envs)
        OR pass setup_script/run_script overrides inline
        Note: working_dir is auto-set from client_cwd for training configs
Step 4: run_oumi_job("job.yaml", "train", client_cwd=CWD, cloud="gcp")       # dry_run to verify
Step 5: run_oumi_job("job.yaml", "train", client_cwd=CWD, cloud="gcp",
        dry_run=False, confirm=True, user_confirmation="EXECUTE")
Step 6: get_job_status(job_id)                           # poll status
Step 7: get_job_logs(job_id, lines=200)                  # check logs
Step 8: [when done] stop_cluster("gcp", cluster_name)   # pause OR
         down_cluster("gcp", cluster_name, confirm=True, user_confirmation="DOWN")  # delete
```

**Key fields to customize in your cloud job YAML:**
- `resources.accelerators` — GPU type and count (e.g. `"A10G:1"`, `"A100:8"`)
- `working_dir` — use `.` (default); resolved to `client_cwd` at launch time
- `run` — your oumi command (path relative to `working_dir`)
- `envs` — API keys (WANDB_API_KEY, HF_TOKEN) that won't be forwarded automatically
- `file_mounts` — credential files auto-included; **add local dataset files** if not git-tracked
  - Example: `~/sky_workdir/data/train.jsonl: /Users/you/data/train.jsonl`

**Tip:** Read `guidance://cloud-launch` for detailed job config field explanations and common
setup patterns (dataset downloads, extra packages, storage mounts).

## Local Quickstart Workflow

1. **Discover models**: `list_categories()` -> see model_families
2. **Find recipes**: `search_configs(model="llama3_1", task="sft")`
3. **Study reference**: `get_config("llama3_1/sft/8b_lora", include_content=True)` — read for structure and defaults, do NOT copy verbatim
4. **Build config**: Create a new config for the user's specific model, dataset, hardware, and goals — use the reference to inform field names and reasonable values
5. **Validate**: `validate_config("configs/train.yaml", "training", client_cwd="/home/user/project")`
6. **Preview**: `run_oumi_job("configs/train.yaml", "train", client_cwd="/home/user/project")` -> dry-run (default)
7. **Execute**: `run_oumi_job("configs/train.yaml", "train", client_cwd="/home/user/project", dry_run=False, confirm=True, user_confirmation="EXECUTE")`
8. **Check status**: `get_job_status("train_20260206_...")`
9. **Get logs**: `get_job_logs("train_20260206_...", lines=200)`

## Execution Pattern

```
CWD = "/home/user/project"  # always pass client_cwd

Step 1 (preview):  run_oumi_job(config, "train", client_cwd=CWD)                                    # dry_run=True
Step 2 (execute):  run_oumi_job(config, "train", client_cwd=CWD, dry_run=False, confirm=True, user_confirmation="EXECUTE")
Step 3 (status):   get_job_status(job_id)
Step 4 (logs):     get_job_logs(job_id, lines=200)
Step 5 (cancel):   cancel_job(job_id)
Step 5b (force):   cancel_job(job_id, force=True)
```

## Cluster Lifecycle

After a cloud job finishes (or to manage costs), use cluster lifecycle tools:

```
Step 1: get_job_status(job_id)                        # → see "cluster" and "cloud" fields
Step 2a (pause):  stop_cluster("gcp", "sky-xxxx")     # keeps infra, lower cost, can restart
Step 2b (delete): down_cluster("gcp", "sky-xxxx", confirm=True, user_confirmation="DOWN")
```

- **`stop_cluster`**: Pauses compute. Storage costs may still apply. Cluster restartable.
- **`down_cluster`**: Permanently deletes everything. No more billing. Irreversible.
- ⚠️  Always `down_cluster` when training is fully done to avoid storage charges.

## Search Parameters

- **task**: sft, dpo, grpo, kto, eval, infer, pretrain
- **model**: llama3_1, llama3_2, llama4, qwen3, phi4, gemma3, deepseek_r1, smollm
- **query**: Any text (e.g., "8b", "lora", "qlora", "instruct")
- **keyword**: Content match inside YAML (e.g., "packing", "flash_attn", "gradient_checkpointing")

## Config Key Settings

When customizing a config, these are the key fields to modify:
- `model.model_name`: HuggingFace model ID
- `data.train.datasets`: Dataset list (see dataset_name below)
- `training.output_dir`: Where to save checkpoints
- `training.learning_rate`: Start with recipe default
- `training.per_device_train_batch_size`: Adjust for your GPU memory

## ⚠️  dataset_name: Use Registry Names, NOT Class Names

For local JSONL files, use `dataset_name: "text_sft_jsonl"` with `dataset_path` pointing to the file:
```yaml
data:
  train:
    datasets:
      - dataset_name: "text_sft_jsonl"
        dataset_path: "pubmedqa/train.jsonl"
```
- WRONG: `TextSftJsonLinesDataset`, `TextSftJsonlDataset` — these are Python class names, not registry names
- For HuggingFace datasets: use the full HF ID (e.g. `yahma/alpaca-cleaned`) as dataset_name
- Use `get_docs(["dataset"])` to search for other registered dataset names

## ⚠️  LoRA/QLoRA: MUST set `use_peft: True`

When using LoRA or QLoRA, you MUST set BOTH:
1. The `peft:` config block (lora_r, lora_alpha, lora_target_modules, etc.)
2. `training.use_peft: True`

Without `use_peft: True`, the `peft:` block is **silently ignored** and full fine-tuning runs
instead — using ~4x more VRAM and likely OOMing on smaller GPUs (e.g. A10G with 8B model).

## GPU VRAM Quick Reference

| Model Size | Full Fine-Tune | LoRA | QLoRA |
|-----------|---------------|------|-------|
| 3B | 24 GB | 12 GB | 8 GB |
| 7-8B | 60 GB | 20 GB | 14 GB |
| 13B | 100 GB | 32 GB | 20 GB |
| 70B | 400 GB+ | 80 GB | 48 GB |

Common cloud GPUs: A10G (22 GB), L4 (24 GB), A100 (40/80 GB), H100 (80 GB).
"""


def _jobconfig_to_yaml(jc: launcher.JobConfig) -> str:
    """Render a JobConfig as compact YAML for dry-run display.

    Omits None values and empty dicts/lists so the preview stays readable.
    """
    d = {k: v for k, v in dataclasses.asdict(jc).items() if v not in (None, {}, [], "")}
    if "resources" in d and isinstance(d["resources"], dict):
        d["resources"] = {
            k: v for k, v in d["resources"].items() if v not in (None, False, "")
        }
    return yaml.dump(d, default_flow_style=False, sort_keys=False)


@mcp.tool()
async def run_oumi_job(
    config_path: str,
    command: str,
    client_cwd: str,
    dry_run: bool = True,
    confirm: bool = False,
    user_confirmation: str = "",
    job_name: str | None = None,
    cloud: str = "local",
    cluster_name: str = "",
    accelerators: str = "",
    envs: dict[str, str] | None = None,
    file_mounts: dict[str, str] | None = None,
    disk_size: int | None = None,
    use_spot: bool = False,
    num_nodes: int = 1,
    setup_script: str = "",
    run_script: str = "",
) -> JobSubmissionResponse:
    """Execute an Oumi CLI command with background job tracking.

    Two-step safety: call with dry_run=True (default) to preview, then
    dry_run=False, confirm=True, user_confirmation="EXECUTE" to launch.
    Cloud runs execute a pre-flight check that may block launch.

    Job configs (with ``resources``/``setup``/``run`` keys) pass through
    directly to ``oumi launch up``. Training configs are auto-wrapped.
    HF/WandB credentials are auto-mounted on cloud VMs.

    Args:
        config_path: Absolute path, or relative to client_cwd, to an Oumi YAML config.
        command: Oumi subcommand: train, analyze, synth, evaluate, eval,
            infer, tune, quantize. Ignored for job configs.
        client_cwd: REQUIRED. Absolute path to the client's working directory
            (project root). Resolves relative config paths. For local jobs, the
            Oumi CLI subprocess runs from this directory. For cloud jobs, this
            directory is synced to the remote VM as the working directory.
        dry_run: Preview execution plan without running (default True).
        confirm: Must be True for actual execution.
        user_confirmation: Must be ``"EXECUTE"`` when dry_run=False.
        job_name: Optional name; auto-generated if omitted.
        cloud: ``"local"`` (default) or a cloud provider name.
        cluster_name: Cluster name for cloud launches.
        accelerators: Accelerator spec, e.g. ``"A100:8"``. Multi-GPU
            auto-enables ``oumi distributed torchrun``.
        envs: Env vars for the remote VM.
        file_mounts: Additional local-to-remote file mappings. Use for local
            dataset files not git-tracked in working_dir (e.g.
            ``{"~/sky_workdir/data/train.jsonl": "/abs/path/to/train.jsonl"}``).
        disk_size: Disk size in GB for the remote VM.
        use_spot: Use spot/preemptible instances.
        num_nodes: Node count for distributed training.
        setup_script: Override default cloud setup script (training-config
            wrapping mode only).
        run_script: Override auto-generated run command (training-config
            wrapping mode only).
    """
    command = command.strip().lower()
    cloud = cloud.strip().lower() or "local"
    cluster_name = cluster_name.strip()
    accelerators = accelerators.strip()

    def _error_response(error: str, **overrides: Any) -> JobSubmissionResponse:
        base: JobSubmissionResponse = {
            "success": False,
            "job_id": "",
            "status": "error",
            "dry_run": dry_run,
            "command": command,
            "config_path": config_path,
            "cloud": cloud,
            "cluster_name": cluster_name,
            "model_name": "",
            "message": "",
            "error": error,
        }
        base.update(overrides)  # type: ignore[typeddict-item]
        return base

    if command not in VALID_OUMI_COMMANDS:
        return _error_response(
            f"Invalid command: '{command}'. "
            f"Must be one of: {sorted(VALID_OUMI_COMMANDS)}"
        )

    config_file, path_error = _resolve_config_path(config_path, client_cwd)
    if path_error:
        return _error_response(path_error)

    abs_config = str(config_file)
    parsed_cfg, parse_error = _load_yaml_strict(config_file)
    if parse_error or parsed_cfg is None:
        return _error_response(
            (
                f"{parse_error} "
                "Run validate_config(..., task_type=...) before launching."
            ),
            config_path=abs_config,
        )
    try:
        model_name, output_dir = _extract_job_metadata_from_cfg(parsed_cfg)
    except Exception as exc:
        return _error_response(
            f"Failed to parse config metadata: {exc}",
            config_path=abs_config,
        )

    job_id = make_job_id(command, job_name)

    is_job_config_file = _is_job_config(config_file) if cloud != "local" else False

    num_gpus = _parse_gpu_count(accelerators or None)

    if dry_run:
        if is_job_config_file:
            cmd_preview = f"oumi launch up -c {abs_config}"
        elif num_gpus > 1 or num_nodes > 1:
            cmd_preview = f"oumi distributed torchrun -m oumi {command} -c {abs_config}"
        else:
            cmd_preview = f"oumi {command} -c {abs_config}"

        dry_run_msg_parts = [
            f"Dry run: would execute `{cmd_preview}` on {cloud}",
            f"Model: {model_name}",
            f"Output: {output_dir}",
            f"Config type: {'job config (passthrough)' if is_job_config_file else 'training config (wrapped)'}",
            "Validation: strict YAML parsing passed.",
        ]
        dry_run_msg_parts.append(
            "To execute, re-call with dry_run=False, confirm=True, "
            "user_confirmation='EXECUTE'."
        )
        message = "\n".join(dry_run_msg_parts)
        if cloud != "local":
            if is_job_config_file:
                try:
                    preview_job_cfg = launcher.JobConfig.from_yaml(abs_config)
                    job_cfg_yaml = _jobconfig_to_yaml(preview_job_cfg)
                except Exception:
                    job_cfg_yaml = "(could not parse job config for preview)"
                message = (
                    message
                    + "\n\n--- Generated JobConfig (review before executing) ---\n"
                    + job_cfg_yaml
                    + "-----------------------------------------------------"
                )
            else:
                job_config_template = _generate_job_config_template(
                    abs_config,
                    command,
                    cloud,
                    model_name,
                    client_cwd=client_cwd,
                    job_name=job_id,
                    accelerators=accelerators,
                    num_nodes=num_nodes,
                    envs=envs,
                    setup_script=setup_script,
                    run_script=run_script,
                )
                env_warning = _build_missing_env_warning(envs)
                if env_warning:
                    message = message + env_warning
                message = (
                    message
                    + "\n\n--- Job Config Template (save as YAML, customize TODO sections, re-submit) ---\n"
                    + job_config_template
                    + "\n----------------------------------------------------------------------\n"
                    + "\nNEXT STEPS:\n"
                    + "1. Save the template above as a job config YAML file (e.g., my_job.yaml in the project)\n"
                    + "2. Customize the TODO sections (setup, file_mounts for data, storage_mounts, envs)\n"
                    + "   - Mount local dataset files via file_mounts if they're not git-tracked\n"
                    + "   - If using LoRA/QLoRA, ensure training.use_peft: True is set in your training config\n"
                    + "3. Re-submit with the job config: run_oumi_job('my_job.yaml', '"
                    + command
                    + "', client_cwd=<project_root>, cloud='"
                    + cloud
                    + "')\n"
                    + "\nAlternatively, pass setup_script and run_script overrides inline to skip the file roundtrip.\n"
                    + "Read guidance://cloud-launch for detailed field explanations, GPU sizing, and setup patterns."
                )
        return {
            "success": True,
            "job_id": job_id,
            "status": "dry_run",
            "dry_run": True,
            "command": command,
            "config_path": abs_config,
            "cloud": cloud,
            "cluster_name": cluster_name,
            "model_name": model_name,
            "message": message,
        }

    if not confirm or user_confirmation != "EXECUTE":
        return {
            "success": False,
            "job_id": job_id,
            "status": "blocked",
            "dry_run": False,
            "command": command,
            "config_path": abs_config,
            "cloud": cloud,
            "cluster_name": cluster_name,
            "model_name": model_name,
            "message": "",
            "error": (
                "Execution blocked: launching requires confirm=True and "
                "user_confirmation='EXECUTE'. Run with dry_run=True first to "
                "preview, then execute with explicit user permission."
            ),
        }

    preflight_summary = ""
    preflight_blocking = False
    preflight_errors: list[str] = []
    preflight_warnings: list[str] = []

    if cloud != "local":
        preflight = _pre_flight_check(abs_config, client_cwd=client_cwd, cloud=cloud)
        preflight_summary = preflight.get("summary", "")
        preflight_blocking = bool(preflight.get("blocking"))
        preflight_errors = preflight.get("errors", []) or []
        preflight_warnings = list(preflight.get("warnings", []) or [])

        if not is_job_config_file:
            hf_token_path = Path("~/.cache/huggingface/token").expanduser()
            if (
                not hf_token_path.exists()
                and preflight.get("hf_authenticated") is False
            ):
                preflight_warnings.append(
                    "HF token not found locally (~/.cache/huggingface/token). "
                    "Gated model downloads will fail on the remote VM."
                )

            if num_gpus > 1 or num_nodes > 1:
                preflight_warnings.append(
                    f"Multi-GPU/multi-node job detected (accelerators={accelerators!r}, "
                    f"num_nodes={num_nodes}). Using `oumi distributed torchrun` automatically."
                )

        compat_messages = [
            msg
            for msg in [*preflight_errors, *preflight_warnings]
            if "SkyPilot API compatibility" in msg
        ]
        if preflight_blocking:
            return _error_response(
                f"Pre-flight checks failed: {preflight_summary}",
                status="blocked",
                job_id=job_id,
                config_path=abs_config,
                model_name=model_name,
                preflight_summary=preflight_summary,
                preflight_blocking=preflight_blocking,
                preflight_errors=preflight_errors,
                preflight_warnings=preflight_warnings,
            )
        if compat_messages:
            return _error_response(
                "Pre-flight detected a SkyPilot compatibility issue. "
                "Align Oumi/SkyPilot versions and run `sky check` before launching.",
                status="blocked",
                job_id=job_id,
                config_path=abs_config,
                model_name=model_name,
                preflight_summary=preflight_summary,
                preflight_blocking=True,
                preflight_errors=preflight_errors,
                preflight_warnings=preflight_warnings,
            )

    submit_time = datetime.now(tz=timezone.utc).isoformat()
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
    reg = get_registry()
    reg.add(record)

    rt = get_runtime(job_id)
    rt.log_dir = JOB_LOGS_DIR / job_id
    rt.run_dir = JOB_RUNS_DIR / job_id

    is_local = cloud == "local"
    if is_local:
        try:
            start_local_job(record, rt, client_cwd=client_cwd)
        except Exception as exc:
            rt.error_message = str(exc)
            return _error_response(
                f"Failed to start job: {exc}",
                job_id=job_id,
                config_path=abs_config,
                model_name=model_name,
            )
        runner = asyncio.create_task(
            wait_local_completion(record, rt),
            name=f"oumi-job-{job_id}",
        )
    else:
        runner = asyncio.create_task(
            launch_job(
                record,
                rt,
                client_cwd=client_cwd,
                accelerators=accelerators or None,
                envs=envs,
                file_mounts=file_mounts,
                disk_size=disk_size,
                use_spot=use_spot,
                num_nodes=num_nodes,
                setup_script=setup_script or None,
                run_script=run_script or None,
            ),
            name=f"oumi-job-{job_id}",
        )
    rt.runner_task = runner

    launch_confirmed = False
    if not is_local:
        try:
            await asyncio.wait_for(asyncio.shield(runner), timeout=10.0)
            launch_confirmed = rt.error_message is None
        except asyncio.TimeoutError:
            launch_confirmed = False
        except Exception:
            launch_confirmed = False

    logger.info(
        "Job %s submitted on %s — launching `oumi %s` in background",
        job_id,
        cloud,
        command,
    )

    record = reg.get(job_id) or record

    if rt.error_message and not is_local:
        return _error_response(
            f"Failed to launch cloud job: {rt.error_message}",
            status="failed",
            job_id=job_id,
            config_path=abs_config,
            model_name=model_name,
            preflight_summary=preflight_summary,
            preflight_blocking=preflight_blocking,
            preflight_errors=preflight_errors,
            preflight_warnings=preflight_warnings,
            launch_confirmed=launch_confirmed,
            oumi_job_id=record.oumi_job_id,
            cluster=record.cluster_name,
        )

    message = (
        f"Job {job_id} submitted on {cloud}. "
        f"Use get_job_status('{job_id}') for status and "
        f"get_job_logs('{job_id}', lines=200) for logs."
    )
    if not is_local and not launch_confirmed:
        message = message + " Launch confirmation is pending; re-check status shortly."

    return {
        "success": True,
        "job_id": job_id,
        "status": "submitted",
        "dry_run": False,
        "command": command,
        "config_path": abs_config,
        "cloud": cloud,
        "cluster_name": cluster_name,
        "model_name": model_name,
        "launch_confirmed": launch_confirmed if not is_local else True,
        "preflight_summary": preflight_summary,
        "preflight_blocking": preflight_blocking,
        "preflight_errors": preflight_errors,
        "preflight_warnings": preflight_warnings,
        "oumi_job_id": record.oumi_job_id,
        "cluster": record.cluster_name,
        "message": message,
    }


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


def _build_status_response(
    record: JobRecord,
    rt: JobRuntime,
    *,
    log_file: str = "",
) -> JobStatusResponse:
    """Build a ``JobStatusResponse`` from a ``JobRecord`` and ``JobRuntime``."""
    status = rt.oumi_status
    is_local = record.cloud == "local"

    status_str = _job_status_str(record, rt)
    if is_local:
        oumi_job_id = record.oumi_job_id
        state_str = status_str.upper()
        cluster_str = "local"
    else:
        oumi_job_id = status.id if status else record.oumi_job_id
        state_str = status.state.name if status and status.state else ""
        cluster_str = status.cluster if status else record.cluster_name

    base: JobStatusResponse = {
        "success": True,
        "job_id": record.job_id,
        "oumi_job_id": oumi_job_id,
        "status": status_str,
        "state": state_str,
        "command": record.command,
        "config_path": record.config_path,
        "cloud": record.cloud,
        "cluster": cluster_str,
        "model_name": record.model_name,
        "is_done": _is_job_done(record, rt),
        "error": rt.error_message,
    }

    if status and status.metadata:
        base["metadata"] = (
            status.metadata
            if isinstance(status.metadata, dict)
            else {"raw": str(status.metadata)}
        )
    if log_file:
        base["log_file"] = log_file

    return base


def _not_found_response(job_id: str) -> JobStatusResponse:
    """Return a ``JobStatusResponse`` for a missing job ID."""
    return {
        "success": False,
        "job_id": job_id,
        "oumi_job_id": "",
        "status": "not_found",
        "state": "",
        "command": "",
        "config_path": "",
        "cloud": "",
        "cluster": "",
        "model_name": "",
        "is_done": False,
        "error": (
            f"Job '{job_id}' not found. "
            "Use list_jobs() for MCP-managed jobs, or provide "
            "oumi_job_id + cloud (+ cluster_name) for direct cloud lookup."
        ),
    }


def _resolve_job_record(
    *,
    job_id: str = "",
    oumi_job_id: str = "",
    cloud: str = "",
    cluster_name: str = "",
) -> JobRecord | None:
    reg = get_registry()
    if job_id:
        record = reg.get(job_id)
        if record:
            return record
    if oumi_job_id and cloud:
        return reg.find_by_cloud_identity(cloud, oumi_job_id)
    return None


async def _fetch_cloud_status_direct(
    *,
    oumi_job_id: str,
    cloud: str,
    cluster_name: str = "",
) -> Any | None:
    try:
        statuses_by_cloud = await asyncio.to_thread(
            launcher.status,
            cloud=cloud,
            cluster=cluster_name or None,
            id=oumi_job_id,
        )
    except Exception:
        return None
    for _cloud_name, statuses in statuses_by_cloud.items():
        for status in statuses:
            if status.id == oumi_job_id:
                return status
    return None


@mcp.tool()
async def get_job_status(
    job_id: str = "",
    oumi_job_id: str = "",
    cloud: str = "",
    cluster_name: str = "",
) -> JobStatusResponse:
    """Return a single status snapshot for an Oumi job.

    Lookup precedence:
      1) MCP ``job_id`` (recommended for jobs launched by this MCP)
      2) Direct cloud identity: ``oumi_job_id`` + ``cloud`` (+ optional ``cluster_name``)
    """
    cloud = cloud.strip().lower()
    cluster_name = cluster_name.strip()
    job_id = job_id.strip()
    oumi_job_id = oumi_job_id.strip()

    if not job_id and not oumi_job_id:
        return _not_found_response("")

    record = _resolve_job_record(
        job_id=job_id,
        oumi_job_id=oumi_job_id,
        cloud=cloud,
        cluster_name=cluster_name,
    )
    if not record:
        if not oumi_job_id or not cloud:
            return _not_found_response(job_id or oumi_job_id)
        direct_status = await _fetch_cloud_status_direct(
            oumi_job_id=oumi_job_id,
            cloud=cloud,
            cluster_name=cluster_name,
        )
        if not direct_status:
            return _not_found_response(job_id or oumi_job_id)
        return {
            "success": True,
            "job_id": job_id or oumi_job_id,
            "oumi_job_id": direct_status.id,
            "status": direct_status.status,
            "state": direct_status.state.name if direct_status.state else "",
            "command": "",
            "config_path": "",
            "cloud": cloud,
            "cluster": direct_status.cluster or cluster_name,
            "model_name": "",
            "is_done": bool(direct_status.done),
            "metadata": direct_status.metadata if direct_status.metadata else {},
            "error": None,
        }

    rt = get_runtime(record.job_id)
    await poll_status(record, rt)
    log_paths = get_log_paths(record, rt)
    return _build_status_response(
        record,
        rt,
        log_file=str(log_paths["stdout"]) if log_paths["stdout"] else "",
    )


def _read_log_tail(stdout_path: Path, lines: int) -> tuple[str, int]:
    """Read the trailing *lines* from *stdout_path* efficiently."""
    if lines <= 0:
        return ("", 0)
    block_size = 8192
    data = b""
    newline_count = 0

    with stdout_path.open("rb") as f:
        pos = f.seek(0, 2)
        while pos > 0 and newline_count <= lines:
            read_size = min(block_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            data = chunk + data
            newline_count = data.count(b"\n")

    text = data.decode("utf-8", errors="replace")
    all_lines = text.splitlines()
    if not all_lines:
        return ("", 0)
    tail_lines = all_lines[-lines:]
    return ("\n".join(tail_lines), len(tail_lines))


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


@mcp.tool()
async def get_job_logs(
    job_id: str = "",
    lines: int = DEFAULT_STREAM_LINES,
    oumi_job_id: str = "",
    cloud: str = "",
    cluster_name: str = "",
) -> JobLogsResponse:
    """Return a bounded log snapshot for an Oumi job.

    Lookup precedence:
      1) MCP ``job_id`` for MCP-managed local log files
      2) Direct cloud identity: ``oumi_job_id`` + ``cloud`` (+ optional ``cluster_name``)

    Note: Direct cloud identities do not map to local MCP log files unless the
    job is already tracked by this MCP instance.
    """
    if lines < 0:
        lines = 0

    cloud = cloud.strip().lower()
    cluster_name = cluster_name.strip()
    job_id = job_id.strip()
    oumi_job_id = oumi_job_id.strip()

    record = _resolve_job_record(
        job_id=job_id,
        oumi_job_id=oumi_job_id,
        cloud=cloud,
        cluster_name=cluster_name,
    )
    if not record:
        # Direct cloud log retrieval for untracked jobs (bypasses registry)
        if oumi_job_id and cloud and cluster_name:
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
            cloud_result = await _get_cloud_logs(ephemeral, JobRuntime(), lines)
            if cloud_result is not None:
                cloud_logs, cloud_lines = cloud_result
                return {
                    "success": True,
                    "job_id": oumi_job_id,
                    "lines_requested": lines,
                    "lines_returned": cloud_lines,
                    "log_file": f"cloud:{cloud}/{cluster_name}",
                    "logs": cloud_logs,
                    "error": None,
                }
            return {
                "success": False,
                "job_id": oumi_job_id,
                "lines_requested": lines,
                "lines_returned": 0,
                "log_file": "",
                "logs": "",
                "error": (
                    f"Cloud log retrieval failed for {cloud}/{cluster_name}. "
                    f"The cluster may no longer exist or SSH timed out."
                ),
            }
        if oumi_job_id and cloud:
            return {
                "success": False,
                "job_id": job_id or oumi_job_id,
                "lines_requested": lines,
                "lines_returned": 0,
                "log_file": "",
                "logs": "",
                "error": (
                    "cluster_name is required for direct cloud log retrieval. "
                    "Provide oumi_job_id + cloud + cluster_name."
                ),
            }
        return {
            "success": False,
            "job_id": job_id or oumi_job_id,
            "lines_requested": lines,
            "lines_returned": 0,
            "log_file": "",
            "logs": "",
            "error": f"Job '{job_id or oumi_job_id}' not found.",
        }

    rt = get_runtime(record.job_id)
    await poll_status(record, rt)
    log_paths = get_log_paths(record, rt)
    stdout_path = log_paths.get("stdout")
    resolved_job_id = record.job_id

    if not stdout_path or not stdout_path.exists():
        # Cloud fallback: fetch logs from cluster via get_logs_stream
        if record.cloud and record.cloud != "local":
            cloud_result = await _get_cloud_logs(record, rt, lines)
            if cloud_result is not None:
                cloud_logs, cloud_lines = cloud_result
                return {
                    "success": True,
                    "job_id": resolved_job_id,
                    "lines_requested": lines,
                    "lines_returned": cloud_lines,
                    "log_file": f"cloud:{record.cloud}/{record.cluster_name}",
                    "logs": cloud_logs,
                    "error": None,
                }
            return {
                "success": False,
                "job_id": resolved_job_id,
                "lines_requested": lines,
                "lines_returned": 0,
                "log_file": "",
                "logs": "",
                "error": (
                    "No local log file and cloud log retrieval failed. "
                    f"The cluster '{record.cluster_name}' may no longer exist. "
                    f"Try `sky logs {record.cluster_name}` directly."
                ),
            }
        return {
            "success": False,
            "job_id": resolved_job_id,
            "lines_requested": lines,
            "lines_returned": 0,
            "log_file": "",
            "logs": "",
            "error": "No stdout log file available yet for this job.",
        }

    try:
        logs, lines_returned = await asyncio.to_thread(
            _read_log_tail, stdout_path, lines
        )
    except OSError as exc:
        return {
            "success": False,
            "job_id": resolved_job_id,
            "lines_requested": lines,
            "lines_returned": 0,
            "log_file": str(stdout_path),
            "logs": "",
            "error": f"Failed to read log file: {exc}",
        }

    return {
        "success": True,
        "job_id": resolved_job_id,
        "lines_requested": lines,
        "lines_returned": lines_returned,
        "log_file": str(stdout_path),
        "logs": logs,
        "error": None,
    }


@mcp.tool()
async def cancel_job(
    job_id: str = "",
    force: bool = False,
    oumi_job_id: str = "",
    cloud: str = "",
    cluster_name: str = "",
) -> JobCancelResponse:
    """Cancel a running or pending Oumi job.

    Local jobs: SIGTERM (or SIGKILL with force=True).
    Cloud jobs: delegates to ``oumi.launcher.cancel()``.

    Args:
        job_id: MCP job ID (preferred).
        force: SIGKILL instead of SIGTERM (local only).
        oumi_job_id: Cluster-side job ID for direct cloud cancellation.
        cloud: Cloud provider when using ``oumi_job_id``.
        cluster_name: Cluster name when using ``oumi_job_id``.
    """
    cloud = cloud.strip().lower()
    cluster_name = cluster_name.strip()
    job_id = job_id.strip()
    oumi_job_id = oumi_job_id.strip()

    record = _resolve_job_record(
        job_id=job_id,
        oumi_job_id=oumi_job_id,
        cloud=cloud,
        cluster_name=cluster_name,
    )

    if not record:
        if oumi_job_id and cloud:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        launcher.cancel,
                        oumi_job_id,
                        cloud,
                        cluster_name,
                    ),
                    timeout=30.0,
                )
            except TimeoutError:
                return {
                    "success": False,
                    "error": (
                        f"Cancel timed out after 30s "
                        f"(cloud={cloud}, cluster={cluster_name}, id={oumi_job_id}). "
                        "The cancellation may still be in progress. "
                        "Check cloud console or retry."
                    ),
                }
            except Exception as exc:
                return {
                    "success": False,
                    "error": (
                        "Failed to cancel cloud job by direct identity "
                        f"(cloud={cloud}, cluster={cluster_name}, id={oumi_job_id}): {exc}"
                    ),
                }
            return {
                "success": True,
                "message": (
                    "Cancel requested by direct cloud identity "
                    f"(cloud={cloud}, cluster={cluster_name}, id={oumi_job_id})."
                ),
            }
        return {
            "success": False,
            "error": f"Job '{job_id or oumi_job_id}' not found.",
        }

    rt = get_runtime(record.job_id)
    return await cancel(record, rt, force=force)


@mcp.tool()
async def stop_cluster(cloud: str, cluster_name: str) -> ClusterLifecycleResponse:
    """Stop a running cluster, preserving infra and reducing compute cost.

    Restart by submitting a new job with the same cluster_name.
    Use ``down_cluster`` to fully delete and stop all billing.

    Args:
        cloud: Cloud provider (e.g. ``"gcp"``, ``"aws"``).
        cluster_name: Name of the cluster to stop.
    """
    cloud = cloud.strip().lower()
    cluster_name = cluster_name.strip()
    if not cloud or not cluster_name:
        return {
            "success": False,
            "error": "cloud and cluster_name are required.",
        }
    try:
        await asyncio.to_thread(launcher.stop, cloud, cluster_name)
        return {
            "success": True,
            "message": (
                f"Cluster '{cluster_name}' on {cloud} stopped. "
                "Infra is preserved; restart by submitting a new job with "
                f"cluster_name='{cluster_name}'. Storage costs may still apply. "
                f"Use down_cluster to fully delete."
            ),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to stop cluster '{cluster_name}' on {cloud}: {exc}",
        }


@mcp.tool()
async def down_cluster(
    cloud: str,
    cluster_name: str,
    confirm: bool = False,
    user_confirmation: str = "",
) -> ClusterLifecycleResponse:
    """IRREVERSIBLE: delete a cluster and all its resources.

    Requires ``confirm=True`` and ``user_confirmation="DOWN"``.
    Without these, returns a dry-run description.
    Use ``stop_cluster`` to pause without deleting.

    Args:
        cloud: Cloud provider (e.g. ``"gcp"``, ``"aws"``).
        cluster_name: Name of the cluster to delete.
        confirm: Must be True for actual deletion.
        user_confirmation: Must be exactly ``"DOWN"``.
    """
    cloud = cloud.strip().lower()
    cluster_name = cluster_name.strip()
    if not cloud or not cluster_name:
        return {
            "success": False,
            "error": "cloud and cluster_name are required.",
        }
    if not confirm:
        return {
            "success": True,
            "message": (
                f"Dry run: would permanently delete cluster '{cluster_name}' on {cloud}. "
                "IRREVERSIBLE — all cluster resources and data will be deleted and "
                "billing will stop. To confirm, re-call with "
                "confirm=True, user_confirmation='DOWN'."
            ),
        }
    if user_confirmation != "DOWN":
        return {
            "success": False,
            "error": "Confirmation phrase must be exactly 'DOWN'. Deletion blocked.",
        }
    try:
        await asyncio.to_thread(launcher.down, cloud, cluster_name)
        return {
            "success": True,
            "message": (
                f"Cluster '{cluster_name}' on {cloud} deleted. "
                "All resources have been removed and billing has stopped."
            ),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to delete cluster '{cluster_name}' on {cloud}: {exc}",
        }


@mcp.tool()
async def list_jobs(
    status: str = "all",
) -> list[JobSummary]:
    """List running and completed jobs.

    Args:
        status: ``"all"`` (default), ``"running"``, or ``"completed"``.
    """
    return await _list_job_summaries(status_filter=status)


@mcp.tool()
def get_docs(
    query: list[str],
    module: str = "",
    kind: str = "",
    limit: int = 10,
    summarize: bool = False,
) -> DocsSearchResponse:
    """Search Oumi's indexed Python API docs.

    Matches by: (1) exact qualified name, (2) exact short name, then
    (3) relevance-ranked keyword search over names, fields, and docstrings.

    Args:
        query: Exact names or keywords, e.g. ["TrainingConfig"] or
            ["learning_rate", "lora"].
        module: Module prefix filter (e.g. "oumi.core.configs").
        kind: Kind filter: "class", "dataclass", "function", or "method".
        limit: Max results (default 10).
        summarize: Compact output omitting fields and docstring sections.
    """
    return search_docs(
        query=query, module=module, kind=kind, limit=limit, summarize=summarize
    )


@mcp.tool()
def list_modules() -> ListModulesResponse:
    """List indexed API modules available for ``get_docs`` searches."""
    return get_module_list()


@mcp.resource("jobs://running", mime_type="application/json")
async def list_running_jobs() -> str:
    """List all currently running Oumi jobs.

    Returns a JSON array of job summaries with job_id, command, status,
    cloud, cluster, model_name, and is_done.
    """
    summaries = await _list_job_summaries(status_filter="running")
    return json.dumps(summaries, indent=2)


@mcp.resource("jobs://completed", mime_type="application/json")
async def list_completed_jobs() -> str:
    """List recently completed, failed, or cancelled Oumi jobs.

    Returns a JSON array of job summaries with job_id, command, status,
    cloud, cluster, model_name, and is_done.
    """
    summaries = await _list_job_summaries(status_filter="completed")
    return json.dumps(summaries, indent=2)


@mcp.resource("jobs://{job_id}/logs", mime_type="text/plain")
async def get_job_logs_resource(job_id: str) -> str:
    """Full log output for a specific job.

    For local jobs, returns the contents of the stdout log file on disk.
    For cloud jobs, returns metadata about how to access logs
    (e.g. via ``sky logs``).
    """
    record = get_registry().get(job_id)
    if not record:
        return json.dumps({"error": f"Job '{job_id}' not found"})

    rt = get_runtime(record.job_id)
    status = await poll_status(record, rt)

    header_parts = [f"Job: {record.job_id}"]
    if status:
        header_parts.append(f"Oumi ID: {status.id}")
        header_parts.append(f"Status: {status.status}")
        header_parts.append(f"Cluster: {status.cluster}")
        header_parts.append(f"Done: {status.done}")
        if status.metadata:
            header_parts.append(f"Metadata: {status.metadata}")
    else:
        header_parts.append(f"Cloud: {record.cloud}")
        if rt.error_message:
            header_parts.append(f"Error: {rt.error_message}")
        else:
            header_parts.append("Status: launching (no status available yet)")

    header = "\n".join(header_parts)

    log_paths = get_log_paths(record, rt)
    stdout_path = log_paths.get("stdout")
    if stdout_path and stdout_path.exists():
        try:
            log_content = stdout_path.read_text(encoding="utf-8", errors="replace")
            return f"{header}\nLog file: {stdout_path}\n\n--- stdout ---\n{log_content}"
        except OSError as exc:
            header += f"\nFailed to read log file: {exc}"

    if record.cloud != "local" and status:
        header += (
            f"\n\nFor cloud jobs, use `sky logs {status.cluster}` "
            f"to stream full logs, or call "
            f"`get_job_logs('{record.job_id}', lines=200)` for a snapshot."
        )

    return header


def _read_version_marker() -> str:
    """Read the oumi version that the cached configs were synced for."""
    marker = get_cache_dir() / CONFIGS_VERSION_MARKER
    try:
        return marker.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _write_version_marker(version: str) -> None:
    """Record which oumi version the cached configs correspond to."""
    marker = get_cache_dir() / CONFIGS_VERSION_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(version, encoding="utf-8")


def _is_cache_stale() -> bool:
    """Check whether the cached configs need to be refreshed.

    Returns True if the cache directory doesn't exist, has no YAML files,
    the last sync was more than CONFIGS_SYNC_INTERVAL_HOURS ago, or the
    installed oumi version no longer matches the cached version marker.
    """
    cache_dir = get_cache_dir()
    marker = cache_dir / CONFIGS_SYNC_MARKER

    if not cache_dir.exists() or not any(cache_dir.rglob("*.yaml")):
        return True

    if not marker.exists():
        return True

    cached_version = _read_version_marker()
    current_version = get_oumi_version()
    if cached_version and current_version != "unknown":
        if cached_version != current_version:
            logger.info(
                "Oumi version changed (%s -> %s); cache is stale",
                cached_version,
                current_version,
            )
            return True

    try:
        age_hours = (time.time() - marker.stat().st_mtime) / 3600
        return age_hours > CONFIGS_SYNC_INTERVAL_HOURS
    except Exception:
        return True


def _touch_sync_marker() -> None:
    """Write/update the sync timestamp marker file."""
    marker = get_cache_dir() / CONFIGS_SYNC_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("")


def get_configs_source() -> str:
    """Describe which source the current configs directory comes from.

    Possible values: ``"cache:<version>"``, ``"cache:main"``,
    ``"bundled:<version>"``, ``"env:<path>"``, or ``"unknown"``.
    """
    env_dir = os.environ.get("OUMI_MCP_CONFIGS_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir() and any(p.rglob("*.yaml")):
            return f"env:{env_dir}"

    cache = get_cache_dir()
    if cache.is_dir() and any(cache.rglob("*.yaml")):
        cached_ver = _read_version_marker()
        return f"cache:{cached_ver}" if cached_ver else "cache:main"

    bundled = get_bundled_configs_dir()
    if bundled.is_dir():
        return f"bundled:{BUNDLED_OUMI_VERSION}"

    return "unknown"


def config_sync(force: bool = False) -> dict:
    """Sync configs from the Oumi repository, matching the installed version.

    Release builds download from the matching Git tag; dev builds use main.
    Skips download if cache is fresh (unless force=True).

    Args:
        force: Sync regardless of cache age.
    """
    if not force and not _is_cache_stale():
        logger.info("Config cache is fresh, skipping sync")
        return {
            "ok": True,
            "skipped": True,
            "error": None,
            "configs_synced": 0,
            "source": get_configs_source(),
        }

    cache_dir = get_cache_dir()
    temp_dir = None
    oumi_ver = get_oumi_version()
    tag = get_oumi_git_tag()

    zip_url, zip_prefix = get_configs_zip_url(tag)
    source_label = f"tag:{tag}" if tag else "main"

    try:
        logger.info(
            "Starting config sync from oumi-ai/oumi (%s, oumi=%s)",
            source_label,
            oumi_ver,
        )
        temp_dir = Path(tempfile.mkdtemp(prefix="oumi_config_sync_"))
        zip_path = temp_dir / "oumi.zip"

        logger.info("Downloading configs from %s", zip_url)
        with httpx.Client(
            follow_redirects=True,
            timeout=CONFIG_SYNC_TIMEOUT_SECONDS,
        ) as client:
            response = client.get(zip_url)

            if response.status_code == 404 and tag:
                logger.warning(
                    "Tag %s not found (404); falling back to main branch", tag
                )
                zip_url, zip_prefix = get_configs_zip_url(None)
                source_label = "main"
                response = client.get(zip_url)

            response.raise_for_status()
            zip_path.write_bytes(response.content)

        logger.info("Downloaded %d bytes from %s", len(response.content), source_label)
        logger.info("Extracting configs from archive")

        archive_root = zip_prefix.split("/")[0]

        with ZipFile(zip_path, "r") as zip_ref:
            config_files = [
                name for name in zip_ref.namelist() if name.startswith(zip_prefix)
            ]

            if not config_files:
                return {
                    "ok": False,
                    "skipped": False,
                    "error": "No configs directory found in repository archive",
                    "configs_synced": 0,
                    "source": source_label,
                }

            for file in config_files:
                zip_ref.extract(file, temp_dir)

        extracted_configs = temp_dir / archive_root / "configs"
        if not extracted_configs.exists():
            return {
                "ok": False,
                "skipped": False,
                "error": f"Extracted configs not found at {extracted_configs}",
                "configs_synced": 0,
                "source": source_label,
            }

        backup_dir = cache_dir.parent / (cache_dir.name + ".backup")
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
        if cache_dir.exists():
            logger.info("Backing up old cache: %s -> %s", cache_dir, backup_dir)
            shutil.move(str(cache_dir), str(backup_dir))
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("Moving new configs to %s", cache_dir)
            shutil.move(str(extracted_configs), str(cache_dir))
        except Exception:
            if backup_dir.exists():
                logger.warning("New cache install failed; restoring backup")
                shutil.move(str(backup_dir), str(cache_dir))
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

        clear_config_caches()

        _touch_sync_marker()
        _write_version_marker(oumi_ver)

        config_count = len(list(cache_dir.rglob("*.yaml")))
        logger.info(
            "Successfully synced %d config files (%s)", config_count, source_label
        )

        return {
            "ok": True,
            "skipped": False,
            "error": None,
            "configs_synced": config_count,
            "source": source_label,
        }

    except httpx.HTTPError as e:
        error_msg = f"Failed to download configs: {e}"
        logger.error(error_msg)
        return {
            "ok": False,
            "skipped": False,
            "error": error_msg,
            "configs_synced": 0,
            "source": source_label,
        }

    except Exception as e:
        error_msg = f"Config sync failed: {e}"
        logger.error(error_msg, exc_info=True)
        return {
            "ok": False,
            "skipped": False,
            "error": error_msg,
            "configs_synced": 0,
            "source": source_label,
        }

    finally:
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning("Failed to clean up temp directory %s: %s", temp_dir, e)


def main() -> None:
    """Run the MCP server.

    On startup:
    1. Attempts to sync configs from GitHub (skipped if cache is fresh).
    2. Falls back to bundled configs if sync fails and no cache exists.
    3. Starts the MCP server.
    """
    _configure_logging()
    _strip_oumi_env_overrides()
    logger.info("Starting Oumi Config MCP Server")

    sync_result = config_sync()
    if sync_result["ok"]:
        if sync_result.get("skipped"):
            logger.info("Using cached configs (still fresh)")
        else:
            logger.info(f"Config sync completed: {sync_result['configs_synced']} files")
    else:
        logger.warning(
            f"Config sync failed: {sync_result['error']}. "
            "Falling back to bundled/cached configs."
        )

    configs_dir = get_configs_dir()
    yaml_count = len(list(configs_dir.rglob("*.yaml"))) if configs_dir.exists() else 0
    logger.info(f"Serving {yaml_count} configs from {configs_dir}")

    if yaml_count == 0:
        logger.error("No configs available. Server may not function correctly.")

    start_background_indexing()

    mcp.run()


if __name__ == "__main__":
    main()
