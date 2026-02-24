"""Oumi MCP Server - ML Training Config Discovery and Execution.

This server helps you set up ML training workflows using Oumi's library of ~500
ready-to-use YAML configs for fine-tuning LLMs (Llama, Qwen, Phi, Gemma, etc.).

Jobs are executed locally via ``subprocess.Popen`` (running the Oumi CLI).

⚠️  CRITICAL: WORKING DIRECTORY AND PATHS
    Jobs run from a different working directory than your project root.
    - **Config file paths** MUST be absolute (e.g., /home/user/config.yaml)
    - **Paths inside configs** (dataset_path, output_dir, etc.) should also be
      absolute, or relative paths will fail with "file not found" errors.
    - Use absolute paths or expand ~ (which gets converted to absolute).
    - Relative paths like "data/train.jsonl" will fail unless the job's cwd
      is set to your project root, which it is not.

GETTING STARTED:
    Call get_started() to see all capabilities and recommended workflow.

TOOLS -- Discovery:
    - get_started: Overview of capabilities and quickstart guide
    - list_categories: Discover available models and config types
    - search_configs: Find training configs by query, task, or model
    - get_config: Get config details and full YAML content
    - validate_config: Validate configs before running
    - pre_flight_check: Catch issues before launch (HF auth, hardware, paths, cloud credentials)
    - get_docs: Search Oumi Python API docs (classes, fields, functions)
      for tool discovery and parameter lookup. Supports exact qualified-name
      match, exact short-name match, and relevance-ranked keyword search.
    - list_modules: List indexed API modules with class/function counts.
      Use this first to discover available namespaces before calling get_docs.

TOOLS -- Execution:
    - run_oumi_job: Execute any Oumi command (local or cloud)
      with dry-run safety and background execution
    - get_job_status: Snapshot status for a running/completed job
    - get_job_logs: Snapshot tail logs for a running/completed job
    - cancel_job: Cancel a running job
    - list_jobs: List running and completed jobs

RESOURCES:
    - guidance://mle-workflow, guidance://mle-train, etc.
    - jobs://running, jobs://completed -- live job listings
    - jobs://{job_id}/logs -- full log output for any job

EXAMPLE WORKFLOW:
    1. get_started()                                                  # See capabilities
    2. list_categories()                                              # See available models
    3. search_configs(model="llama3_1", task="sft")                  # Find recipes
    4. get_config("llama3_1/sft/8b_lora", include_content=True)      # Get YAML
    5. validate_config("/path/to/config.yaml", "training")           # Validate
    6. pre_flight_check("/path/to/config.yaml", cloud="gcp")          # Pre-flight
    7. run_oumi_job("/path/to/config.yaml", "train")                 # Preview (dry-run)
    8. run_oumi_job("/path/to/config.yaml", "train", confirm=True, user_confirmation="EXECUTE")  # Execute
    9. get_job_status("train_20260206_...")                           # Status snapshot
   10. get_job_logs("train_20260206_...", lines=200)                  # Log snapshot
   11. list_jobs(status="running")                                    # See running jobs
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
    _build_cloud_job_config,
    _default_file_mounts,
    _is_job_config,
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
    EVAL_COMMAND_RESOURCE,
    INFER_COMMAND_RESOURCE,
    MLE_WORKFLOW_RESOURCE,
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

mcp = FastMCP("Oumi Config Server")


def _configure_logging() -> None:
    """Reduce noisy third-party INFO logs on stderr in MCP clients."""
    logger.setLevel(logging.INFO)
    for noisy_logger in (
        "mcp.server.lowlevel.server",
        "mcp.server.lowlevel",
        "mcp.shared.session",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _resolve_config_path(config: str) -> tuple[Path, str | None]:
    """Resolve and validate a config file path.

    Requires an absolute path (after ``~`` expansion). Returns a clear error
    when a relative path is given so callers surface an actionable message
    instead of silently resolving against the MCP server's working directory.

    Returns:
        ``(resolved_path, None)`` on success, or ``(Path(), error_message)``
        on failure.
    """
    p = Path(config).expanduser()
    if not p.is_absolute():
        return Path(), (
            f"Path must be absolute, got relative path: '{config}'. "
            f"Provide the full path (e.g. '/home/user/configs/{Path(config).name}')."
        )
    p = p.resolve()
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


@mcp.tool()
def search_configs(
    query: str = "",
    task: str = "",
    model: str = "",
    keyword: str | list[str] = "",
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[ConfigMetadata]:
    """Search the Oumi config library for ML training, evaluation, and inference configs.

    Use this as your first step when looking for example configs. The library contains
    ~500 ready-to-use YAML configs for fine-tuning LLMs (Llama, Qwen, Phi, Gemma, etc.)
    with various techniques (SFT, DPO, GRPO, LoRA, QLoRA).

    All filters are case-insensitive substring matches on the file path. Combine
    multiple filters to narrow results.

    Args:
        query: General search term matching any part of the path.
               Use for size ("8b", "70b"), variant ("instruct"), or technique ("lora", "qlora").
               Space-separated words are all required (AND logic).
        task: Filter by training/task type.
              Options: sft (supervised fine-tuning), dpo (direct preference optimization),
                       kto, grpo (group relative policy optimization), eval, infer, pretrain
        model: Filter by model family.
               Options: llama3_1, llama3_2, llama4, qwen3, phi4, gemma3, deepseek_r1, etc.
        keyword: Case-insensitive substring match on config file content.
                 Pass a list to require all keywords to be present (AND logic).
                 E.g. "packing" or ["packing", "flash_attention"].
        limit: Maximum results to return (default 20).

    Returns:
        List of matching configs, each with:
        - path: Relative path to use with get_config()
        - description: What this config does
        - model_name: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        - task_type: Training type (sft/dpo/grpo/evaluation/inference)
        - datasets: Dataset names used for training
        - peft_type: "lora", "qlora", or null for full fine-tuning

    Examples:
        - search_configs(model="llama3_1", task="sft") -> All Llama 3.1 SFT configs
        - search_configs(query="8b", task="sft") -> All 8B model SFT configs
        - search_configs(query="lora", model="qwen3") -> Qwen3 LoRA configs
        - search_configs(task="grpo") -> All GRPO/RLHF training configs
        - search_configs(query="qlora") -> All QLoRA (quantized LoRA) configs
        - search_configs(keyword="packing") -> Configs mentioning packing in YAML
        - search_configs(keyword=["packing", "flash_attention"]) -> Configs with both
    """
    configs = get_all_configs()
    return search_configs_service(configs, query, task, model, keyword, limit)


@mcp.tool()
def get_config(path: str, include_content: bool = False) -> ConfigDetail:
    """Get detailed information about a specific Oumi config file.

    Use this after search_configs() to get full details about a config, including
    hyperparameters and optionally the complete YAML content. This helps you
    understand the config before using it as a template or running training.

    Args:
        path: Config path from search_configs() results, or a partial path.
              Full path example: "recipes/llama3_1/sft/8b_lora/train.yaml"
              Partial paths work too: "llama3_1/sft/8b_lora" will match.
        include_content: Set True to get the full YAML content for copying/modifying.
                        Default False returns only metadata.

    Returns:
        Dict with full config details:
        - path: Relative path to the config file
        - description: What this config does (from file header)
        - model_name: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        - task_type: Training type (sft/dpo/grpo/evaluation/inference)
        - datasets: List of dataset names used
        - reward_functions: Reward functions for RLHF (empty for non-RLHF)
        - peft_type: "lora", "qlora", or null for full fine-tuning
        - key_settings: Important hyperparameters (learning_rate, batch_size, epochs, etc.)
        - content: Full YAML text (only if include_content=True)
        - error: Error message if config not found

    Examples:
        - get_config("recipes/llama3_1/sft/8b_lora/train.yaml") -> Metadata only
        - get_config("llama3_1/sft/8b_lora", include_content=True) -> With full YAML
        - get_config("qwen3/grpo") -> First matching Qwen3 GRPO config
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
    """List all available config categories, model families, and API providers.

    Use this to discover what's available in the Oumi config library before searching.
    Helpful for understanding the scope of supported models and training approaches.

    Returns:
        Dict with:
        - categories: Top-level config directories
          ["recipes", "apis", "examples", "projects"]
        - model_families: All supported model families in recipes/
          e.g., ["llama3_1", "llama3_2", "llama4", "qwen3", "phi4", "gemma3", ...]
        - api_providers: Available API providers in apis/
          e.g., ["anthropic", "openai", "together", ...]
        - total_configs: Total number of configs in the library (~500)
        - oumi_version: Installed oumi library version
        - configs_source: Where configs are loaded from
        - version_warning: Non-empty if configs may be mismatched with the library

    Examples:
        - list_categories() -> See all available models and categories
        - Then use search_configs(model="llama3_1") to find specific configs
    """
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
def pre_flight_check(config: str, cloud: str = "") -> PreFlightCheckResponse:
    """Run pre-flight checks on a config to catch issues before launching.

    Validates four areas and returns errors (will crash) vs warnings (may be
    fine if you are configuring locally but targeting a remote GPU cluster):

    1. HuggingFace auth — token validity and gated model/dataset access.
    2. Hardware — missing packages (flash-attn, bitsandbytes, deepspeed),
       torch version requirements, GPU presence, and compute capability.
    3. Paths — whether local directories referenced in the config exist.
    4. Cloud readiness — SkyPilot installation and cloud credential
       validation. Performs actual API calls to verify credentials (not just
       file existence). When ``cloud`` is specified, also validates that the
       target cloud provider is ready.

    IMPORTANT: When the response has `blocking=True`, there are hard blockers
    that WILL prevent the run from succeeding. You MUST surface these to the
    user as showstoppers and instruct them to resolve the issues before
    proceeding. Do NOT treat blocking issues as informational notes.

    Args:
        config: Absolute path to the YAML config file.
        cloud: Optional cloud provider to target (e.g. "gcp", "aws", "azure",
               "lambda"). When provided, validates that credentials for this
               cloud are configured and working, and returns ``suggested_configs``
               — a list of config paths from the Oumi library that match the
               model in your config. Use these with ``get_config(path,
               include_content=True)`` to retrieve reference YAML examples you
               can adapt for your cloud job. Leave empty for local runs.

    Returns:
        PreFlightCheckResponse with:
        - blocking: True if there are hard blockers — the run WILL fail.
          When True, you MUST tell the user to fix these before proceeding.
        - summary: One-line verdict to surface to the user.
        - hf_authenticated: True if a valid HuggingFace token was found.
        - repo_access: Per-repo status — "ok", "gated", "not_found", or "error".
        - hardware: Detected accelerator, GPU info, and installed package versions.
        - cloud_readiness: SkyPilot status — which clouds have valid credentials,
          and whether the target cloud (if specified) is ready.
        - errors: Issues that will cause the run to crash (missing packages, etc.).
        - warnings: Issues that may be fine for remote clusters (no local GPU, etc.).
        - paths: Local paths from the config mapped to whether they exist.
        - suggested_configs: (cloud only) Relative paths of similar configs from
          the library. Call ``get_config(path, include_content=True)`` on these
          to get full YAML you can use as a reference or starting point.

    Examples:
        - pre_flight_check("/home/user/train.yaml")
        - pre_flight_check("/home/user/train.yaml", cloud="gcp")
    """
    return _pre_flight_check(config, cloud=cloud)


def _pre_flight_check(config: str, cloud: str = "") -> PreFlightCheckResponse:
    """Run pre-flight checks (internal implementation)."""
    errors: list[str] = []
    warnings: list[str] = []
    repo_access: dict[str, str] = {}

    config_path, path_error = _resolve_config_path(config)
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
        "paths": validate_paths(cfg, config_path.parent),
    }

    if target_cloud:
        model_name = cfg.get("model", {}).get("model_name", "")
        all_cfgs = get_all_configs()
        suggested = search_configs_service(all_cfgs, query=model_name, limit=5)
        result["suggested_configs"] = [c["path"] for c in suggested]

    return result


def _looks_like_hf_repo(val: str) -> bool:
    """Return True if *val* looks like an HF repo ID (org/name)."""
    return bool(val) and val.count("/") == 1 and not val.startswith(("/", ".", "~"))


def validate_paths(cfg: dict, base_dir: Path) -> dict[str, bool]:
    """Extract all local paths from config and validate they exist."""
    paths: dict[str, bool] = {}

    def _extract(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if isinstance(val, str) and key.endswith(
                    ("_dir", "_path", "_file", "_folder")
                ):
                    if _looks_like_hf_repo(val):
                        continue
                    p = Path(val).expanduser()
                    if not p.is_absolute():
                        p = base_dir / p
                        paths[f"{val} (resolved to {p})"] = p.exists()
                    else:
                        paths[val] = p.exists()
                else:
                    _extract(val)
        elif isinstance(obj, list):
            for item in obj:
                _extract(item)

    _extract(cfg)
    return paths


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
    config: str, task_type: ValidatorTaskType
) -> ValidateConfigResponse:
    """Validate an Oumi YAML config file against its schema.

    Use this to check if a config file is valid before running training,
    evaluation, or inference. The validator checks required fields, types,
    and cross-field constraints.

    Args:
        config: Absolute path to the YAML config file.
                Example: "/path/to/train.yaml"
        task_type: The type of config to validate against.
                Options: training, evaluation, inference, tuning, synthesis,
                         quantization, job, judge, analyze, async_evaluation

    Returns:
        Dict with:
        - ok: True if the config is valid, False otherwise.
        - error: Error message describing what's wrong (null if valid).

    Examples:
        - validate_config("/path/to/sft/train.yaml", "training")
        - validate_config("/path/to/eval.yaml", "evaluation")
        - validate_config("/path/to/infer.yaml", "inference")
    """
    config_path, path_error = _resolve_config_path(config)
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
    """Get started with Oumi MCP - returns guidance for ML training workflows.

    Call this first when working with Oumi. Returns a guide explaining
    how to use the available tools for finding and customizing training configs.

    Returns:
        Formatted guide with available tools, workflow steps, and examples.
    """
    return """# Oumi MCP - ML Training Config Server

## Available Tools

### Discovery
| Tool | Purpose | Example |
|------|---------|---------|
| `list_categories()` | See available models & config types | Start here |
| `search_configs(query, task, model, keyword)` | Find training configs | `search_configs(model="llama3_1", task="sft")` |
| `get_config(path, include_content)` | Get config details/YAML | `get_config("llama3_1/sft/8b_lora", include_content=True)` |
| `validate_config(config, task_type)` | Validate before training | `validate_config("/path/to/config.yaml", "training")` |
| `pre_flight_check(config, cloud)` | Catch issues before launch | `pre_flight_check("/path/to/config.yaml", cloud="gcp")` |
| `get_docs(query, module, kind)` | Search Oumi Python API docs | `get_docs(["TrainingConfig"])` |
| `list_modules()` | List indexed API modules | `list_modules()` |

### Execution
| Tool | Purpose | Example |
|------|---------|---------|
| `run_oumi_job(config, cmd)` | Execute Oumi command (dry-run by default) | `run_oumi_job("/path/to/config.yaml", "train")` |
| `get_job_status(job_id)` | Status snapshot (no streaming) | `get_job_status("train_20260206_...")` |
| `get_job_logs(job_id, lines)` | Tail log snapshot | `get_job_logs("train_20260206_...", lines=200)` |
| `cancel_job(job_id)` | Cancel a running job | `cancel_job("train_20260206_...")` |
| `list_jobs()` | List running and completed jobs | `list_jobs(status="running")` |
| `stop_cluster(cloud, cluster_name)` | Stop cluster (preserves infra, reduces compute cost) | `stop_cluster("gcp", "sky-xxxx")` |
| `down_cluster(cloud, cluster_name, confirm, user_confirmation)` | Delete cluster entirely — irreversible | `down_cluster("gcp", "sky-xxxx", confirm=True, user_confirmation="DOWN")` |

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

### Jobs
| Resource | What it contains |
|----------|-----------------|
| `jobs://running` | Currently running jobs (JSON array) |
| `jobs://completed` | Recently finished jobs (JSON array) |
| `jobs://{job_id}/logs` | Full log output for a specific job |

## ⚠️  CRITICAL: Working Directory and Paths

**Jobs run from a different working directory than your project root.**
- **Config file path**: MUST be absolute (e.g., `/home/user/config.yaml` or `~/my_config.yaml`)
- **Paths inside config**: Also use absolute paths (dataset_path, output_dir, validation_path, etc.)
  - ❌ BAD: `dataset_path: pubmedqa/train.jsonl` → will fail with "file not found"
  - ✅ GOOD: `dataset_path: /home/user/data/pubmedqa/train.jsonl`
  - ✅ GOOD: `dataset_path: ~/data/pubmedqa/train.jsonl` (~ is expanded)

For **cloud jobs**, paths inside the job config are relative to `working_dir` (synced to the remote VM).

## ☁️  Cloud Job Workflow (REQUIRED — follow this order)

**When a user asks to run a cloud training job, ALWAYS follow these steps:**

```
Step 1: pre_flight_check("~/my_config.yaml", cloud="gcp")
        # → check credentials, then use suggested_configs paths with get_config() for reference YAMLs
Step 2: [create ~/my_job.yaml based on a reference config or from scratch]
Step 3: [customize: resources.accelerators, working_dir, run, envs, file_mounts]
Step 4: run_oumi_job("~/my_job.yaml", "train")           # dry_run=True (default) — shows full JobConfig
Step 5: run_oumi_job("~/my_job.yaml", "train", dry_run=False, confirm=True, user_confirmation="EXECUTE")
Step 6: get_job_status(job_id)                           # poll status
Step 7: get_job_logs(job_id, lines=200)                  # check logs
Step 8: [when done] stop_cluster("gcp", cluster_name)   # pause OR
         down_cluster("gcp", cluster_name, confirm=True, user_confirmation="DOWN")  # delete
```

**Key fields to customize in your cloud job YAML:**
- `resources.accelerators` — GPU type and count (e.g. `"A100:8"`, `"H100:4"`)
- `working_dir` — absolute path to your local project (synced to remote)
- `run` — your oumi command (path relative to `working_dir`)
- `envs` — API keys (WANDB_API_KEY, HF_TOKEN) that won't be forwarded automatically
- `file_mounts` — credential files to sync (HF token, .netrc auto-included)

**Tip:** `pre_flight_check(config, cloud="gcp")` returns `suggested_configs` — a list of
library config paths for the same model family. Call `get_config(path, include_content=True)`
on any of those paths to get a full YAML reference you can adapt for your cloud job.

## Local Quickstart Workflow

1. **Discover models**: `list_categories()` -> see model_families
2. **Find recipes**: `search_configs(model="llama3_1", task="sft")`
3. **Get YAML**: `get_config("llama3_1/sft/8b_lora", include_content=True)`
4. **Customize**: Modify model_name, datasets, output_dir — use absolute paths
5. **Validate**: `validate_config("/your/config.yaml", "training")`
6. **Preview**: `run_oumi_job("/your/config.yaml", "train")` -> dry-run (default)
7. **Execute**: `run_oumi_job(config, "train", dry_run=False, confirm=True, user_confirmation="EXECUTE")`
8. **Check status**: `get_job_status("train_20260206_...")`
9. **Get logs**: `get_job_logs("train_20260206_...", lines=200)`

## Execution Pattern

```
Step 1 (preview):  run_oumi_job(config, "train")                                    # dry_run=True
Step 2 (execute):  run_oumi_job(config, "train", dry_run=False, confirm=True, user_confirmation="EXECUTE")
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
- `data.train.datasets`: Your dataset name/path
- `training.output_dir`: Where to save checkpoints
- `training.learning_rate`: Start with recipe default
- `training.per_device_train_batch_size`: Adjust for your GPU memory
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
) -> JobSubmissionResponse:
    """Execute an Oumi CLI command with background job tracking.

    **Two-step safety pattern:**
    1. Call with ``dry_run=True`` (default) to preview the execution plan
       without running anything.  Returns model name,
       output directory, and the exact CLI command that would be invoked.
    2. Call with ``dry_run=False, confirm=True`` to actually launch the job.
       You must also provide ``user_confirmation="EXECUTE"`` to explicitly
       authorize execution. The process runs in the background and returns
       a ``job_id`` immediately.
       For cloud runs, a pre-flight check is executed and may block launch.

    **Config type auto-detection:**
    If *config_path* is a launcher job config (contains ``resources``, ``setup``,
    or ``run`` keys at the top level), it is passed directly to ``oumi launch up``
    preserving all cloud-specific fields (``envs``, ``file_mounts``,
    ``storage_mounts``, ``disk_size``, ``use_spot``, etc.).  Otherwise the config
    is treated as a training config and wrapped in a minimal cloud job config,
    enriched with any caller-supplied *envs*, *file_mounts*, and other params.

    **Credential propagation:**
    Common credential files (``~/.cache/huggingface/token``, ``~/.netrc``) are
    automatically mounted on the remote VM when they exist locally, so HuggingFace
    and WandB auth work without manual configuration. Environment variables from
    your local shell are **not** forwarded automatically to cloud jobs; pass them
    explicitly via ``envs``.

    Use ``get_job_status(job_id)`` for a status snapshot.
    Use ``get_job_logs(job_id, lines=...)`` for a bounded log snapshot.
    Use ``cancel_job(job_id)`` to stop a running job.
    Use ``list_jobs()`` to see all running and completed jobs.

    Args:
        config_path: Absolute path to a validated Oumi YAML config file.
                     IMPORTANT: Must be absolute (e.g., /home/user/config.yaml)
                     or expanded with ~ (e.g., ~/my_config.yaml).
                     Relative paths will be rejected.
                     **Also ensure paths inside the config (dataset_path, output_dir, etc.)
                     are absolute**, since the job runs from a different working
                     directory than your project root.
                     If this is already a launcher job config (has ``resources``/
                     ``setup``/``run`` keys), it is passed through directly.
        command: Which Oumi subcommand to run.  One of:
            train, analyze, synth, evaluate, eval, infer, tune, quantize.
            Ignored when *config_path* is a job config (the run script is
            taken from the config itself).
        dry_run: If True (default), validate and return an execution plan
            without running anything.  Always start here.
        confirm: Must be True for actual execution.  Acts as a safety gate
            so the agent cannot accidentally launch GPU-hours of training.
        user_confirmation: Required explicit approval phrase for execution.
            Must be exactly ``"EXECUTE"`` when ``dry_run=False``.
        job_name: Optional human-readable name for the job.  If omitted, a
            name is generated from the command and timestamp.
        cloud: Execution target cloud. ``"local"`` runs locally; any other
            value launches through ``oumi.launcher``.
        cluster_name: Optional cluster name for cloud launches.
        accelerators: Optional accelerator request string for cloud launches
            (e.g. ``"A100:8"``).  Multi-GPU specs automatically enable
            ``oumi distributed torchrun``.
        envs: Environment variables to set on the remote VM
            (e.g. ``{"WANDB_PROJECT": "my-project"}``).
        file_mounts: Additional local→remote file mappings
            (e.g. ``{"~/.ssh/id_rsa": "~/.ssh/id_rsa"}``).
            Common credential files are auto-mounted by default.
        disk_size: Disk size in GB for the remote VM.
        use_spot: Use spot/preemptible instances (cheaper but can be
            preempted).  Default: False.
        num_nodes: Number of nodes for distributed training.  Default: 1.
            Values > 1 automatically enable ``oumi distributed torchrun``.
    Returns:
        JobSubmissionResponse with job_id, status, model info, and either
        an execution plan (dry_run) or a submission confirmation.

    Examples:
        # Step 1: Preview what would happen
        run_oumi_job("/path/to/train.yaml", "train")

        # Step 2: Execute for real
        run_oumi_job(
            "/path/to/train.yaml",
            "train",
            dry_run=False,
            confirm=True,
            user_confirmation="EXECUTE",
        )

        # Step 3: Monitor progress
        get_job_status("train_20260206_143022_a1b2c3")

        # Cloud job with job config passthrough (all cloud fields preserved)
        run_oumi_job(
            "~/configs/gcp_job.yaml",
            "train",
            cloud="gcp",
            dry_run=False,
            confirm=True,
            user_confirmation="EXECUTE",
        )

        # Cloud job with multi-GPU distributed training
        run_oumi_job(
            "~/configs/train.yaml",
            "train",
            cloud="gcp",
            accelerators="A100:8",
            envs={"WANDB_PROJECT": "my-run"},
            dry_run=False,
            confirm=True,
            user_confirmation="EXECUTE",
        )
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

    config_file, path_error = _resolve_config_path(config_path)
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

    num_gpus_preview = 0
    if accelerators:
        try:
            num_gpus_preview = (
                int(accelerators.split(":")[-1]) if ":" in accelerators else 1
            )
        except (ValueError, IndexError):
            num_gpus_preview = 1

    if dry_run:
        if is_job_config_file:
            cmd_preview = f"oumi launch up -c {abs_config}"
        elif num_gpus_preview > 1 or num_nodes > 1:
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
            else:
                preview_mounts = _default_file_mounts()
                if file_mounts:
                    preview_mounts.update(file_mounts)
                preview_job_cfg = _build_cloud_job_config(
                    "config.yaml",
                    command,
                    cloud=cloud,
                    working_dir="<staging dir set at launch>",
                    accelerators=accelerators or None,
                    job_name=job_id,
                    envs=envs,
                    file_mounts=preview_mounts,
                    disk_size=disk_size,
                    use_spot=use_spot,
                    num_nodes=num_nodes,
                )
                job_cfg_yaml = _jobconfig_to_yaml(preview_job_cfg)
                env_warning = _build_missing_env_warning(envs)
                if env_warning:
                    message = message + env_warning
            message = (
                message
                + "\n\n--- Generated JobConfig (review before executing) ---\n"
                + job_cfg_yaml
                + "-----------------------------------------------------"
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
        preflight = _pre_flight_check(abs_config, cloud=cloud)
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

            num_gpus_for_check = (
                int(accelerators.split(":")[-1])
                if accelerators and ":" in accelerators
                else (1 if accelerators else 0)
            )
            if num_gpus_for_check > 1 or num_nodes > 1:
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
        status="running",
        submit_time=submit_time,
    )
    reg = get_registry()
    reg.add(record)

    rt = get_runtime(job_id)
    rt.log_dir = JOB_LOGS_DIR / job_id
    rt.run_dir = JOB_RUNS_DIR / job_id

    is_local = cloud == "local"
    if is_local:
        try:
            start_local_job(record, rt)
        except Exception as exc:
            rt.error_message = str(exc)
            reg.update(record.job_id, status="failed")
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
                accelerators=accelerators or None,
                envs=envs,
                file_mounts=file_mounts,
                disk_size=disk_size,
                use_spot=use_spot,
                num_nodes=num_nodes,
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

    # Re-read record in case cloud launch updated it
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
        return "cancel_requested"
    is_local = record.cloud == "local"
    if is_local:
        proc = rt.process
        if proc is None:
            if record.status in {"running", "launching"}:
                return "launching"
            return "failed" if rt.error_message else record.status
        rc = proc.poll()
        if rc is None:
            return "running"
        if rt.error_message and "Cancelled" in rt.error_message:
            return "cancelled"
        return "completed" if rc == 0 else "failed"
    if rt.oumi_status:
        return rt.oumi_status.status
    if rt.error_message:
        return "failed"
    return record.status


def _is_job_done(record: JobRecord, rt: JobRuntime) -> bool:
    """Return True if the job is in a terminal state."""
    if record.status in {"completed", "failed"}:
        return True
    is_local = record.cloud == "local"
    if is_local and rt.process is not None:
        return rt.process.poll() is not None
    if rt.oumi_status and rt.oumi_status.done:
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
    """Build normalized job summaries for tools and resources."""
    reg = get_registry()
    records = reg.all()
    if records:
        await asyncio.gather(
            *(poll_status(r, get_runtime(r.job_id)) for r in records),
            return_exceptions=True,
        )

    if status_filter == "running":
        records = [r for r in records if not _is_job_done(r, get_runtime(r.job_id))]
    elif status_filter == "completed":
        records = [r for r in records if _is_job_done(r, get_runtime(r.job_id))]

    return [
        {
            "job_id": r.job_id,
            "command": r.command,
            "status": _job_status_str(r, get_runtime(r.job_id)),
            "cloud": r.cloud,
            "cluster": r.cluster_name,
            "model_name": r.model_name,
            "is_done": _is_job_done(r, get_runtime(r.job_id)),
        }
        for r in records
    ]


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
        if oumi_job_id and cloud:
            cluster_hint = cluster_name or "<cluster>"
            return {
                "success": False,
                "job_id": job_id or oumi_job_id,
                "lines_requested": lines,
                "lines_returned": 0,
                "log_file": "",
                "logs": "",
                "error": (
                    "No MCP-managed local log file for this cloud job identity. "
                    f"Use `sky logs {cluster_hint}` to stream logs directly."
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

    For local jobs, sends SIGTERM to the subprocess.  Use ``force=True``
    to send SIGKILL instead if the process won't stop.

    For cloud jobs, delegates to ``oumi.launcher.cancel()``.

    Args:
        job_id: The MCP job ID to cancel (preferred).
        force: If True, send SIGKILL instead of SIGTERM (local jobs only).
        oumi_job_id: Optional cluster-side job ID for direct cloud cancellation.
        cloud: Cloud provider name when using ``oumi_job_id``.
        cluster_name: Optional cluster name when using ``oumi_job_id``.

    Returns:
        Dict with success status and message/error.

    Examples:
        # Graceful cancellation
        cancel_job("train_20260206_143022_a1b2c3")

        # Force kill
        cancel_job("train_20260206_143022_a1b2c3", force=True)
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

    Stopped clusters keep their storage but no longer consume compute resources.
    They can be restarted by submitting a new job with the same ``cluster_name``.

    Use ``down_cluster`` to fully delete the cluster and stop all billing.

    Get the cluster name from ``get_job_status(job_id)["cluster"]`` or ``list_jobs()``.

    Args:
        cloud: Cloud provider (e.g. ``"gcp"``, ``"aws"``, ``"azure"``).
        cluster_name: Name of the cluster to stop.

    Returns:
        ClusterLifecycleResponse with ``success`` and ``message`` or ``error``.
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
    """Delete a cluster and all its resources. This is irreversible.

    WARNING: All data on the cluster is permanently deleted and billing stops
    immediately. This cannot be undone. Use ``stop_cluster`` to pause instead.

    Requires ``confirm=True`` and ``user_confirmation="DOWN"`` to execute.
    Call without these to see a dry-run description of what would be deleted.

    Get the cluster name from ``get_job_status(job_id)["cluster"]`` or ``list_jobs()``.

    Args:
        cloud: Cloud provider (e.g. ``"gcp"``, ``"aws"``, ``"azure"``).
        cluster_name: Name of the cluster to delete.
        confirm: Must be ``True`` for actual deletion.
        user_confirmation: Must be exactly ``"DOWN"`` to authorize deletion.

    Returns:
        ClusterLifecycleResponse with ``success`` and ``message`` or ``error``.
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

    Use this to see what jobs have been submitted, their current status,
    and whether they are still running or have finished.

    Args:
        status: Filter by job status.  One of:
            ``"all"`` (default) -- all jobs,
            ``"running"`` -- only in-progress jobs,
            ``"completed"`` -- only finished jobs.

    Returns:
        List of job summaries with job_id, command, status, cloud,
        cluster, model_name, and is_done.

    Examples:
        list_jobs()                  # All jobs
        list_jobs(status="running")  # Only running jobs
        list_jobs(status="completed")  # Only finished jobs
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
    """Search Oumi's indexed Python API docs for agent tool discovery.

    This tool is optimized for agents that need to discover API surface area
    quickly and with minimal context switching. It searches an in-memory index
    of classes, dataclasses, functions, and methods across the modules listed
    by ``list_modules()``.

    Matching strategy (in order):
    1. Exact qualified name match
       Example: ``oumi.core.configs.params.training_params.TrainingParams``
    2. Exact short-name match (case-insensitive)
       Example: ``TrainingParams``
    3. Relevance-ranked keyword search over names, dataclass fields, summaries,
       and docstring sections
       Example: ``learning_rate``

    Agent discovery workflow:
    1. Call ``list_modules()`` to discover indexed namespaces.
    2. Call ``get_docs(query, module=..., kind=...)`` to narrow results.
    3. Read ``signature``, ``fields``, and ``sections`` from returned entries
       to infer valid parameters and usage patterns.

    Args:
        query: Search terms. Include one or more exact qualified names,
               short names, or keywords. Examples:
               ["oumi.core.configs.TrainingConfig"], ["TrainingConfig"],
               ["learning_rate", "lora"].
        module: Optional module prefix filter (e.g. "oumi.core.configs").
        kind: Optional kind filter: "class", "dataclass", "function", or "method".
        limit: Maximum number of results to return (default 10).
        summarize: If True, return compact entries that focus on high-level
            metadata and summary text (omits fields and docstring sections).

    Returns:
        DocsSearchResponse with:
        - results: Matching documentation entries with name, kind, summary,
          fields (for dataclasses), signature, and parsed docstring sections.
        - query: The normalized query terms used for this search.
        - total_matches: Total matches before limiting.
        - index_ready: False if background indexing hasn't finished yet.
        - error: Error message if something went wrong.

    Examples:
        - get_docs(["TrainingConfig"]) -> Full TrainingConfig docs with fields
        - get_docs(["ModelParams"]) -> ModelParams with per-field docstrings
        - get_docs(["learning_rate", "lora"]) -> Multi-keyword search
        - get_docs(["lora"], module="oumi.core.configs") -> Filtered search
        - get_docs(["infer"], kind="function") -> Kind-filtered search
    """
    return search_docs(
        query=query, module=module, kind=kind, limit=limit, summarize=summarize
    )


@mcp.tool()
def list_modules() -> ListModulesResponse:
    """List modules currently indexed for ``get_docs`` tool discovery.

    This is the entry point for documentation discovery. It returns the module
    namespaces available to ``get_docs`` plus light inventory metadata so
    agents can choose a narrow search scope.

    Recommended usage:
    1. Call ``list_modules()`` once to discover candidate namespaces.
    2. Select one module prefix (for example ``oumi.core.configs``).
    3. Call ``get_docs(query, module=<prefix>)`` for precise retrieval.
    4. If no results, broaden to fewer filters or a higher-level module prefix.

    Returns:
        ListModulesResponse with:
        - modules: Per-module summaries (module path, description, counts).
        - total_entries: Total number of indexed documentation entries.
        - index_ready: Whether background indexing has completed.

    Examples:
        - list_modules() -> See all indexed modules with class counts
        - Then use get_docs(["TrainingConfig"]) to look up specific classes
    """
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

    For release builds (e.g. oumi 0.7), downloads configs from the ``v0.7``
    Git tag so they stay compatible with the installed library.  For dev
    builds (e.g. ``0.8.dev35+g...``), downloads from the ``main`` branch.
    Falls back to ``main`` when the tag archive returns a 404.

    The sync process:
    1. Check if cache is stale (skip download if fresh, unless force=True)
    2. Resolve the correct Git tag (or main) for the installed oumi version
    3. Download the archive and extract only the configs/ directory
    4. Write timestamp and version markers for staleness tracking
    5. Clean up temporary files

    Args:
        force: If True, sync regardless of cache age.

    Returns:
        Dict with:
        - ok: True if sync succeeded or cache is fresh
        - skipped: True if sync was skipped (cache is fresh)
        - error: Error message if sync failed, None otherwise
        - configs_synced: Number of config files synced (0 if skipped)
        - source: Description of where configs came from
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
