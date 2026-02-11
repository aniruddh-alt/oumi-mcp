"""Oumi MCP Server - ML Training Config Discovery and Execution.

This server helps you set up ML training workflows using Oumi's library of ~500
ready-to-use YAML configs for fine-tuning LLMs (Llama, Qwen, Phi, Gemma, etc.).

Jobs are executed locally via ``subprocess.Popen`` (running the Oumi CLI).

GETTING STARTED:
    Call get_started() to see all capabilities and recommended workflow.

TOOLS -- Discovery:
    - get_started: Overview of capabilities and quickstart guide
    - list_categories: Discover available models and config types
    - search_configs: Find training configs by query, task, or model
    - get_config: Get config details and full YAML content
    - validate_config: Validate configs before running
    - pre_flight_check: Catch issues before launch (HF auth, hardware, paths)
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
    6. pre_flight_check("/path/to/config.yaml")                      # Pre-flight
    7. run_oumi_job("/path/to/config.yaml", "train")                 # Preview (dry-run)
    8. run_oumi_job("/path/to/config.yaml", "train", confirm=True, user_confirmation="EXECUTE")  # Execute
    9. get_job_status("train_20260206_...")                           # Status snapshot
   10. get_job_logs("train_20260206_...", lines=200)                  # Log snapshot
   11. list_jobs(status="running")                                    # See running jobs
"""

import asyncio
import json
import logging
import shutil
import sys
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import httpx
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
    extract_key_settings,
    find_config_match,
    get_all_configs,
    get_categories,
    get_configs_dir,
    parse_yaml,
)
from oumi_mcp_server.config_service import (
    search_configs as search_configs_service,
)
from oumi_mcp_server.constants import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_STREAM_LINES,
    HARDWARE_PACKAGES,
    HF_API_TIMEOUT_SECONDS,
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
    cancel,
    extract_job_metadata,
    get_log_paths,
    launch_job,
    make_job_id,
    poll_status,
    registry,
    start_local_job,
    wait_local_completion,
)
from oumi_mcp_server.models import (
    CategoriesResponse,
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
)
from oumi_mcp_server.prompts.mle_prompt import (
    ANALYZE_COMMAND_RESOURCE,
    EVAL_COMMAND_RESOURCE,
    INFER_COMMAND_RESOURCE,
    MLE_WORKFLOW_RESOURCE,
    SYNTH_COMMAND_RESOURCE,
    TRAIN_COMMAND_RESOURCE,
)


def get_package_version(package_name: str) -> str | None:
    """Return the installed version string for *package_name*, or None."""
    try:
        return _pkg_version(package_name)
    except PackageNotFoundError:
        return None


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
        import torch

        if torch.cuda.is_available():
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
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["accelerator_type"] = "mps"
            info["accelerator_count"] = 1
    except Exception:
        pass
    return info


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

mcp = FastMCP("Oumi Config Server")


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
    keyword: str = "",
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
        task: Filter by training/task type.
              Options: sft (supervised fine-tuning), dpo (direct preference optimization),
                       kto, grpo (group relative policy optimization), eval, infer, pretrain
        model: Filter by model family.
               Options: llama3_1, llama3_2, llama4, qwen3, phi4, gemma3, deepseek_r1, etc.
        keyword: Case-insensitive substring match on config file content.
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

    Examples:
        - list_categories() -> See all available models and categories
        - Then use search_configs(model="llama3_1") to find specific configs
    """
    configs_dir = get_configs_dir()
    configs = get_all_configs()
    return get_categories(configs_dir, len(configs))


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
def pre_flight_check(config: str) -> PreFlightCheckResponse:
    """Run pre-flight checks on a config to catch issues before launching training.

    Validates three areas and returns errors (will crash) vs warnings (may be
    fine if you are configuring locally but targeting a remote GPU cluster):

    1. HuggingFace auth — token validity and gated model/dataset access.
    2. Hardware — missing packages (flash-attn, bitsandbytes, deepspeed),
       torch version requirements, GPU presence, and compute capability.
    3. Paths — whether local directories referenced in the config exist.

    IMPORTANT: When the response has `blocking=True`, there are hard blockers
    that WILL prevent the run from succeeding. You MUST surface these to the
    user as showstoppers and instruct them to resolve the issues before
    proceeding. Do NOT treat blocking issues as informational notes.

    Args:
        config: Absolute path to the YAML config file.

    Returns:
        PreFlightCheckResponse with:
        - blocking: True if there are hard blockers — the run WILL fail.
          When True, you MUST tell the user to fix these before proceeding.
        - summary: One-line verdict to surface to the user.
        - hf_authenticated: True if a valid HuggingFace token was found.
        - repo_access: Per-repo status — "ok", "gated", "not_found", or "error".
        - hardware: Detected accelerator, GPU info, and installed package versions.
        - errors: Issues that will cause the run to crash (missing packages, etc.).
        - warnings: Issues that may be fine for remote clusters (no local GPU, etc.).
        - paths: Local paths from the config mapped to whether they exist.

    Examples:
        - pre_flight_check("/home/user/train.yaml")
        - pre_flight_check("/workspace/configs/llama3_sft.yaml")
    """
    return _pre_flight_check(config)


def _pre_flight_check(config: str) -> PreFlightCheckResponse:
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
            "errors": errors,
            "warnings": [],
            "paths": {},
        }

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if not cfg:
        errors.append("Config file is empty or invalid YAML")
        return {
            "blocking": True,
            "summary": "BLOCKED: Config file is empty or invalid YAML",
            "hf_authenticated": False,
            "repo_access": {},
            "hardware": _empty_hardware(),
            "errors": errors,
            "warnings": [],
            "paths": {},
        }

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

    return {
        "blocking": is_blocking,
        "summary": summary,
        "hf_authenticated": hf_authenticated,
        "repo_access": repo_access,
        "hardware": hardware,
        "errors": errors,
        "warnings": warnings,
        "paths": validate_paths(cfg, config_path.parent),
    }


def validate_paths(cfg: dict, base_dir: Path) -> dict[str, bool]:
    """Extract all local paths from config and validate they exist."""
    paths: dict[str, bool] = {}

    def _looks_like_hf_repo(val: str) -> bool:
        """Return True if val looks like an HF repo ID (org/name)."""
        return val.count("/") == 1 and not val.startswith(("/", ".", "~"))

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

    def _looks_like_hf_repo(val: str) -> bool:
        """Return True if val looks like an HF repo ID (org/name)."""
        return bool(val) and val.count("/") == 1 and not val.startswith("/")

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
def validate_config(config: str, task_type: ValidatorTaskType) -> dict:
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
| `pre_flight_check(config)` | Catch issues before launch (HF auth, hardware, paths) | `pre_flight_check("/path/to/config.yaml")` |
| `get_docs(query, module, kind)` | Search Oumi Python API docs | `get_docs("TrainingConfig")` |
| `list_modules()` | List indexed API modules | `list_modules()` |

### Execution
| Tool | Purpose | Example |
|------|---------|---------|
| `run_oumi_job(config, cmd)` | Execute Oumi command (dry-run by default) | `run_oumi_job("/path/to/config.yaml", "train")` |
| `get_job_status(job_id)` | Status snapshot (no streaming) | `get_job_status("train_20260206_...")` |
| `get_job_logs(job_id, lines)` | Tail log snapshot with requested lines | `get_job_logs("train_20260206_...", lines=200)` |
| `cancel_job(job_id)` | Cancel a running job | `cancel_job("train_20260206_...")` |
| `list_jobs()` | List running and completed jobs | `list_jobs(status="running")` |

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

## Quickstart Workflow

1. **Discover models**: `list_categories()` -> see model_families
2. **Find recipes**: `search_configs(model="llama3_1", task="sft")`
3. **Get YAML**: `get_config("llama3_1/sft/8b_lora", include_content=True)`
4. **Customize**: Modify model_name, datasets, output_dir for your use case
5. **Validate**: `validate_config("/your/config.yaml", "training")`
6. **Pre-flight**: `pre_flight_check("/your/config.yaml")` -> verify HF auth, hardware, paths
7. **Preview**: `run_oumi_job("/your/config.yaml", "train")` -> dry-run (default)
8. **Execute**: `run_oumi_job(config, "train", dry_run=False, confirm=True, user_confirmation="EXECUTE")`
9. **Check status**: `get_job_status("train_20260206_...")` -> single snapshot
10. **Get logs**: `get_job_logs("train_20260206_...", lines=200)` -> bounded tail
11. **List running**: `list_jobs(status="running")`

## Execution Pattern

The execution tools follow a **two-step safety pattern**:

```
Step 1 (preview):  run_oumi_job(config, "train")                                    # dry_run=True
Step 2 (execute):  run_oumi_job(config, "train", dry_run=False, confirm=True, user_confirmation="EXECUTE")  # launch
Step 3 (status):   get_job_status(job_id)                                            # snapshot
Step 4 (logs):     get_job_logs(job_id, lines=200)                                   # tail snapshot
Step 5 (cancel):   cancel_job(job_id)                                                # if needed
Step 5b (force):   cancel_job(job_id, force=True)                                    # SIGKILL
```

Jobs are launched locally or on cloud and return immediately
with a job ID.

`get_job_status()` and `get_job_logs()` are snapshot reads, so they do
not hold open streaming sessions.

Log files are written to `~/.cache/oumi-mcp/job-logs/{job_id}/` and
can also be accessed via the `jobs://{job_id}/logs` resource.

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

    Use ``get_job_status(job_id)`` for a status snapshot.
    Use ``get_job_logs(job_id, lines=...)`` for a bounded log snapshot.
    Use ``cancel_job(job_id)`` to stop a running job.
    Use ``list_jobs()`` to see all running and completed jobs.

    Args:
        config_path: Absolute path to a validated Oumi YAML config file.
        command: Which Oumi subcommand to run.  One of:
            train, analyze, synth, evaluate, eval, infer, tune, quantize.
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
        accelerators: Optional accelerator request for cloud launches.

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
            "output_dir": "",
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
    try:
        meta = extract_job_metadata(abs_config)
        model_name = meta["model_name"]
        output_dir = meta["output_dir"]
    except Exception as exc:
        return _error_response(
            f"Failed to parse config metadata: {exc}",
            config_path=abs_config,
        )

    job_id = make_job_id(command, job_name)

    if dry_run:
        try:
            parse_yaml(abs_config)
        except Exception as exc:
            return _error_response(
                f"Invalid YAML config: {exc}",
                job_id=job_id,
                config_path=abs_config,
                model_name=model_name,
                output_dir=output_dir,
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
            "output_dir": output_dir,
            "message": (
                f"Dry run: would execute `oumi {command} -c {abs_config}` on {cloud}\n"
                f"Model: {model_name}\n"
                f"Output: {output_dir}\n"
                "To execute, re-call with dry_run=False, confirm=True, "
                "user_confirmation='EXECUTE'."
            ),
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
            "output_dir": output_dir,
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
        preflight = _pre_flight_check(abs_config)
        preflight_summary = preflight.get("summary", "")
        preflight_blocking = bool(preflight.get("blocking"))
        preflight_errors = preflight.get("errors", []) or []
        preflight_warnings = preflight.get("warnings", []) or []
        if preflight_blocking:
            return _error_response(
                f"Pre-flight checks failed: {preflight_summary}",
                status="blocked",
                job_id=job_id,
                config_path=abs_config,
                model_name=model_name,
                output_dir=output_dir,
                preflight_summary=preflight_summary,
                preflight_blocking=preflight_blocking,
                preflight_errors=preflight_errors,
                preflight_warnings=preflight_warnings,
            )

    record = JobRecord(
        job_id=job_id,
        command=command,
        config_path=abs_config,
        cloud=cloud,
        cluster_name=cluster_name,
        model_name=model_name,
        output_dir=output_dir,
    )
    try:
        await registry.register(record)
    except ValueError as exc:
        return _error_response(
            str(exc),
            job_id=job_id,
            config_path=abs_config,
            model_name=model_name,
            output_dir=output_dir,
        )

    if record.is_local:
        try:
            start_local_job(record)
        except Exception as exc:
            record.error_message = str(exc)
            return _error_response(
                f"Failed to start job: {exc}",
                job_id=job_id,
                config_path=abs_config,
                model_name=model_name,
                output_dir=output_dir,
            )
        runner = asyncio.create_task(
            wait_local_completion(record),
            name=f"oumi-job-{job_id}",
        )
    else:
        runner = asyncio.create_task(
            launch_job(record, accelerators=accelerators or None),
            name=f"oumi-job-{job_id}",
        )
    record.runner_task = runner

    launch_confirmed = False
    if not record.is_local:
        try:
            await asyncio.wait_for(asyncio.shield(runner), timeout=10.0)
            launch_confirmed = record.error_message is None
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

    if record.error_message and not record.is_local:
        return _error_response(
            f"Failed to launch cloud job: {record.error_message}",
            status="failed",
            job_id=job_id,
            config_path=abs_config,
            model_name=model_name,
            output_dir=output_dir,
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
    if not record.is_local and not launch_confirmed:
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
        "output_dir": output_dir,
        "launch_confirmed": launch_confirmed if not record.is_local else True,
        "preflight_summary": preflight_summary,
        "preflight_blocking": preflight_blocking,
        "preflight_errors": preflight_errors,
        "preflight_warnings": preflight_warnings,
        "oumi_job_id": record.oumi_job_id,
        "cluster": record.cluster_name,
        "message": message,
    }


def _local_status_str(record: JobRecord) -> str:
    """Derive a human-readable status string for a local job."""
    proc = record.process
    if proc is None:
        return "launching" if not record.error_message else "failed"
    rc = proc.poll()
    if rc is None:
        return "running"
    if record.error_message and "Cancelled" in record.error_message:
        return "cancelled"
    return "completed" if rc == 0 else "failed"


def _build_status_response(
    record: JobRecord,
    *,
    log_file: str = "",
) -> JobStatusResponse:
    """Build a ``JobStatusResponse`` from a ``JobRecord``."""
    status = record.oumi_status

    if record.is_local:
        status_str = _local_status_str(record)
        oumi_job_id = record.oumi_job_id
        state_str = status_str.upper()
        cluster_str = "local"
    else:
        status_str = (
            status.status
            if status
            else ("launching" if not record.error_message else "failed")
        )
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
        "is_done": record.is_done,
        "error": record.error_message,
    }

    if status and status.metadata:
        base["metadata"] = status.metadata
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
        "error": f"Job '{job_id}' not found. Check the ID and try again.",
    }


@mcp.tool()
async def get_job_status(
    job_id: str,
) -> JobStatusResponse:
    """Return a single status snapshot for an Oumi job."""
    record = await registry.get(job_id)
    if not record:
        return _not_found_response(job_id)

    await poll_status(record)
    log_paths = get_log_paths(record)
    return _build_status_response(
        record,
        log_file=str(log_paths["stdout"]) if log_paths["stdout"] else "",
    )


def _read_log_tail(stdout_path: Path, lines: int) -> tuple[str, int]:
    """Read the trailing *lines* from *stdout_path* efficiently."""
    if lines <= 0:
        return ("", 0)

    with stdout_path.open("r", encoding="utf-8", errors="replace") as f:
        tail = deque(f, maxlen=lines)

    if not tail:
        return ("", 0)

    content = "".join(tail)
    if tail[-1].endswith("\n"):
        content = content.rstrip("\n")
    return (content, len(tail))


@mcp.tool()
async def get_job_logs(
    job_id: str,
    lines: int = DEFAULT_STREAM_LINES,
) -> JobLogsResponse:
    """Return a bounded log snapshot for an Oumi job."""
    if lines < 0:
        lines = 0

    record = await registry.get(job_id)
    if not record:
        return {
            "success": False,
            "job_id": job_id,
            "lines_requested": lines,
            "lines_returned": 0,
            "log_file": "",
            "logs": "",
            "error": f"Job '{job_id}' not found.",
        }

    await poll_status(record)
    log_paths = get_log_paths(record)
    stdout_path = log_paths.get("stdout")

    if not stdout_path or not stdout_path.exists():
        return {
            "success": False,
            "job_id": job_id,
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
            "job_id": job_id,
            "lines_requested": lines,
            "lines_returned": 0,
            "log_file": str(stdout_path),
            "logs": "",
            "error": f"Failed to read log file: {exc}",
        }

    return {
        "success": True,
        "job_id": job_id,
        "lines_requested": lines,
        "lines_returned": lines_returned,
        "log_file": str(stdout_path),
        "logs": logs,
        "error": None,
    }


@mcp.tool()
async def cancel_job(
    job_id: str,
    force: bool = False,
) -> JobCancelResponse:
    """Cancel a running or pending Oumi job.

    For local jobs, sends SIGTERM to the subprocess.  Use ``force=True``
    to send SIGKILL instead if the process won't stop.

    For cloud jobs, delegates to ``oumi.launcher.cancel()``.

    Args:
        job_id: The job ID to cancel.
        force: If True, send SIGKILL instead of SIGTERM (local jobs only).

    Returns:
        Dict with success status and message/error.

    Examples:
        # Graceful cancellation
        cancel_job("train_20260206_143022_a1b2c3")

        # Force kill
        cancel_job("train_20260206_143022_a1b2c3", force=True)
    """
    record = await registry.get(job_id)

    if not record:
        return {
            "success": False,
            "error": f"Job '{job_id}' not found.",
        }

    return await cancel(record, force=force)


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
    records = await registry.all_jobs()
    if records:
        await asyncio.gather(*(poll_status(r) for r in records), return_exceptions=True)

    if status == "running":
        records = [r for r in records if not r.is_done]
    elif status == "completed":
        records = [r for r in records if r.is_done]

    def _status_str(r: JobRecord) -> str:
        if r.is_local:
            proc = r.process
            if proc is None:
                return "launching" if not r.error_message else "failed"
            rc = proc.poll()
            if rc is None:
                return "running"
            if r.error_message and "Cancelled" in r.error_message:
                return "cancelled"
            return "completed" if rc == 0 else "failed"
        return r.oumi_status.status if r.oumi_status else "launching"

    return [
        {
            "job_id": r.job_id,
            "command": r.command,
            "status": _status_str(r),
            "cloud": r.cloud,
            "cluster": r.cluster_name,
            "model_name": r.model_name,
            "is_done": r.is_done,
        }
        for r in records
    ]


@mcp.tool()
def get_docs(
    query: str,
    module: str = "",
    kind: str = "",
    limit: int = 10,
    examples: bool = False,
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
        query: Search term. Can be an exact qualified name
               (e.g. "oumi.core.configs.TrainingConfig"), a short class name
               (e.g. "TrainingConfig"), or a keyword (e.g. "learning_rate").
        module: Optional module prefix filter (e.g. "oumi.core.configs").
        kind: Optional kind filter: "class", "dataclass", "function", or "method".
        limit: Maximum number of results to return (default 10).

    Returns:
        DocsSearchResponse with:
        - results: Matching documentation entries with name, kind, summary,
          fields (for dataclasses), signature, and parsed docstring sections.
        - query: The original search query.
        - total_matches: Total matches before limiting.
        - index_ready: False if background indexing hasn't finished yet.
        - error: Error message if something went wrong.

    Examples:
        - get_docs("TrainingConfig") -> Full TrainingConfig docs with fields
        - get_docs("ModelParams") -> ModelParams with per-field docstrings
        - get_docs("learning_rate") -> Find fields named learning_rate
        - get_docs("lora", module="oumi.core.configs") -> Filtered search
        - get_docs("infer", kind="function") -> Kind-filtered search
    """
    return search_docs(query=query, module=module, kind=kind, limit=limit)


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
        - Then use get_docs("TrainingConfig") to look up specific classes
    """
    return get_module_list()


@mcp.resource("jobs://running", mime_type="application/json")
async def list_running_jobs() -> str:
    """List all currently running Oumi jobs.

    Returns a JSON array of job summaries with job_id, command, status,
    cloud, cluster, model_name, and is_done.
    """
    records = await registry.all_jobs()
    if records:
        await asyncio.gather(*(poll_status(r) for r in records), return_exceptions=True)
    records = [r for r in records if not r.is_done]
    summaries: list[JobSummary] = [
        {
            "job_id": r.job_id,
            "command": r.command,
            "status": _local_status_str(r)
            if r.is_local
            else (r.oumi_status.status if r.oumi_status else "launching"),
            "cloud": r.cloud,
            "cluster": r.cluster_name,
            "model_name": r.model_name,
            "is_done": False,
        }
        for r in records
    ]
    return json.dumps(summaries, indent=2)


@mcp.resource("jobs://completed", mime_type="application/json")
async def list_completed_jobs() -> str:
    """List recently completed, failed, or cancelled Oumi jobs.

    Returns a JSON array of job summaries with job_id, command, status,
    cloud, cluster, model_name, and is_done.
    """
    records = await registry.all_jobs()
    if records:
        await asyncio.gather(*(poll_status(r) for r in records), return_exceptions=True)
    records = [r for r in records if r.is_done]
    summaries: list[JobSummary] = [
        {
            "job_id": r.job_id,
            "command": r.command,
            "status": _local_status_str(r)
            if r.is_local
            else (r.oumi_status.status if r.oumi_status else "unknown"),
            "cloud": r.cloud,
            "cluster": r.cluster_name,
            "model_name": r.model_name,
            "is_done": True,
        }
        for r in records
    ]
    return json.dumps(summaries, indent=2)


@mcp.resource("jobs://{job_id}/logs", mime_type="text/plain")
async def get_job_logs_resource(job_id: str) -> str:
    """Full log output for a specific job.

    For local jobs, returns the contents of the stdout log file on disk.
    For cloud jobs, returns metadata about how to access logs
    (e.g. via ``sky logs``).
    """
    record = await registry.get(job_id)
    if not record:
        return json.dumps({"error": f"Job '{job_id}' not found"})

    status = await poll_status(record)

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
        if record.error_message:
            header_parts.append(f"Error: {record.error_message}")
        else:
            header_parts.append("Status: launching (no status available yet)")

    header = "\n".join(header_parts)

    log_paths = get_log_paths(record)
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


def _is_cache_stale() -> bool:
    """Check whether the cached configs need to be refreshed.

    Returns True if the cache directory doesn't exist, has no YAML files,
    or the last sync was more than CONFIGS_SYNC_INTERVAL_HOURS ago.
    """
    from oumi_mcp_server.config_service import get_cache_dir
    from oumi_mcp_server.constants import (
        CONFIGS_SYNC_INTERVAL_HOURS,
        CONFIGS_SYNC_MARKER,
    )

    cache_dir = get_cache_dir()
    marker = cache_dir / CONFIGS_SYNC_MARKER

    if not cache_dir.exists() or not any(cache_dir.rglob("*.yaml")):
        return True

    if not marker.exists():
        return True

    try:
        import time

        age_hours = (time.time() - marker.stat().st_mtime) / 3600
        return age_hours > CONFIGS_SYNC_INTERVAL_HOURS
    except Exception:
        return True


def _touch_sync_marker() -> None:
    """Write/update the sync timestamp marker file."""
    from oumi_mcp_server.config_service import get_cache_dir
    from oumi_mcp_server.constants import CONFIGS_SYNC_MARKER

    marker = get_cache_dir() / CONFIGS_SYNC_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("")


def config_sync(force: bool = False) -> dict:
    """Sync configs from the Oumi main repository.

    Downloads the latest configs from the Oumi GitHub repository main branch
    and updates the local cache directory. Uses a staleness check to avoid
    redundant downloads -- configs are only re-fetched if the cache is missing
    or older than CONFIGS_SYNC_INTERVAL_HOURS (default 24h).

    The sync process:
    1. Check if cache is stale (skip download if fresh, unless force=True)
    2. Download the main branch as a ZIP archive
    3. Extract only the configs/ directory to ~/.cache/oumi-mcp/configs
    4. Write a timestamp marker for staleness tracking
    5. Clean up temporary files

    Args:
        force: If True, sync regardless of cache age.

    Returns:
        Dict with:
        - ok: True if sync succeeded or cache is fresh
        - skipped: True if sync was skipped (cache is fresh)
        - error: Error message if sync failed, None otherwise
        - configs_synced: Number of config files synced (0 if skipped)
    """
    from oumi_mcp_server.config_service import get_cache_dir
    from oumi_mcp_server.constants import GITHUB_CONFIGS_ZIP_URL, GITHUB_ZIP_PREFIX

    if not force and not _is_cache_stale():
        logger.info("Config cache is fresh, skipping sync")
        return {"ok": True, "skipped": True, "error": None, "configs_synced": 0}

    cache_dir = get_cache_dir()
    temp_dir = None

    try:
        logger.info("Starting config sync from oumi-ai/oumi main branch")
        temp_dir = Path(tempfile.mkdtemp(prefix="oumi_config_sync_"))
        zip_path = temp_dir / "oumi.zip"

        logger.info(f"Downloading configs from {GITHUB_CONFIGS_ZIP_URL}")
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            response = client.get(GITHUB_CONFIGS_ZIP_URL)
            response.raise_for_status()
            zip_path.write_bytes(response.content)

        logger.info(f"Downloaded {len(response.content)} bytes")
        logger.info("Extracting configs from archive")
        with ZipFile(zip_path, "r") as zip_ref:
            config_files = [
                name
                for name in zip_ref.namelist()
                if name.startswith(GITHUB_ZIP_PREFIX)
            ]

            if not config_files:
                return {
                    "ok": False,
                    "skipped": False,
                    "error": "No configs directory found in repository archive",
                    "configs_synced": 0,
                }

            for file in config_files:
                zip_ref.extract(file, temp_dir)

        extracted_configs = temp_dir / "oumi-main" / "configs"
        if not extracted_configs.exists():
            return {
                "ok": False,
                "skipped": False,
                "error": f"Extracted configs not found at {extracted_configs}",
                "configs_synced": 0,
            }

        if cache_dir.exists():
            logger.info(f"Removing old cache: {cache_dir}")
            shutil.rmtree(cache_dir)

        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Moving configs to {cache_dir}")
        shutil.move(str(extracted_configs), str(cache_dir))

        _touch_sync_marker()

        config_count = len(list(cache_dir.rglob("*.yaml")))
        logger.info(f"Successfully synced {config_count} config files")

        return {
            "ok": True,
            "skipped": False,
            "error": None,
            "configs_synced": config_count,
        }

    except httpx.HTTPError as e:
        error_msg = f"Failed to download configs: {e}"
        logger.error(error_msg)
        return {"ok": False, "skipped": False, "error": error_msg, "configs_synced": 0}

    except Exception as e:
        error_msg = f"Config sync failed: {e}"
        logger.error(error_msg, exc_info=True)
        return {"ok": False, "skipped": False, "error": error_msg, "configs_synced": 0}

    finally:
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")


def main() -> None:
    """Run the MCP server.

    On startup:
    1. Attempts to sync configs from GitHub (skipped if cache is fresh).
    2. Falls back to bundled configs if sync fails and no cache exists.
    3. Starts the MCP server.
    """
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
