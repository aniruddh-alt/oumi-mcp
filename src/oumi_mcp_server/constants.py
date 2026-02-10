"""Constants and type definitions for Oumi MCP server."""

from pathlib import Path
from typing import Literal

# Task types supported by Oumi
TaskType = Literal[
    "grpo",
    "dpo",
    "kto",
    "sft",
    "evaluation",
    "inference",
    "pretraining",
    "synthesis",
    "quantization",
    "",
]

# PEFT types
PeftType = Literal["lora", "qlora"]

# Validator task types (for validate_config tool)
ValidatorTaskType = Literal[
    "analyze",
    "async_evaluation",
    "evaluation",
    "inference",
    "job",
    "judge",
    "quantization",
    "synthesis",
    "training",
    "tuning",
]

# Training hyperparameter keys
TRAINING_KEYS = [
    "learning_rate",
    "num_train_epochs",
    "max_steps",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
]

MODEL_KEYS = [
    "model_max_length",
    "torch_dtype_str",
]

# Data split names
DATA_SPLITS = ["train", "validation", "test"]

# Comment prefixes to skip when extracting description
COMMENT_PREFIXES_TO_SKIP = [
    "Usage:",
    "See Also:",
    "Requirements:",
]

# Config categories
CATEGORIES_DIR = "categories"
MODEL_FAMILIES_DIR = "recipes"
API_PROVIDERS_DIR = "apis"

# Special file names
TRAIN_YAML = "train.yaml"

# Cache sizes
YAML_CACHE_SIZE = 500
CONFIGS_CACHE_SIZE = 1

# Default search limit
DEFAULT_SEARCH_LIMIT = 20

# Config sync settings
CONFIGS_SYNC_INTERVAL_HOURS = 24
CONFIGS_SYNC_MARKER = ".last_sync"
GITHUB_CONFIGS_ZIP_URL = "https://github.com/oumi-ai/oumi/archive/refs/heads/main.zip"
GITHUB_ZIP_PREFIX = "oumi-main/configs/"

# HuggingFace API timeout (seconds) for pre-flight checks
HF_API_TIMEOUT_SECONDS = 10

# Hardware check constants
MIN_CC_BF16 = 8.0  # Ampere+ required for native bf16
MIN_CC_FLASH_ATTN = 8.0  # Ampere+ required for flash attention 2
MIN_TORCH_VERSION_SDPA = "2.0"  # SDPA requires torch >= 2.0
MIN_TORCH_VERSION_COMPILE = "2.0"  # torch.compile requires torch >= 2.0

# Packages to check for hardware compatibility
HARDWARE_PACKAGES = frozenset(["torch", "flash-attn", "bitsandbytes", "deepspeed"])

VALID_OUMI_COMMANDS = frozenset(
    ["train", "analyze", "synth", "evaluate", "eval", "infer", "tune", "quantize"]
)

# Maximum number of completed/failed/cancelled jobs to keep in the registry.
# Oldest finished jobs are evicted first when this limit is exceeded.
MAX_COMPLETED_JOBS = 50

# ---------------------------------------------------------------------------
# Job log streaming constants
# ---------------------------------------------------------------------------

# Directory for job log files (set as OUMI_LOGGING_DIR for local jobs).
# The oumi.launcher LocalClient writes subprocess stdout/stderr here.
JOB_LOGS_DIR: Path = Path.home() / ".cache" / "oumi-mcp" / "job-logs"

# How often to poll launcher status while streaming logs (seconds).
LOG_POLL_INTERVAL_SECONDS: float = 2.0

# How often to check for new log lines when tailing a file (seconds).
LOG_TAIL_INTERVAL_SECONDS: float = 1.0

# Default maximum number of log lines to stream to the client per
# get_job_status call.  After the cap, only progress updates are sent.
DEFAULT_STREAM_LINES: int = 200
