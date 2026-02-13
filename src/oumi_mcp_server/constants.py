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

# Modules to index for the get_docs / list_modules tools.
# Each entry is (module_path, human_description).
DOCS_MODULES: list[tuple[str, str]] = [
    ("oumi.core.configs", "TrainingConfig and all top-level config dataclasses"),
    ("oumi.core.configs.analyze_config", "AnalyzeConfig and analysis settings"),
    ("oumi.core.configs.job_config", "JobConfig and job settings"),
    ("oumi.core.configs.synthesis_config", "SynthesisConfig and synthesis settings"),
    ("oumi.core.configs.params.model_params", "ModelParams and model settings"),
    (
        "oumi.core.configs.params.training_params",
        "TrainingParams and training settings",
    ),
    (
        "oumi.core.configs.params.data_params",
        "DataParams, DatasetParams, DatasetSplitParams",
    ),
    ("oumi.core.configs.params.peft_params", "PeftParams for LoRA/QLoRA"),
    ("oumi.core.configs.params.fsdp_params", "FSDPParams for FSDP"),
    ("oumi.core.configs.params.generation_params", "GenerationParams for inference"),
    ("oumi.core.configs.params.grpo_params", "GrpoParams for GRPO training"),
    ("oumi.core.configs.params.evaluation_params", "Evaluation parameters"),
    ("oumi.core.configs.params.deepspeed_params", "DeepSpeed parameters"),
    ("oumi.core.configs.params.synthesis_params", "Synthesis parameters"),
    ("oumi.core.distributed", "Distributed training utilities"),
    ("oumi.core.evaluation.metrics", "Evaluation metrics"),
    ("oumi.core.launcher.base_cluster", "Launcher base cluster APIs"),
    ("oumi.core.registry.registry", "Registry utilities"),
    ("oumi.core.tokenizers.utils", "Tokenizer helpers"),
    ("oumi.core.trainers.oumi_trainer", "Core training loop"),
    ("oumi.core.types.conversation", "Conversation data types"),
    ("oumi.core.datasets.base_dpo_dataset", "Base DPO dataset"),
    ("oumi.core.datasets.base_rubric_dataset", "Base rubric dataset"),
    ("oumi.core.feature_generators.base_feature_generator", "Feature generator base"),
    ("oumi.core.analyze.column_types", "Analysis column and content types"),
    ("oumi.core.analyze.dataframe_analyzer", "DataFrame analysis utilities"),
    ("oumi.core.analyze.dataset_analyzer", "Dataset analysis utilities"),
    ("oumi.core.analyze.length_analyzer", "Length analyzer"),
    ("oumi.core.analyze.sample_analyzer", "Sample analyzer"),
    ("oumi.cli.analyze", "CLI entrypoint for dataset analysis"),
    ("oumi.evaluate_async", "Asynchronous evaluation entrypoint"),
    ("oumi.infer", "Inference entrypoint"),
    ("oumi.inference", "Inference engine classes"),
    ("oumi.inference.adaptive_semaphore", "Adaptive inference concurrency"),
    ("oumi.inference.remote_inference_engine", "Remote inference engine"),
    ("oumi.judge", "Judging entrypoint"),
    ("oumi.datasets", "Dataset implementations"),
    ("oumi.datasets.debug", "Debug dataset helpers"),
    ("oumi.datasets.evaluation.utils", "Evaluation dataset utilities"),
    ("oumi.datasets.grpo.rar_dataset", "GRPO RAR dataset"),
    ("oumi.datasets.grpo.rewards.completion_length_rewards", "GRPO reward helpers"),
    ("oumi.datasets.sft.chatqa", "ChatQA dataset"),
    ("oumi.datasets.sft.magpie", "Magpie dataset"),
    ("oumi.builders.callbacks", "Training callbacks"),
    ("oumi.builders.collators", "Data collators"),
    ("oumi.builders.data", "Data builders"),
    ("oumi.builders.inference_engines", "Inference engine builders"),
    ("oumi.builders.lr_schedules", "Learning rate schedules"),
    ("oumi.builders.metrics", "Metrics builders"),
    ("oumi.builders.models", "Model builders"),
    ("oumi.builders.optimizers", "Optimizer builders"),
    ("oumi.builders.oumi_data", "Oumi data builders"),
    ("oumi.builders.processors", "Processor builders"),
    ("oumi.builders.quantizers", "Quantizer builders"),
    ("oumi.builders.rewards", "Reward builders"),
    ("oumi.builders.rollouts", "Rollout builders"),
    ("oumi.builders.training", "Training builders"),
    ("oumi.builders.tuning", "Tuning builders"),
    ("oumi.launcher.clouds.frontier_cloud", "Frontier cloud launcher"),
    ("oumi.launcher.clouds.local_cloud", "Local cloud launcher"),
    ("oumi.launcher.clouds.perlmutter_cloud", "Perlmutter cloud launcher"),
    ("oumi.launcher.clouds.polaris_cloud", "Polaris cloud launcher"),
    ("oumi.launcher.clouds.sky_cloud", "Sky cloud launcher"),
    ("oumi.launcher.clouds.slurm_cloud", "Slurm cloud launcher"),
    ("oumi.quantize.base", "Quantization base"),
    ("oumi.quantize.utils", "Quantization utilities"),
    ("oumi.judges", "Judge implementations"),
    ("oumi.judges.base_judge", "Base judge interfaces"),
    ("oumi.launcher", "Job launching infrastructure"),
]
DOCS_MAX_RESULTS: int = 10
DOCS_MAX_METHODS_PER_CLASS: int = 10
