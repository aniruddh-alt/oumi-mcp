"""TypedDict models for Oumi MCP server data structures."""

from typing import NotRequired, TypedDict

from oumi_mcp_server.constants import PeftType, TaskType


class ConfigMetadata(TypedDict):
    """Metadata extracted from an Oumi config file.

    Attributes:
        path: Relative path to the config file.
        description: Description extracted from header comments.
        model_name: HuggingFace model ID or model name.
        task_type: Type of task (sft, dpo, grpo, evaluation, etc.).
        datasets: List of dataset names used in the config.
        reward_functions: List of reward functions for RLHF training.
        peft_type: Type of PEFT (lora/qlora) if applicable.
    """

    path: str
    description: str
    model_name: str
    task_type: TaskType
    datasets: list[str]
    reward_functions: list[str]
    peft_type: str  # "lora", "qlora", or "" for full fine-tuning


class KeySettings(TypedDict):
    """Important training hyperparameters extracted from config.

    Attributes:
        learning_rate: Learning rate for training.
        num_train_epochs: Number of training epochs.
        max_steps: Maximum number of training steps.
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        model_max_length: Maximum sequence length.
        torch_dtype: PyTorch data type (e.g., bfloat16, float32).
    """

    learning_rate: NotRequired[float]
    num_train_epochs: NotRequired[int]
    max_steps: NotRequired[int]
    per_device_train_batch_size: NotRequired[int]
    gradient_accumulation_steps: NotRequired[int]
    model_max_length: NotRequired[int]
    torch_dtype: NotRequired[str]


class ConfigDetail(ConfigMetadata):
    """Full config details including key settings and optional content.

    Attributes:
        key_settings: Important training hyperparameters.
        content: Full YAML content (empty string if not requested).
        error: Error message if config not found (empty string if no error).
    """

    key_settings: KeySettings
    content: str
    error: str


class CategoriesResponse(TypedDict):
    """Response from list_categories tool.

    Attributes:
        categories: Top-level directories.
        model_families: Available model families in recipes/.
        api_providers: Available API providers in apis/.
        total_configs: Total number of configs available.
    """

    categories: list[str]
    model_families: list[str]
    api_providers: list[str]
    total_configs: int


class SearchResult(TypedDict):
    """Result item from search_configs tool.

    This is an alias for ConfigMetadata for clarity in tool responses.
    """

    path: str
    description: str
    model_name: str
    task_type: TaskType
    datasets: list[str]
    reward_functions: list[str]
    peft_type: str  # "lora", "qlora", or "" for full fine-tuning


class HardwareInfo(TypedDict):
    """Detected hardware and installed ML packages on the local machine.

    Attributes:
        accelerator_type: "cuda", "mps", or "none".
        accelerator_count: Number of accelerators detected.
        gpu_name: GPU device name (None if no CUDA GPU).
        gpu_memory_gb: Total GPU memory in GB (None if no CUDA GPU).
        compute_capability: CUDA compute capability e.g. "8.0" (None if no CUDA GPU).
        cuda_version: CUDA toolkit version (None if no CUDA GPU).
        packages: Installed packages relevant to hardware checks, e.g. {"torch": "2.3.0"}.
    """

    accelerator_type: str
    accelerator_count: int
    gpu_name: str | None
    gpu_memory_gb: float | None
    compute_capability: str | None
    cuda_version: str | None
    packages: dict[str, str]


class PreFlightCheckResponse(TypedDict):
    """Response from pre_flight_check tool.

    Attributes:
        blocking: True when errors contain hard blockers that WILL prevent the
            run from succeeding. When True, the user MUST resolve these issues
            before proceeding. Do NOT treat blocking issues as informational.
        summary: One-line human-readable verdict ("ready", "blocked: …", etc.).
            Surface this to the user prominently.
        hf_authenticated: Whether a valid HuggingFace token was found.
        repo_access: Per-repo access status: "ok", "gated", "not_found", or "error".
        hardware: Detected local hardware and installed packages.
        errors: Issues that will cause the training run to crash.
        warnings: Potential issues that may be fine if targeting a remote cluster.
        paths: Local filesystem paths from the config mapped to whether they exist.
    """

    blocking: bool
    summary: str
    hf_authenticated: bool
    repo_access: dict[str, str]
    hardware: HardwareInfo
    errors: list[str]
    warnings: list[str]
    paths: dict[str, bool]


class JobSubmissionResponse(TypedDict):
    """Response from run_oumi_job tool.

    Returned immediately when a job is submitted (or dry-run previewed).

    Attributes:
        success: Whether the submission/dry-run succeeded.
        job_id: Unique job identifier for use with get_job_status / cancel_job.
        status: "dry_run", "submitted", or an error indicator.
        dry_run: True if this was a dry-run preview (no actual execution).
        command: The Oumi CLI subcommand (train, evaluate, etc.).
        config_path: Absolute path to the config file.
        cloud: Cloud provider (e.g. "local", "gcp", "aws").
        cluster_name: Cluster name (empty if auto-generated).
        model_name: HuggingFace model ID extracted from config.
        output_dir: Output directory extracted from config.
        message: Human-readable summary of what happened or will happen.
        error: Error message if success is False.
    """

    success: bool
    job_id: str
    status: str
    dry_run: bool
    command: str
    config_path: str
    cloud: str
    cluster_name: str
    model_name: str
    output_dir: str
    message: str
    error: NotRequired[str]


class JobStatusResponse(TypedDict):
    """Response from ``get_job_status`` snapshot tool.

    Attributes:
        success: Whether the status lookup succeeded.
        job_id: The MCP job identifier.
        oumi_job_id: The job ID on the cluster (from oumi.launcher).
        status: Current status string from the launcher.
        state: Job state enum name (QUEUED, RUNNING, COMPLETED, FAILED, CANCELED).
        command: The Oumi CLI subcommand.
        config_path: Absolute path to the config.
        cloud: Cloud provider name.
        cluster: Cluster name the job is running on.
        model_name: Model being trained/evaluated.
        is_done: True if the job is in a terminal state.
        metadata: Additional metadata from the launcher.
        log_file: Absolute path to the full stdout log file on disk (if available).
        error: Error message if the job failed or lookup failed.
    """

    success: bool
    job_id: str
    oumi_job_id: str
    status: str
    state: str
    command: str
    config_path: str
    cloud: str
    cluster: str
    model_name: str
    is_done: bool
    metadata: NotRequired[str]
    log_file: NotRequired[str]
    error: str | None


class JobCancelResponse(TypedDict):
    """Response from cancel_job tool.

    Attributes:
        success: Whether cancellation succeeded.
        message: Human-readable result description.
        error: Error message if cancellation failed.
    """

    success: bool
    message: NotRequired[str]
    error: NotRequired[str]


class JobSummary(TypedDict):
    """Compact job summary for resource listings.

    Attributes:
        job_id: Unique MCP job identifier.
        command: Oumi CLI subcommand.
        status: Current lifecycle state.
        cloud: Cloud provider name.
        cluster: Cluster name.
        model_name: Model being trained/evaluated.
        is_done: Whether the job has finished.
    """

    job_id: str
    command: str
    status: str
    cloud: str
    cluster: str
    model_name: str
    is_done: bool


class JobLogsResponse(TypedDict):
    """Response from ``get_job_logs`` snapshot tool.

    Attributes:
        success: Whether log retrieval succeeded.
        job_id: The MCP job identifier.
        lines_requested: Number of trailing lines requested.
        lines_returned: Number of trailing lines returned.
        log_file: Absolute path to the stdout log file (if available).
        logs: Tail content for the requested number of lines.
        error: Error message if lookup or reading failed.
    """

    success: bool
    job_id: str
    lines_requested: int
    lines_returned: int
    log_file: str
    logs: str
    error: str | None
