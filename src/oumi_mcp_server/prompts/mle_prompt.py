"""Oumi MCP - ML Engineer Training Workflow Prompt.

Provides a structured, XML-tagged prompt that guides agents through ML training
workflows using Oumi MCP tools. Designed for clear section parsing and role definition.
"""

MLE_WORKFLOW_RESOURCE = """# Oumi ML Engineering Workflow Guidance

This resource provides an end-to-end ML engineering workflow for LLM training with Oumi.
Use it to reason like an MLE: start with requirements, validate data, pick recipes via
MCP tools, customize configs, validate, and evaluate before iterating.

## Principles
- Requirements first: Clarify success criteria before implementation
- Data-centric: Data quality determines model quality
- Iterative: Start small, validate assumptions, then scale
- Tool-first: Use MCP tools to find and adapt existing configs rather than building from scratch

## When to use each tool
1. search_configs: First step for finding training recipes
2. get_config: Retrieve full YAML after finding candidates
3. validate_config: ALWAYS validate before training
4. launch_training: Execute after validation passes (if available)

## Decision guidelines
- Model < 10B: Full fine-tuning viable
- Model 10B-30B: Use LoRA (r=16)
- Model > 30B: Use QLoRA (4-bit)

## Critical rules
- Never skip validation before launch_training
- Always check GPU memory requirements

## MCP tools overview
- list_categories(): Discover available model families and config categories
- search_configs(query, task, model, limit): Find training/eval/inference recipes
- get_config(path, include_content): Retrieve YAML and metadata for a recipe
- validate_config(config, task_type): Validate before running

## Workflow
### Phase 1: Requirements Gathering
Establish clarity before designing anything.
- Task: What should the model do?
- Data: Format, size, quality status
- Compute: GPU type and count, time budget
- Success criteria: Metrics and targets
- Constraints: Model size limits, latency requirements, deployment target

### Phase 2: Recipe Selection
Call list_categories() then search_configs() to find matching recipes.

Task mapping (examples):
- Instruction following: sft (TRL_SFT)
- Domain adaptation: pretrain (OUMI)
- Preference alignment: dpo (TRL_DPO)
- Reward optimization: grpo (TRL_GRPO)

PEFT selection:
- Use LoRA/QLoRA when GPU memory is limited or rapid iteration is needed
- Use full fine-tuning when maximum quality and compute are available

### Phase 3: Data Validation
- Consistent schema across examples
- Manual review of 50+ samples
- Duplicates removed
- No data leakage between train/val/test splits
- Token lengths within model context
- Labels/outputs are correct

Red flags (fix before training):
- Too few examples (less than 500 for narrow tasks or 5000 for general capability)
- More than 5% duplicates
- Significant class imbalance (greater than 10:1)
- P95 token length exceeds max context
- More than 10% low-quality examples

### Phase 4: Config Customization
1. get_config("path", include_content=True)
2. Adapt key settings
3. Save modified config
4. validate_config("/path/to/config.yaml", "training")

Key settings to customize:
- model.model_name
- data.train.datasets
- training.output_dir
- training.learning_rate
- training.per_device_train_batch_size
- training.gradient_accumulation_steps

### Phase 5: Evaluation Strategy
- During training: monitor train/val loss
- Post-training benchmarks: lm-eval-harness or similar
- Task-specific metrics on held-out test set
- Qualitative review of 50-100 samples

Success criteria:
- Val loss near or below train loss
- Primary metric exceeds baseline
- No regression on general capabilities
- 90%+ manual review quality

### Phase 6: Iteration and Troubleshooting
Common issues:
- Loss NaN/Inf: lower LR, check data
- OOM: reduce batch size, increase grad accum
- Slow training: data loading bottleneck
- Tokenizer mismatch: verify tokenizer/model alignment
- Overfitting: reduce epochs, add regularization, add data

Pivot guidance:
- Stop when success criteria met or diminishing returns
- Continue with a clear hypothesis for improvement
- Pivot if data quality is the bottleneck after multiple iterations

## Quick reference
VRAM rough guidance:
- 3B: 24GB full / 12GB LoRA / 8GB QLoRA
- 7B: 60GB full / 20GB LoRA / 14GB QLoRA
- 13B: 100GB full / 32GB LoRA / 20GB QLoRA
- 70B: 400GB+ full / 80GB LoRA / 48GB QLoRA
"""

TRAIN_COMMAND_RESOURCE = """<resource>
<title>MLE Train Guide (Oumi)</title>
<purpose>Plan and run training to change model behavior or adapt to domain data.</purpose>

<usage>
<cli>oumi train -c path/to/train.yaml</cli>
<override>oumi train -c path/to/train.yaml --training.max_steps 20</override>
</usage>

<workflow>
<step name="intent">
<what>Clarify the user's objective, success criteria, and constraints.</what>
<why>Task type, data requirements, and model choices flow directly from intent.</why>
<oumi>Map intent to `task` (sft/dpo/grpo/pretrain) and find recipes with `search_configs`.</oumi>
</step>

<step name="data">
<what>Confirm dataset format, size, and quality.</what>
<why>Quality issues propagate into model behavior.</why>
<oumi>Set `data.train.datasets` and run `oumi analyze` before training.</oumi>
</step>

<step name="model">
<what>Select a base model and parameter size.</what>
<why>Model size drives VRAM needs, runtime, and expected capability.</why>
<oumi>Set `model.model_name` and choose LoRA/QLoRA vs full fine-tuning.</oumi>
</step>

<step name="compute">
<what>Match batch size and accumulation to GPU memory.</what>
<why>Effective batch size controls stability and throughput.</why>
<oumi>Tune `training.per_device_train_batch_size` and `training.gradient_accumulation_steps`.</oumi>
</step>

<step name="run">
<what>Validate and execute the config.</what>
<why>Schema errors waste GPU time.</why>
<oumi>Run `validate_config(..., \"training\")` then `oumi train`.</oumi>
</step>
</workflow>

<key_config>
<field name="model.model_name">HF model ID or local path.</field>
<field name="data.train.datasets">Dataset list; include dataset_name/path and splits.</field>
<field name="training.learning_rate">Start from recipe; tune if unstable or slow.</field>
<field name="training.max_steps">Limit steps for quick iteration.</field>
<field name="training.output_dir">Per-run output directory for checkpoints/logs.</field>
<field name="training.per_device_train_batch_size">Tune to avoid OOM.</field>
<field name="training.gradient_accumulation_steps">Increase if batch size is small.</field>
</key_config>

<common_overrides>
<override>--training.max_steps</override>
<override>--training.learning_rate</override>
<override>--training.output_dir</override>
<override>--data.train.datasets</override>
</common_overrides>

<outputs>
<item>Checkpoints and logs under `training.output_dir`.</item>
<item>Trainer state and metrics for monitoring convergence.</item>
</outputs>
</resource>
"""

SYNTH_COMMAND_RESOURCE = """<resource>
<title>MLE Synthesis Guide (Oumi)</title>
<purpose>Generate synthetic training data when real data is scarce, noisy, or needs coverage.</purpose>

<usage>
<cli>oumi synth -c path/to/synth.yaml</cli>
<override>oumi synth -c path/to/synth.yaml --num_samples 1000</override>
</usage>

<workflow>
<step name="intent">
<what>Define what the synthetic data should teach the model.</what>
<why>Generation targets must align with downstream training goals.</why>
<oumi>Encode intent in `strategy_params.generated_attributes` prompt templates.</oumi>
</step>

<step name="attributes">
<what>Define which attributes should vary across samples.</what>
<why>Attributes control diversity and coverage.</why>
<oumi>Specify `sampled_attributes` (topic, difficulty, persona, style).</oumi>
</step>

<step name="templates">
<what>Describe how to generate outputs from attributes.</what>
<why>Templates enforce schema consistency and label quality.</why>
<oumi>Use `generated_attributes` with `instruction_messages` prompts.</oumi>
</step>

<step name="scale">
<what>Run a small sample and review before scaling up.</what>
<why>Early review prevents low-quality data at scale.</why>
<oumi>Start with low `num_samples`, inspect JSONL, then increase.</oumi>
</step>
</workflow>

<key_config>
<field name="strategy">Synthesis strategy (typically `GENERAL`).</field>
<field name="num_samples">Total samples to generate.</field>
<field name="output_path">JSONL output file path.</field>
<field name="strategy_params.sampled_attributes">Attribute variations for diversity.</field>
<field name="strategy_params.generated_attributes">Prompt templates and expected outputs.</field>
<field name="inference_config">LLM model + generation parameters.</field>
</key_config>

<outputs>
<item>JSONL dataset at `output_path` for use in training.</item>
</outputs>
</resource>
"""

ANALYZE_COMMAND_RESOURCE = """<resource>
<title>MLE Dataset Analysis Guide (Oumi)</title>
<purpose>Profile datasets, compute metrics, and flag outliers before training.</purpose>

<usage>
<cli>oumi analyze --config path/to/analyze.yaml</cli>
</usage>

<workflow>
<step name="intent">
<what>Validate data quality and suitability for training.</what>
<why>Bad data causes instability and poor generalization.</why>
<oumi>Run analyze before training and after synthesis.</oumi>
</step>

<step name="coverage">
<what>Check distribution coverage and length ranges.</what>
<why>Outliers and long examples cause truncation and bias.</why>
<oumi>Use length/token analyzers and inspect percentiles.</oumi>
</step>

<step name="quality">
<what>Find duplicates, empty samples, and label issues.</what>
<why>Duplicates and noise reduce effective data and increase overfitting.</why>
<oumi>Export results and filter problematic rows.</oumi>
</step>
</workflow>

<key_config>
<field name="dataset_path|dataset_name">Input dataset.</field>
<field name="sample_count">Limit for quick scans.</field>
<field name="output_path">Directory for analysis outputs.</field>
<field name="format">csv/json/parquet export format.</field>
<field name="analyzers">Built-in or custom analyzers.</field>
</key_config>

<outputs>
<item>`analysis_summary.json` plus message/conversation analysis files.</item>
</outputs>
</resource>
"""

EVAL_COMMAND_RESOURCE = """<resource>
<title>MLE Evaluation Guide (Oumi)</title>
<purpose>Benchmark model performance on standard or custom evaluation tasks.</purpose>

<usage>
<cli>oumi evaluate -c path/to/eval.yaml</cli>
<override>oumi evaluate -c path/to/eval.yaml --model.model_name my/model</override>
</usage>

<workflow>
<step name="intent">
<what>Define which behaviors to measure and what baseline to beat.</what>
<why>Evaluation tasks must match user goals.</why>
<oumi>Configure `tasks` with the correct backend and task names.</oumi>
</step>

<step name="comparability">
<what>Keep evaluation settings consistent across runs.</what>
<why>Different generation settings invalidate comparisons.</why>
<oumi>Lock `generation` parameters and use a per-run `output_dir`.</oumi>
</step>

<step name="scale">
<what>Plan for multi-GPU or sharded eval on larger models.</what>
<why>Large models may not fit on one GPU.</why>
<oumi>Set `model.shard_for_eval` and use distributed launch if needed.</oumi>
</step>
</workflow>

<key_config>
<field name="model.model_name">Model to evaluate.</field>
<field name="tasks">Benchmarks and backends (lm_harness, alpaca_eval, custom).</field>
<field name="generation">Batch size and decoding parameters.</field>
<field name="output_dir">Where results and metadata are stored.</field>
</key_config>

<outputs>
<item>Metrics and run metadata in `output_dir` (e.g., task_result.json).</item>
</outputs>
</resource>
"""

INFER_COMMAND_RESOURCE = """<resource>
<title>MLE Inference Guide (Oumi)</title>
<purpose>Run inference for quick checks or batch generation.</purpose>

<usage>
<cli>oumi infer -c path/to/infer.yaml</cli>
<interactive>oumi infer -c path/to/infer.yaml --interactive</interactive>
</usage>

<workflow>
<step name="intent">
<what>Decide between quick checks and structured batch outputs.</what>
<why>Interactive mode is best for debugging; batch mode is for datasets and evaluation.</why>
<oumi>Use `--interactive` or set `input_path`/`output_path` for batch JSONL.</oumi>
</step>

<step name="alignment">
<what>Ensure prompts and templates match the model’s chat format.</what>
<why>Mismatched templates degrade responses.</why>
<oumi>Confirm tokenizer and chat template for the selected `model.model_name`.</oumi>
</step>

<step name="decoding">
<what>Set generation parameters appropriate for your task.</what>
<why>Temperature and max tokens change output quality and length.</why>
<oumi>Configure `generation.max_new_tokens` and `generation.temperature`.</oumi>
</step>
</workflow>

<key_config>
<field name="model.model_name">Model used for inference.</field>
<field name="generation">max_new_tokens, temperature, batch_size.</field>
<field name="input_path">Batch JSONL input (optional).</field>
<field name="output_path">Batch JSONL output (optional).</field>
</key_config>

<outputs>
<item>Console responses (interactive) or JSONL outputs (batch).</item>
</outputs>
</resource>
"""
