# Oumi MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that gives AI coding assistants access to Oumi's library of ~500 ready-to-use YAML configs for fine-tuning LLMs.

When connected to Cursor, Claude Desktop, or any MCP-compatible client, the server lets the AI search for training recipes, retrieve full YAML configs, validate them, and follow guided ML engineering workflows -- all without you having to browse docs manually.

## What it does

The server exposes **5 tools** and **6 resources** over MCP:

| Tool | Purpose |
|------|---------|
| `get_started()` | Overview of capabilities and quickstart guide |
| `list_categories()` | Discover available model families and config types |
| `search_configs(query, task, model, keyword)` | Find training configs by filters |
| `get_config(path, include_content)` | Get config details and full YAML content |
| `validate_config(config, task_type)` | Validate a config file before running |

| Resource | Purpose |
|----------|---------|
| `guidance://mle-workflow` | End-to-end ML engineering workflow guide |
| `guidance://mle-train` | Training command usage and sizing heuristics |
| `guidance://mle-synth` | Synthetic data generation guidance |
| `guidance://mle-analyze` | Dataset analysis and quality checks |
| `guidance://mle-eval` | Evaluation strategies and benchmarks |
| `guidance://mle-infer` | Inference best practices |

### Supported models

Llama 3.1/3.2/4, Qwen 3, Phi 4, Gemma 3, DeepSeek R1, SmolLM, and more.

### Supported training techniques

SFT, DPO, GRPO, KTO, LoRA, QLoRA, full fine-tuning, pretraining, evaluation, inference.

## Installation

### As part of Oumi (recommended)

```bash
pip install oumi[mcp]
```

### Standalone

```bash
pip install oumi-mcp
```

### From source (development)

```bash
git clone https://github.com/oumi-ai/oumi.git
cd projects/oumi-mcp
pip install -e .
```

## Running the server

```bash
oumi-mcp
```

Or run as a Python module:

```bash
python -m oumi_mcp_server
```

## Connecting to an MCP client

### Cursor

Add to your Cursor MCP settings (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "oumi": {
      "command": "oumi-mcp"
    }
  }
}
```

### Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "oumi": {
      "command": "oumi-mcp"
    }
  }
}
```

### Any MCP client (stdio transport)

The server uses stdio transport by default. Point your MCP client to the `oumi-mcp` command.

## How configs work

The server ships with a bundled snapshot of Oumi's ~500 YAML config files. On startup, it checks for a fresher cached copy and syncs from GitHub if the cache is stale (older than 24 hours). The resolution order is:

1. **`OUMI_MCP_CONFIGS_DIR`** environment variable (explicit override)
2. **`~/.cache/oumi-mcp/configs`** (synced from GitHub, refreshed every 24h)
3. **Bundled configs** shipped with the package (always-available fallback)

This means:
- The server works immediately after install, even offline
- Configs stay up-to-date automatically via lazy background sync
- You can pin a specific config directory with the env var if needed

### Force a sync

To manually refresh configs, delete the cache and restart:

```bash
rm -rf ~/.cache/oumi-mcp
oumi-mcp
```

## Example workflow

Once connected, ask your AI assistant something like:

> "Find me a LoRA config for fine-tuning Llama 3.1 8B on my custom dataset"

The assistant will use the MCP tools to:

1. `search_configs(model="llama3_1", query="8b_lora", task="sft")` -- find matching recipes
2. `get_config("llama3_1/sft/8b_lora", include_content=True)` -- retrieve the full YAML
3. Help you customize `model_name`, `datasets`, `output_dir`, etc.
4. `validate_config("/path/to/your/config.yaml", "training")` -- validate before running

## Configuration

| Environment variable | Default | Description |
|---------------------|---------|-------------|
| `OUMI_MCP_CONFIGS_DIR` | _(unset)_ | Override the configs directory path |

## Project structure

```
oumi-mcp/
  src/oumi_mcp_server/
    __init__.py          # Package metadata
    __main__.py          # python -m entry point
    server.py            # MCP server, tools, resources, config sync
    config_service.py    # Config parsing, search, metadata extraction
    constants.py         # Type definitions and constants
    models.py            # TypedDict data models
    prompts/
      mle_prompt.py      # ML engineering workflow guidance resources
    configs/             # Bundled YAML configs (~500 files)
      recipes/           # Model-specific training recipes
      apis/              # API provider configs
      examples/          # Example configs
  pyproject.toml
```

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run the server
oumi-mcp

# Run tests
pytest
```

## Versioning

This package follows [semantic versioning](https://semver.org/). The version is independent from the main `oumi` package but tracks compatibility:

- **oumi-mcp 0.x.y** is compatible with **oumi >= 0.6.0**
- Configs are synced from the oumi `main` branch and stay current regardless of package version
- Bump the oumi-mcp version when the server code, tools, or resources change

## License

Apache-2.0 -- see the main [Oumi repository](https://github.com/oumi-ai/oumi) for details.
