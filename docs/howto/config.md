# How to Customize Agent Configurations with Hydra

This guide shows you how to customize agent configurations using [Hydra](https://hydra.cc/docs/intro/), a powerful configuration composition framework.

## Configuration Examples

All example configuration files are available at [`docs/configs/`](https://github.com/TencentCloudADP/youtu-agent/tree/main/docs/configs/):

### 1. `agent_simple.yaml` - Full Configuration

A complete agent configuration with all fields explicitly defined. This example shows:

- Agent type and basic settings (`type`, `max_turns`)
- Model configuration with environment variable interpolation using `${oc.env:VAR_NAME}`
- Agent profile (name and instructions)
- Context manager and environment setup
- Toolkit configuration (search toolkit example)

Use this as a reference for understanding all available configuration options.

### 2. `agent_with_hydra.yaml` - Hydra Composition

A simplified version using Hydra's composition features. Key concepts:

**Defaults list**: Load and compose predefined configs
```yaml
defaults:
  - /model/base@model                  # Load from configs/model/base.yaml
  - /tools/search@toolkits.search      # Load from configs/tools/search.yaml
  - _self_
```

**Override loaded configs**: Customize parameters after loading
```yaml
model:
  model_settings:
    temperature: 0                     # Override default temperature
toolkits:
  search:
    search_engine: jina                # Override default search engine
```

This approach reduces redundancy and makes configs more maintainable.

### 3. `toolkit_builtin.yaml` - Builtin Toolkit

Configuration for builtin toolkits (implemented in `utu/tools/`). Example shows:

- Toolkit mode: `builtin` | `mcp` | `customized`
- Tool activation: specify which tools to enable or use all with `null`
- Toolkit-specific configs (e.g., `search_engine`, `crawl_engine` for search toolkit)
- Optional LLM configuration for tools that need it

### 4. `toolkit_mcp.yaml` - MCP Toolkit

Configuration for Model Context Protocol (MCP) toolkits. Example shows:

- MCP transport types: `stdio` | `sse` | `streamable_http`
- Connection parameters (URL, headers, timeout)
- Environment variable interpolation for secrets like `${oc.env:GITHUB_TOKEN}`

See [`configs/tools/mcp/`](https://github.com/TencentCloudADP/youtu-agent/tree/main/configs/tools/mcp) for more MCP examples.

## Quick Start

1. **Copy an example**: Start with `agent_with_hydra.yaml` as a template
2. **Modify agent profile**: Update `agent.name` and `agent.instructions`
3. **Choose toolkits**: Update the `defaults` list to include your desired toolkits
4. **Override defaults**: Customize specific parameters as needed
5. **Save to `configs/agents/`**: Create your own config file (e.g., `configs/agents/my_agent.yaml`)

## Loading Your Config

```bash
# Load config by name (without .yaml extension)
python scripts/cli_chat.py --config my_agent

# Or programmatically
from utu.config import ConfigLoader
config = ConfigLoader.load_agent_config("my_agent")
```

## References

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [Agent Config Structure](https://github.com/TencentCloudADP/youtu-agent/blob/main/utu/config/agent_config.py)
- [Builtin Toolkits](https://tencentcloudadp.github.io/youtu-agent/tools/)
