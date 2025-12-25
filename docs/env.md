# Agent Environments

An Environment (`Env`) represents the world in which the agent operates. Its primary responsibilities are to provide the agent with a sense of its current **state** and a set of **tools** to interact with that world.

The framework uses a factory function, `get_env`, to create the appropriate environment based on the agent's configuration file.

## Core Concepts

All environments inherit from the abstract base class `Env`, which defines the core interface:

- `get_state() -> str`: Returns a string describing the current state of the environment. This information is injected into the agent's prompt to provide context.
- `get_tools() -> list[Tool]`: Returns a list of `Tool` objects that the agent can use to interact with the environment.
- `build()` / `cleanup()`: Methods to manage the lifecycle of the environment, such as starting services or cleaning up resources.


## ⭐ Recommended Environments

For production use and maximum security, we recommend using **E2B-based environments**:

- **[E2BEnv](#e2benv)**: For code execution and file manipulation tasks
- **[BrowserE2BEnv](#browsere2benv)**: For web automation and browser interaction tasks

These cloud-based sandboxes provide enterprise-grade isolation, eliminating risks associated with running untrusted or AI-generated code on your local machine. They offer automatic resource cleanup, GUI monitoring capabilities (NoVNC), and seamless integration with modern development workflows. In practice, we use Tencent Cloud's [Agent Sandbox service](https://cloud.tencent.com/product/agentsandbox) to host these environments securely.


## Available Environments

Here are the currently available environment implementations.

| Config Name | Environment Class | Description | Recommendation |
|-------------|-------------------|-------------|----------------|
| `e2b` | `E2BEnv` | Cloud-based code interpreter | ⭐ **Recommended** for code execution |
| `browser_e2b` | `BrowserE2BEnv` | Cloud-based browser automation | ⭐ **Recommended** for web tasks |
| `shell_local` | `ShellLocalEnv` | Local shell with workspace | Use for local development only |
| `browser_docker` | `BrowserEnv` | Docker-based browser | Use if E2B is unavailable |
| `base` | `BasicEnv` | No environment | For simple tasks without execution |


### BasicEnv

This is the simplest and default environment. It can be considered a "null" environment.

- **State**: Provides no state information (returns an empty string).
- **Tools**: Provides no tools (returns an empty list).
- **Use Case**: Used when the agent's task does not require any specific environmental interaction.

### ShellLocalEnv

This environment provides the agent with an isolated workspace on the local filesystem.

- **State**: The state string includes the current time, the absolute path to the isolated workspace, and a crucial instruction for the agent: `You can only run bash commands in your workspace!!!`. This helps guide and constrain the agent's behavior.
- **Tools**: This environment does **not** provide tools directly. The agent must be configured separately with tools capable of executing shell commands (e.g., a `bash` tool). The environment's role is to provide the context and workspace for those tools.
- **Isolation**: A unique workspace directory is created for each run session, preventing interference between different tasks.

### BrowserEnv

This is an environment that gives the agent control over a fully-featured, interactive web browser.

- **Architecture**: `BrowserEnv` runs a browser automation service inside a **Docker container**. This ensures that each agent session is completely isolated and has a clean, predictable browser environment.
- **State**: The state represents the current content of the web page. It is updated after every action (e.g., clicking an element, navigating to a URL), giving the agent feedback on the result of its last action.
- **Tools**: Tools are provided dynamically by the browser service running in the container. `BrowserEnv` acts as a **proxy**: it discovers the available tools (e.g., `go_to_url`, `click_element`, `input_text`) and makes them available to the agent. When the agent calls a tool, `BrowserEnv` forwards the request to the Docker container for execution.

### E2BEnv

`E2BEnv` is a cloud-based code execution environment powered by [E2B](https://e2b.dev/docs), offering superior security and isolation compared to local environments.

- **Architecture**: Uses E2B's Code Interpreter sandbox (`code-interpreter-v1` template) running in a secure, isolated cloud environment. Each session creates a fresh AsyncSandbox instance.
- **Tools**: You can config python, bash, file editing tools with E2BEnv.
- **Use Case**: **Recommended for all code execution tasks** where security is a concern, especially when running untrusted or AI-generated code.
- **Configuration**: See `configs/agents/examples/e2b/e2b_python.yaml` for a complete setup example with Python, Bash, and file editing tools.

```yaml
# @package _global_
defaults:
  - /tools/e2b/python_executor@toolkits.PythonTool
  - /tools/e2b/bash@toolkits.BashTool
  - /tools/e2b/file_edit@toolkits.FileTool
  - _self_

env:
  name: e2b
  config:
    request_timeout: 5  # Optional: timeout in seconds for file operations (default: 5)

agent: ...
```

### BrowserE2BEnv

This is an advanced browser automation environment built on E2B, providing secure browser control with GUI access.

- **Architecture**: Leverages E2B's SDK combined with TencentCloud's [Agent Sandbox service](https://cloud.tencent.com/product/agentsandbox). Integrates Playwright MCP server for tool provisioning via Chrome DevTools Protocol (CDP).
- **Tools**: Dynamically provided by the Playwright MCP server (`@playwright/mcp`), including:
  - Browser navigation and page interaction
  - Element clicking, text input, and form filling
  - Screenshot capture and page content extraction
  - Multi-tab and window management
- **Configuration**: See `configs/agents/examples/e2b/e2b_browser.yaml` for a complete setup example. Note that browser tools are auto-discovered from the MCP server—no manual tool configuration required.

```yaml
# @package _global_
defaults:
  - /model/base@model
  - _self_

env:
  name: browser_e2b

agent: ...
```
