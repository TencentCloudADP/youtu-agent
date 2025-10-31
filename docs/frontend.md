# WebUI

We provide a simple WebUI for visualizing the conversation.

## Basic Usage

> It is required to install the `utu_agent_ui` package before using the WebUI. Refer to [Installation](#installation) for more details.

After installing `utu_agent_ui`, The `WebUIAgents` class (from `utu.ui.webui_agents`) should work. Here is a simple usage example:

```python
from utu.ui.webui_agents import WebUIAgents

webui = WebUIAgents(default_config="simple/base.yaml")
print(f"Server started at http://127.0.0.1:8848/")
webui.launch(ip="127.0.0.1", port=8848)
```

### Examples with WebUI Integration

The `examples` directory contains some examples for you to play with. You can run the `main_web.py` files to start the WebUI. For example:

```bash
python examples/data_analysis/main_web.py
```

By default, the WebUI uses port `8848` and IP `127.0.0.1`. You can customize these settings by adding the following configuration to your `.env` file:

```env
# =============================================
# frontend
# =============================================
UTU_WEBUI_PORT=8848
UTU_WEBUI_IP=127.0.0.1
```

If you change the default port or IP, make sure to update the WebSocket URL in your frontend configuration to match the new settings. The default WebSocket URL is `ws://localhost:8848/ws`, and you can find and modify this setting in the right top corner of the frontend page.

### WebUIChatbot (Deprecated)

!!! warning
    The `WebUIChatbot` class (from `utu.ui.webui_chatbot`) is deprecated. Please use `WebUIAgents` instead.

Here is a simple usage example for `WebUIChatbot`:

```python
from utu.ui.webui_chatbot import WebUIChatbot
from utu.agents import SimpleAgent

simple_agent = SimpleAgent(name="demo")

chatbot = WebUIChatbot(
    simple_agent,
    example_query="Hello, how are you?",
)
chatbot.launch(port=8848, ip="127.0.0.1")
```

It also works with an `OrchestraAgent`. The following pseudo-code shows the basic usage:

```python
from utu.agents import OrchestraAgent
from utu.ui.webui_chatbot import WebUIChatbot

# ... load the config ...

orchestra_agent = OrchestraAgent(config=config)
chatbot = WebUIChatbot(
    orchestra_agent,
    example_query="Hello, how are you?",
)
chatbot.launch(port=8848, ip="127.0.0.1")
```

## Installation

We ship the static web pages in the `utu_agent_ui` package, which can be installed by either `pip install`ing the prebuilt wheel file or compiling from source.

### Installing the Prebuilt `*.whl` file

Download prebuilt wheel file (`utu_agent_ui-0.3.0-py3-none-any.whl`) from [releases](https://github.com/TencentCloudADP/youtu-agent/releases) and run the following command:

```bash
uv pip install utu_agent_ui-0.3.0-py3-none-any.whl
```

### Compiling from Source

Before compiling, make sure you have `npm` installed.

`cd` into the `utu/ui/frontend` directory and run the following commands:

```bash
npm install
uv pip install build
bash ./build.sh
```

The wheel file will be in the `build` directory. Refer to [Installing the Prebuilt `*.whl` file](#installing-the-prebuilt-whl-file) to install it.

## Implementation Details

The `WebUIAgents` class is basically a tornado based WebSocket server, which translate the model responses to JSON events and send them through a WebSocket connection to the frontend.

The frontend (in `utu/ui/frontend`, or installed as `utu_agent_ui` package) is a React application, which visualizes the events, and provides a simple UI for users to interact with the agent.

You can customize the port and IP of `WebUIAgents` by setting the `UTU_WEBUI_PORT` and `UTU_WEBUI_IP` environment variables (default: `127.0.0.1:8848`). The default WebSocket URL is `ws://localhost:8848/ws`, and you can find and modify this setting in the right top corner of the frontend page.

For development, you can set `UTU_WEBUI_AUTOLOAD=true` to enable auto reload for tornado server.
