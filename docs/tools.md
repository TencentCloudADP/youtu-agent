# Toolkits

Toolkits are collections of related tools that an agent can use to perform actions. They are the primary way to extend an agent's capabilities.

There are three main types of toolkits:
- **builtin**: The toolkits provided by default in the framework.
- **mcp**: The toolkits accessible via the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP).
- **customized**: The toolkits created by users.

## Builtin Toolkits

All builtin toolkits inherit from the `AsyncBaseToolkit` abstract base class. This class provides a standardized interface for creating and managing tools.

Here is a summary of some key toolkits available in the framework:

| Toolkit Class | Provided Tools (Functions) | Core Functionality & Mechanism |
| :--- | :--- | :--- |
| **[SearchToolkit][utu.tools.search_toolkit.SearchToolkit]** | `search`, `web_qa` | Performs web searches using the Serper API and reads webpage content using the Jina API. It can use an LLM to answer questions based on page content. |
| **[DocumentToolkit][utu.tools.document_toolkit.DocumentToolkit]** | `document_qa` | Processes local or remote documents (PDF, DOCX, etc.). It uses the `chunkr.ai` service to parse the document and an LLM to answer questions or provide a summary. |
| **[PythonExecutorToolkit][utu.tools.python_executor_toolkit.PythonExecutorToolkit]** | `execute_python_code` | Executes Python code snippets in an isolated environment using `IPython.core.interactiveshell`. It runs in a separate thread to prevent blocking and can capture outputs, errors, and even `matplotlib` plots. |
| **[BashToolkit][utu.tools.bash_toolkit.BashToolkit]** | `run_bash` | Provides a persistent local shell session using the `pexpect` library. This allows the agent to run a series of commands that maintain state (e.g., current directory). |
| **[ImageToolkit][utu.tools.image_toolkit.ImageToolkit]** | `image_qa` | Answers questions about an image or provides a detailed description. It uses a vision-capable LLM to analyze the image content. |
| **[AudioToolkit][utu.tools.audio_toolkit.AudioToolkit]** | `audio_qa` | Transcribes audio files using an audio model and then uses an LLM to answer questions based on the transcription. |
| **[CodesnipToolkit][utu.tools.codesnip_toolkit.CodesnipToolkit]** | `run_code` | Executes code in various languages (Python, C++, JS, etc.) by sending it to a remote sandbox service (like SandboxFusion) and returning the result. |
<!-- 
| **[FileEditToolkit][utu.tools.file_edit_toolkit.FileEditToolkit]** | `edit_file` | Edits local files by applying a specific `SEARCH/REPLACE` diff format. It includes safety features like filename sanitization and automatic backups. |
| **[ArxivToolkit][utu.tools.arxiv_toolkit.ArxivToolkit]** | `search_papers`, `download_papers` | A wrapper around the `arxiv.py` library to search for and download academic papers from arXiv.org. |
| **[GitHubToolkit][utu.tools.github_toolkit.GitHubToolkit]** | `get_repo_info` | Fetches repository metadata (stars, forks, language, etc.) from the GitHub REST API. |
-->


## MCP Toolkits

We provide examples of MCP toolkits in the `examples/mcp` directory, including transports of `stdio`, `sse` and `streamable_http`. You should easily run them. E.g., for the stdio example:

```sh
python examples/mcp/stdio_example/main.py
```

If you are not familiar with MCP, please refer to the [MCP documentation](https://modelcontextprotocol.io/docs/getting-started/intro).


## How to Inspect Toolkits

We provide two useful scripts to inspect and test toolkits:

### Dump Tool Schemas

This script dumps add tools registered in `TOOLKIT_MAP` into a `.xlsx` file. The default output file is `tools.xlsx`.

```sh
python scripts/utils/dump_tool_schemas.py
```

### Inspect & Test Tools

You can start a local MCP server that exposes the tools via HTTP. This allows you to interactively test the tools using an MCP client. E.g. [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector).

```sh
# start the MCP server
python scripts/utils/start_tools_mcp.py --toolkits search image github

# start the MCP inspector
npx @modelcontextprotocol/inspector
# ... connect to the MCP server with Streamable HTTP transport URL http://localhost:3005/mcp
```
