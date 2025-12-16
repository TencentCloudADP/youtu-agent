# Examples

Check out examples in the `examples` directory.

| Example | Core Architecture | Implementation | Key Tools | Use Case / Features |
| :--- | :--- | :--- | :--- | :--- |
| **`research`** | Manual Multi-Agent Orchestration | Orchestrates 3 independent `SimpleAgent`s in code to create a "Plan-and-Execute" workflow. | [`SearchToolkit`][utu.tools.search_toolkit.SearchToolkit] | Demonstrates building a complex workflow from scratch using basic [SimpleAgent][utu.agents.simple_agent.SimpleAgent] blocks. |
| **`wide_research`** | "Agent-as-Tool" Pattern | A single `SimpleAgent` makes decisions and calls a custom tool that encapsulates parallel sub-agents. | [`SearchToolkit`][utu.tools.search_toolkit.SearchToolkit] | Shows how to encapsulate parallelism and complex logic within a tool, simplifying the main agent's logic. |
| **`paper_collector`** | Standard [OrchestraAgent][utu.agents.orchestra_agent.OrchestraAgent] | Configuration-driven; uses few-shot examples (`planner_examples_data.json`) to guide the Planner. | [`DocumentToolkit`][utu.tools.document_toolkit.DocumentToolkit], [`SearchToolkit`][utu.tools.search_toolkit.SearchToolkit] | A standard, "out-of-the-box" application of [OrchestraAgent][utu.agents.orchestra_agent.OrchestraAgent] for multi-step document analysis. |
| **`file_manager`** | [SimpleAgent][utu.agents.simple_agent.SimpleAgent] + UI | A configuration-driven `SimpleAgent` wrapped in an interactive `Gradio` web UI. | [`BashToolkit`][utu.tools.bash_toolkit.BashToolkit] | A practical example of an agent interacting with the local file system, with a focus on UI and safety. |
| **`data_analysis`** | Customized [OrchestraAgent][utu.agents.orchestra_agent.OrchestraAgent] | Extends [OrchestraAgent][utu.agents.orchestra_agent.OrchestraAgent]'s core components: <br>1. **`DAPlannerAgent`**: Proactively inspects data schema before planning. <br>2. **Reporter**: Uses a custom template to generate a rich HTML report. | [`TabularDataToolkit`][utu.tools.tabular_data_toolkit.TabularDataToolkit] | An advanced [OrchestraAgent][utu.agents.orchestra_agent.OrchestraAgent] use case, showing deep customization for a complex, domain-specific problem. |
| **`ppt_gen`** | [SimpleAgent][utu.agents.simple_agent.SimpleAgent] | A configuration-driven `SimpleAgent` that synthesizes content from a given document and generates a PowerPoint presentation based on the given json schema and content page by page. | [`SearchToolkit`][utu.tools.search_toolkit.SearchToolkit] | An experimental demo of using [SimpleAgent][utu.agents.simple_agent.SimpleAgent] to generate a PowerPoint presentation. |

## Run the Examples

In each example directory, you can run the `main.py` files to start the examples in command line. For some examples, you can also run the `main_web.py` files to start the examples with WebUI. Refer to corresponding `README.md` files in the example directories for more details.

> Note: To use the WebUI, you need to install the `utu_agent_ui` package. Refer to [Installation](frontend.md#installation) for more details.

