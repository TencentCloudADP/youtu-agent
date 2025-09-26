"""
Usage:

    python scripts/chat_ui.py --config_name orchestrator/universal
"""

from utu.agents import get_agent
from utu.ui.webui_chatbot import WebUIChatbot
from utu.utils.script_utils import parse_cli_args


def main():
    config = parse_cli_args()
    agent = get_agent(config)

    question = "请分析论文 https://www.arxiv.org/pdf/2507.12883，整理出它的相关工作，并且进行简单的比较。"

    ui = WebUIChatbot(agent, example_query=question)
    ui.launch()


if __name__ == "__main__":
    main()
