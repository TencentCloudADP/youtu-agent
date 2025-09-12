import os

from utu.agents import SimpleAgent
from utu.config import ConfigLoader
from utu.ui.webui_chatbot import WebUIChatbot


def main():
    # Load the agent configured with the game_config toolkit
    config = ConfigLoader.load_agent_config("simple_agents/game_ops_config_agent")

    agent = SimpleAgent(config=config)
    example = "请先初始化活动，例如：初始化活动ID为 act_demo，然后查看物品数据表的字段定义。"

    port = int(os.getenv("UTU_WEBUI_PORT", "8848"))
    ip = os.getenv("UTU_WEBUI_IP", "127.0.0.1")

    WebUIChatbot(agent, example_query=example).launch(port=port, ip=ip)


if __name__ == "__main__":
    main()
