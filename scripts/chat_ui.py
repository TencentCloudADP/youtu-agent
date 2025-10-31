"""
Usage:

    python scripts/chat_ui.py --default_config orchestrator/universal
"""

import argparse

from utu.ui.webui_agents import WebUIAgents
from utu.utils import EnvUtils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default=EnvUtils.get_env("UTU_WEBUI_IP", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=EnvUtils.get_env("UTU_WEBUI_PORT", "8848"))
    parser.add_argument("--autoload", type=bool, default=EnvUtils.get_env("UTU_WEBUI_AUTOLOAD", "false") == "true")

    parser.add_argument("--default_config", type=str, default="simple/base", help="Configuration name")
    parser.add_argument("--example_query", type=str, default="What can you do?", help="Example query to show in the UI")
    args = parser.parse_args()

    ui = WebUIAgents(args.default_config, example_query=args.example_query)
    ui.launch(ip=args.ip, port=args.port, autoload=args.autoload)


if __name__ == "__main__":
    main()
