import argparse

from utu.ui.webui_agents import WebUIAgents
from utu.utils.env import EnvUtils

DEFAULT_CONFIG = "simple/base.yaml"
DEFAULT_IP = EnvUtils.get_env("UTU_WEBUI_IP", "127.0.0.1")
DEFAULT_PORT = EnvUtils.get_env("UTU_WEBUI_PORT", "8848")
DEFAULT_AUTOLOAD = EnvUtils.get_env("UTU_WEBUI_AUTOLOAD", "false") == "true"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--ip", type=str, default=DEFAULT_IP)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--autoload", type=bool, default=DEFAULT_AUTOLOAD)
    args = parser.parse_args()

    webui = WebUIAgents(default_config=args.config)
    print(f"Server started at http://{args.ip}:{args.port}/")
    webui.launch(ip=args.ip, port=args.port, autoload=args.autoload)
