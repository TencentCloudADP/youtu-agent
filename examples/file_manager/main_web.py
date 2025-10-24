import argparse

from utu.ui.webui_agents import WebUIAgents
from utu.utils.env import EnvUtils

DEFAULT_CONFIG = "examples/file_manager"
DEFAULT_IP = EnvUtils.get_env("UTU_WEBUI_IP", "127.0.0.1")
DEFAULT_PORT = EnvUtils.get_env("UTU_WEBUI_PORT", "8848")
DEFAULT_AUTOLOAD = EnvUtils.get_env("UTU_WEBUI_AUTOLOAD", "false") == "true"

EXAMPLE_QUERY = (
    "整理一下当前文件夹下面的所有文件，按照 学号-姓名 的格式重命名。"
    "我只接受学生提交的pdf，如果不是pdf文件，归档到一个文件夹里面。"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default=DEFAULT_IP)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--autoload", type=bool, default=DEFAULT_AUTOLOAD)
    args = parser.parse_args()

    webui = WebUIAgents(default_config=DEFAULT_CONFIG, example_query=EXAMPLE_QUERY)
    print(f"Server started at http://{args.ip}:{args.port}/")
    webui.launch(ip=args.ip, port=args.port, autoload=args.autoload)
