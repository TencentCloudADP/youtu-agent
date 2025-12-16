import argparse

from ..config import AgentConfig, ConfigLoader
from .print_utils import PrintUtils


def parse_cli_args() -> AgentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="simple/base", help="Configuration name")
    parser.add_argument("--agent_model", type=str, default=None, help="Agent model.")
    args = parser.parse_args()

    config: AgentConfig = ConfigLoader.load_agent_config(args.config_name)
    # Override basic configs
    if args.agent_model:
        assert config.type == "simple", f"--agent_model only support SimpleAgent, get {config.type}"
        config.model.model_provider.model = args.agent_model
    if config.type == "workforce":
        PrintUtils.print_info("Error: Workforce agent is not supported in CLI mode yet.")
        raise NotImplementedError

    return config
