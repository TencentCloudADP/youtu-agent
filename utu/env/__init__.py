from ..config import AgentConfig
from .base_env import BaseEnv, _BaseEnv
from .browser_env import BrowserEnv
from .browser_env_e2b import BrowserE2BEnv
from .browser_tione_env import BrowserTioneEnv
from .e2b_env import E2BEnv
from .shell_local_env import ShellLocalEnv


async def get_env(config: AgentConfig, trace_id: str) -> _BaseEnv:
    if (not config.env) or (not config.env.name):
        return BaseEnv()
    match config.env.name:
        case "base":
            return BaseEnv()
        case "shell_local":
            return ShellLocalEnv(config.env.config, trace_id)
        case "e2b":
            return E2BEnv(config.env.config)
        case "browser_docker":
            return BrowserEnv(trace_id)
        case "browser_e2b":
            return BrowserE2BEnv(config.env.config)
        case "browser_tione":
            return BrowserTioneEnv(config.env.config)
        case _:
            raise ValueError(f"Unknown env name: {config.env.name}")
