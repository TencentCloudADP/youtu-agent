from ..utils import DIR_ROOT
from .base_env import BaseEnv

TEMPLATE = r"""<env>
{env}
</env>
<instructions>
1. You can only run bash commands in your workspace!!!
</instructions>
"""


class ShellLocalEnv(BaseEnv):
    workspace: str

    def __init__(self, config: dict = None, trace_id: str = None):
        config = config or {}
        workspace = config.get("workspace_root")
        if not workspace:
            workspace = DIR_ROOT / "workspace" / trace_id
            workspace.mkdir(parents=True, exist_ok=True)
        print(f"> Workspace: {workspace}")
        self.workspace = workspace

    def get_state(self) -> str:
        env_strs = [
            f"Time: {self.get_time()}",
            f"Workspace: {self.workspace}",
        ]
        sp_prefix = TEMPLATE.format(env="\n".join(env_strs))
        return sp_prefix
