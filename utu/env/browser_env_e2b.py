import logging

from agents import Tool

from ..config import ToolkitConfig
from ..tools.utils import AgentsMCPUtils
from .base_env import BasicEnv

logger = logging.getLogger(__name__)


class BrowserE2BEnv(BasicEnv):
    """Browser environment extended from E2B (https://e2b.dev/docs).
    Here we used TencentCloud's agent sandbox service. https://cloud.tencent.com/product/agentsandbox
    Sample config: configs/agents/examples/browser_e2b/browser_e2b.yaml

    Variables:
        sandbox: AsyncSandbox instance
        browser_cdp_url: CDP endpoint URL for the browser
        sandbox_novnc_url: NoVNC URL for accessing the browser GUI
        mcp_server: MCP server instance
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    async def build(self):
        """Build the environment. We use docker to run a browser container."""
        from e2b import AsyncSandbox

        # start browser sandbox
        self.sandbox: AsyncSandbox = await AsyncSandbox.create(template="browser-v1", timeout=3600)
        sandbox_url = self.sandbox.get_host(9000)
        self.sandbox_novnc_url = (
            f"https://{sandbox_url}/novnc/vnc_lite.html?&path=websockify?access_token={self.sandbox._envd_access_token}"
        )
        self.browser_cdp_url = f"https://{sandbox_url}/cdp"
        logger.info(f"browser sandbox created: {self.sandbox.sandbox_id}. vnc url: {self.sandbox_novnc_url}")

        # run mcp server
        config = ToolkitConfig(
            mode="mcp",
            mcp_transport="stdio",
            config={
                "command": "npx",
                "args": [
                    "-y",
                    "@playwright/mcp@latest",
                    "--cdp-endpoint",
                    self.browser_cdp_url,
                    "--cdp-header",
                    f"X-Access-Token: {self.sandbox._envd_access_token}",
                ],
            },
        )
        self.mcp_server = AgentsMCPUtils.get_mcp_server(config)
        await self.mcp_server.connect()

    async def cleanup(self):
        await self.mcp_server.cleanup()
        await self.sandbox.kill()

    async def get_tools(self) -> list[Tool]:
        """Get the tools available in the environment."""
        return await AgentsMCPUtils.get_tools_agents(self.mcp_server)
