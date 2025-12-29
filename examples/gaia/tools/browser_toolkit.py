"""
- [x] wrap browse toolkit from https://github.com/camel-ai/camel/blob/master/camel/toolkits/hybrid_browser_toolkit_py/hybrid_browser_toolkit.py
- [ ] special logic for Web Search Agent
"""

from typing import Any

from camel.models import ModelFactory
from camel.toolkits import HybridBrowserToolkit
from camel.toolkits.hybrid_browser_toolkit_py import HybridBrowserToolkit as PyToolkit
from camel.types import ModelPlatformType

from utu.config import ToolkitConfig
from utu.tools import AsyncBaseToolkit, register_tool
from utu.utils import get_logger

logger = get_logger(__name__)


class BrowserToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)

        llm_config = self.config.config_llm.model_provider
        web_agent_model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENROUTER,
            model_type=llm_config.model,  # model name
            api_key=llm_config.api_key,
            url=llm_config.base_url,
            model_config_dict={"temperature": 0},
        )
        self.toolkit: PyToolkit = HybridBrowserToolkit(
            mode="python",
            headless=True,
            web_agent_model=web_agent_model,
            cache_dir="tmp/shared_browser",
            navigation_timeout=60000,
            default_timeout=60000,
            default_start_url="www.google.com",
            # NOTE: the actual enabled tools in youtu-agent are set in YAML config!
            enabled_tools=[
                # "browser_open",
                "browser_visit_page",
                "browser_click",
                "browser_type",
                "browser_switch_tab",
                "browser_forward",
                "browser_back",
                # "browser_close",
            ],
        )

    @register_tool
    async def browser_open(self) -> dict[str, Any]:
        r"""Starts a new browser session. This must be the first browser
        action.

        This method initializes the browser and navigates to a default start
        page. To visit a specific URL, use `visit_page` after this.

        Returns:
            Dict[str, Any]: A dictionary with the result of the action:
                - "result" (str): Confirmation of the action.
                - "snapshot" (str): A textual snapshot of interactive
                elements.
                - "tabs" (List[Dict]): Information about all open tabs.
                - "current_tab" (int): Index of the active tab.
                - "total_tabs" (int): Total number of open tabs.
        """
        return await self.toolkit.browser_open()

    @register_tool
    async def browser_close(self) -> str:
        r"""Closes the browser session, releasing all resources.

        This should be called at the end of a task for cleanup.

        Returns:
            str: A confirmation message.
        """
        return await self.toolkit.browser_close()

    @register_tool
    async def browser_visit_page(self, url: str) -> dict[str, Any]:
        r"""Opens a URL in a new browser tab and switches to it.

        Args:
            url (str): The web address to load. This should be a valid and
                existing URL.

        Returns:
            Dict[str, Any]: A dictionary with the result of the action:
                - "result" (str): Confirmation of the action.
                - "snapshot" (str): A textual snapshot of the new page.
                - "tabs" (List[Dict]): Information about all open tabs.
                - "current_tab" (int): Index of the new active tab.
                - "total_tabs" (int): Total number of open tabs.
        """
        return await self.toolkit.browser_visit_page(url)

    @register_tool
    async def browser_click(self, *, ref: str) -> dict[str, Any]:
        r"""Performs a click on an element on the page.

        Args:
            ref (str): The `ref` ID of the element to click. This ID is
                obtained from a page snapshot (`get_page_snapshot` or
                `get_som_screenshot`).

        Returns:
            Dict[str, Any]: A dictionary with the result of the action:
                - "result" (str): Confirmation of the action.
                - "snapshot" (str): A textual snapshot of the page after the
                  click.
                - "tabs" (List[Dict]): Information about all open tabs.
                - "current_tab" (int): Index of the active tab.
                - "total_tabs" (int): Total number of open tabs.
        """
        return await self.toolkit.browser_click(ref=ref)

    @register_tool
    async def browser_type(self, *, ref: str, text: str) -> dict[str, Any]:
        r"""Types text into an input element on the page.

        Args:
            ref (str): The `ref` ID of the input element, from a snapshot.
            text (str): The text to type into the element.

        Returns:
            Dict[str, Any]: A dictionary with the result of the action:
                - "result" (str): Confirmation of the action.
                - "snapshot" (str): A textual snapshot of the page after
                  typing.
                - "tabs" (List[Dict]): Information about all open tabs.
                - "current_tab" (int): Index of the active tab.
                - "total_tabs" (int): Total number of open tabs.
        """
        return await self.toolkit.browser_type(ref=ref, text=text)

    @register_tool
    async def browser_switch_tab(self, *, tab_id: str) -> dict[str, Any]:
        r"""Switches to a different browser tab using its ID.

        After switching, all actions will apply to the new tab. Use
        `get_tab_info` to find the ID of the tab you want to switch to.

        Args:
            tab_id (str): The ID of the tab to activate.

        Returns:
            Dict[str, Any]: A dictionary with the result of the action:
                - "result" (str): Confirmation of the action.
                - "snapshot" (str): A snapshot of the newly active tab.
                - "tabs" (List[Dict]): Information about all open tabs.
                - "current_tab" (int): Index of the new active tab.
                - "total_tabs" (int): Total number of open tabs.
        """
        return await self.toolkit.browser_switch_tab(tab_id=tab_id)

    @register_tool
    async def browser_forward(self) -> dict[str, Any]:
        r"""Goes forward to the next page in the browser history.

        This action simulates using the browser's "forward" button in the
        currently active tab.

        Returns:
            Dict[str, Any]: A dictionary with the result of the action:
                - "result" (str): Confirmation of the action.
                - "snapshot" (str): A textual snapshot of the next page.
                - "tabs" (List[Dict]): Information about all open tabs.
                - "current_tab" (int): Index of the active tab.
                - "total_tabs" (int): Total number of open tabs.
        """
        return await self.toolkit.browser_forward()

    @register_tool
    async def browser_back(self) -> dict[str, Any]:
        r"""Goes back to the previous page in the browser history.

        This action simulates using the browser's "back" button in the
        currently active tab.

        Returns:
            Dict[str, Any]: A dictionary with the result of the action:
                - "result" (str): Confirmation of the action.
                - "snapshot" (str): A textual snapshot of the previous page.
                - "tabs" (List[Dict]): Information about all open tabs.
                - "current_tab" (int): Index of the active tab.
                - "total_tabs" (int): Total number of open tabs.
        """
        return await self.toolkit.browser_back()
