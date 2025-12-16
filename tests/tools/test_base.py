from utu.config import ConfigLoader
from utu.tools import SearchToolkit
from utu.tools.utils import get_tools_map, get_tools_schema


async def test_tools_map():
    config = ConfigLoader.load_toolkit_config("search")
    toolkit = SearchToolkit(config=config)
    print(toolkit.tools_map)
    print(toolkit.get_tools_in_agents())


def test_get_tools_map():
    print(get_tools_map(SearchToolkit))


def test_get_tools_schema():
    print(get_tools_schema(SearchToolkit))
