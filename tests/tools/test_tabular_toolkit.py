import pytest

from utu.config import ConfigLoader
from utu.tools import TabularDataToolkit


@pytest.fixture
def tabular_toolkit() -> TabularDataToolkit:
    config = ConfigLoader.load_toolkit_config("tabular")
    return TabularDataToolkit(config=config)


async def test_tabular_toolkit(tabular_toolkit: TabularDataToolkit):
    fn = "examples/data_analysis/demo_data_cat_breeds_clean.csv"
    result = tabular_toolkit.get_tabular_columns(fn)
    print(result)
    result = await tabular_toolkit.get_column_info(fn)
    print(result)
