"""Utils to inspect tools
- load all tools in TOOLKIT_MAP;
- save the tool infos into .xlsx;"""

import pandas as pd
from agents.function_schema import FuncSchema

from utu.tools import TOOLKIT_MAP, get_tools_schema


def get_tool_schema() -> dict[str, FuncSchema]:
    tool_schemas = {}
    for _, toolkit in TOOLKIT_MAP.items():
        tool_schemas.update(get_tools_schema(toolkit))
    save_tools_info(tool_schemas.values())
    return tool_schemas


def save_tools_info(tools: list[FuncSchema]):
    tools_schema_list = []
    for tool in tools:
        tools_schema_list.append(
            {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.params_json_schema,
            }
        )
    df = pd.DataFrame(tools_schema_list)
    # df.to_csv("tools.csv", index=False)
    df.to_excel("tools.xlsx", index=False)
    print(df)


if __name__ == "__main__":
    get_tool_schema()
