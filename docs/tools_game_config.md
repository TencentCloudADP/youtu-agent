# GameConfigToolkit

A schema-driven toolkit to help game ops configure four related tables via dialogue:

- c礼包普通掉落项数据表
- c礼包投放数据表
- c系统玩法物品数据表
- 物品使用参数表

It reads column-list JSON files in `json_schemas/` and normalizes them into JSON Schema for validation. It maintains in-memory working data, supports preview and export, and enforces `check_schema` before writing files.

## Tools
- list_schemas
- read_schema
- start_new_activity(activity_id)
- set_field(table, field_path, value)
- upsert_row(table, selector, updates)
- remove_row(table, selector)
- preview(table|all)
- check_schema(table|all)
- export(table|all, activity_id?)

## Config
See `configs/tools/game_config.yaml`. Add to an agent like `configs/agents/simple_agents/game_ops_config_agent.yaml`.
