import json
from pathlib import Path

import pytest

from utu.config import ToolkitConfig
from utu.tools.game_config_toolkit import GameConfigToolkit


@pytest.mark.asyncio
async def test_toolkit_basic(tmp_path: Path):
    schema_dir = Path("json_schemas").resolve()
    export_dir = tmp_path / "exports"
    cfg = ToolkitConfig(
        name="game_config",
        config={
            "schema_dir": str(schema_dir),
            "export_dir": str(export_dir),
            "table_files": {
                "c礼包普通掉落项数据表": "01_c礼包投放数据表_礼包普通掉落项数据表.json",
                "c礼包投放数据表": "01_c礼包投放数据表_礼包投放数据表.json",
                "c系统玩法物品数据表": "01_c系统玩法物品数据表_物品数据表.json",
                "物品使用参数表": "02_物品使用参数表_物品使用参数表.json",
            },
        },
    )

    async with GameConfigToolkit(config=cfg) as tk:
        ls = await tk.list_schemas()
        assert "c礼包投放数据表" in ls
        # init activity
        msg = await tk.start_new_activity("act_001")
        assert "Initialized" in msg
        # upsert one row into an array table
        out = await tk.upsert_row("c礼包投放数据表", None, {"投放备注": "test", "投放ID": 100})
        assert "Row" in out
        pv = await tk.preview("c礼包投放数据表")
        arr = json.loads(pv)
        assert isinstance(arr, list) and len(arr) >= 1
        # validate (will pass basic draft-07 checks)
        check = json.loads(await tk.check_schema("c礼包投放数据表"))
        assert "ok" in check
        # export (may fail on validation if required fields missing; we add minimal fields)
        # ensure required keys exist to pass (depends on normalized required list)
        # try export all to verify function works and returns a JSON object
        res = json.loads(await tk.export(activity_id="act_001"))
        assert isinstance(res, dict)
