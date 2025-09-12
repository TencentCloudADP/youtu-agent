import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import ToolkitConfig
from ..utils import get_logger
from .base import AsyncBaseToolkit, register_tool

logger = get_logger(__name__)


@dataclass
class _TableState:
    name: str
    schema_path: Path
    schema: dict
    data: Any | None = None  # current working JSON for this table


class GameConfigToolkit(AsyncBaseToolkit):
    """Schema-driven editor for game operation activity configuration tables."""

    def __init__(self, config: ToolkitConfig | None = None) -> None:
        super().__init__(config)
        cfg = self.config.config or {}
        # where schemas are stored
        self.schema_dir = Path(cfg.get("schema_dir", "./json_schemas")).resolve()
        # where exports will be written
        self.export_dir = Path(cfg.get("export_dir", "./output/game_configs")).resolve()
        # logical table name -> filename
        self.table_files: dict[str, str] = cfg.get(
            "table_files",
            {
                "c礼包普通掉落项数据表": "01_c礼包投放数据表_礼包普通掉落项数据表.json",
                "c礼包投放数据表": "01_c礼包投放数据表_礼包投放数据表.json",
                "c系统玩法物品数据表": "01_c系统玩法物品数据表_物品数据表.json",
                "物品使用参数表": "02_物品使用参数表_物品使用参数表.json",
            },
        )
        # runtime state
        self.activity_id: str | None = None
        self._tables: dict[str, _TableState] = {}

        # jsonschema availability flag
        try:
            import jsonschema  # noqa: F401

            self._jsonschema_available = True
        except Exception:  # pragma: no cover
            self._jsonschema_available = False

    async def build(self):
        await super().build()
        self._load_all_schemas()
        self.export_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "GameConfigToolkit initialized. schema_dir=%s export_dir=%s tables=%s",
            self.schema_dir,
            self.export_dir,
            list(self.table_files.keys()),
        )

    # --------------------------- helpers ---------------------------
    def _load_all_schemas(self) -> None:
        assert self.table_files, "config.table_files must be provided"
        self._tables.clear()
        for logical_name, filename in self.table_files.items():
            path = (self.schema_dir / filename).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Schema file not found: {path}")
            with open(path, encoding="utf-8") as f:
                try:
                    raw = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON schema in {path}: {e}") from e
            schema = self._normalize_schema(raw, logical_name)
            self._tables[logical_name] = _TableState(name=logical_name, schema_path=path, schema=schema, data=None)

    @staticmethod
    def _normalize_schema(raw: Any, title: str) -> dict:
        # If already JSON Schema-like
        if isinstance(raw, dict) and ("type" in raw or "properties" in raw or "items" in raw):
            return raw

        if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "name" in raw[0]:
            def map_type(t: str) -> str:
                t = (t or "").lower()
                if t in ["int", "integer"]:
                    return "integer"
                if t in ["float", "double", "number"]:
                    return "number"
                if t in ["bool", "boolean"]:
                    return "boolean"
                return "string"

            properties: dict[str, dict] = {}
            required: list[str] = []
            for col in raw:
                name = str(col.get("name"))
                jtype = map_type(str(col.get("type")))
                desc = col.get("desc") or col.get("description")
                properties[name] = {"type": jtype}
                if desc:
                    properties[name]["description"] = desc
                req = str(col.get("required", "")).lower()
                if req in ["required", "true", "yes", "y"]:
                    required.append(name)
            return {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "title": title,
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": True,
                },
            }

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": title,
            "type": "object",
        }

    def _ensure_activity(self) -> None:
        if not self.activity_id:
            raise ValueError("No activity initialized. Call start_new_activity first with an activity_id.")

    def _ensure_table(self, table: str) -> _TableState:
        if table not in self._tables:
            raise KeyError(f"Unknown table '{table}'. Available: {list(self._tables.keys())}")
        return self._tables[table]

    @staticmethod
    def _schema_top_type(schema: dict) -> str | None:
        t = schema.get("type")
        if isinstance(t, list):
            return t[0]
        return t

    @staticmethod
    def _pretty(obj: Any, compact: bool = False) -> str:
        if compact:
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        return json.dumps(obj, ensure_ascii=False, indent=2)

    @staticmethod
    def _coerce_value(value: Any, json_type: str) -> Any:
        try:
            if json_type == "integer":
                if isinstance(value, bool):
                    return int(value)
                return int(str(value).strip())
            if json_type == "number":
                if isinstance(value, bool):
                    return int(value)
                return float(str(value).strip())
            if json_type == "boolean":
                if isinstance(value, bool):
                    return value
                if isinstance(value, int | float):
                    return bool(value)
                if isinstance(value, str):
                    v = value.strip().lower()
                    if v in ["true", "1", "yes", "y"]:
                        return True
                    if v in ["false", "0", "no", "n"]:
                        return False
                return value
            return value if isinstance(value, str) else str(value)
        except Exception:
            return value

    @staticmethod
    def _set_by_path(obj: Any, path: str, value: Any) -> None:
        parts: list[str] = []
        buf = ""
        i = 0
        while i < len(path):
            ch = path[i]
            if ch == ".":
                if buf:
                    parts.append(buf)
                    buf = ""
                i += 1
                continue
            if ch == "[":
                if buf:
                    parts.append(buf)
                    buf = ""
                j = path.find("]", i)
                if j == -1:
                    raise ValueError(f"Invalid path: {path}")
                idx = path[i + 1 : j]
                parts.append(f"[{idx}]")
                i = j + 1
                continue
            buf += ch
            i += 1
        if buf:
            parts.append(buf)

        ref = obj
        for k in parts[:-1]:
            if k.startswith("["):
                idx = int(k[1:-1])
                ref = ref[idx]
            else:
                ref = ref.setdefault(k, {})
        last = parts[-1]
        if last.startswith("["):
            idx = int(last[1:-1])
            ref[idx] = value
        else:
            ref[last] = value

    # --------------------------- tools ---------------------------
    @register_tool()
    async def list_schemas(self) -> str:
        lines = []
        for k, t in self._tables.items():
            title = t.schema.get("title", "")
            top = self._schema_top_type(t.schema) or "unknown"
            lines.append(f"- {k}: type={top}, title={title}, file={t.schema_path.name}")
        return "\n".join(lines)

    @register_tool()
    async def read_schema(self, table: str) -> str:
        ts = self._ensure_table(table)
        schema = ts.schema
        top = self._schema_top_type(schema)
        info = {"top_type": top, "title": schema.get("title"), "description": schema.get("description")}
        if top == "array":
            items = schema.get("items", {})
            props = items.get("properties", {})
            required = items.get("required", [])
        else:
            props = schema.get("properties", {})
            required = schema.get("required", [])
        fields = []
        for name, prop in props.items():
            fields.append(
                {
                    "name": name,
                    "type": prop.get("type"),
                    "required": name in required,
                    "description": prop.get("description"),
                    "enum": prop.get("enum"),
                }
            )
        info["fields"] = fields
        return self._pretty(info)

    @register_tool()
    async def start_new_activity(self, activity_id: str) -> str:
        self.activity_id = activity_id
        for ts in self._tables.values():
            top = self._schema_top_type(ts.schema)
            ts.data = [] if top == "array" else {}
        return f"Initialized activity: {activity_id}. Tables reset."

    @register_tool()
    async def set_field(self, table: str, field_path: str, value: Any) -> str:
        self._ensure_activity()
        ts = self._ensure_table(table)
        top = self._schema_top_type(ts.schema)
        if top == "array":
            return "Error: set_field is for object top-level schemas. Use upsert_row for array tables."
        props = ts.schema.get("properties", {})
        head = field_path.split(".")[0]
        if head in props and isinstance(props[head], dict):
            jtype = props[head].get("type", "string")
            value = self._coerce_value(value, jtype)
        if ts.data is None:
            ts.data = {}
        self._set_by_path(ts.data, field_path, value)
        return f"Set {table}.{field_path} = {value}"

    @register_tool()
    async def upsert_row(self, table: str, selector: dict | None, updates: dict) -> str:
        self._ensure_activity()
        ts = self._ensure_table(table)
        top = self._schema_top_type(ts.schema)
        if top != "array":
            return "Error: upsert_row is only for array top-level tables."
        if ts.data is None:
            ts.data = []
        items = ts.schema.get("items", {})
        props = items.get("properties", {})
        coerced = {}
        for k, v in updates.items():
            jtype = props.get(k, {}).get("type", "string") if isinstance(props.get(k, {}), dict) else "string"
            coerced[k] = self._coerce_value(v, jtype)
        row = None
        idx = -1
        if selector:
            for i, r in enumerate(ts.data):
                if isinstance(r, dict) and all(r.get(sk) == sv for sk, sv in selector.items()):
                    row = r
                    idx = i
                    break
        if row is None:
            ts.data.append(coerced.copy())
            idx = len(ts.data) - 1
            action = "inserted"
        else:
            row.update(coerced)
            action = "updated"
        return f"Row {action} at index {idx} in {table}."

    @register_tool()
    async def remove_row(self, table: str, selector: dict) -> str:
        self._ensure_activity()
        ts = self._ensure_table(table)
        if self._schema_top_type(ts.schema) != "array":
            return "Error: remove_row is only for array top-level tables."
        if not ts.data:
            return "No data."
        before = len(ts.data)
        ts.data = [r for r in ts.data if not (isinstance(r, dict) and all(r.get(k) == v for k, v in selector.items()))]
        after = len(ts.data)
        return f"Removed {before - after} rows from {table}."

    @register_tool()
    async def preview(self, table: str | None = None, compact: bool = False) -> str:
        self._ensure_activity()
        if table:
            ts = self._ensure_table(table)
            data = ts.data if ts.data is not None else ([] if self._schema_top_type(ts.schema) == "array" else {})
            return self._pretty(data, compact=compact)
        result = {
            k: (
                t.data
                if t.data is not None
                else ([] if self._schema_top_type(t.schema) == "array" else {})
            )
            for k, t in self._tables.items()
        }
        return self._pretty(result, compact=compact)

    def _validate_one(self, table: str) -> list[dict[str, Any]]:
        ts = self._ensure_table(table)
        data = ts.data if ts.data is not None else ([] if self._schema_top_type(ts.schema) == "array" else {})
        if not self._jsonschema_available:
            return [
                {
                    "table": table,
                    "path": "",
                    "message": "jsonschema not installed. Please install 'jsonschema' to enable validation.",
                }
            ]
        from jsonschema import Draft7Validator

        validator = Draft7Validator(ts.schema)
        errors = []
        for err in validator.iter_errors(data):
            path = "/".join([str(p) for p in err.path])
            errors.append({"table": table, "path": path, "message": err.message})
        return errors

    @register_tool()
    async def check_schema(self, table: str | None = None) -> str:
        self._ensure_activity()
        tables = [table] if table else list(self._tables.keys())
        all_errors: list[dict[str, Any]] = []
        for t in tables:
            all_errors.extend(self._validate_one(t))
        result = {"ok": len(all_errors) == 0, "errors": all_errors}
        return self._pretty(result)

    @register_tool()
    async def export(self, table: str | None = None, activity_id: str | None = None) -> str:
        if activity_id:
            self.activity_id = activity_id
        self._ensure_activity()
        check = json.loads(await self.check_schema(table))
        if not check.get("ok"):
            return self._pretty({"exported": False, "reason": "validation_failed", "errors": check.get("errors")})
        ts_id = self.activity_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        base = (self.export_dir / ts_id)
        base.mkdir(parents=True, exist_ok=True)
        written = []
        tables = [table] if table else list(self._tables.keys())
        for tname in tables:
            ts = self._ensure_table(tname)
            data = ts.data if ts.data is not None else ([] if self._schema_top_type(ts.schema) == "array" else {})
            fname = f"{tname}.json"
            out_path = base / fname
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            written.append(str(out_path))
        return self._pretty({"exported": True, "files": written})
