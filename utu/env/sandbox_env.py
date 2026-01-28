import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from agents import FunctionTool, RunContextWrapper, TContext, Tool

from ..utils import DIR_ROOT
from .base_env import BaseEnv

logger = logging.getLogger(__name__)


class SandboxEnv(BaseEnv):
    """Sandbox environment backed by a remote session-based service.

    Config conventions:
    - sandbox_type: Logical sandbox subtype (e.g. "envscale").
    - base_url: Sandbox API base URL (e.g. https://...).
    - access_token / access_token_env: Adds X-Access-Token header.
    - run_id_source: "trace_id" | "explicit" (default trace_id).
    - run_id: Explicit run_id used when run_id_source="explicit".
    - task_id: Optional task_id for session creation.
    - session_create_mode: "run_id" | "task_id" to force which payload is used.
    - data_dir: Path containing interface_plan.json and generated_tasks.json.
    - interface_plan_path: Direct path to interface_plan.json (overrides data_dir).
    """

    def __init__(self, config: dict | None = None, trace_id: str | None = None, sandbox_type: str | None = None):
        config = config or {}
        # sandbox_type is a logical classifier (e.g. "envscale") for multi-sandbox support.
        self.sandbox_type = sandbox_type or config.get("sandbox_type") or "default"
        self.base_url = str(config.get("base_url", "http://localhost:8848")).rstrip("/")
        self.timeout = int(config.get("timeout", 60))
        self.headers = config.get("headers") or {}
        self._apply_access_token(config)

        # Session identity can be created from run_id or task_id.
        self.run_id_source = config.get("run_id_source", "trace_id")
        self.task_id = config.get("task_id")
        self.session_create_mode = config.get("session_create_mode")
        self.run_id = None
        self._resolve_session_identity(config.get("run_id"), trace_id)

        # Tool schemas are read from local files (interface_plan.json by default).
        self.data_dir = config.get("data_dir")
        self.interface_plan_path = config.get("interface_plan_path")
        self.tasks_path = config.get("tasks_path")
        self._resolve_paths()

        self.session_id: str | None = None
        self.tools_cache: list[Tool] | None = None
        self.last_state: dict | str | None = None
        self.last_delta: Any | None = None
        self.last_raw_response: str | None = None
        self.http_client: httpx.AsyncClient | None = None

    async def build(self) -> None:
        """Create a remote session and fetch initial state."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(base_url=self.base_url, headers=self.headers, timeout=self.timeout)

        # Session creation supports run_id or task_id payloads.
        logger.info(
            "Creating sandbox session type=%s mode=%s run_id=%s task_id=%s base_url=%s",
            self.sandbox_type,
            self.session_create_mode,
            self.run_id,
            self.task_id,
            self.base_url,
        )
        try:
            response = await self.http_client.post("/sessions", json=self._session_payload())
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Failed to create session: %s", exc, exc_info=True)
            raise

        try:
            data = response.json()
        except ValueError as exc:
            logger.error("Session creation response is not JSON: %s", response.text)
            raise RuntimeError("Invalid session creation response") from exc

        self.session_id = data.get("session_id")
        if not self.session_id:
            raise RuntimeError("Missing session_id in session creation response")
        logger.info("Sandbox session created session_id=%s", self.session_id)

        # Best-effort initial state fetch for prompt context.
        try:
            state_resp = await self.http_client.get(f"/sessions/{self.session_id}/state")
            state_resp.raise_for_status()
            self.last_state = self._parse_json_or_text(state_resp)
            logger.info("Fetched initial sandbox state")
        except httpx.HTTPError as exc:
            logger.warning("Failed to fetch initial state: %s", exc)

    async def cleanup(self) -> None:
        """Terminate the remote session and close HTTP client."""
        if self.session_id and self.http_client:
            try:
                response = await self.http_client.delete(f"/sessions/{self.session_id}")
                response.raise_for_status()
                logger.info("Sandbox session deleted session_id=%s", self.session_id)
            except httpx.HTTPError as exc:
                logger.warning("Failed to delete sandbox session %s: %s", self.session_id, exc)

        if self.http_client:
            await self.http_client.aclose()

        self.session_id = None
        self.http_client = None
        self.tools_cache = None
        self.last_state = None
        self.last_delta = None
        self.last_raw_response = None

    def get_state(self) -> str:
        if self.last_state is None:
            return ""
        if isinstance(self.last_state, str):
            return self.last_state
        return json.dumps(self.last_state, ensure_ascii=False)

    async def get_tools(self) -> list[Tool]:
        if self.tools_cache is not None:
            return self.tools_cache

        interface_plan = self._load_interface_plan()
        # Convert interface_plan.json definitions into JSON schema for FunctionTool.
        tool_schemas = [self._interface_plan_to_schema(tool_def) for tool_def in interface_plan]
        tools = [self._create_tool(schema) for schema in tool_schemas]
        self.tools_cache = tools
        logger.info("Loaded %s sandbox tools from %s", len(tools), self.interface_plan_path)
        return tools

    def _resolve_paths(self) -> None:
        data_dir = self._resolve_path(self.data_dir) if self.data_dir else None
        interface_plan_path = self._resolve_path(self.interface_plan_path) if self.interface_plan_path else None
        tasks_path = self._resolve_path(self.tasks_path) if self.tasks_path else None

        if interface_plan_path is None:
            if data_dir is None:
                raise ValueError("interface_plan_path or data_dir must be provided")
            # Default file name for tool schema.
            interface_plan_path = data_dir / "interface_plan.json"

        if tasks_path is None and data_dir is not None:
            tasks_path = data_dir / "generated_tasks.json"

        self.data_dir = data_dir
        self.interface_plan_path = interface_plan_path
        self.tasks_path = tasks_path
        logger.info(
            "Sandbox paths resolved interface_plan_path=%s tasks_path=%s",
            self.interface_plan_path,
            self.tasks_path,
        )

    @staticmethod
    def _resolve_path(path_value: str | Path) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            path = (DIR_ROOT / path).resolve()
        return path

    def _apply_access_token(self, config: dict) -> None:
        token = config.get("access_token")
        token_env = config.get("access_token_env")
        if not token and token_env:
            token = os.getenv(token_env)

        if token:
            if "X-Access-Token" not in self.headers:
                self.headers["X-Access-Token"] = token
                logger.info("Sandbox access token loaded from %s", "config" if config.get("access_token") else token_env)
        elif token_env:
            logger.warning("Sandbox access token env var %s is not set", token_env)

    def _resolve_session_identity(self, explicit_run_id: str | None, trace_id: str | None) -> None:
        # session_create_mode forces which identity (task_id/run_id) to use.
        if self.session_create_mode:
            if self.session_create_mode not in {"task_id", "run_id"}:
                raise ValueError(f"Unsupported session_create_mode: {self.session_create_mode}")
        if self.session_create_mode == "task_id":
            if not self.task_id:
                raise ValueError("task_id is required when session_create_mode is 'task_id'")
            return

        if self.session_create_mode == "run_id" or not self.task_id:
            self.run_id = self._resolve_run_id(self.run_id_source, explicit_run_id, trace_id)
            self.session_create_mode = "run_id"
            return

        self.session_create_mode = "task_id"

    def _session_payload(self) -> dict:
        # Only one of task_id/run_id is sent, matching the server API.
        if self.session_create_mode == "task_id":
            return {"task_id": self.task_id}
        return {"run_id": self.run_id}

    def _load_interface_plan(self) -> list[dict]:
        if not self.interface_plan_path:
            raise RuntimeError("interface_plan_path is not configured")
        if not self.interface_plan_path.exists():
            raise FileNotFoundError(f"interface_plan.json not found: {self.interface_plan_path}")

        data = json.loads(self.interface_plan_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("interface_plan.json must contain a list of tool definitions")
        return data

    def _interface_plan_to_schema(self, tool_def: dict) -> dict:
        # interface_plan.json is converted into an OpenAI-compatible JSON schema.
        name = tool_def.get("name")
        if not name:
            raise ValueError("Tool definition missing name field")
        description = tool_def.get("doc", "")
        returns = tool_def.get("returns")
        if returns:
            description = f"{description}\nReturns: {returns}" if description else f"Returns: {returns}"

        params = tool_def.get("params") or []
        properties: dict[str, dict] = {}
        required: list[str] = []

        for param in params:
            param_name = param.get("name")
            if not param_name:
                continue
            type_hint = param.get("type_hint", "")
            json_type = self._type_hint_to_json_type(type_hint)
            description_line = param.get("description", "")
            description_line = self._append_type_hint(description_line, type_hint, json_type)

            properties[param_name] = {
                "type": json_type,
                "description": description_line,
            }
            if self._is_required(type_hint):
                required.append(param_name)

        input_schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        return {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
        }

    @staticmethod
    def _append_type_hint(description: str, type_hint: str, json_type: str) -> str:
        if not type_hint:
            return description
        normalized = type_hint.strip()
        if json_type == "string" and normalized not in {"str", "string"}:
            suffix = f" (type_hint: {normalized})"
            return f"{description}{suffix}" if description else suffix.strip()
        return description

    @staticmethod
    def _is_required(type_hint: str) -> bool:
        if not type_hint:
            return True
        lower_hint = type_hint.lower()
        return not ("optional" in lower_hint or "none" in lower_hint)

    @staticmethod
    def _type_hint_to_json_type(type_hint: str) -> str:
        if not type_hint:
            return "string"
        normalized = type_hint.strip().lower()
        normalized = normalized.replace("optional[", "").replace("]", "")
        normalized = normalized.replace("| none", "").replace("none |", "").strip()

        if "list" in normalized or "[]" in normalized:
            return "array"
        if "dict" in normalized or "mapping" in normalized:
            return "object"
        if "int" in normalized:
            return "integer"
        if "float" in normalized or "double" in normalized or "number" in normalized:
            return "number"
        if "bool" in normalized:
            return "boolean"
        if "str" in normalized or "string" in normalized:
            return "string"
        return "string"

    def _create_tool(self, tool_schema: dict) -> FunctionTool:
        tool_name = tool_schema.get("name")
        description = tool_schema.get("description", "")
        params_schema = tool_schema.get("inputSchema", {})

        def create_on_invoke(name: str):
            async def on_invoke_tool(ctx: RunContextWrapper[TContext], input_json: str) -> str:
                # The agent passes params as JSON string; parse defensively.
                try:
                    params = json.loads(input_json) if input_json else {}
                except json.JSONDecodeError:
                    logger.warning("Tool input is not valid JSON for %s: %s", name, input_json)
                    params = {}
                return await self._call_step(name, params)

            return on_invoke_tool

        return FunctionTool(
            name=tool_name,
            description=description,
            params_json_schema=params_schema,
            on_invoke_tool=create_on_invoke(tool_name),
        )

    async def _call_step(self, tool_name: str, params: dict) -> str:
        if not self.session_id:
            return "Error: session is not initialized"
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(base_url=self.base_url, headers=self.headers, timeout=self.timeout)

        # Server expects a list of actions, even for a single tool call.
        payload = {"actions": [{"name": tool_name, "params": params}]}
        logger.info("Calling sandbox tool %s", tool_name)
        logger.debug("Sandbox tool params %s", params)

        try:
            response = await self.http_client.post(f"/sessions/{self.session_id}/step", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Sandbox tool call failed %s: %s", tool_name, exc, exc_info=True)
            return f"Error: {exc}"

        self.last_raw_response = response.text
        data = self._parse_json_or_text(response)
        if isinstance(data, str):
            # Non-JSON responses are returned as-is.
            return data

        executed = data.get("executed") or []
        if executed:
            first = executed[0]
            error = first.get("error")
            if error:
                return f"Error: {error}"
            result = first.get("result")
        else:
            result = data.get("result")

        # Update state caches for agent prompt/context.
        self.last_state = data.get("final_state", self.last_state)
        self.last_delta = data.get("delta")
        return self._format_result(result)

    @staticmethod
    def _parse_json_or_text(response: httpx.Response) -> dict | str:
        try:
            return response.json()
        except ValueError:
            return response.text

    @staticmethod
    def _format_result(result: Any) -> str:
        if isinstance(result, str):
            return result
        if result is None:
            return ""
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _resolve_run_id(run_id_source: str, explicit_run_id: str | None, trace_id: str | None) -> str:
        if run_id_source == "explicit":
            if not explicit_run_id:
                raise ValueError("run_id must be provided when run_id_source is 'explicit'")
            return str(explicit_run_id)
        if run_id_source == "trace_id":
            if not trace_id:
                raise ValueError("trace_id is required when run_id_source is 'trace_id'")
            return str(trace_id)
        raise ValueError(f"Unsupported run_id_source: {run_id_source}")
