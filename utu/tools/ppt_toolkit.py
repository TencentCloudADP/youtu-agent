import json
import os
import yaml

from typing import TYPE_CHECKING, Any

from agents import FunctionTool, RunContextWrapper, TContext, TResponseInputItem, function_tool

from ..config import ToolkitConfig
from ..utils import get_logger, SimplifiedAsyncOpenAI
from ..utils.agents_utils import ChatCompletionConverter
from .base import AsyncBaseToolkit, register_tool
from .ppt_utils import fill_template_with_yaml_config

logger = get_logger(__name__)

JSON_SCHEMA_TEMPLATE = """
## JSON Schema

```json
{schema}
```
"""


def remove_additional_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively remove 'additionalProperties' from a JSON schema.
    This is needed because FunctionTool requires strict schema without additionalProperties.
    
    Args:
        schema: The JSON schema to clean
        
    Returns:
        The cleaned schema
    """
    if not isinstance(schema, dict):
        return schema
    
    # Create a copy to avoid modifying the original
    cleaned = {}
    for key, value in schema.items():
        if key == 'additionalProperties':
            # Skip additionalProperties
            continue
        elif isinstance(value, dict):
            # Recursively clean nested dicts
            cleaned[key] = remove_additional_properties(value)
        elif isinstance(value, list):
            # Clean items in lists
            cleaned[key] = [
                remove_additional_properties(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    
    return cleaned


def _map_field(field_name: str, spec: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    spec = spec or {}
    t = spec.get("type")
    desc = spec.get("description")
    min_len = spec.get("min_len")
    max_len = spec.get("max_len")
    optional_flag = spec.get("optional")
    is_required = not optional_flag if optional_flag is not None else True

    def add_len_constraints(obj: dict[str, Any]) -> dict[str, Any]:
        if max_len is not None:
            # Interpret "words" limits as maxLength as a simple approximation
            obj["maxLength"] = max_len
        if min_len is not None and obj.get("type") == "string":
            obj["minLength"] = min_len
        return obj

    if t in ("str", "string"):
        prop = {"type": "string"}
        if desc:
            prop["description"] = desc
        return add_len_constraints(prop), is_required

    if t in ("int", "integer"):
        prop = {"type": "integer"}
        if desc:
            prop["description"] = desc
        return prop, is_required

    if t == "str_list":
        items: dict[str, Any] = {"type": "string"}
        # If exactly one char per element is desired, YAML example uses description; add a pattern helper if max_len==1
        if max_len == 1:
            items["pattern"] = "^.{1}$"
        prop = {"type": "array", "items": items}
        if desc:
            prop["description"] = desc
        if min_len is not None:
            prop["minItems"] = min_len
        if max_len is not None:
            prop["maxItems"] = max_len
        return prop, is_required

    if t == "item_list":
        prop = {
            "type": "array",
            "items": {"$ref": "#/$defs/Item"},
        }
        if desc:
            prop["description"] = desc
        if min_len is not None:
            prop["minItems"] = min_len
        if max_len is not None:
            prop["maxItems"] = max_len
        return prop, is_required

    if t == "content_list":
        prop = {
            "type": "array",
            "items": {"$ref": "#/$defs/BaseContent"},
        }
        if desc:
            prop["description"] = desc
        if min_len is not None:
            prop["minItems"] = min_len
        if max_len is not None:
            prop["maxItems"] = max_len
        return prop, is_required

    if t == "content":
        prop = {"$ref": "#/$defs/BaseContent"}
        if desc:
            prop["description"] = desc
        return prop, is_required

    if t == "image":
        prop = {"$ref": "#/$defs/BasicImage"}
        if desc:
            prop["description"] = desc
        return prop, is_required

    # Fallback: treat unknown as free-form string
    prop = {"type": "string"}
    if desc:
        prop["description"] = desc
    return prop, is_required


def _page_to_schema(page_key: str, page_spec: dict[str, Any], allowed_types: set[str]) -> dict[str, Any]:
    title_text = page_spec.get("description") or page_key

    properties: dict[str, Any] = {}
    required = ["type"]

    # fixed type const with validation against allowed types
    page_type = page_spec.get("type")
    if not page_type:
        raise ValueError(f"Page '{page_key}' must specify a 'type' and it must be one of: {sorted(allowed_types)}")
    if page_type not in allowed_types:
        raise ValueError(
            f"Page '{page_key}' has type '{page_type}' not present in type_map. Allowed types: {sorted(allowed_types)}"
        )
    properties["type"] = {"const": str(page_type)}

    # other fields
    for field, spec in page_spec.items():
        if field in {"description", "type"}:
            continue
        properties[field], field_required = _map_field(field, spec)
        if field_required:
            required.append(field)

    return {
        "type": "object",
        "title": title_text,
        "properties": properties,
        "required": sorted(set(required)),
        "additionalProperties": False,
    }


def build_schema(yaml_root: dict[str, Any]) -> dict[str, Any]:
    # build allowed types from type_map in YAML (list of single-key mappings)
    allowed_types: set[str] = set()
    tm = yaml_root.get("type_map")
    if isinstance(tm, list):
        for item in tm:
            if isinstance(item, dict):
                for t in item.keys():
                    allowed_types.add(str(t))
    # Fallback: infer from page specs if type_map missing
    if not allowed_types:
        for key, value in yaml_root.items():
            if isinstance(value, dict) and key != "type_map":
                t = value.get("type")
                if t:
                    allowed_types.add(str(t))

    one_of = []
    # iterate keys excluding type_map
    for key, value in yaml_root.items():
        if key == "type_map":
            continue
        if not isinstance(value, dict):
            continue
        one_of.append(_page_to_schema(key, value, allowed_types))

    schema: dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Slide Deck Structure",
        "description": "Schema to represent a structured set of slides for presentations.",
        "type": "object",
        "properties": {
            "slides": {
                "type": "array",
                "items": {"oneOf": one_of},
            }
        },
        "required": ["slides"],
        "additionalProperties": False,
        "$defs": {
            # Minimal defs aligned with template.schema.json
            "BaseContent": {
                "type": "object",
                "discriminator": {
                    "propertyName": "content_type",
                    "mapping": {
                        "text": "#/$defs/TextContent",
                        "image": "#/$defs/ImageContent",
                        "table": "#/$defs/TableContent",
                    },
                },
                "properties": {
                    "content_type": {
                        "type": "string",
                        "enum": ["text", "image", "table"],
                    }
                },
                "required": ["content_type"],
            },
            "BasicImage": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "format": "uri"},
                },
                "required": ["image_url"],
                "additionalProperties": False,
            },
            "Paragraph": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "bullet": {"type": "boolean", "default": False},
                    "level": {"type": "integer", "minimum": 0},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            "Item": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "maxLength": 4},
                    "content": {"type": "string", "maxLength": 10},
                },
                "required": ["title", "content"],
                "additionalProperties": False,
            },
            "TextContent": {
                "allOf": [
                    {"$ref": "#/$defs/BaseContent"},
                    {
                        "type": "object",
                        "properties": {
                            "paragraph": {
                                "oneOf": [
                                    {"type": "array", "items": {"$ref": "#/$defs/Paragraph"}, "minItems": 1},
                                    {"type": "string"},
                                ]
                            }
                        },
                        "required": ["paragraph"],
                    },
                ]
            },
            "ImageContent": {
                "allOf": [
                    {"$ref": "#/$defs/BaseContent"},
                    {
                        "type": "object",
                        "properties": {
                            "image_url": {"type": "string", "format": "uri"},
                            "caption": {"type": "string", "maxLength": 20},
                        },
                        "required": ["image_url"],
                    },
                ],
            },
            "TableContent": {
                "allOf": [
                    {"$ref": "#/$defs/BaseContent"},
                    {
                        "type": "object",
                        "properties": {
                            "header": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                            "rows": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                                "minItems": 1,
                                "maxItems": 7,
                            },
                            "caption": {"type": "string", "maxLength": 20},
                            "n_rows": {"type": "integer", "minimum": 1, "maximum": 7},
                            "n_cols": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "required": ["header", "rows", "n_rows", "n_cols"],
                    },
                ]
            },
        },
    }
    return schema


class PPTToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig | dict | None = None):
        super().__init__(config)
        # Initialize LLM
        self.llm = SimplifiedAsyncOpenAI(
            **self.config.config_llm.model_provider.model_dump() if self.config.config_llm else {}
        )
        # Load prompts from config
        self.ppt_generation_prompt = self.config.config.get("ppt_generation_prompt")
        # Load default config
        self.workdir = self.config.config.get("workdir")
        self.default_template_path = os.path.join(self.workdir, self.config.config.get("default_template_path"))
        self.default_template_config_path = os.path.join(self.workdir, self.config.config.get("default_template_config_path"))
        self.default_schema_path = os.path.join(self.workdir, self.config.config.get("default_schema_path"))

        # init generation config
        self.template_path = self.default_template_path
        with open(self.default_template_config_path, "r") as f:
            self.yaml_config = yaml.safe_load(f)
        self.schema = build_schema(self.yaml_config)
        self.instructions_to_generate_ppt = self.ppt_generation_prompt.format(schema=json.dumps(self.schema, indent=2))
    
    def _prepare_messages(self, trajectory: list[TResponseInputItem] | None, prompt: str) -> list[dict]:
        """
        Prepare messages by combining trajectory and current prompt.
        
        Args:
            trajectory: Historical conversation trajectory (from RunContextWrapper)
            prompt: Current prompt to add
            
        Returns:
            list[dict]: Combined messages list
        """
        messages = []
        if trajectory:
            # Convert TResponseInputItem to standard OpenAI message format
            messages.extend(ChatCompletionConverter.items_to_messages(trajectory))
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        return messages
    
    def get_tools_in_agents(self) -> list[FunctionTool]:
        """
        Override to create custom FunctionTool with trajectory auto-injection.
        """
        tools_map = self.get_tools_map_func()
        tools = []
        
        for tool_name, tool_func in tools_map.items():
            # Get the function schema
            schema = function_tool(tool_func, strict_mode=False)
            
            # Remove 'trajectory' from the schema so LLM won't generate it
            if schema.params_json_schema and 'properties' in schema.params_json_schema:
                schema.params_json_schema['properties'].pop('trajectory', None)
                if 'required' in schema.params_json_schema and 'trajectory' in schema.params_json_schema['required']:
                    schema.params_json_schema['required'].remove('trajectory')
            
            # Clean the schema to remove additionalProperties (required for strict mode)
            cleaned_schema = remove_additional_properties(schema.params_json_schema) if schema.params_json_schema else {}
            
            # Create custom on_invoke_tool that injects trajectory
            def create_on_invoke_tool(func):
                async def on_invoke_tool(ctx: RunContextWrapper[TContext], input_json: str) -> str:
                    import json
                    # Parse the arguments from LLM
                    arguments = json.loads(input_json)
                    
                    # Get trajectory from context
                    trajectory = None
                    if ctx and ctx.context:
                        # Try to get input_list from context
                        # The input_list is the current conversation history (trajectory)
                        trajectory = ctx.context.get('input_list', None)
                    
                    # Inject trajectory into arguments
                    arguments['trajectory'] = trajectory
                    
                    # Call the actual tool function
                    result = await func(**arguments)
                    return result
                
                return on_invoke_tool
            
            # Create FunctionTool with custom on_invoke_tool and cleaned schema
            tools.append(
                FunctionTool(
                    name=schema.name,
                    description=schema.description,
                    params_json_schema=cleaned_schema,
                    on_invoke_tool=create_on_invoke_tool(tool_func),
                )
            )
        
        return tools

    @register_tool
    async def load_template(self, config_path: str = None, template_path: str = None) -> str:
        if config_path:
            with open(config_path, "r") as f:
                self.yaml_config = yaml.safe_load(f)
            self.schema = build_schema(self.yaml_config)
            self.instructions_to_generate_ppt = self.ppt_generation_prompt.format(schema=json.dumps(self.schema, indent=2))
        if template_path:
            self.template_path = template_path
        return "Template loaded successfully."

    @register_tool
    async def generate_ppt(self, task: str, output_file_name: str, trajectory: list[TResponseInputItem] | None = None) -> str:
        messages = self._prepare_messages(trajectory, self.instructions_to_generate_ppt + "\n" + task)
        json_schema = await self.llm.query_one(
            messages=messages, **self.config.config_llm.model_params.model_dump()
        )
        json_schema = json_schema.replace("```json", "").replace("```", "")
        fill_template_with_yaml_config(
            template_path=self.template_path,
            output_path=output_file_name,
            json_data=json_schema,
            yaml_config=self.yaml_config,
        )
        return f"PPT generated successfully. Outline is: {json_schema}. Saved as {output_file_name}."
        