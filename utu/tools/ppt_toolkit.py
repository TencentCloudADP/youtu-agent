import json
import uuid
import os
import requests
import yaml

from typing import TYPE_CHECKING, Any
from PIL import Image

from agents import FunctionTool, RunContextWrapper, TContext, TResponseInputItem, function_tool

from ..config import ToolkitConfig
from ..utils import get_logger, SimplifiedAsyncOpenAI
from ..utils.agents_utils import ChatCompletionConverter
from .base import AsyncBaseToolkit, register_tool
from .pptx_utils.fill_template import fill_template_with_yaml_config
from .pptx_utils.gen_schema import build_schema

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


class PPTToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig | dict | None = None):
        super().__init__(config)
        # Initialize LLM
        self.llm = SimplifiedAsyncOpenAI(
            **self.config.config_llm.model_provider.model_dump() if self.config.config_llm else {}
        )
        # Load prompts from config
        self.ppt_generation_prompt = self.config.config.get("ppt_generation_prompt")
        # Load default workdirs
        self.workdir = self.config.config.get("workdir")
        self.template_dir = self.config.config.get("template_dir")
        # Load default pptx config
        self.yaml_config_path = os.path.join(self.workdir, self.config.config.get("default_yaml_config_path"))
        # Load template config
        self.template_index = str(self.config.config.get("default_template_index"))
        self.template_yaml_config_path = os.path.join(self.template_dir, self.template_index, f"{self.template_index}.yaml")
        self.template_path = os.path.join(self.template_dir, self.template_index, f"{self.template_index}.pptx")
        # Init generation config
        with open(self.yaml_config_path, "r") as f:
            self.yaml_config = yaml.safe_load(f)
        # Overwrite yaml_config with template_yaml_config
        if os.path.exists(self.template_yaml_config_path):
            with open(self.template_yaml_config_path, "r") as f:
                self.yaml_config.update(yaml.safe_load(f))
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
    async def load_template(self, template_index: str, trajectory: list[TResponseInputItem] | None = None) -> str:
        raise NotImplementedError

    @register_tool
    async def download_image_url(self, image_url: str, image_description: str, trajectory: list[TResponseInputItem] | None = None) -> str:
        """
        Check if an image URL is accessible and can be used for PPT generation.

        Args:
            url: The image URL to check
            image_description: Description of the image content (e.g., "company logo", "product screenshot")
            trajectory: Conversation history (auto-injected)

        Returns:
            Status message indicating if the image is available, including the description if provided
        """
        if image_url.startswith("http"):
            headers = {"Accept": "image/*, */*"}
            response = requests.get(image_url, headers=headers)
            extension_name = image_url.split(".")[-1]
            if extension_name not in ["png", "jpg", "jpeg", "gif", "bmp", "webp"]:
                extension_name = "png"
            if response.status_code == 200:
                file_name = f"{uuid.uuid4()}.{extension_name}"
                with open(file_name, "wb") as f:
                    f.write(response.content)
                image = Image.open(file_name)
                width, height = image.size
                if width > 800 and height > 600:
                    return f"✓ Image is available and can be used for PPT generation: {file_name}. Description: {image_description}"
        return f"✗ Image is NOT available {image_url}."

    @register_tool
    async def generate_json_schema(self, task: str, trajectory: list[TResponseInputItem] | None = None) -> str:
        messages = self._prepare_messages(trajectory, self.instructions_to_generate_ppt + "\n" + task)
        json_schema = await self.llm.query_one(
            messages=messages, **self.config.config_llm.model_params.model_dump()
        )
        json_schema = json_schema.replace("```json", "").replace("```", "").strip()
        return json_schema
    
    @register_tool
    async def fill_template(self, json_schema: str, output_file_name: str, trajectory: list[TResponseInputItem] | None = None) -> str:
        fill_template_with_yaml_config(
            template_path=self.template_path,
            output_path=output_file_name,
            json_data=json_schema,
            yaml_config=self.yaml_config,
        )
        return f"PPT generated successfully. Outline is: {json_schema}. Saved as {output_file_name}."
        