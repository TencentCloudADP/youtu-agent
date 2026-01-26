import os
from typing import TYPE_CHECKING, Any

from agents import FunctionTool, RunContextWrapper, TContext, TResponseInputItem, function_tool

from ..config import ToolkitConfig
from ..utils import get_logger, SimplifiedAsyncOpenAI
from ..utils.agents_utils import ChatCompletionConverter
from .base import AsyncBaseToolkit, register_tool

logger = get_logger(__name__)


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


class PPTHtmlBasedToolkit(AsyncBaseToolkit):
    """Toolkit for generating HTML-based PowerPoint presentations."""

    def __init__(self, config: ToolkitConfig | dict | None = None):
        super().__init__(config)
        # Initialize LLM
        self.llm = SimplifiedAsyncOpenAI(
            **self.config.config_llm.model_provider.model_dump() if self.config.config_llm else {}
        )
        # Load prompts from config
        self.ppt_generation_prompt = self.config.config.get("ppt_generation_prompt", "")
        self.ppt_refinement_prompt = self.config.config.get("ppt_refinement_prompt", "")
        # Load HTML template path from config or use default
        self.html_template_path = self.config.config.get("html_template_path", "")
        # Load HTML template
        self.html_template = self._load_html_template()

    def _load_html_template(self) -> str:
        """Load HTML template from file."""
        if self.html_template_path and os.path.exists(self.html_template_path):
            with open(self.html_template_path, "r", encoding="utf-8") as f:
                return f.read()
        logger.warning(f"HTML template not found at {self.html_template_path}, using empty template")
        return ""

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

    def _extract_html_content(self, response: str) -> str:
        """
        Extract HTML content from LLM response.

        Args:
            response: The raw response from LLM

        Returns:
            str: Extracted HTML content
        """
        # Remove markdown code blocks if present
        content = response.strip()
        if content.startswith("```html"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()

    @register_tool
    async def generate_html_ppt(
        self,
        task: str,
        output_file_name: str,
        trajectory: list[TResponseInputItem] | None = None,
    ) -> str:
        """
        Generate an HTML-based PowerPoint presentation.

        Args:
            task: The main task description about the PowerPoint presentation.
            output_file_name: The name of the PowerPoint file to be generated (e.g., example.html).
            trajectory: Conversation history (auto-injected).

        Returns:
            str: Status message indicating the result of the generation.
        """
        try:
            # Build prompt for generation (include html_template and user_query)
            prompt = self.ppt_generation_prompt.format(
                html_template=self.html_template,
                user_query=task
            )
            
            # Prepare messages with trajectory
            messages = self._prepare_messages(trajectory, prompt)

            # Query LLM to generate HTML content
            llm_params = (
                self.config.config_llm.model_params.model_dump()
                if self.config.config_llm and self.config.config_llm.model_params
                else {}
            )
            response = await self.llm.query_one(messages=messages, **llm_params)

            # Extract HTML content from response
            html_content = self._extract_html_content(response)

            # Ensure output file has .html extension
            if not output_file_name.endswith(".html"):
                output_file_name = output_file_name + ".html"

            # Write HTML content to file
            with open(output_file_name, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"HTML PPT generated successfully: {output_file_name}")
            return f"HTML PPT generated successfully. Saved as {output_file_name}."

        except Exception as e:
            logger.error(f"Failed to generate HTML PPT: {e}")
            return f"Failed to generate HTML PPT: {str(e)}"

    @register_tool
    async def refine_html_ppt(
        self,
        input_file_name: str,
        output_file_name: str,
        trajectory: list[TResponseInputItem] | None = None,
    ) -> str:
        """
        Refine an existing HTML-based PowerPoint presentation.

        Args:
            input_file_name: The name of the PowerPoint file to be refined (e.g., example.html).
            output_file_name: The name of the refined PowerPoint file to be generated (e.g., example_refined.html).
            trajectory: Conversation history (auto-injected).

        Returns:
            str: Status message indicating the result of the refinement.
        """
        try:
            # Check if input file exists
            if not os.path.exists(input_file_name):
                return f"Input file not found: {input_file_name}"

            # Read existing HTML content
            with open(input_file_name, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Build prompt for refinement
            prompt = self.ppt_refinement_prompt.format(html_content=html_content)
            
            # Prepare messages with trajectory
            messages = self._prepare_messages(trajectory, prompt)

            # Query LLM to refine HTML content
            llm_params = (
                self.config.config_llm.model_params.model_dump()
                if self.config.config_llm and self.config.config_llm.model_params
                else {}
            )
            response = await self.llm.query_one(messages=messages, **llm_params)

            # Extract refined HTML content from response
            refined_html_content = self._extract_html_content(response)

            # Ensure output file has .html extension
            if not output_file_name.endswith(".html"):
                output_file_name = output_file_name + ".html"

            # Write refined HTML content to file
            with open(output_file_name, "w", encoding="utf-8") as f:
                f.write(refined_html_content)

            logger.info(f"HTML PPT refined successfully: {output_file_name}")
            return f"HTML PPT refined successfully. Saved as {output_file_name}."

        except Exception as e:
            logger.error(f"Failed to refine HTML PPT: {e}")
            return f"Failed to refine HTML PPT: {str(e)}"
