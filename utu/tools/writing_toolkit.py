from datetime import datetime
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


class WritingToolkit(AsyncBaseToolkit):
    """
    A tool for different writing tasks.
    """
    def __init__(self, config: ToolkitConfig | dict | None = None):
        super().__init__(config)
        # Initialize LLM
        self.llm = SimplifiedAsyncOpenAI(
            **self.config.config_llm.model_provider.model_dump() if self.config.config_llm else {}
        )
        # Load prompts from config
        self.deep_research_outline_prompt = self.config.config.get("deep_research_outline_prompt")
        self.deep_research_chapter_prompt = self.config.config.get("deep_research_chapter_prompt")
        self.fiction_writing_prompt = self.config.config.get("fiction_writing_prompt")
        self.general_writing_prompt = self.config.config.get("general_writing_prompt")

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
    
    async def write_deep_research(self, task: str, trajectory: list[TResponseInputItem] | None = None) -> str:
        """
        Generate a comprehensive deep research report through an iterative process.
        
        This method follows a three-step workflow:
        1. Generate an outline based on the research task
        2. Iteratively generate content for each chapter in the outline until the model 
           determines all chapters are complete and outputs "FINISHED"
        3. Return the complete research report
        
        Args:
            task (str): The research task or topic to write about
            trajectory (list[TResponseInputItem] | None): Historical conversation trajectory
            
        Returns:
            str: A complete research report with all chapters filled in according to the outline
        """
        outline_messages = self._prepare_messages(trajectory, self.deep_research_outline_prompt.format(task=task))
        outline = await self.llm.query_one(
            messages=outline_messages, **self.config.config_llm.model_params.model_dump()
        )
        report = ""
        chapter_messages = self._prepare_messages(trajectory, self.deep_research_chapter_prompt.format(outline=outline, report=report))
        chapter = await self.llm.query_one(
            messages=chapter_messages, **self.config.config_llm.model_params.model_dump()
        )
        while chapter != "FINISHED":
            report += "\n" + chapter.strip()
            chapter_messages = self._prepare_messages(trajectory, self.deep_research_chapter_prompt.format(outline=outline, report=report))
            chapter = await self.llm.query_one(
                messages=chapter_messages, **self.config.config_llm.model_params.model_dump()
            )
        return report

    async def write_fiction(self, task: str, trajectory: list[TResponseInputItem] | None = None) -> str:
        """
        Generate fiction writing based on the task.
        
        Args:
            task (str): The fiction writing task
            trajectory (list[TResponseInputItem] | None): Historical conversation trajectory
            
        Returns:
            str: Generated fiction content
        """
        messages = self._prepare_messages(trajectory, self.fiction_writing_prompt.format(task=task))
        return await self.llm.query_one(
            messages=messages, **self.config.config_llm.model_params.model_dump()
        )

    async def write_general_report(self, task: str, trajectory: list[TResponseInputItem] | None = None) -> str:
        """
        Generate a general report based on the task.
        
        Args:
            task (str): The report writing task
            trajectory (list[TResponseInputItem] | None): Historical conversation trajectory
            
        Returns:
            str: Generated report content
        """
        messages = self._prepare_messages(trajectory, self.general_writing_prompt.format(task=task))
        return await self.llm.query_one(
            messages=messages, **self.config.config_llm.model_params.model_dump()
        )

    @register_tool
    async def write_text(self, genre: str, task: str, trajectory: list[TResponseInputItem] | None = None) -> str:
        """
        Write text based on the specified genre.
        
        Args:
            genre (str): The genre of writing ("deep_research", "fiction", or general)
            task (str): The writing task description
            trajectory (list[TResponseInputItem] | None): Historical conversation trajectory
            
        Returns:
            str: Generated text content
        """
        match genre:
            case "deep_research":
                return await self.write_deep_research(task, trajectory)
            case "fiction":
                return await self.write_fiction(task, trajectory)
            case _:
                return await self.write_general_report(task, trajectory)
