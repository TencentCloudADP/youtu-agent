import argparse
import asyncio

from utu.meta import ToolGenerator
from utu.utils import AgentsUtils, PrintUtils


async def do_gen():
    """Generate a new tool with optional auto-debugging."""
    generator = ToolGenerator()
    task = await PrintUtils.async_print_input("Enter your tool requirements: ")
    task_recorder = generator.run_streamed(task)
    await AgentsUtils.print_stream_events(task_recorder.stream_events())
    PrintUtils.print_info(f"Generated tool config saved to {task_recorder.output_file}", color="green")


async def do_debug(tool_name: str):
    """Debug an existing tool."""
    generator = ToolGenerator(auto_debug=False)
    task_recorder = generator.run_debug_streamed(tool_name)
    await AgentsUtils.print_stream_events(task_recorder.stream_events())


async def main():
    parser = argparse.ArgumentParser(description="Generate and debug MCP tools")
    parser.add_argument("--debug", action="store_true", help="Debug an existing tool")
    parser.add_argument("--tool_name", type=str, default=None, help="Tool name to debug (required with --debug)")
    args = parser.parse_args()

    if args.debug:
        if not args.tool_name:
            PrintUtils.print_info("Error: --tool_name is required when using --debug", color="red")
            return
        await do_debug(args.tool_name)
    else:
        await do_gen()


if __name__ == "__main__":
    asyncio.run(main())
