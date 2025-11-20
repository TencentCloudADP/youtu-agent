import argparse
import asyncio
import datetime
import json

import yaml
from fill_template import extract_json, fill_template_with_yaml_config
from gen_schema import build_schema

from utu.agents import SimpleAgent
from utu.config import ConfigLoader

JSON_SCHEMA_TEMPLATE = """
## JSON Schema

```json
{schema}
```
"""

config = ConfigLoader.load_agent_config("examples/ppt_generator.yaml")


async def main():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--extra_prompt", type=str, default="")
    parser.add_argument("--pages", type=int, default=15)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--template_path", type=str, default="templates/0.pptx")
    parser.add_argument("--yaml_path", type=str, default="yaml_example.yaml")
    parser.add_argument("--output_path", type=str, default=f"output-{current_date}.pptx")
    parser.add_argument("--output_json", type=str, default=f"output-{current_date}.json")
    parser.add_argument("--disable_tooluse", action="store_true")
    args = parser.parse_args()

    with open(args.yaml_path) as f:
        yaml_config = yaml.safe_load(f)
    schema = build_schema(yaml_config)

    # add json schema to instructions
    config.agent.instructions += JSON_SCHEMA_TEMPLATE.format(schema=json.dumps(schema, indent=2))
    # remove toolkits if disable_tooluse is True
    if args.disable_tooluse:
        config.toolkits.clear()

    agent = SimpleAgent(config=config)
    await agent.build()

    if args.file:
        with open(args.file) as f:
            input_file = f.read()
        query = f"""
        把这个文件做成{args.pages}页左右的演讲PPT。{args.extra_prompt}

        {input_file}
        """
    elif args.url:
        url = args.url
        query = f"""
        把这个网页做成{args.pages}页左右的演讲PPT。{args.extra_prompt}

        {url}
        """
    else:
        raise ValueError("Please provide either --file or --url")

    result = await agent.chat_streamed(query)
    final_result = result.final_output
    print(final_result)

    with open(args.output_json, "w") as f:
        f.write(final_result)

    json_data = extract_json(final_result)
    fill_template_with_yaml_config(
        template_path=args.template_path,
        output_path=args.output_path,
        json_data=json_data,
        yaml_config=yaml_config,
    )


if __name__ == "__main__":
    asyncio.run(main())
