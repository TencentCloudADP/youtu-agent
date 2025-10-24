import argparse
import asyncio
import datetime

from fill_template import extract_json, fill_template

from utu.agents import SimpleAgent
from utu.config import ConfigLoader

config = ConfigLoader.load_agent_config("examples/ppt_generator.yaml")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--extra_prompt", type=str, default="")
    parser.add_argument("--pages", type=int, default=15)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--template_path", type=str, default="template/template_ori.pptx")
    parser.add_argument(
        "--output_path", type=str, default=f"output-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pptx"
    )
    parser.add_argument(
        "--output_json", type=str, default=f"output-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"
    )
    args = parser.parse_args()

    agent = SimpleAgent(config=config)
    await agent.build()

    if args.file:
        with open(args.file) as f:
            input_file = f.read()
        query = f"""
        把这个网页做成{args.pages}页左右的演讲PPT。{args.extra_prompt}

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
    fill_template(
        template_path=args.template_path,
        output_path=args.output_path,
        json_data=json_data,
    )


if __name__ == "__main__":
    asyncio.run(main())
