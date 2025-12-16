from agents import trace
from pydantic import BaseModel

from utu.config import ConfigLoader
from utu.utils import OpenAIUtils, SimplifiedAsyncOpenAI

config = ConfigLoader.load_model_config("base")
openai_client = SimplifiedAsyncOpenAI(**config.model_provider.model_dump())
print(f"Testing {config.model_provider.model} [{config.model_provider.type}], {config.model_provider.base_url}")
# messages = [{"role": "user", "content": "Tell a joke. And what is the weather like in Bogotá and Shanghai?"}]
messages = [{"role": "user", "content": "给我讲两个笑话, 然后帮我查一下北京天津的天气"}]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City and country e.g. Bogotá, Colombia"}},
                "required": ["location"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]
tools_response = [OpenAIUtils.tool_chatcompletion_to_responses(t) for t in tools]


# test ModelConfig -----------------------------------------------------------------------
async def test_model_config():
    # test the `ModelParamsConfig` config -- NOTE passing temperature=None will lead to SGLang error!
    config = ConfigLoader.load_model_config("base")
    client = SimplifiedAsyncOpenAI(**config.model_provider.model_dump())
    res = await client.query_one(messages=messages, tools=tools, **config.model_params.model_dump())
    print(res)


# test chat completions -----------------------------------------------------------------------
# for model test, use the .chat.completions.create API
async def test_model():
    res = await openai_client.chat.completions.create(
        messages=messages, model=config.model_provider.model, tools=tools, stream=False
    )
    print(res.choices[0].message)


async def test_model_stream():
    stream = await openai_client.chat.completions.create(
        messages=messages, model=config.model_provider.model, tools=tools, stream=True
    )
    async for chunk in stream:
        print(f"chat completion chunk: {chunk}")
        # print(chunk.choices[0].delta)


async def test_query_one():
    res = await openai_client.query_one(messages=messages, tools=tools, stream=False)
    print(res)


async def test_print():
    res = await openai_client.chat_completions_create(messages=messages, tools=tools, stream=False)
    OpenAIUtils.print_message(res.choices[0].message)


async def test_print_stream():
    stream = await openai_client.chat_completions_create(messages=messages, tools=tools, stream=True)
    message = await OpenAIUtils.print_stream(stream)
    print()
    print(message)
    print()


# test responses -----------------------------------------------------------------------
async def test_responses():
    res = await openai_client.responses.create(input=messages, tools=tools_response, model=config.model_provider.model)
    for item in res.output:
        print(f"> responses item: {item}")


async def test_responses_stream():
    stream = await openai_client.responses.create(
        input=messages, tools=tools_response, stream=True, model=config.model_provider.model
    )
    async for event in stream:
        print(f"> responses stream event: {event}")


async def test_print_response():
    with trace(workflow_name="test_print_response"):
        res = await openai_client.responses_create(input=messages, tools=tools_response)
        OpenAIUtils.print_response(res)
        print(OpenAIUtils.get_response_configs(res))


async def test_print_response_stream():
    print("test_print_response_stream")
    with trace(workflow_name="test_print_response_stream"):
        stream = await openai_client.responses.create(
            input=messages, tools=tools_response, stream=True, tool_choice="auto"
        )
        print(f"input: {messages}\ntools: {tools_response}")
        async for event in stream:
            print(f"> responses event: {event}")


async def test_output_schema():
    class ExtractedEvent(BaseModel):
        date: str
        """date of the event, in the format of YYYY-MM-DD"""
        summary: str
        """summary of the event"""

    response_format = {
        "format": {
            "type": "json_schema",
            "name": "extracted_event",
            "schema": ExtractedEvent.model_json_schema(),
            "strict": True,
        }
    }

    input = "extract the event from the following text: Bob is going to have a birthday party on 2025-08-12."
    res = await openai_client.responses.create(input=input, text=response_format)
    OpenAIUtils.print_response(res)
    print(OpenAIUtils.get_response_configs(res))


# test extra bodys -----------------------------------------------------------------------
async def test_retry():
    res = await openai_client.chat_completions_create(
        messages=messages,
        tools=tools,
        stream=False,
        # manually trigger retry with 500 (internal server error)
        extra_body={"return_status_code": 500, "enable_format_check": True},
    )
    print(res)


async def test_parallel_tool_calls():
    res = await openai_client.chat_completions_create(
        messages=messages,
        tools=tools,
        stream=False,
        # manually trigger retry with 500 (internal server error)
        extra_body={"parallel_tool_calls": False},
    )
    print(res)
