import asyncio
import json
import traceback
from importlib import resources

import agents as ag
import tornado.web
import tornado.websocket

from utu.agents import OrchestratorAgent, SimpleAgent
from utu.agents.orchestra import OrchestraStreamEvent
from utu.agents.orchestra_agent import OrchestraAgent
from utu.agents.orchestrator import OrchestratorStreamEvent
from utu.utils import EnvUtils

from .common import (
    Event,
    ExampleContent,
    UserRequest,
    handle_new_agent,
    handle_orchestra_events,
    handle_raw_stream_events,
    handle_tool_call_output,
)


class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, agent: SimpleAgent | OrchestraAgent | OrchestratorAgent, example_query: str = ""):
        self.agent: SimpleAgent | OrchestraAgent | OrchestratorAgent = agent
        self.history = None  # recorder for multi-turn chat. Now only used for OrchestraAgent
        self.example_query = example_query

    def check_origin(self, origin):
        # Allow all origins to connect
        return True

    def open(self):
        # print("WebSocket opened")
        # send example query
        self.write_message(
            Event(type="example", data=ExampleContent(type="example", query=self.example_query)).model_dump()
        )

    async def send_event(self, event: Event):
        # print in green color
        # print(f"\033[92mSending event: {event.model_dump()}\033[0m")
        self.write_message(event.model_dump())

    async def on_message(self, message: str):
        try:
            data = json.loads(message)
            user_request = UserRequest(**data)
            if user_request.type == "query":
                try:
                    content = user_request.content
                    # check query not empty
                    if content.query.strip() == "":
                        raise ValueError("Query cannot be empty")

                    # print(f"Received query: {query.query}")
                    # Echo back the query in the response
                    if isinstance(self.agent, OrchestraAgent):
                        stream = self.agent.run_streamed(content.query)
                    elif isinstance(self.agent, SimpleAgent):
                        self.agent.input_items.append({"role": "user", "content": content.query})
                        # print in red color
                        print(f"\033[91mInput items: {self.agent.input_items}\033[0m")
                        stream = self.agent.run_streamed(self.agent.input_items)
                    elif isinstance(self.agent, OrchestratorAgent):
                        stream = self.agent.run_streamed(content.query, self.history)
                        self.history = stream
                    else:
                        raise ValueError(f"Unsupported agent type: {type(self.agent).__name__}")

                    async for event in stream.stream_events():
                        event_to_send = None
                        # print(f"--------------------\n{event}")
                        if isinstance(event, ag.RawResponsesStreamEvent):
                            event_to_send = await handle_raw_stream_events(event)
                        elif isinstance(event, ag.RunItemStreamEvent):
                            event_to_send = await handle_tool_call_output(event)
                        elif isinstance(event, ag.AgentUpdatedStreamEvent):
                            event_to_send = await handle_new_agent(event)
                        elif isinstance(event, OrchestraStreamEvent):
                            event_to_send = await handle_orchestra_events(event)
                        elif isinstance(event, OrchestratorStreamEvent):
                            print(f"> {event}")
                        else:
                            pass
                        if event_to_send:
                            # print(f"Sending event: {asdict(event_to_send)}")
                            await self.send_event(event_to_send)
                    else:
                        pass
                    event_to_send = Event(type="finish")
                    # self.write_message(asdict(event_to_send))
                    await self.send_event(event_to_send)
                    if isinstance(self.agent, SimpleAgent):
                        input_list = stream.to_input_list()
                        self.agent.input_items = input_list
                        # print in red
                        print(f"\033[91mInput list: {input_list}\033[0m")
                        self.agent.current_agent = stream.last_agent
                except TypeError as e:
                    print(f"Invalid query format: {e}")
                    # stack trace
                    print(traceback.format_exc())
                    self.close(1002, "Invalid query format")
            else:
                # print(f"Unhandled message type: {data.get('type')}")
                # self.close(1002, "Unhandled message type")
                pass
        except json.JSONDecodeError:
            # print(f"Invalid JSON received: {message}")
            self.close(1002, "Invalid JSON received")
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            # stack trace
            print(traceback.format_exc())
            self.close(1002, "Error processing message")

    def on_close(self):
        # print("WebSocket closed")
        pass


class WebUIChatbot:
    def __init__(self, agent: SimpleAgent | OrchestraAgent, example_query: str = ""):
        self.agent = agent
        self.example_query = example_query
        # hack
        with resources.as_file(resources.files("utu_agent_ui.static").joinpath("index.html")) as static_dir:
            self.static_path = str(static_dir).replace("index.html", "")

    def make_app(self, autoload: bool | None = None) -> tornado.web.Application:
        if autoload is None:
            autoload = EnvUtils.get_env("UTU_WEBUI_AUTOLOAD", "false") == "true"
        return tornado.web.Application(
            [
                (r"/ws", WebSocketHandler, {"agent": self.agent, "example_query": self.example_query}),
                (
                    r"/",
                    tornado.web.RedirectHandler,
                    {"url": "/index.html"},
                ),
                (
                    r"/(.*)",
                    tornado.web.StaticFileHandler,
                    {"path": self.static_path, "default_filename": "index.html"},
                ),
            ],
            debug=autoload,
        )

    async def __launch(self, port: int = 8848, ip: str = "127.0.0.1", autoload: bool | None = None):
        if hasattr(self.agent, "build"):
            await self.agent.build()
        app = self.make_app()
        app.listen(port, address=ip)
        print(f"Server started at http://{ip}:{port}/")
        await asyncio.Event().wait()

    async def launch_async(self, port: int = 8848, ip: str = "127.0.0.1", autoload: bool | None = None):
        await self.__launch(port=port, ip=ip, autoload=autoload)

    def launch(self, port: int = 8848, ip: str = "127.0.0.1", autoload: bool | None = None):
        asyncio.run(self.__launch(port=port, ip=ip, autoload=autoload))


if __name__ == "__main__":
    webui = WebUIChatbot()
    webui.launch()
