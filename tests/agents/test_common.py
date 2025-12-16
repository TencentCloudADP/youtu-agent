import asyncio

from utu.agents.common import DataClassWithStreamEvents, QueueCompleteSentinel


async def test_stream_events():
    data_with_stream = DataClassWithStreamEvents()

    async def producer():
        try:
            for i in range(3):
                await asyncio.sleep(0.1)
                await data_with_stream._event_queue.put(f"event-{i}")
            raise Exception("Test exception")
            data_with_stream._is_complete = True
        # NOTE: should 1) set _is_complete to True; 2) put QueueCompleteSentinel to stop the stream
        except Exception as e:
            data_with_stream._is_complete = True
            data_with_stream._event_queue.put_nowait(QueueCompleteSentinel())
            raise e

    data_with_stream._run_impl_task = asyncio.create_task(producer())

    try:
        async for event in data_with_stream.stream_events():
            print(f"Received: {event}")
    except Exception as e:
        print(f"Exception: {e}")
    print(f"data: {data_with_stream}")
