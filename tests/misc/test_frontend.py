import asyncio
import json
from contextlib import suppress

import pytest
import torch
from minisgl.distributed import DistributedInfo
from minisgl.message import BatchFrontendMsg, UserReply
from minisgl.server import api_server
from minisgl.server.args import ServerArgs
from starlette.requests import Request


@pytest.mark.parametrize("response_type", ["generate", "chat_stream", "chat_nonstream"])
def test_finished_response_releases_request(response_type, monkeypatch):
    async def run():
        # Keep the real frontend listener and response consumers; only replace the
        # tokenizer process transport with in-memory queues.
        send_tokenizer = asyncio.Queue()
        recv_tokenizer = asyncio.Queue()
        state = api_server.FrontendManager(
            config=ServerArgs(
                model_path="test-model", tp_info=DistributedInfo(0, 1), dtype=torch.float32
            ),
            send_tokenizer=send_tokenizer,
            recv_tokenizer=recv_tokenizer,
            initialized=True,
        )
        monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)
        listener = asyncio.create_task(state.listen())
        live_uid = state.new_user()

        async def collect(stream):
            return [chunk async for chunk in stream]

        try:
            if response_type == "chat_nonstream":
                response = asyncio.create_task(
                    api_server.v1_completions(
                        api_server.OpenAICompletionRequest(model="test-model", prompt="Hello"),
                        Request({"type": "http"}),
                    )
                )
                msg = await asyncio.wait_for(send_tokenizer.get(), timeout=5)
                uid = msg.uid
            else:
                uid = state.new_user()
                stream = (
                    state.stream_generate(uid)
                    if response_type == "generate"
                    else state.stream_chat_completions(uid)
                )
                response = asyncio.create_task(collect(stream))

            await recv_tokenizer.put(
                BatchFrontendMsg([UserReply(uid, "Hello", False), UserReply(uid, " world", True)])
            )
            result = await asyncio.wait_for(response, timeout=5)
            if response_type == "chat_nonstream":
                assert result["choices"][0]["message"]["content"] == "Hello world"
            elif response_type == "generate":
                assert result[-1].decode().strip() == "data: [DONE]"
                assert (
                    "".join(
                        chunk.decode().removeprefix("data: ").removesuffix("\n")
                        for chunk in result[:-1]
                    )
                    == "Hello world"
                )
            else:
                assert result[-1] == b"data: [DONE]\n\n"
                chunks = [
                    json.loads(chunk.decode().removeprefix("data: ")) for chunk in result[:-1]
                ]
                assert (
                    "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
                    == "Hello world"
                )

            # A completed response must not leave per-request state behind,
            # even though each consumer breaks upon receiving its final reply.
            assert uid not in state.ack_map
            assert uid not in state.event_map
            assert live_uid in state.ack_map
            assert live_uid in state.event_map
        finally:
            listener.cancel()
            with suppress(asyncio.CancelledError):
                await listener

    asyncio.run(run())
