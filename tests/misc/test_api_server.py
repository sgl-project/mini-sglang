"""CPU-only tests for the OpenAI-compatible endpoints, using a fake tokenizer/backend."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import List

import httpx
import pytest
from minisgl.message import BaseFrontendMsg, BatchFrontendMsg, TokenizeMsg, UserReply
from minisgl.server import api_server

PROMPT_TOKENS = 5
PIECES = ["Hello", ",", " world"]


class FakeBackend:
    """Replies to each TokenizeMsg with PIECES, finishing because of `finish_reason`."""

    def __init__(self, finish_reason: str) -> None:
        self.finish_reason = finish_reason
        self.replies: asyncio.Queue[BaseFrontendMsg] = asyncio.Queue()
        self.received: List[TokenizeMsg] = []

    async def put(self, msg) -> None:
        if not isinstance(msg, TokenizeMsg):
            return
        self.received.append(msg)
        replies = [
            UserReply(
                uid=msg.uid,
                incremental_output=piece,
                finished=i == len(PIECES) - 1,
                finish_reason=self.finish_reason if i == len(PIECES) - 1 else None,
                prompt_tokens=PROMPT_TOKENS,
                completion_tokens=i + 1,
            )
            for i, piece in enumerate(PIECES)
        ]
        # send the first reply alone and the rest batched, like the detokenizer may do
        await self.replies.put(replies[0])
        await self.replies.put(BatchFrontendMsg(data=replies[1:]))

    async def get(self) -> BaseFrontendMsg:
        return await self.replies.get()


def _run(finish_reason: str, test):
    async def main():
        backend = FakeBackend(finish_reason)
        api_server._GLOBAL_STATE = api_server.FrontendManager(
            config=SimpleNamespace(model_path="fake-model"),  # type: ignore
            send_tokenizer=backend,  # type: ignore
            recv_tokenizer=backend,  # type: ignore
        )
        transport = httpx.ASGITransport(app=api_server.app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                await test(client, backend)
        finally:
            api_server._GLOBAL_STATE = None

    asyncio.run(main())


def _sse_events(body: str) -> List[dict]:
    lines = [line[len("data: ") :] for line in body.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    return [json.loads(line) for line in lines[:-1]]


EXPECTED_USAGE = {
    "prompt_tokens": PROMPT_TOKENS,
    "completion_tokens": len(PIECES),
    "total_tokens": PROMPT_TOKENS + len(PIECES),
}


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_completions(finish_reason: str):
    async def test(client: httpx.AsyncClient, backend: FakeBackend):
        r = await client.post(
            "/v1/completions", json={"model": "m", "prompt": "Hi", "max_tokens": 3}
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["object"] == "text_completion"
        assert data["choices"][0]["text"] == "".join(PIECES)
        assert data["choices"][0]["finish_reason"] == finish_reason
        assert data["usage"] == EXPECTED_USAGE
        # a completion prompt is passed through as raw text, without a chat template
        assert backend.received[0].text == "Hi"
        assert backend.received[0].sampling_params.max_tokens == 3

    _run(finish_reason, test)


def test_completions_stream():
    async def test(client: httpx.AsyncClient, _):
        r = await client.post(
            "/v1/completions", json={"model": "m", "prompt": "Hi", "stream": True}
        )
        events = _sse_events(r.text)
        assert all(e["object"] == "text_completion" for e in events)
        assert "".join(e["choices"][0]["text"] for e in events) == "".join(PIECES)
        assert [e["choices"][0]["finish_reason"] for e in events] == [None, None, "length"]

    _run("length", test)


def test_completions_requires_prompt():
    async def test(client: httpx.AsyncClient, _):
        r = await client.post(
            "/v1/completions", json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]}
        )
        assert r.status_code == 400

    _run("stop", test)


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_chat_completions(finish_reason: str):
    async def test(client: httpx.AsyncClient, backend: FakeBackend):
        messages = [{"role": "user", "content": "Hi"}]
        r = await client.post("/v1/chat/completions", json={"model": "m", "messages": messages})
        data = r.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"] == {"role": "assistant", "content": "".join(PIECES)}
        assert data["choices"][0]["finish_reason"] == finish_reason
        assert data["usage"] == EXPECTED_USAGE
        assert backend.received[0].text == messages

    _run(finish_reason, test)


def test_chat_completions_stream():
    async def test(client: httpx.AsyncClient, _):
        messages = [{"role": "user", "content": "Hi"}]
        r = await client.post(
            "/v1/chat/completions", json={"model": "m", "messages": messages, "stream": True}
        )
        events = _sse_events(r.text)
        assert all(e["object"] == "chat.completion.chunk" for e in events)
        assert all(e["id"].startswith("chatcmpl-") for e in events)
        assert events[0]["choices"][0]["delta"]["role"] == "assistant"
        content = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        assert content == "".join(PIECES)
        assert events[-1]["choices"][0]["finish_reason"] == "length"

    _run("length", test)


def test_openai_client():
    openai = pytest.importorskip("openai")

    async def test(client: httpx.AsyncClient, _):
        oai = openai.AsyncOpenAI(api_key="x", base_url="http://test/v1", http_client=client)
        completion = await oai.completions.create(model="m", prompt="Hi")
        assert completion.choices[0].text == "".join(PIECES)
        assert completion.choices[0].finish_reason == "stop"
        assert completion.usage is not None and completion.usage.total_tokens == 8

        chat = await oai.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "Hi"}]
        )
        assert chat.choices[0].message.content == "".join(PIECES)

        stream = await oai.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "Hi"}], stream=True
        )
        chunks = [chunk async for chunk in stream]
        assert chunks[-1].choices[0].finish_reason == "stop"

    _run("stop", test)


def test_reply_fields_roundtrip():
    from minisgl.message import BaseTokenizerMsg, DetokenizeMsg

    msg = DetokenizeMsg(
        uid=1,
        next_token=2,
        finished=True,
        finish_reason="length",
        prompt_tokens=3,
        completion_tokens=4,
    )
    assert BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg)) == msg
    reply = UserReply(uid=1, incremental_output="a", finished=True, finish_reason="stop")
    assert BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(reply)) == reply
