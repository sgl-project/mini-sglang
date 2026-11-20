"""Verify SSE framing and shell consumption without starting model workers."""

import asyncio
import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

import httpx
import psutil
import pytest
from fastapi.testclient import TestClient
from httpx_sse import EventSource


@pytest.fixture
def frontend_api(monkeypatch):
    # ServerArgs is only a frontend annotation, but importing it loads the CUDA
    # engine. Substitute that annotation while importing the real frontend.
    args_module = ModuleType("minisgl.server.args")
    args_module.ServerArgs = object
    monkeypatch.setitem(sys.modules, "minisgl.server.args", args_module)
    return importlib.import_module("minisgl.server.api_server")


@pytest.fixture
def frontend_factory(frontend_api, monkeypatch):
    """Supply tokenizer replies while retaining the real frontend consumers."""

    def create(reply_texts):
        manager = frontend_api.FrontendManager(
            config=None, send_tokenizer=None, recv_tokenizer=None
        )
        submitted_requests = []

        async def send_request(message):
            submitted_requests.append(message)
            manager.ack_map[message.uid] = [
                frontend_api.UserReply(
                    uid=message.uid,
                    incremental_output=text,
                    finished=index == len(reply_texts) - 1,
                )
                for index, text in enumerate(reply_texts)
            ]
            manager.event_map[message.uid].set()

        monkeypatch.setattr(manager, "send_one", send_request)
        monkeypatch.setattr(manager, "shutdown", lambda: None)
        monkeypatch.setattr(frontend_api, "_GLOBAL_STATE", manager)
        return manager, submitted_requests

    return create


def read_sse_data(raw_body):
    """Decode the emitted bytes with an independent SSE client."""
    response = httpx.Response(200, headers={"content-type": "text/event-stream"}, content=raw_body)
    return [event.data for event in EventSource(response).iter_sse()]


@pytest.mark.parametrize(
    ("reply_text", "expected_text"),
    [
        pytest.param("hello", "hello", id="plain-text"),
        pytest.param("hello\nworld", "hello\nworld", id="multiline-text"),
        pytest.param("\nleading", "\nleading", id="leading-newline"),
        pytest.param("trailing\n", "trailing\n", id="trailing-newline"),
        pytest.param("\n\n", "\n\n", id="blank-lines"),
        pytest.param("", "", id="empty-text"),
        pytest.param("  spaces  ", "  spaces  ", id="surrounding-spaces"),
        pytest.param(
            "caf\u00e9 \U0001f642\nna\u00efve",
            "caf\u00e9 \U0001f642\nna\u00efve",
            id="unicode-text",
        ),
        pytest.param("a\r\nb\rc", "a\nb\nc", id="crlf-and-cr"),
        pytest.param("a\v\fb\u2028c", "a\v\fb\u2028c", id="non-sse-line-separators"),
        pytest.param(
            "hello\n\nevent: injected\ndata: extra",
            "hello\n\nevent: injected\ndata: extra",
            id="protocol-like-text",
        ),
    ],
)
def test_generate_preserves_reply_events(frontend_api, frontend_factory, reply_text, expected_text):
    """Each reply, including empty or multiline text, must be a complete event."""
    _, submitted_requests = frontend_factory([reply_text, " tail"])
    with TestClient(frontend_api.app) as client:
        response = client.post("/generate", json={"prompt": "test", "max_tokens": 8})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert read_sse_data(response.content) == [expected_text, " tail", "[DONE]"]
    assert response.content.endswith(b"data: [DONE]\n\n")
    assert submitted_requests[0].text == "test"
    assert submitted_requests[0].sampling_params.max_tokens == 8


def test_each_reply_dispatches_before_stream_finishes(frontend_api, frontend_factory):
    """A client must receive a reply without waiting for a later stream chunk."""

    async def consume_stream():
        manager, _ = frontend_factory(["first", "second"])
        uid = manager.new_user()
        await manager.send_one(
            frontend_api.TokenizeMsg(
                uid=uid, text="test", sampling_params=frontend_api.SamplingParams(max_tokens=8)
            )
        )
        stream = manager.stream_generate(uid)
        try:
            assert read_sse_data(await anext(stream)) == ["first"]
            assert read_sse_data(await anext(stream)) == ["second"]
            assert read_sse_data(await anext(stream)) == ["[DONE]"]
        finally:
            await stream.aclose()

    asyncio.run(consume_stream())


def test_shell_preserves_output_and_history(frontend_api, frontend_factory, monkeypatch, capsys):
    """The shell must display and remember text without SSE framing artifacts."""
    reply_texts = ["  hello\n", "\nworld \U0001f642\n", " tail  "]
    _, submitted_requests = frontend_factory(reply_texts)
    commands = iter(["first prompt", "second prompt", "/exit"])

    class ScriptedSession:
        async def prompt_async(self):
            return next(commands)

    monkeypatch.setattr(frontend_api, "PromptSession", lambda *args, **kwargs: ScriptedSession())
    monkeypatch.setattr(psutil, "Process", lambda: SimpleNamespace(children=lambda **kwargs: []))
    asyncio.run(frontend_api.shell())

    expected_output = "".join(reply_texts)
    assert capsys.readouterr().out == (expected_output + "\n") * 2 + "Exiting shell...\n"
    assert submitted_requests[1].text == [
        {"role": "user", "content": "first prompt"},
        {"role": "assistant", "content": expected_output},
        {"role": "user", "content": "second prompt"},
    ]


def test_chat_stream_still_emits_json_and_done(frontend_api, frontend_factory):
    """The existing chat SSE response must remain compatible with its client."""
    frontend_factory(["hello\nworld"])
    with TestClient(frontend_api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

    assert response.status_code == 200
    events = read_sse_data(response.content)
    assert json.loads(events[0])["choices"][0]["delta"]["content"] == "hello\nworld"
    assert json.loads(events[-2])["choices"][0]["finish_reason"] == "stop"
    assert events[-1] == "[DONE]"
