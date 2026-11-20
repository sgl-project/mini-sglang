"""Exercise the real frontend with synthetic tokenizer replies; no model or GPU needed."""

import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from httpx_sse import EventSource


@pytest.fixture
def api(monkeypatch):
    # ServerArgs imports the CUDA engine transitively. Only its type annotation is
    # used by the frontend under test; keep that backend out of this CPU test.
    args = ModuleType("minisgl.server.args")
    args.ServerArgs = object
    monkeypatch.setitem(sys.modules, "minisgl.server.args", args)
    return importlib.import_module("minisgl.server.api_server")


def make_state(api, monkeypatch, texts):
    state = api.FrontendManager(config=None, send_tokenizer=None, recv_tokenizer=None)
    requests = []

    async def send(msg):
        requests.append(msg)
        state.ack_map[msg.uid] = [
            api.UserReply(msg.uid, text, i == len(texts) - 1) for i, text in enumerate(texts)
        ]
        state.event_map[msg.uid].set()

    monkeypatch.setattr(state, "send_one", send)
    monkeypatch.setattr(state, "shutdown", lambda: None)
    monkeypatch.setattr(api, "_GLOBAL_STATE", state)
    return state, requests


def parse_events(raw):
    response = httpx.Response(200, headers={"content-type": "text/event-stream"}, content=raw)
    return [event.data for event in EventSource(response).iter_sse()]


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "hello\nworld",
        "\nleading",
        "trailing\n",
        "\n\n",
        "",
        "  spaces  ",
        "你好🙂\n世界",
        "a\r\nb\rc",
        "a\v\fb\u2028c",
        "hello\n\nevent: injected\ndata: extra",
    ],
)
def test_generate_http_events(api, monkeypatch, text):
    state, requests = make_state(api, monkeypatch, [text, " tail"])
    with TestClient(api.app) as client:
        response = client.post("/generate", json={"prompt": "test", "max_tokens": 8})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    expected = text.replace("\r\n", "\n").replace("\r", "\n")
    assert parse_events(response.content) == [expected, " tail", "[DONE]"]
    assert response.content.endswith(b"data: [DONE]\n\n")
    assert requests[0].text == "test"
    assert requests[0].sampling_params.max_tokens == 8


def test_each_reply_dispatches_before_stream_finishes(api, monkeypatch):
    async def run():
        state, _ = make_state(api, monkeypatch, ["first", "second"])
        uid = state.new_user()
        await state.send_one(SimpleNamespace(uid=uid))
        stream = state.stream_generate(uid)
        assert parse_events(await anext(stream)) == ["first"]
        assert parse_events(await anext(stream)) == ["second"]
        assert parse_events(await anext(stream)) == ["[DONE]"]
        await stream.aclose()

    asyncio.run(run())


def test_shell_preserves_output_and_history(api, monkeypatch, capsys):
    texts = ["  hello\n", "\n世界🙂\n", " tail  "]
    state, requests = make_state(api, monkeypatch, texts)
    commands = iter(["first prompt", "second prompt", "/exit"])

    class Session:
        async def prompt_async(self):
            return next(commands)

    monkeypatch.setattr(api, "PromptSession", lambda *a, **kw: Session())
    monkeypatch.setattr(state, "shutdown", lambda: None)
    import psutil

    monkeypatch.setattr(psutil, "Process", lambda: SimpleNamespace(children=lambda **kw: []))
    asyncio.run(api.shell())
    expected = "".join(texts)
    assert capsys.readouterr().out == (expected + "\n") * 2 + "Exiting shell...\n"
    assert requests[1].text == [
        {"role": "user", "content": "first prompt"},
        {"role": "assistant", "content": expected},
        {"role": "user", "content": "second prompt"},
    ]


def test_chat_stream_still_emits_json_and_done(api, monkeypatch):
    make_state(api, monkeypatch, ["hello\n世界"])
    with TestClient(api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    import json

    events = parse_events(response.content)
    assert json.loads(events[0])["choices"][0]["delta"]["content"] == "hello\n世界"
    assert json.loads(events[-2])["choices"][0]["finish_reason"] == "stop"
    assert events[-1] == "[DONE]"
