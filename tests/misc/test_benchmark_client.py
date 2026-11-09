import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from minisgl.benchmark import client as benchmark
from openai import AsyncOpenAI


@pytest.mark.parametrize("header_delay", [0.0, 0.25, 3.0])
def test_request_setup_time_is_included_in_latency(monkeypatch, header_delay):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(benchmark, "time", SimpleNamespace(perf_counter=lambda: clock.now))

    class ResponseStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for content, finish_reason in [("Hello", None), (" world", None), (None, "length")]:
                clock.now += 0.1
                chunk = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    async def handle_request(request):
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["max_tokens"] == 2
        assert body["ignore_eos"] is True
        clock.now += header_delay
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ResponseStream(),
        )

    async def run():
        async with AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle_request)),
        ) as sdk:
            return await benchmark.benchmark_one(
                sdk, "Hi", 2, "test-model", pbar=False, input_length=1
            )

    result = asyncio.run(run())
    assert result.tics == pytest.approx(
        [100.0, 100.1 + header_delay, 100.2 + header_delay, 100.3 + header_delay]
    )
    # These are the same differences consumed by process_benchmark_results.
    assert result.tics[1] - result.tics[0] == pytest.approx(header_delay + 0.1)
    assert result.tics[-1] - result.tics[0] == pytest.approx(header_delay + 0.3)
    assert result.input_len == 1
    assert result.output_len == 2
    assert result.message == "Hi"
