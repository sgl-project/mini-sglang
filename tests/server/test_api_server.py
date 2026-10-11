from __future__ import annotations

import minisgl.server.api_server as api_server
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def stub_global_state():
    """Requests rejected by validation must never reach the backend."""
    old_state = api_server._GLOBAL_STATE
    api_server._GLOBAL_STATE = object()  # any backend access fails loudly
    yield
    api_server._GLOBAL_STATE = old_state


def test_chat_completions_without_prompt_or_messages_is_rejected():
    client = TestClient(api_server.app)
    response = client.post("/v1/chat/completions", json={"model": "dummy"})
    assert response.status_code == 400
    assert "messages" in response.json()["detail"]
