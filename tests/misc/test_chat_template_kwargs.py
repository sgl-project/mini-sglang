from __future__ import annotations

import json
from typing import Any, Dict

import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.server.api_server import OpenAICompletionRequest
from minisgl.tokenizer.tokenize import TokenizeManager
from pydantic import ValidationError

MESSAGES = [{"role": "user", "content": "Hello!"}]


class RecordingTokenizer:
    """Stands in for a HF tokenizer and records what `apply_chat_template` receives."""

    def __init__(self) -> None:
        self.kwargs: Dict[str, Any] = {}

    def apply_chat_template(self, conversation, **kwargs) -> str:
        self.kwargs = kwargs
        return "prompt"

    def encode(self, prompt: str, return_tensors: str) -> torch.Tensor:
        return torch.tensor([[1, 2, 3]])


def _msg(kwargs: Dict[str, Any]) -> TokenizeMsg:
    return TokenizeMsg(
        uid=0,
        text=MESSAGES,
        sampling_params=SamplingParams(),
        chat_template_kwargs=json.dumps(kwargs),
    )


def test_request_rejects_chat_template():
    with pytest.raises(ValidationError, match="must not contain 'chat_template'"):
        OpenAICompletionRequest(
            model="", messages=MESSAGES, chat_template_kwargs={"chat_template": "{{ messages }}"}
        )


def test_tokenize_forwards_kwargs():
    tokenizer = RecordingTokenizer()
    TokenizeManager(tokenizer).tokenize([_msg({"enable_thinking": False})])  # type: ignore[arg-type]
    assert tokenizer.kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_qwen3_enable_thinking_false():
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as e:
        pytest.skip(f"Qwen3 tokenizer unavailable: {e}")

    def render(kwargs: Dict[str, Any]) -> str:
        input_ids = TokenizeManager(tokenizer).tokenize([_msg(kwargs)])[0]
        return tokenizer.decode(input_ids)

    assert render({}).endswith("<|im_start|>assistant\n")
    assert render({"enable_thinking": False}).endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
