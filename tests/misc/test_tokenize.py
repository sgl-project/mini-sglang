import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.tokenizer.tokenize import TokenizeManager
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast


def _tokenizer(add_eos: bool) -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "user": 3, "assistant": 4, "hello": 5},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]" if add_eos else "[BOS] $A",
        special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        chat_template=(
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{{ message['role'] }} {{ message['content'] }} {{ eos_token }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}assistant {% endif %}"
        ),
    )


@pytest.mark.parametrize("add_eos", [False, True])
@pytest.mark.parametrize("multi_turn", [False, True])
def test_chat_template_tokens_are_not_added_twice(add_eos: bool, multi_turn: bool) -> None:
    tokenizer = _tokenizer(add_eos)
    chat = [{"role": "user", "content": "hello"}]
    if multi_turn:
        chat += [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "hello"}]
    message = TokenizeMsg(uid=0, text=chat, sampling_params=SamplingParams())

    (actual,) = TokenizeManager(tokenizer).tokenize([message])
    expected = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)

    assert actual.tolist() == expected
    assert actual.dtype == torch.int32
    assert actual.device.type == "cpu"
    assert actual.ndim == 1


@pytest.mark.parametrize("add_eos", [False, True])
def test_mixed_messages_keep_plain_text_special_tokens(add_eos: bool) -> None:
    tokenizer = _tokenizer(add_eos)
    chat = [{"role": "user", "content": "hello"}]
    prompts = ["hello", chat, "hello hello"]
    messages = [
        TokenizeMsg(uid=uid, text=text, sampling_params=SamplingParams())
        for uid, text in enumerate(prompts)
    ]

    actual = TokenizeManager(tokenizer).tokenize(messages)

    assert actual[0].tolist() == tokenizer.encode(prompts[0])
    assert actual[1].tolist() == tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True
    )
    assert actual[2].tolist() == tokenizer.encode(prompts[2])


def test_empty_tokenization_batch() -> None:
    assert TokenizeManager(_tokenizer(False)).tokenize([]) == []
