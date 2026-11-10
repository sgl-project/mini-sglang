import json
from pathlib import Path

import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.message import TokenizeMsg
from minisgl.tokenizer.tokenize import TokenizeManager
from minisgl.utils import hf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

CHAT_TEMPLATE = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + ' ' }}{% endfor %}{{ 'assistant:' if add_generation_prompt else '' }}"


@pytest.fixture
def local_tokenizer(tmp_path):
    backend = Tokenizer(
        WordLevel({"[UNK]": 0, "user": 1, "assistant": 2, ":": 3, "hello": 4}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    model_dir = tmp_path / "models" / "test-tokenizer"
    tokenizer.save_pretrained(model_dir)
    return model_dir


@pytest.mark.parametrize("relative", [False, True])
def test_local_json_template_formats_chat(local_tokenizer, monkeypatch, relative):
    (local_tokenizer / "chat_template.json").write_text(
        json.dumps({"chat_template": CHAT_TEMPLATE}), encoding="utf-8"
    )
    if relative:
        monkeypatch.chdir(local_tokenizer.parent.parent)
        model_path = str(local_tokenizer.relative_to(Path.cwd()))
    else:
        model_path = str(local_tokenizer)

    tokenizer = hf.load_tokenizer(model_path)
    messages = [{"role": "user", "content": "hello"}]
    assert (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        == "user: hello assistant:"
    )
    output = TokenizeManager(tokenizer).tokenize(
        [TokenizeMsg(uid=1, text=messages, sampling_params=SamplingParams())]
    )[0]
    assert output.tolist() == [1, 3, 4, 2, 3]
    assert output.dtype == torch.int32
    assert output.device.type == "cpu"


def test_existing_template_takes_precedence(local_tokenizer, monkeypatch):
    (local_tokenizer / "chat_template.jinja").write_text(CHAT_TEMPLATE, encoding="utf-8")
    (local_tokenizer / "chat_template.json").write_text(
        json.dumps({"chat_template": "wrong"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        hf, "hf_hub_download", lambda *args, **kwargs: pytest.fail("No fallback should be needed")
    )
    assert hf.load_tokenizer(str(local_tokenizer)).chat_template == CHAT_TEMPLATE


@pytest.mark.parametrize("contents", [None, "not json", "{}"])
def test_optional_local_template_still_allows_raw_text(local_tokenizer, contents):
    if contents is not None:
        (local_tokenizer / "chat_template.json").write_text(contents, encoding="utf-8")
    tokenizer = hf.load_tokenizer(str(local_tokenizer))
    assert tokenizer.chat_template is None
    assert tokenizer.encode("hello") == [4]


def test_remote_json_template_fallback_is_preserved(local_tokenizer, monkeypatch):
    template_path = local_tokenizer / "chat_template.json"
    template_path.write_text(json.dumps({"chat_template": CHAT_TEMPLATE}), encoding="utf-8")
    tokenizer = hf.AutoTokenizer.from_pretrained(str(local_tokenizer))
    calls = []
    monkeypatch.setattr(hf.AutoTokenizer, "from_pretrained", lambda model_path: tokenizer)

    def download(*, repo_id, filename):
        calls.append((repo_id, filename))
        return str(template_path)

    monkeypatch.setattr(hf, "hf_hub_download", download)
    assert hf.load_tokenizer("example/remote-tokenizer").chat_template == CHAT_TEMPLATE
    assert calls == [("example/remote-tokenizer", "chat_template.json")]
