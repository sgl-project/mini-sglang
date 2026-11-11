import pytest
from minisgl.message import DetokenizeMsg
from minisgl.tokenizer.detokenize import DetokenizeManager
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


@pytest.fixture
def tokenizer():
    # Use the real byte-level decoder, including a token ending mid-codepoint.
    byte_chars = bytes_to_unicode()
    pieces = [b"Hello", b" \xf0\x9f\x98", b"\x8a", b"\xe0\xa5\x8d\xe0\xa4"]
    vocab = {"".join(byte_chars[b] for b in piece): i for i, piece in enumerate(pieces)}
    vocab["<eos>"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab, []))
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>")


@pytest.mark.parametrize("ids", [[1], [0, 1], [3], [0, 3]])
@pytest.mark.parametrize("finish_with_eos", [False, True])
def test_finished_request_flushes_buffered_text(tokenizer, ids, finish_with_eos):
    manager = DetokenizeManager(tokenizer)
    messages = ids + ([tokenizer.eos_token_id] if finish_with_eos else [])
    output = []
    for i, token in enumerate(messages):
        output.extend(
            manager.detokenize(
                [DetokenizeMsg(uid=7, next_token=token, finished=i == len(messages) - 1)]
            )
        )
    assert "".join(output) == tokenizer.decode(ids)
    assert manager.decode_map == {}


def test_unfinished_request_waits_for_complete_codepoint(tokenizer):
    manager = DetokenizeManager(tokenizer)
    output = manager.detokenize([DetokenizeMsg(uid=7, next_token=1, finished=False)])
    assert output == [" "]
    assert "\ufffd" not in "".join(output)
    output.extend(manager.detokenize([DetokenizeMsg(uid=7, next_token=2, finished=True)]))
    assert "".join(output) == tokenizer.decode([1, 2]) == " 😊"
    assert manager.decode_map == {}


@pytest.mark.parametrize("ids", [[0], [0, 1, 2], []])
def test_complete_text_and_empty_eos_are_unchanged(tokenizer, ids):
    manager = DetokenizeManager(tokenizer)
    output = []
    for token in ids:
        output.extend(manager.detokenize([DetokenizeMsg(uid=7, next_token=token, finished=False)]))
    output.extend(
        manager.detokenize([DetokenizeMsg(uid=7, next_token=tokenizer.eos_token_id, finished=True)])
    )
    assert "".join(output) == tokenizer.decode(ids)
    assert manager.decode_map == {}


def test_independent_requests_keep_separate_buffers(tokenizer):
    manager = DetokenizeManager(tokenizer)
    chunks = manager.detokenize(
        [
            DetokenizeMsg(uid=0, next_token=1, finished=True),
            DetokenizeMsg(uid=1, next_token=1, finished=False),
        ]
    )
    assert chunks == [tokenizer.decode([1]), " "]
    assert set(manager.decode_map) == {1}
    assert manager.detokenize([DetokenizeMsg(uid=1, next_token=2, finished=True)]) == ["😊"]
    assert manager.decode_map == {}
