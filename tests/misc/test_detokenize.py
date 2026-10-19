import pytest
from minisgl.message import DetokenizeMsg
from minisgl.tokenizer.detokenize import DetokenizeManager
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast


@pytest.mark.parametrize("uids", [[0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1]])
def test_detokenize_queued_tokens(uids):
    backend = Tokenizer(models.BPE({"Hello": 0, "Ġworld": 1, "!": 2, "<eos>": 3}, []))
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>")
    manager = DetokenizeManager(tokenizer)
    counts = {0: 0, 1: 0}
    messages = []
    expected = []
    for uid in uids:
        token = counts[uid]
        messages.append(DetokenizeMsg(uid=uid, next_token=token, finished=False))
        expected.append(["Hello", " world", "!"][token])
        counts[uid] += 1

    chunks = manager.detokenize(messages)
    assert chunks == expected
    for uid in counts:
        assert "".join(chunk for msg, chunk in zip(messages, chunks) if msg.uid == uid) == (
            tokenizer.decode([0, 1, 2])
        )
    assert manager.detokenize(
        [DetokenizeMsg(uid=uid, next_token=tokenizer.eos_token_id, finished=True) for uid in counts]
    ) == ["", ""]
    assert manager.decode_map == {}
