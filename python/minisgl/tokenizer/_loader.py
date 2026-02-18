from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(
    model_path: str,
    model_source: str = "huggingface",
    **kwargs,
) -> PreTrainedTokenizerBase:
    if model_source == "modelscope":
        from modelscope import snapshot_download

        model_path = snapshot_download(model_path)
    return AutoTokenizer.from_pretrained(model_path, **kwargs)
