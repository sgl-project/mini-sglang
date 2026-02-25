import time
import urllib.request
import shutil
from random import seed
from pathlib import Path

import pyarrow.parquet as pq
from minisgl.core import SamplingParams
from minisgl.llm import LLM
from transformers import AutoTokenizer

LANGS = {"English", "Chinese"}

BASE_DIR = Path(__file__).resolve().parent
WILDCHAT_FIRST_SHARD = "train-00000-of-00086.parquet"
WILDCHAT_BASE_URL = "https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/data/"


def download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    print(f"Downloading {url} -> {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as resp, path.open("wb") as out:
        shutil.copyfileobj(resp, out)


def iter_filtered_prompt_ids(tokenizer):
    shard_path = BASE_DIR / WILDCHAT_FIRST_SHARD
    download_if_missing(WILDCHAT_BASE_URL + WILDCHAT_FIRST_SHARD, shard_path)

    parquet = pq.ParquetFile(shard_path)
    for batch in parquet.iter_batches(batch_size=256, columns=["conversation"]):
        prompts = []
        for conv in batch.to_pydict()["conversation"]:
            if not conv:
                continue

            first_user = None
            for turn in conv:
                if turn.get("role") == "user":
                    first_user = turn
                    break
            if first_user is None:
                continue

            text = (first_user.get("content") or "").strip()
            if not text:
                continue
            if first_user.get("language") not in LANGS:
                continue
            if bool(first_user.get("redacted")) or bool(first_user.get("toxic")):
                continue
            prompts.append(text)

        if not prompts:
            continue

        for prompt in prompts:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            yield ids


def print_len_stats(name: str, lengths: list[int]) -> None:
    if not lengths:
        print(f"{name}: no data")
        return
    arr = sorted(lengths)
    n = len(arr)
    print(
        f"{name}: count={n}, min={arr[0]}, p50={arr[int(0.50*n)]}, "
        f"p90={arr[int(0.90*n)]}, p99={arr[int(0.99*n)]}, max={arr[-1]}"
    )


def main() -> None:
    seed(0)
    MODEL = "Qwen/Qwen3-0.6B"
    NUM_SEQS = 256 * 1
    MAX_OUTPUT_LEN = 4096
    IGNORE_EOS = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    prompt_token_ids = []
    for ids in iter_filtered_prompt_ids(tokenizer):
        prompt_token_ids.append(ids)
        if len(prompt_token_ids) >= NUM_SEQS:
            break

    num_reqs = len(prompt_token_ids)
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=IGNORE_EOS, max_tokens=MAX_OUTPUT_LEN)
        for _ in range(num_reqs)
    ]

    llm = LLM(MODEL)

    # warm up
    warmup_result = llm.generate(
        [prompt_token_ids[0]],
        sampling_params[0],
    )[0]
    templated_input_preview = tokenizer.decode(
        prompt_token_ids[0],
        skip_special_tokens=False,
    )
    templated_input_preview = templated_input_preview.replace("\n", "\\n")
    warmup_token_ids = warmup_result["token_ids"]
    warmup_text = warmup_result["text"]
    print(
        "Warmup sample: "
        f"input={len(prompt_token_ids[0])}tok, "
        f"templated_input_preview='{templated_input_preview}', "
        f"output={len(warmup_token_ids)}tok, "
        f"preview='{warmup_text}'"
    )

    t = time.time()
    bench_results = llm.generate(prompt_token_ids, sampling_params)
    t = time.time() - t

    total_output_budget = sum(sp.max_tokens for sp in sampling_params)
    output_lens = []
    for result in bench_results:
        token_ids = result["token_ids"]
        output_lens.append(len(token_ids))
    total_output_tokens = sum(output_lens)

    print_len_stats("Input length", [len(x) for x in prompt_token_ids])
    print_len_stats("Output length", output_lens)
    throughput = total_output_tokens / t if t > 0 else 0.0
    print(f"Bench requests: {len(prompt_token_ids)}")
    print(f"Output budget: {total_output_budget}tok, " f"Actual output: {total_output_tokens}tok")
    print(f"Total: {total_output_tokens}tok, Time: {t:.2f}s, " f"Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
