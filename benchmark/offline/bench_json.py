from __future__ import annotations

import argparse
import time
from random import seed

from minisgl.benchmark.json import (
    collect_filtered_json_samples,
    render_json_prompt_ids,
    validate_json_output,
)
from minisgl.core import SamplingParams
from minisgl.llm import LLM
from transformers import AutoTokenizer


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["constrained", "unconstrained"],
        default="constrained",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed(0)
    # NOTE: Using a small, unaligned model makes the diff easier to observe
    MODEL = "Qwen/Qwen2-0.5B"
    NUM_SEQS = 100
    MAX_OUTPUT_LEN = 4096
    IGNORE_EOS = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    samples = collect_filtered_json_samples(NUM_SEQS)
    prompt_token_ids = [render_json_prompt_ids(tokenizer, sample) for sample in samples]

    assert prompt_token_ids, "No valid json-mode-eval samples found"

    sampling_params = []
    for sample in samples:
        json_schema = sample.json_schema if args.mode == "constrained" else None
        sampling_params.append(
            SamplingParams(
                temperature=0.0,
                top_k=1,
                ignore_eos=IGNORE_EOS,
                max_tokens=MAX_OUTPUT_LEN,
                json_schema=json_schema,
            )
        )

    llm = LLM(MODEL)

    warmup_result = llm.generate(
        [prompt_token_ids[-1]],
        sampling_params[-1],
    )[0]
    templated_input_preview = tokenizer.decode(
        prompt_token_ids[-1],
        skip_special_tokens=False,
    )
    templated_input_preview = templated_input_preview.replace("\n", "\\n")
    warmup_token_ids = warmup_result["token_ids"]
    warmup_text = warmup_result["text"]
    print(
        "Warmup sample: "
        f"mode={args.mode}, "
        f"input={len(prompt_token_ids[-1])}tok, "
        f"templated_input_preview='{templated_input_preview}', "
        f"output={len(warmup_token_ids)}tok, "
        f"preview='{warmup_text}'"
    )

    t = time.time()
    bench_results = llm.generate(prompt_token_ids, sampling_params)
    t = time.time() - t

    output_lens = []
    parse_ok = 0
    schema_ok = 0
    schema_checked = 0
    req_outputs = []
    for i, (sample, result) in enumerate(zip(samples, bench_results)):
        token_ids = result["token_ids"]
        output_lens.append(len(token_ids))
        parsed, valid = validate_json_output(result["text"], sample.json_schema)
        parse_ok += int(parsed)
        if valid is not None:
            schema_checked += 1
            schema_ok += int(valid)
        req_outputs.append(
            {
                "index": i,
                "output_len": len(token_ids),
                "parsed": parsed,
                "schema_valid": valid,
                "text": result["text"],
            }
        )

    total_output_budget = sum(sp.max_tokens for sp in sampling_params)
    total_output_tokens = sum(output_lens)

    print(f"Mode: {args.mode}")
    print_len_stats("Input length", [len(x) for x in prompt_token_ids])
    print_len_stats("Output length", output_lens)
    print(f"Bench requests: {len(prompt_token_ids)}")
    print(f"Output budget: {total_output_budget}tok, " f"Actual output: {total_output_tokens}tok")
    print(f"JSON parse: {parse_ok}/{len(bench_results)}")
    print(f"Schema valid: {schema_ok}/{schema_checked}")
    throughput = total_output_tokens / t if t > 0 else 0.0
    print(f"Total: {total_output_tokens}tok, Time: {t:.2f}s, " f"Throughput: {throughput:.2f}tok/s")
    print("==== Request Outputs ====")
    for item in req_outputs:
        print(
            f"[Request {item['index']}] "
            f"output_len={item['output_len']}tok "
            f"parsed={item['parsed']} "
            f"schema_valid={item['schema_valid']}"
        )
        print(item["text"])
        print("====")


if __name__ == "__main__":
    main()
