# LLM Serving FAQ

These answers focus on text generation in Mini-SGLang. For process layout and
request flow, see [System Architecture](./structures.md).

## What is a loaded model instance?

A loaded instance is one initialized model replica: its architecture is
instantiated from configuration and its weights are in memory, ready for
inference. One replica can span multiple GPUs through tensor parallelism (TP).

Many requests share the same weights but have separate logical generation state.
Inference updates that state, not the weights. Conversation history comes from
the request context, not from the model learning previous requests.
Mini-SGLang's [Engine](../python/minisgl/engine/engine.py) initializes and executes
the model.

## Why does the serving framework handle tokenization?

The neural network consumes token IDs and produces logits (next-token scores).
The runtime samples tokens from those scores and converts them back into text.
The tokenizer must match the model, but need not run inside the neural network
or its GPU process. Separate workers let CPU text processing overlap with GPU
inference.

Mini-SGLang uses
[`AutoTokenizer.from_pretrained`](../python/minisgl/utils/hf.py).
Many models reuse tokenizer algorithms with different vocabularies, merge rules,
and chat templates supplied in model files. These usually require no serving-code
changes; a genuinely new tokenizer implementation may require a library update
or explicit integration.

## Is supporting a new model just a matter of loading its weights?

Only if its architecture, configuration, and checkpoint format are already
supported. A fine-tuned checkpoint can often reuse the original model
implementation. A different architecture may need new layers, configuration
parsing, weight-loading/sharding rules, or cache and kernel support.

For example, [Qwen3](../python/minisgl/models/qwen3.py) enables Q/K normalization
in shared attention code, while the [Llama](../python/minisgl/models/llama.py)
implementation does not. Loading different weights cannot replace that operation.

Check Mini-SGLang's [model registry](../python/minisgl/models/register.py) for
architecture identifiers. Full SGLang supports more families; consult its
[model catalog](https://docs.sglang.io/docs/supported-models/generative_models)
for the version you use. Successfully loading a tokenizer does not establish
support for running the model.

## What does the scheduler worker actually schedule?

It schedules work within a model replica, rather than choosing an arbitrary GPU
for each request. In Mini-SGLang, each scheduler worker owns an engine bound to a
TP rank; the ranks cooperate on the same logical batch.

Generation has two phases:

- **Prefill:** process uncached prompt tokens and populate the attention
  key-value (KV) cache.
- **Decode:** use that cache to generate subsequent tokens, normally one token
  per active request per step.

The [Scheduler](../python/minisgl/scheduler/scheduler.py) admits requests according
to memory limits, builds batches within token budgets, allocates cache space,
and handles completion or cancellation. Long prefills can be split into chunks.
The engine executes the network and samples tokens; the scheduler coordinates
the request lifecycle.

## What is shared across models, and why use a serving framework?

Request handling, batching, memory-management mechanisms, streaming, and CPU/GPU
overlap are useful across many models. Continuous batching lets requests join
or leave between generation steps instead of waiting for an entire fixed batch
to finish. Combined with efficient kernels and cache reuse, these mechanisms
improve utilization beyond simply calling a model's forward pass per request.

The network operations, weight layouts, cache requirements, and specialized
kernels remain model-dependent. Even shared scheduling logic needs correct
model-specific resource limits. Recurrent or hybrid models may require different
state management; sharing a runtime does not make every architecture compatible.

## What is prefix reuse?

Prefix reuse retains KV-cache entries for identical leading token sequences
under the same model weights and compatible execution settings. For example:

```text
Request A: [system instructions][long document][Summarize it]
Request B: [system instructions][long document][What are the risks?]
           <------------- shared prefix ---------------->
```

Request B can reuse cached state for the shared instructions and document,
reducing prefill work and time to first token. See the
[CacheManager](../python/minisgl/scheduler/cache.py).

Matching is by token IDs, not meaning. After prompts diverge, later matching
text generally cannot reuse the same state because its preceding context differs.
This is not answer caching: the new question and answer still require computation,
and new tokens still attend to the cached context. Cache entries consume memory
and may be evicted.
