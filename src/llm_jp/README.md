# LLM-jp 4.1 32B-A3B: experimental gfx906 work

This is an early, correctness-first prototype. The token-ID decoder runs LLM-jp, but is not wired into the Strata server.

The model config describes 32 layers, Qwen3-MoE, hidden width 2560, 40 query heads, 4 KV heads,
128-dimensional heads, 128 experts, and top-8 routing. The separate Qwen3.5/3.6 engine is a
hybrid architecture and is disabled in the gfx906 build; enabling it is not sufficient.

`attention.cuh` is a deliberately simple, single-token grouped-query decode attention kernel.
It expects normalized, rotary-encoded Q and K and a populated KV cache; it does not perform those
steps itself. The O(heads × context × head_dim²) reference implementation recomputes logits and is
not a performance claim. The caller must enforce positive dimensions, length > 0, head_dim <= the
device block limit, and n_head divisible by n_kv_head.

`tests/llm_jp/attention_test.hip` compares its results against a double-precision CPU reference
using the model's 40/4/128 geometry. On the Radeon VII / TheRock 10.3.0a20261102, 17 synthetic
KV cells gave max absolute error 2.23e-6. This does not establish whole-model correctness.
`norm_rope.cuh` adds a similarly simple weighted Q/K RMSNorm and NeoX-style RoPE kernel;
`norm_rope_test.hip` checks 40 synthetic heads at position 123 against a double-precision reference.
It is not yet checked against the real model's logits or long-position rounding.

```bash
source /path/to/rocm-env.sh
hipcc --offload-arch=gfx906 -O2 tests/llm_jp/attention_test.hip -o /tmp/llm-jp-attention-test
/tmp/llm-jp-attention-test
hipcc --offload-arch=gfx906 -O2 tests/llm_jp/norm_rope_test.hip -o /tmp/llm-jp-norm-rope-test
/tmp/llm-jp-norm-rope-test
```

The existing `native_router_topk` also accepts 128 experts / top 8. Its gfx906 test compares
19 rows against a CPU reference, including ties, and repeats the check for the existing
256/8 and 512/10 shapes. This tests routing, not quantized expert execution.

```bash
hipcc --offload-arch=gfx906 -O2 -Iinclude/strata/platform/hip_compat -Iinclude \
  -x hip src/kernels/cuda/native_router.cu tests/llm_jp/router_test.hip -o /tmp/llm-jp-router-test
/tmp/llm-jp-router-test
```

`inspect.cpp` inventories GGUF metadata and tensor types using Strata's own GGUF reader; it
returns an error if any payload lies outside the file or uses an unknown block type. This is an
inventory, not a full schema or checksum validation. The downloaded model's header names
387 tensors: F32, Q4_K, Q5_0, Q6_K and Q8_0. Do not run inference on an incomplete download.

Opt-in CMake targets: configure with `STRATA_HIP_GFX906=ON` and `STRATA_LLM_JP_PROTOTYPE=ON`,
then build `llm_jp_inspect`, `llm_jp_attention_test`, `llm_jp_norm_rope_test`,
`llm_jp_router_test`, and `llm_jp_real_mmvq`; run the first three GPU tests with
`ctest --test-dir BUILD -R '^llm_jp_' --output-on-failure`.

The real-weight smoke test runs 32 rows of a Q4_K tensor through Strata's native MMVQ on the GPU,
then compares against gguf-py's dequantization with the original float input. It accepts a verified
GGUF and an explicit tensor name; the GPU input is Q8_1, so expect bounded numerical error.

```bash
BUILD/llm_jp_real_mmvq MODEL.gguf blk.0.ffn_gate_exps.weight /tmp/gpu.bin
STRATA_GGUF_PY=/path/to/gguf-py python tests/llm_jp/check_real_mmvq.py \
  MODEL.gguf blk.0.ffn_gate_exps.weight /tmp/gpu.bin
```

On the Radeon VII, Q, gate and up (first expert, first 32 rows each) gave relative RMSE 0.00345,
0.00322 and 0.00200, respectively. This is a single synthetic input and is not a layer or
whole-model correctness check.

Remaining work includes stronger model and numerical validation, batched prefill, efficient
RAM/GPU expert caching and KV attention, server integration, and performance and stability testing.

The opt-in `llm_jp_real_expert` test also runs one complete real expert (gate,
up, SwiGLU, down) using Strata's grouped HIP kernels. Against independently
dequantized weights and float activations, relative RMSE was 0.58% for layer 0
(Q8_0 down) and 1.85% for layer 4 (Q5_0 down). Both activation quantization
steps contribute error; this is not an end-to-end model quality measurement.

```
./build-llm-jp/llm_jp_real_expert MODEL.gguf 4 0 /tmp/expert.bin
STRATA_GGUF_PY=/path/to/gguf-py python tests/llm_jp/check_real_expert.py MODEL.gguf 4 0 /tmp/expert.bin
```

The actual model has Q4_K gate/up in all 32 layers, and Q8_0 or Q5_0 down
in 16 layers each; both down variants have been checked above.

`expert_store.hpp` validates all 32 layer layouts and exposes per-expert views
into the GGUF mapping without copying the whole model. The mapping must stay
alive until GPU transfers complete. Layer 4 expert 127 was independently checked
against the CPU reference (relative RMSE 2.12%), exercising the final expert's
offset as well as the Q5_0 path.

`Moe` is a correctness-first, single-token routed FFN: it computes the F32
router logits, selects and normalizes the top 8, copies those experts from the
mapping, runs the grouped Strata kernels, and combines their outputs. The input
must already be FFN-normalized; residual and attention are not included. It
synchronizes for the host expert selection and transfers all eight experts per
call, so it is a baseline rather than a performance-optimized engine.

Against a separate NumPy/dequantized-weights reference, the chosen IDs matched
and the output relative RMSE was 0.58% at layer 0 and 1.54% at layer 4:

```
./build-llm-jp/llm_jp_real_moe MODEL.gguf 4 /tmp/moe.bin
STRATA_GGUF_PY=/path/to/gguf-py python tests/llm_jp/check_real_moe.py MODEL.gguf 4 /tmp/moe.bin
```

## Full decoder experiment

`llm_jp_token_probe` is an experimental greedy decoder that joins embedding,
all 32 attention and MoE layers, the final norm and output projection. It takes
and emits token IDs, supports consecutive positions, and is a separate opt-in
binary: it is **not yet** wired into the Strata HTTP server. The current code
streams weights from the GGUF and synchronizes often, so it prioritizes a
verifiable baseline over speed. KV cache uses F32 and requires about 128 KiB
per context token across the model; the attention kernel is not optimized for
long contexts despite the later optimized attention kernel. Restart the process for a new sequence.

The official Q4_K_M GGUF on gfx906 produced IDs `98335 642 11938` from input
`1`, matching three independently evaluated greedy next-token IDs from a local
llama.cpp build. This small smoke test is not a general numerical or quality
validation. A layer-4 test showed activation quantization can change the
boundary expert: for one input, the eighth selected expert differed from the
float reference, yielding 16% whole-layer relative RMSE; conditioning the CPU
reference on the GPU-selected IDs reduced it to about 1.1%. This is an expected
risk of the current activation quantization and requires broader evaluation.

```
./build-llm-jp/llm_jp_token_probe MODEL.gguf 3 1
```

`chat.py` is a single-turn wrapper using the official `tokenizer.json` and
`chat_template.jinja`; it requires Python packages `tokenizers` and `jinja2`.
It launches the experimental decoder, not llama.cpp. Download those two files
from the official `llm-jp/llm-jp-4.1-32b-a3b-thinking` repository, then run:

```
python src/llm_jp/chat.py --engine ./build-llm-jp/llm_jp_token_probe \
  --model MODEL.gguf --tokenizer-dir TOKENIZER_DIR --max-tokens 128 'こんにちは'
```

`--final-only` starts generation in the final channel; omit it to let the
thinking model choose its own analysis/final channel. The wrapper currently
has no conversation history, tool calling, streaming API, or sampling controls.

A longer test exposed a correctness/quality limitation: with the official chat
template, a Japanese greeting prompt and forced final channel, the first output
ID (`609`) matched the local llama.cpp oracle, while the second was `5371`
in this prototype versus `118157` in the oracle. The oracle logits for those
candidates were 14.2286 and 14.2830 respectively. Their closeness and the
observed routing changes suggest numerical sensitivity, but the cause has not
been established; do not treat the prototype as numerically equivalent to
llama.cpp. In that test the prototype generated Japanese text that did not
properly answer the request; the oracle also failed to give just the requested
greeting in that forced-final test. With the unmodified official generation
prompt, the oracle reasoned and then answered "こんにちは！". The token wrapper
currently provides a way to experiment, not evidence of dependable instruction
following.

## Diagnosing numerical differences

Setting `STRATA_LLMJP_TRACE_DIR` to an existing directory writes intermediate
activation rows for every token and layer. Use only on diagnostic inputs: these
files contain model activations derived from the prompt. `tests/llm_jp/llama_trace_oracle.cpp`
is an optional reference utility linked against a separate llama.cpp build; it
is never linked into or called by the decoder. Both accept token IDs so the
comparison does not depend on the tokenizer. For example, after compiling that
utility as `llama_trace_oracle` with the matching external headers and libraries:

```
mkdir -p /tmp/strata-trace /tmp/llama-trace
STRATA_LLMJP_TRACE_DIR=/tmp/strata-trace ./build-llm-jp/llm_jp_token_probe MODEL.gguf 1 1
./llama_trace_oracle MODEL.gguf /tmp/llama-trace 1
STRATA_GGUF_PY=/path/to/gguf-py python tests/llm_jp/compare_q8k_projection.py \
  MODEL.gguf /tmp/strata-trace /tmp/llama-trace
```

For the BOS token, the layer-0 input normalization matched the reference and
its V projection differed by 1.71% relative RMSE. Against dequantized weights
and float input, the Strata Q8_1 path had 0.78% error and the reference had
1.79%. Emulating Q8_K activation quantization in NumPy reproduced the reference
V projection within 2.6e-7 relative RMSE. Thus this particular discrepancy
comes from different activation quantization (Q8_1 per 32 values vs reference
Q8_K per 256), not evidence by itself of a faulty GPU matrix product. Different
rounding can still alter top-8 routing and subsequent generation; this single
projection comparison does not establish that all remaining differences are
explained or that either end-to-end response is better.

With the unmodified official chat template, the Q4_K_M model answered
“こんにちは！” to “こんにちは。短く挨拶して。” on the experimental GPU decoder.
The independent reference gave the same final answer; the generated token
sequences diverged during analysis. The test took 61 generated tokens in the
prototype, so short generation limits can stop before a final answer.
`--show-analysis` displays all generated channels and special tokens; by
default the wrapper prints only the final channel.

## Performance experiment

The default MoE path uses approximately 41 MiB of pinned host staging memory
and issues one asynchronous copy per selected expert, allowing host copies and
GPU transfers to overlap. If the allocation fails it falls back to pageable
transfers. Set `STRATA_LLMJP_PINNED=0` to force the original path; `bulk` tests
a single large transfer. It does not reserve additional persistent VRAM.
A 128 MiB LRU VRAM-cache experiment produced no hits when traversing all 32
layers in order and was therefore not retained.

Attention uses a wave-size-independent block reduction with online softmax,
computing each query/key dot product once rather than once per output element.
`STRATA_LLMJP_REFERENCE_ATTN=1` selects the original reference kernel for
comparison. With synthetic 40-query-head / 4-KV-head inputs on the Radeon VII,
`llm_jp_attention_perf` measured the original versus optimized kernels at
3.73 ms vs 0.40 ms (256 cached positions), and 15.02 ms vs 1.78 ms (1024).
Maximum output absolute error was below 3e-7 in those tests. These are kernel
measurements, not end-to-end generation throughput; results vary with load.

On the shared machine with the Qwen server running, two alternating-order
runs processing the same 32 input IDs and generating one token took 10.34–11.20 s
with the original transfer and attention paths and 8.33–10.90 s with both
optimizations. These are small, cache-sensitive experiments including startup,
not a general throughput claim. `STRATA_LLMJP_PROFILE=1` prints aggregate
phase times; staging enqueue and GPU completion times reflect asynchronous
execution rather than independent, non-overlapping hardware measurements.

The new attention changes floating-point reduction order. Both kernels match
an independent reference for the tested attention inputs, but MoE routing can
amplify small differences: in one 32-token greedy test, the first 25 generated
IDs matched and later IDs differed. The natural Japanese greeting test still
returned “こんにちは！” with the optimized path. This is a small smoke test,
not proof of equivalent generation quality or long-context correctness.
