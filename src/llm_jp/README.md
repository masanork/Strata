# LLM-jp 4.1 32B-A3B: experimental gfx906 work

This is an early prototype. It does **not** load or run LLM-jp yet and is not wired into the Strata server.

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

Remaining work: GGUF metadata and quantized tensor validation; embeddings and normalization; RoPE integration;
Q/K/V and output projections; causal prefill and KV management; router and quantized expert execution
with RAM/GPU placement; output projection and tokenizer; server protocol; per-layer and end-to-end
logit parity against an independent reference; and real-model performance and stability testing.

The opt-in `llm_jp_real_expert` test also runs one complete real expert (gate,
up, SwiGLU, down) using Strata's grouped HIP kernels. Against independently
dequantized weights and float activations, relative RMSE was 0.58% for layer 0
(Q8_0 down) and 1.85% for layer 4 (Q5_0 down). Both activation quantization
steps contribute error; this is not an end-to-end model quality measurement.

```
./build-llm-jp/llm_jp_real_expert MODEL.gguf 4 /tmp/expert.bin
STRATA_GGUF_PY=/path/to/gguf-py python tests/llm_jp/check_real_expert.py MODEL.gguf 4 /tmp/expert.bin
```

The actual model has Q4_K gate/up in all 32 layers, and Q8_0 or Q5_0 down
in 16 layers each; both down variants have been checked above.
