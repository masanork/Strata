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

Remaining work: GGUF metadata and quantized tensor validation; embeddings and normalization; RoPE integration;
Q/K/V and output projections; causal prefill and KV management; router and quantized expert execution
with RAM/GPU placement; output projection and tokenizer; server protocol; per-layer and end-to-end
logit parity against an independent reference; and real-model performance and stability testing.
