#pragma once
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

// Correctness-first Qwen3-MoE single-token decode attention. Layouts:
// q [n_head, head_dim], cache [position, n_kv_head, head_dim], output [n_head, head_dim].
// One lane computes one output dimension; recomputing logits avoids shared-memory assumptions
// and makes the initial wave64 implementation independent of hardware warp size.
namespace strata::llm_jp {
__global__ void attention_decode_reference(const float* q, const float* keys, const float* values,
                                 float* output, int n_head, int n_kv_head, int head_dim,
                                 int length, float scale) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (h >= n_head || d >= head_dim) return;
    const int kvh = h / (n_head / n_kv_head);
    const float* query = q + h * head_dim;
    float maximum = -INFINITY;
    for (int t = 0; t < length; ++t) {
        const float* key = keys + (t * n_kv_head + kvh) * head_dim;
        float dot = 0;
        for (int k = 0; k < head_dim; ++k) dot += query[k] * key[k];
        maximum = fmaxf(maximum, dot * scale);
    }
    float numerator = 0, denominator = 0;
    for (int t = 0; t < length; ++t) {
        const float* key = keys + (t * n_kv_head + kvh) * head_dim;
        float dot = 0;
        for (int k = 0; k < head_dim; ++k) dot += query[k] * key[k];
        const float weight = expf(dot * scale - maximum);
        denominator += weight;
        numerator += weight * values[(t * n_kv_head + kvh) * head_dim + d];
    }
    output[h * head_dim + d] = numerator / denominator;
}

// Wave-size-independent online softmax: one block per query head. Each lane
// accumulates one output component; dot products are reduced once per token.
// Caller supplies exactly head_dim threads, where head_dim is a power of two.
__global__ void attention_decode(const float* q, const float* keys, const float* values,
                                 float* output, int n_head, int n_kv_head, int head_dim,
                                 int length, float scale) {
    const int h=blockIdx.x,d=threadIdx.x;
    if(h>=n_head || d>=head_dim) return;
    __shared__ float partial[128], maximum, denominator, old_scale, new_scale;
    if(d==0){maximum=-INFINITY;denominator=0;}
    __syncthreads();
    const int kvh=h/(n_head/n_kv_head);
    const float query=q[h*head_dim+d];float acc=0;
    for(int t=0;t<length;t++) {
        const size_t offset=(size_t(t)*n_kv_head+kvh)*head_dim+d;
        partial[d]=query*keys[offset];__syncthreads();
        for(int stride=head_dim/2;stride>0;stride/=2) {
            if(d<stride)partial[d]+=partial[d+stride];
            __syncthreads();
        }
        if(d==0){
            float score=partial[0]*scale, next=fmaxf(maximum,score);
            old_scale=expf(maximum-next);new_scale=expf(score-next);
            denominator=denominator*old_scale+new_scale;maximum=next;
        }
        __syncthreads();
        acc=acc*old_scale+values[offset]*new_scale;
        __syncthreads();
    }
    output[h*head_dim+d]=acc/denominator;
}
}
