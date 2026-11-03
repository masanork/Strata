#pragma once
#include "strata/artifact/gguf_reader.hpp"
#include <hip/hip_runtime.h>
#include <cstddef>
#include <array>
namespace strata::llm_jp {
// Single-token attention baseline. Input is already attention-normalized. A
// caller must write positions 0..position consecutively for each layer; output
// excludes the residual. The object and device buffers are not reentrant.
class Attention {
public:
    Attention(const GgufFile& file, int context);
    ~Attention();
    Attention(const Attention&)=delete;
    Attention& operator=(const Attention&)=delete;
    void run(int layer, int position, const float* input, float* output, hipStream_t stream);
private:
    void release() noexcept;
    const GgufFile& file_;
    int context_;
    std::array<int,32> next_position_{};
    size_t weight_capacity_=0;
    void *weight_=nullptr,*xq_=nullptr;
    float *q_=nullptr,*qnorm_=nullptr,*k_=nullptr,*v_=nullptr,*attn_=nullptr;
    float *qgamma_=nullptr,*kgamma_=nullptr,*keys_=nullptr,*values_=nullptr;
};
}
