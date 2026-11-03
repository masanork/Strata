#pragma once
#include "expert_store.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>

namespace strata::llm_jp {
// Correctness-first single-token MoE. Input is the already RMS-normalized FFN
// input on the device. run() synchronizes to fetch route IDs and before return.
// File, store, stream and device input/output must outlive each call. Not reentrant.
class Moe {
public:
    explicit Moe(const GgufFile& file, const ExpertStore& store);
    ~Moe();
    Moe(const Moe&) = delete;
    Moe& operator=(const Moe&) = delete;
    void run(int layer, const float* input, float* output, hipStream_t stream);
    const int32_t* ids() const { return ids_; }
    const float* weights() const { return weights_; }
private:
    void release() noexcept;
    const GgufFile& file_;
    const ExpertStore& store_;
    void *blobs_=nullptr, *xq_=nullptr, *scratch_=nullptr, *staging_=nullptr;
    float *router_=nullptr, *logits_=nullptr, *weights_dev_=nullptr, *parts_=nullptr;
    int32_t *ids_dev_=nullptr, *start_=nullptr, *groups_=nullptr, *dst_=nullptr, *tok_=nullptr;
    unsigned long long* ptr_=nullptr;
    int32_t ids_[8]{};
    float weights_[8]{};
    bool split_copy_=false;
};
}
