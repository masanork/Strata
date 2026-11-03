#pragma once
#include "attention.hpp"
#include "expert_store.hpp"
#include "moe.hpp"
namespace strata::llm_jp {
// One decode layer, including both residual connections. Each layer must be
// visited in position order. Single stream, single caller; input/output device.
class Layer {
public:
    Layer(const GgufFile& file, const ExpertStore& store, int context);
    ~Layer();
    Layer(const Layer&)=delete;
    Layer& operator=(const Layer&)=delete;
    void run(int layer,int position,const float* input,float* output,hipStream_t stream);
    const int32_t* expert_ids() const { return moe_.ids(); }
private:
    const GgufFile& file_;
    Attention attention_;
    Moe moe_;
    float *norm_=nullptr,*gamma_=nullptr,*residual_=nullptr,*part_=nullptr;
};
}
