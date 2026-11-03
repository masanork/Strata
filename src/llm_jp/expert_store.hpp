#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::llm_jp {
// A view into the read-only GGUF mapping. The file must outlive this object and
// any transfers using the returned pointers. No expert weights are duplicated.
class ExpertStore {
public:
    static constexpr int layers = 32, experts = 128, hidden = 2560, intermediate = 960;
    struct Slice { const uint8_t* gate; const uint8_t* up; const uint8_t* down; };
    struct Layer {
        const TensorInfo *gate, *up, *down;
        kernels::NativeExpertLayout layout;
    };

    explicit ExpertStore(const GgufFile& file) : file_(file) {
        const std::vector<uint64_t> gu_shape{hidden, intermediate, experts};
        const std::vector<uint64_t> down_shape{intermediate, hidden, experts};
        for (int i=0; i<layers; ++i) {
            const auto prefix = "blk." + std::to_string(i) + ".ffn_";
            Layer l{file.find(prefix+"gate_exps.weight"), file.find(prefix+"up_exps.weight"),
                    file.find(prefix+"down_exps.weight"), {}};
            if (!l.gate || !l.up || !l.down || l.gate->shape != gu_shape || l.up->shape != gu_shape ||
                l.down->shape != down_shape || l.gate->type != l.up->type ||
                !kernels::native_expert_supported(l.gate->type,l.down->type,hidden,intermediate))
                throw std::runtime_error("unsupported expert tensor layout at layer " + std::to_string(i));
            l.layout = kernels::native_expert_layout(l.gate->type,l.down->type,hidden,intermediate);
            check(*l.gate,l.layout.up_off); check(*l.up,l.layout.up_off);
            check(*l.down,l.layout.bytes-l.layout.down_off);
            layers_.push_back(l);
        }
    }
    const Layer& layer(int i) const { return layers_.at(i); }
    Slice expert(int layer_index, int expert_index) const {
        if (expert_index < 0 || expert_index >= experts) throw std::out_of_range("expert index");
        const auto& l = layer(layer_index);
        size_t gu = l.layout.up_off, down = l.layout.bytes-l.layout.down_off;
        return {file_.tensor_data(*l.gate)+expert_index*gu,
                file_.tensor_data(*l.up)+expert_index*gu,
                file_.tensor_data(*l.down)+expert_index*down};
    }
private:
    void check(const TensorInfo& t,size_t per_expert) const {
        const auto payload=file_.file_size()-file_.data_start();
        if (t.offset>payload || per_expert>(payload-t.offset)/experts)
            throw std::runtime_error("expert tensor outside GGUF: "+t.name);
    }
    const GgufFile& file_;
    std::vector<Layer> layers_;
};
}
