#include "attention.hpp"
#include "attention.cuh"
#include "norm_rope.cuh"
#include "strata/kernels/native_mmvq.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
namespace strata::llm_jp {
namespace {
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
template<class T> void alloc(T** p,size_t n){check(hipMalloc(reinterpret_cast<void**>(p),n));}
void trace(const char* name,int layer,int position,const float* data,size_t count,hipStream_t stream) {
    const char* dir=std::getenv("STRATA_LLMJP_TRACE_DIR");if(!dir)return;
    std::vector<float> row(count);check(hipMemcpyAsync(row.data(),data,count*sizeof(float),hipMemcpyDeviceToHost,stream));check(hipStreamSynchronize(stream));
    auto path=std::string(dir)+"/"+std::to_string(position)+"_"+std::to_string(layer)+"_"+name+".bin";
    FILE* f=std::fopen(path.c_str(),"wb");if(!f)throw std::runtime_error("cannot open trace: "+path);
    auto n=std::fwrite(row.data(),sizeof(float),count,f);std::fclose(f);if(n!=count)throw std::runtime_error("cannot write trace: "+path);
}
const TensorInfo& tensor(const GgufFile& file,int layer,const char* name,int in,int out,bool gamma=false) {
    const std::string key="blk."+std::to_string(layer)+"."+name+".weight";
    auto* t=file.find(key);
    const std::vector<uint64_t> shape=gamma?std::vector<uint64_t>{(uint64_t)in}:std::vector<uint64_t>{(uint64_t)in,(uint64_t)out};
    if(!t||t->shape!=shape||(gamma?t->type!=0:!kernels::native_mmvq_supported(t->type)))throw std::runtime_error("unsupported attention tensor: "+key);
    size_t bytes=gamma?in*sizeof(float):kernels::native_mmvq_weight_bytes(t->type,in,out);
    auto payload=file.file_size()-file.data_start();
    if(t->offset>payload||bytes>payload-t->offset)throw std::runtime_error("attention tensor outside GGUF: "+key);
    return *t;
}
}
Attention::Attention(const GgufFile& file,int context):file_(file),context_(context) {
    if(context<1||context>65536)throw std::invalid_argument("invalid attention context");
    try {
        for(int l=0;l<32;l++){
            for(const auto& spec: {std::pair{"attn_q",5120}, {"attn_k",512}, {"attn_v",512}}){
                auto& t=tensor(file,l,spec.first,2560,spec.second);
                weight_capacity_=std::max(weight_capacity_,kernels::native_mmvq_weight_bytes(t.type,2560,spec.second));
            }
            auto& o=tensor(file,l,"attn_output",5120,2560);
            weight_capacity_=std::max(weight_capacity_,kernels::native_mmvq_weight_bytes(o.type,5120,2560));
            tensor(file,l,"attn_q_norm",128,0,true); tensor(file,l,"attn_k_norm",128,0,true);
        }
        alloc(&weight_,weight_capacity_); alloc(&xq_,kernels::native_q8_1_bytes(5120));
        alloc(&q_,5120*4); alloc(&qnorm_,5120*4); alloc(&k_,512*4); alloc(&v_,512*4); alloc(&attn_,5120*4);
        alloc(&qgamma_,128*4); alloc(&kgamma_,128*4);
        const size_t cache=size_t(32)*context*512*sizeof(float); alloc(&keys_,cache); alloc(&values_,cache);
    }catch(...){release();throw;}
}
void Attention::release() noexcept {
    (void)hipFree(weight_); (void)hipFree(xq_); (void)hipFree(q_); (void)hipFree(qnorm_); (void)hipFree(k_);
    (void)hipFree(v_); (void)hipFree(attn_); (void)hipFree(qgamma_); (void)hipFree(kgamma_);
    (void)hipFree(keys_); (void)hipFree(values_);
}
Attention::~Attention(){release();}
void Attention::run(int layer,int position,const float* input,float* output,hipStream_t stream){
    if(layer<0||layer>=32||position<0||position>=context_||!input||!output||!stream)throw std::invalid_argument("invalid attention input");
    if(position!=next_position_[layer]) throw std::invalid_argument("attention positions must be consecutive");
    try {
        kernels::native_quantize_q8_1(input,xq_,2560,1,stream);
        auto project=[&](const char* name,int in,int out,float* target){
            auto& t=tensor(file_,layer,name,in,out); auto bytes=kernels::native_mmvq_weight_bytes(t.type,in,out);
            check(hipMemcpyAsync(weight_,file_.tensor_data(t),bytes,hipMemcpyHostToDevice,stream));
            kernels::native_mmvq(t.type,weight_,xq_,target,in,out,1,stream);
        };
        project("attn_q",2560,5120,q_); project("attn_k",2560,512,k_); project("attn_v",2560,512,v_);
        trace("Qraw",layer,position,q_,5120,stream);trace("Kraw",layer,position,k_,512,stream);trace("Vcur",layer,position,v_,512,stream);
        auto& qg=tensor(file_,layer,"attn_q_norm",128,0,true); auto& kg=tensor(file_,layer,"attn_k_norm",128,0,true);
        check(hipMemcpyAsync(qgamma_,file_.tensor_data(qg),128*4,hipMemcpyHostToDevice,stream));
        check(hipMemcpyAsync(kgamma_,file_.tensor_data(kg),128*4,hipMemcpyHostToDevice,stream));
        size_t offset=(size_t(layer)*context_+position)*512;
        hipLaunchKernelGGL(norm_rope,dim3(40),dim3(128),0,stream,q_,qgamma_,qnorm_,40,128,position,1e-6f,500000.f);
        hipLaunchKernelGGL(norm_rope,dim3(4),dim3(128),0,stream,k_,kgamma_,keys_+offset,4,128,position,1e-6f,500000.f);
        check(hipMemcpyAsync(values_+offset,v_,512*4,hipMemcpyDeviceToDevice,stream));
        trace("Qcur",layer,position,qnorm_,5120,stream);trace("Kcur",layer,position,keys_+offset,512,stream);
        if(std::getenv("STRATA_LLMJP_REFERENCE_ATTN"))
            hipLaunchKernelGGL(attention_decode_reference,dim3(40),dim3(128),0,stream,qnorm_,keys_+size_t(layer)*context_*512,values_+size_t(layer)*context_*512,attn_,40,4,128,position+1,1.f/std::sqrt(128.f));
        else
            hipLaunchKernelGGL(attention_decode,dim3(40),dim3(128),0,stream,qnorm_,keys_+size_t(layer)*context_*512,values_+size_t(layer)*context_*512,attn_,40,4,128,position+1,1.f/std::sqrt(128.f));
        check(hipGetLastError());
        kernels::native_quantize_q8_1(attn_,xq_,5120,1,stream);
        project("attn_output",5120,2560,output);trace("attn_output",layer,position,output,2560,stream);
        check(hipStreamSynchronize(stream));
        ++next_position_[layer];
    } catch(...) { (void)hipStreamSynchronize(stream); throw; }
}
}
