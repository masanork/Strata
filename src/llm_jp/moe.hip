#include "moe.hpp"
#include "profile.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>

namespace strata::llm_jp {
namespace {
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
template<class T> void alloc(T** p, size_t bytes) { check(hipMalloc(reinterpret_cast<void**>(p),bytes)); }
__global__ void router_logits(const float* weights, const float* x, float* logits) {
    __shared__ float sums[256];
    float v=0; const int expert=blockIdx.x;
    for(int i=threadIdx.x;i<ExpertStore::hidden;i+=blockDim.x)
        v += weights[expert*ExpertStore::hidden+i]*x[i];
    sums[threadIdx.x]=v; __syncthreads();
    for(int stride=128;stride>0;stride/=2) {
        if(threadIdx.x<(unsigned)stride) sums[threadIdx.x]+=sums[threadIdx.x+stride];
        __syncthreads();
    }
    if(threadIdx.x==0) logits[expert]=sums[0];
}
}
Moe::Moe(const GgufFile& file,const ExpertStore& store):file_(file),store_(store) {
    try {
        size_t max_bytes=0;
        for(int i=0;i<ExpertStore::layers;i++) {
            const auto& l=store.layer(i); if(l.layout.bytes>max_bytes)max_bytes=l.layout.bytes;
            const auto name="blk."+std::to_string(i)+".ffn_gate_inp.weight";
            const auto* t=file.find(name);
            if(!t || t->type!=0 || t->shape!=std::vector<uint64_t>{ExpertStore::hidden,ExpertStore::experts})
                throw std::runtime_error("unsupported router: "+name);
            size_t bytes=ExpertStore::hidden*ExpertStore::experts*sizeof(float);
            const auto payload=file.file_size()-file.data_start();
            if(t->offset>payload||bytes>payload-t->offset) throw std::runtime_error("router outside GGUF: "+name);
        }
        alloc(&blobs_,8*max_bytes);
        const char* mode=std::getenv("STRATA_LLMJP_PINNED");
        if(!mode || std::strcmp(mode,"0")!=0) {
            split_copy_=!mode || std::strcmp(mode,"bulk")!=0;
            if(hipHostMalloc(&staging_,8*max_bytes,hipHostMallocDefault)!=hipSuccess) {
                staging_=nullptr;
                std::fprintf(stderr,"LLM-jp: pinned staging unavailable; using pageable transfers\n");
            }
        }
        alloc(&xq_,(ExpertStore::hidden/32)*36);
        alloc(&scratch_,kernels::native_expert_scratch_bytes(8,ExpertStore::intermediate));
        alloc(&router_,ExpertStore::hidden*ExpertStore::experts*sizeof(float)); alloc(&logits_,128*sizeof(float));
        alloc(&weights_dev_,8*sizeof(float)); alloc(&parts_,8*ExpertStore::hidden*sizeof(float));
        alloc(&ids_dev_,8*sizeof(int32_t)); alloc(&start_,9*sizeof(int32_t)); alloc(&groups_,sizeof(int32_t));
        alloc(&dst_,8*sizeof(int32_t)); alloc(&tok_,8*sizeof(int32_t)); alloc(&ptr_,8*sizeof(unsigned long long));
    } catch(...) { release(); throw; }
}
void Moe::release() noexcept {
    if(staging_) (void)hipHostFree(staging_);
    (void)hipFree(blobs_); (void)hipFree(xq_); (void)hipFree(scratch_); (void)hipFree(router_); (void)hipFree(logits_); (void)hipFree(weights_dev_);
    (void)hipFree(parts_); (void)hipFree(ids_dev_); (void)hipFree(start_); (void)hipFree(groups_); (void)hipFree(dst_); (void)hipFree(tok_); (void)hipFree(ptr_);
}
Moe::~Moe(){ release(); }
void Moe::run(int layer,const float* input,float* output,hipStream_t stream) {
    if(!input || !output || !stream) throw std::invalid_argument("MoE requires device input/output and stream");
    unsigned long long ptr[8]; int32_t start[9],dst[8],tok[8],groups=8;
    try {
    auto t0=ProfileClock::now();
    const auto& l=store_.layer(layer); const auto& L=l.layout;
    auto* router=file_.find("blk."+std::to_string(layer)+".ffn_gate_inp.weight");
    check(hipMemcpyAsync(router_,file_.tensor_data(*router),128*2560*sizeof(float),hipMemcpyHostToDevice,stream));
    hipLaunchKernelGGL(router_logits,dim3(128),dim3(256),0,stream,router_,input,logits_);
    check(hipGetLastError());
    kernels::native_router_topk(logits_,ids_dev_,weights_dev_,128,8,1,stream);
    check(hipMemcpyAsync(ids_,ids_dev_,sizeof(ids_),hipMemcpyDeviceToHost,stream));
    check(hipMemcpyAsync(weights_,weights_dev_,sizeof(weights_),hipMemcpyDeviceToHost,stream));
    check(hipStreamSynchronize(stream));
    auto t1=ProfileClock::now();
    for(int i=0;i<8;i++) {
        auto slice=store_.expert(layer,ids_[i]);auto* blob=static_cast<uint8_t*>(blobs_)+i*L.bytes;
        if(staging_) {
            auto* host=static_cast<uint8_t*>(staging_)+i*L.bytes;
            std::memcpy(host,slice.gate,L.up_off);std::memcpy(host+L.up_off,slice.up,L.up_off);
            std::memcpy(host+L.down_off,slice.down,L.bytes-L.down_off);
            if(split_copy_)check(hipMemcpyAsync(blob,host,L.bytes,hipMemcpyHostToDevice,stream));
        } else {
            check(hipMemcpyAsync(blob,slice.gate,L.up_off,hipMemcpyHostToDevice,stream));
            check(hipMemcpyAsync(blob+L.up_off,slice.up,L.up_off,hipMemcpyHostToDevice,stream));
            check(hipMemcpyAsync(blob+L.down_off,slice.down,L.bytes-L.down_off,hipMemcpyHostToDevice,stream));
        }
        ptr[i]=reinterpret_cast<unsigned long long>(blob);start[i]=i;dst[i]=i;tok[i]=0;
    }
    if(staging_&&!split_copy_)check(hipMemcpyAsync(blobs_,staging_,8*L.bytes,hipMemcpyHostToDevice,stream));
    auto t2=ProfileClock::now();
    start[8]=8;
    check(hipMemcpyAsync(ptr_,ptr,sizeof(ptr),hipMemcpyHostToDevice,stream));
    check(hipMemcpyAsync(start_,start,sizeof(start),hipMemcpyHostToDevice,stream));
    check(hipMemcpyAsync(groups_,&groups,sizeof(groups),hipMemcpyHostToDevice,stream));
    check(hipMemcpyAsync(dst_,dst,sizeof(dst),hipMemcpyHostToDevice,stream));
    check(hipMemcpyAsync(tok_,tok,sizeof(tok),hipMemcpyHostToDevice,stream));
    kernels::quantize_q8_1_rows(input,1,2560,xq_,stream);
    kernels::native_expert_grouped(L,ptr_,start_,groups_,dst_,tok_,8,8,xq_,scratch_,parts_,stream);
    kernels::native_moe_combine(parts_,weights_dev_,nullptr,output,2560,8,stream);
    check(hipStreamSynchronize(stream));
    if(profile.enabled){profile.route+=std::chrono::duration<double,std::milli>(t1-t0).count();profile.stage+=std::chrono::duration<double,std::milli>(t2-t1).count();profile.expert+=Profile::elapsed(t2);}
    } catch (...) {
        // Complete outstanding transfers before local descriptors leave scope.
        (void)hipStreamSynchronize(stream);
        throw;
    }
}
}
