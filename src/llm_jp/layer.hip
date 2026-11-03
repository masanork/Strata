#include "layer.hpp"
#include "profile.hpp"
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
namespace strata::llm_jp {
namespace {
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
__global__ void rms(const float* x,const float* gamma,float* y){
    __shared__ float a[256];float sum=0;
    for(int i=threadIdx.x;i<2560;i+=blockDim.x)sum+=x[i]*x[i];
    a[threadIdx.x]=sum;__syncthreads();
    for(int stride=128;stride>0;stride/=2){if(threadIdx.x<(unsigned)stride)a[threadIdx.x]+=a[threadIdx.x+stride];__syncthreads();}
    float scale=rsqrtf(a[0]/2560.f+1e-6f);
    for(int i=threadIdx.x;i<2560;i+=blockDim.x)y[i]=x[i]*scale*gamma[i];
}
__global__ void add(const float* x,const float* y,float* out){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<2560)out[i]=x[i]+y[i];}
void trace_row(const char* name,int layer,int position,const float* data,hipStream_t stream) {
    const char* dir=std::getenv("STRATA_LLMJP_TRACE_DIR"); if(!dir)return;
    std::vector<float> row(2560);check(hipMemcpyAsync(row.data(),data,row.size()*sizeof(float),hipMemcpyDeviceToHost,stream));check(hipStreamSynchronize(stream));
    const auto path=std::string(dir)+"/"+std::to_string(position)+"_"+std::to_string(layer)+"_"+name+".bin";
    FILE* f=std::fopen(path.c_str(),"wb");if(!f)throw std::runtime_error("cannot open trace: "+path);
    auto n=std::fwrite(row.data(),sizeof(float),row.size(),f);std::fclose(f);if(n!=row.size())throw std::runtime_error("cannot write trace: "+path);
}
const TensorInfo& norm_tensor(const GgufFile& file,int layer,const char* name){
    auto key="blk."+std::to_string(layer)+"."+name+".weight";auto* t=file.find(key);
    if(!t||t->type!=0||t->shape!=std::vector<uint64_t>{2560})throw std::runtime_error("unsupported norm: "+key);
    const auto payload=file.file_size()-file.data_start();
    if(t->offset>payload||2560*sizeof(float)>payload-t->offset)throw std::runtime_error("norm outside GGUF: "+key);
    return *t;
}
}
Layer::Layer(const GgufFile& file,const ExpertStore& store,int context):file_(file),attention_(file,context),moe_(file,store){
    try {
        for(int i=0;i<32;i++){norm_tensor(file,i,"attn_norm");norm_tensor(file,i,"ffn_norm");}
        check(hipMalloc(&norm_,2560*4));check(hipMalloc(&gamma_,2560*4));check(hipMalloc(&residual_,2560*4));check(hipMalloc(&part_,2560*4));
    }catch(...){(void)hipFree(norm_);(void)hipFree(gamma_);(void)hipFree(residual_);(void)hipFree(part_);throw;}
}
Layer::~Layer(){(void)hipFree(norm_);(void)hipFree(gamma_);(void)hipFree(residual_);(void)hipFree(part_);}
void Layer::run(int layer,int position,const float* input,float* output,hipStream_t stream){
    if(layer<0||layer>=32||!input||!output||!stream)throw std::invalid_argument("invalid layer input");
    try {
        auto normalize=[&](const char* name,const float* x){auto& t=norm_tensor(file_,layer,name);check(hipMemcpyAsync(gamma_,file_.tensor_data(t),2560*4,hipMemcpyHostToDevice,stream));hipLaunchKernelGGL(rms,dim3(1),dim3(256),0,stream,x,gamma_,norm_);check(hipGetLastError());};
        auto t0=ProfileClock::now();
        normalize("attn_norm",input);trace_row("attn_norm",layer,position,norm_,stream);auto t1=ProfileClock::now();attention_.run(layer,position,norm_,part_,stream);
        auto t2=ProfileClock::now();
        hipLaunchKernelGGL(add,dim3(10),dim3(256),0,stream,input,part_,residual_);check(hipGetLastError());
        trace_row("ffn_inp",layer,position,residual_,stream);normalize("ffn_norm",residual_);trace_row("ffn_norm",layer,position,norm_,stream);auto t3=ProfileClock::now();moe_.run(layer,norm_,part_,stream);auto t4=ProfileClock::now();
        if(const char* path=std::getenv("STRATA_LLMJP_ROUTE_TRACE")) {
            FILE* f=std::fopen(path,"a");if(!f)throw std::runtime_error("cannot open route trace");
            std::fprintf(f,"%d %d",position,layer);for(int i=0;i<8;i++)std::fprintf(f," %d",moe_.ids()[i]);std::fprintf(f,"\n");std::fclose(f);
        }
        trace_row("ffn_moe_out",layer,position,part_,stream);
        hipLaunchKernelGGL(add,dim3(10),dim3(256),0,stream,residual_,part_,output);check(hipGetLastError());check(hipStreamSynchronize(stream));
        if(profile.enabled){auto t5=ProfileClock::now();profile.attention+=std::chrono::duration<double,std::milli>(t2-t1).count();profile.moe+=std::chrono::duration<double,std::milli>(t4-t3).count();profile.layer_other+=std::chrono::duration<double,std::milli>((t1-t0)+(t3-t2)+(t5-t4)).count();++profile.layers;}
    }catch(...){(void)hipStreamSynchronize(stream);throw;}
}
}
