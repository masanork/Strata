// Experimental Strata-native, token-ID-only greedy decoder. No HTTP server or
// tokenizer. Weights are streamed from GGUF for clarity, not throughput.
#include "layer.hpp"
#include "profile.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
__global__ void rms_final(const float* x,const float* gamma,float* y){
 __shared__ float a[256];float sum=0;for(int i=threadIdx.x;i<2560;i+=blockDim.x)sum+=x[i]*x[i];a[threadIdx.x]=sum;__syncthreads();
 for(int d=128;d>0;d/=2){if(threadIdx.x<(unsigned)d)a[threadIdx.x]+=a[threadIdx.x+d];__syncthreads();}
 float scale=rsqrtf(a[0]/2560.f+1e-6f);for(int i=threadIdx.x;i<2560;i+=blockDim.x)y[i]=x[i]*scale*gamma[i];
}
const strata::TensorInfo& tensor(const strata::GgufFile& file,const char* name,uint32_t type,std::vector<uint64_t> shape,size_t bytes){
 auto* t=file.find(name);if(!t||t->type!=type||t->shape!=shape)throw std::runtime_error(std::string("unsupported tensor: ")+name);
 const auto payload=file.file_size()-file.data_start();if(t->offset>payload||bytes>payload-t->offset)throw std::runtime_error(std::string("tensor outside GGUF: ")+name);return *t;
}
}
int main(int argc,char** argv){
 if(argc<4){std::fprintf(stderr,"usage: llm_jp_token_probe MODEL.gguf GENERATE_COUNT TOKEN_ID [TOKEN_ID ...]\n");return 2;}
 try {
  constexpr int H=2560,V=196608,chunk=4096;
  char* count_end=nullptr;long count=std::strtol(argv[2],&count_end,10);if(!count_end||*count_end||count<1||count>1024)throw std::invalid_argument("invalid generation count");
  std::vector<int> ids;for(int i=3;i<argc;i++){char* end=nullptr;long id=std::strtol(argv[i],&end,10);if(!end||*end||id<0||id>=V)throw std::invalid_argument("invalid token ID");ids.push_back(id);}
  strata::GgufFile file(argv[1]);auto row=strata::kernels::native_mmvq_weight_bytes(12,H,1);auto orow=strata::kernels::native_mmvq_weight_bytes(8,H,1);
  auto& emb=tensor(file,"token_embd.weight",12,{H,V},row*V);auto& head=tensor(file,"output.weight",8,{H,V},orow*V);auto& norm=tensor(file,"output_norm.weight",0,{H},H*4);
  strata::llm_jp::ExpertStore store(file);if(ids.size()+count>65536)throw std::invalid_argument("context exceeds model limit");
  const size_t prompt_size=ids.size();strata::llm_jp::Layer layer(file,store,ids.size()+count);hipStream_t stream;check(hipStreamCreate(&stream));
  float *state,*next,*gamma,*normed,*logits;void *w,*xq;check(hipMalloc(&state,H*4));check(hipMalloc(&next,H*4));check(hipMalloc(&gamma,H*4));check(hipMalloc(&normed,H*4));check(hipMalloc(&logits,chunk*4));check(hipMalloc(&w,chunk*orow));check(hipMalloc(&xq,strata::kernels::native_q8_1_bytes(H)));
  const char* trace=std::getenv("STRATA_LLMJP_TRACE_DIR");
  std::vector<float> host(chunk),trace_row(H);
  for(size_t pos=0;pos<ids.size();pos++){
   check(hipMemcpyAsync(w,file.tensor_data(emb)+ids[pos]*row,row,hipMemcpyHostToDevice,stream));strata::kernels::dequant_f32(12,w,0,1,H,state,stream);
   for(int l=0;l<32;l++){
    layer.run(l,pos,state,next,stream);std::swap(state,next);
    if(trace){
     check(hipMemcpy(trace_row.data(),state,H*4,hipMemcpyDeviceToHost));
     auto path=std::string(trace)+"/"+std::to_string(pos)+"_"+std::to_string(l)+".bin";
     FILE* f=std::fopen(path.c_str(),"wb");if(!f)throw std::runtime_error("cannot open trace: "+path);
     auto written=std::fwrite(trace_row.data(),sizeof(float),H,f);std::fclose(f);
     if(written!=H)throw std::runtime_error("cannot write trace: "+path);
    }
   }
   if(pos+1<prompt_size){
    if((pos+1)%16==0)std::fprintf(stderr,"prompt %zu/%zu tokens\n",pos+1,prompt_size);
    continue;
   }
   auto head_start=strata::llm_jp::ProfileClock::now();
   check(hipMemcpyAsync(gamma,file.tensor_data(norm),H*4,hipMemcpyHostToDevice,stream));hipLaunchKernelGGL(rms_final,dim3(1),dim3(256),0,stream,state,gamma,normed);check(hipGetLastError());
   strata::kernels::native_quantize_q8_1(normed,xq,H,1,stream);
   float best=-INFINITY,second=-INFINITY;int bestid=-1,secondid=-1;
   for(int start=0;start<V;start+=chunk){int n=std::min(chunk,V-start);check(hipMemcpyAsync(w,file.tensor_data(head)+start*orow,n*orow,hipMemcpyHostToDevice,stream));strata::kernels::native_mmvq(8,w,xq,logits,H,n,1,stream);check(hipMemcpyAsync(host.data(),logits,n*4,hipMemcpyDeviceToHost,stream));check(hipStreamSynchronize(stream));
    for(int i=0;i<n;i++){float v=host[i];if(v>best){second=best;secondid=bestid;best=v;bestid=start+i;}else if(v>second){second=v;secondid=start+i;}}
   }
   if(strata::llm_jp::profile.enabled){strata::llm_jp::profile.head+=strata::llm_jp::Profile::elapsed(head_start);++strata::llm_jp::profile.heads;}
   if(std::getenv("STRATA_LLMJP_SHOW_TOP2"))std::fprintf(stderr,"top2 %d %.9g %d %.9g gap=%.9g\n",bestid,best,secondid,second,best-second);
   std::printf("%d\n",bestid);std::fflush(stdout);
   if(bestid==2||ids.size()-prompt_size+1>=size_t(count))break;
   ids.push_back(bestid);
  }
  strata::llm_jp::profile.report();
  check(hipFree(state));check(hipFree(next));check(hipFree(gamma));check(hipFree(normed));check(hipFree(logits));check(hipFree(w));check(hipFree(xq));check(hipStreamDestroy(stream));
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
