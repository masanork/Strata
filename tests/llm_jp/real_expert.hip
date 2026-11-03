#include "../../src/llm_jp/expert_store.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>
#include <string>
#include <cstdlib>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess){std::fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));return 2;} } while(0)
int main(int argc,char** argv) {
 if(argc!=5){std::fprintf(stderr,"usage: real_expert MODEL.gguf LAYER EXPERT OUTPUT.bin\n");return 2;}
 char* end=nullptr; long layer=std::strtol(argv[2],&end,10); if(!end || *end || layer<0 || layer>=32)return 2;
 long expert=std::strtol(argv[3],&end,10); if(!end || *end || expert<0 || expert>=128)return 2;
 try {
  namespace k=strata::kernels; strata::GgufFile f(argv[1]); constexpr int H=2560,F=960;
  strata::llm_jp::ExpertStore store(f);
  const auto& L=store.layer(layer).layout;
  auto slice=store.expert(layer,expert);
  std::vector<float> x(H),y(H); for(int i=0;i<H;i++)x[i]=std::sin(float(i)*0.013f)*0.1f;
  void *blob,*xq,*scratch; float *dx,*out; unsigned long long* ptr; int32_t *start,*ng,*dst,*tok; hipStream_t s;
  CHECK(hipStreamCreate(&s)); CHECK(hipMalloc(&blob,L.bytes)); CHECK(hipMalloc(&xq,(H/32)*36)); CHECK(hipMalloc(&scratch,k::native_expert_scratch_bytes(1,F))); CHECK(hipMalloc(&dx,H*4)); CHECK(hipMalloc(&out,H*4)); CHECK(hipMalloc(&ptr,8)); CHECK(hipMalloc(&start,8)); CHECK(hipMalloc(&ng,4)); CHECK(hipMalloc(&dst,4)); CHECK(hipMalloc(&tok,4));
  CHECK(hipMemcpy(blob,slice.gate,L.up_off,hipMemcpyHostToDevice)); CHECK(hipMemcpy((char*)blob+L.up_off,slice.up,L.up_off,hipMemcpyHostToDevice)); CHECK(hipMemcpy((char*)blob+L.down_off,slice.down,L.bytes-L.down_off,hipMemcpyHostToDevice)); CHECK(hipMemcpy(dx,x.data(),H*4,hipMemcpyHostToDevice));
  unsigned long long p=(unsigned long long)blob; int32_t starts[2]={0,1},one=1,zero=0;
  CHECK(hipMemcpy(ptr,&p,8,hipMemcpyHostToDevice)); CHECK(hipMemcpy(start,starts,8,hipMemcpyHostToDevice)); CHECK(hipMemcpy(ng,&one,4,hipMemcpyHostToDevice)); CHECK(hipMemcpy(dst,&zero,4,hipMemcpyHostToDevice)); CHECK(hipMemcpy(tok,&zero,4,hipMemcpyHostToDevice));
  k::quantize_q8_1_rows(dx,1,H,xq,s); k::native_expert_grouped(L,ptr,start,ng,dst,tok,1,1,xq,scratch,out,s); CHECK(hipStreamSynchronize(s)); CHECK(hipMemcpy(y.data(),out,H*4,hipMemcpyDeviceToHost));
  FILE* file=std::fopen(argv[4],"wb"); if(!file)return 5; size_t n=std::fwrite(y.data(),4,y.size(),file); std::fclose(file); if(n!=y.size())return 6;
  std::printf("computed real expert: %zu bytes\n",L.bytes);
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
