#include "strata/artifact/gguf_reader.hpp"
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
 if(argc!=4){std::fprintf(stderr,"usage: real_expert MODEL.gguf LAYER OUTPUT.bin\n");return 2;}
 char* end=nullptr; long layer=std::strtol(argv[2],&end,10); if(!end || *end || layer<0 || layer>=32)return 2;
 try {
  namespace k=strata::kernels; strata::GgufFile f(argv[1]); constexpr int H=2560,F=960;
  std::string prefix="blk."+std::to_string(layer)+".ffn_";
  auto *g=f.find(prefix+"gate_exps.weight"), *u=f.find(prefix+"up_exps.weight"), *d=f.find(prefix+"down_exps.weight");
  if(!g||!u||!d||g->type!=12||u->type!=12||(d->type!=8&&d->type!=6)||g->shape!=std::vector<uint64_t>{H,F,128}||u->shape!=g->shape||d->shape!=std::vector<uint64_t>{F,H,128}) {std::fprintf(stderr,"unexpected expert tensor layout\n");return 3;}
  auto L=k::native_expert_layout(g->type,d->type,H,F);
  if(!k::native_expert_supported(g->type,d->type,H,F)) {std::fprintf(stderr,"unsupported expert format\n");return 3;}
  auto valid=[&](const auto* t,size_t bytes){return t->offset<=f.file_size()-f.data_start()&&bytes<=f.file_size()-f.data_start()-t->offset;};
  if(!valid(g,L.up_off)||!valid(u,L.up_off)||!valid(d,L.bytes-L.down_off))return 4;
  std::vector<float> x(H),y(H); for(int i=0;i<H;i++)x[i]=std::sin(float(i)*0.013f)*0.1f;
  void *blob,*xq,*scratch; float *dx,*out; unsigned long long* ptr; int32_t *start,*ng,*dst,*tok; hipStream_t s;
  CHECK(hipStreamCreate(&s)); CHECK(hipMalloc(&blob,L.bytes)); CHECK(hipMalloc(&xq,(H/32)*36)); CHECK(hipMalloc(&scratch,k::native_expert_scratch_bytes(1,F))); CHECK(hipMalloc(&dx,H*4)); CHECK(hipMalloc(&out,H*4)); CHECK(hipMalloc(&ptr,8)); CHECK(hipMalloc(&start,8)); CHECK(hipMalloc(&ng,4)); CHECK(hipMalloc(&dst,4)); CHECK(hipMalloc(&tok,4));
  CHECK(hipMemcpy(blob,f.tensor_data(*g),L.up_off,hipMemcpyHostToDevice)); CHECK(hipMemcpy((char*)blob+L.up_off,f.tensor_data(*u),L.up_off,hipMemcpyHostToDevice)); CHECK(hipMemcpy((char*)blob+L.down_off,f.tensor_data(*d),L.bytes-L.down_off,hipMemcpyHostToDevice)); CHECK(hipMemcpy(dx,x.data(),H*4,hipMemcpyHostToDevice));
  unsigned long long p=(unsigned long long)blob; int32_t starts[2]={0,1},one=1,zero=0;
  CHECK(hipMemcpy(ptr,&p,8,hipMemcpyHostToDevice)); CHECK(hipMemcpy(start,starts,8,hipMemcpyHostToDevice)); CHECK(hipMemcpy(ng,&one,4,hipMemcpyHostToDevice)); CHECK(hipMemcpy(dst,&zero,4,hipMemcpyHostToDevice)); CHECK(hipMemcpy(tok,&zero,4,hipMemcpyHostToDevice));
  k::quantize_q8_1_rows(dx,1,H,xq,s); k::native_expert_grouped(L,ptr,start,ng,dst,tok,1,1,xq,scratch,out,s); CHECK(hipStreamSynchronize(s)); CHECK(hipMemcpy(y.data(),out,H*4,hipMemcpyDeviceToHost));
  FILE* file=std::fopen(argv[3],"wb"); if(!file)return 5; size_t n=std::fwrite(y.data(),4,y.size(),file); std::fclose(file); if(n!=y.size())return 6;
  std::printf("computed first real expert: %zu bytes\n",L.bytes);
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
