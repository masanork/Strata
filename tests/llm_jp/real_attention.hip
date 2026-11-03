#include "../../src/llm_jp/attention.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess){std::fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));return 2;} } while(0)
int main(int argc,char** argv){
 if(argc!=4){std::fprintf(stderr,"usage: real_attention MODEL.gguf LAYER OUTPUT.bin\n");return 2;}
 char* end=nullptr;long layer=std::strtol(argv[2],&end,10);if(!end||*end||layer<0||layer>=32)return 2;
 try {
  strata::GgufFile file(argv[1]);strata::llm_jp::Attention attention(file,2);
  std::vector<float> x(2560),y(5120);float *dx,*dy;hipStream_t s;
  CHECK(hipStreamCreate(&s));CHECK(hipMalloc(&dx,2560*4));CHECK(hipMalloc(&dy,2560*4));
  for(int pos=0;pos<2;pos++){
   for(int i=0;i<2560;i++)x[i]=std::sin(float(i)*.013f+float(pos)*.31f)*.1f;
   CHECK(hipMemcpy(dx,x.data(),2560*4,hipMemcpyHostToDevice));attention.run(layer,pos,dx,dy,s);
   CHECK(hipMemcpy(y.data()+pos*2560,dy,2560*4,hipMemcpyDeviceToHost));
  }
  FILE* out=std::fopen(argv[3],"wb");if(!out)return 3;bool ok=std::fwrite(y.data(),4,y.size(),out)==y.size();std::fclose(out);
  CHECK(hipFree(dx));CHECK(hipFree(dy));CHECK(hipStreamDestroy(s));return ok?0:4;
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
