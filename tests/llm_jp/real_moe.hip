#include "../../src/llm_jp/moe.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess){std::fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));return 2;} } while(0)
int main(int argc,char** argv) {
 if(argc!=4){std::fprintf(stderr,"usage: real_moe MODEL.gguf LAYER OUTPUT.bin\n");return 2;}
 char* end=nullptr; long layer=std::strtol(argv[2],&end,10); if(!end || *end || layer<0 || layer>=32)return 2;
 try {
  strata::GgufFile file(argv[1]); strata::llm_jp::ExpertStore store(file); strata::llm_jp::Moe moe(file,store);
  std::vector<float> x(2560),y(2560); for(int i=0;i<2560;i++)x[i]=std::sin(float(i)*0.013f)*0.1f;
  float *dx,*dy; hipStream_t stream; CHECK(hipStreamCreate(&stream)); CHECK(hipMalloc(&dx,x.size()*4)); CHECK(hipMalloc(&dy,y.size()*4));
  CHECK(hipMemcpy(dx,x.data(),x.size()*4,hipMemcpyHostToDevice)); moe.run(layer,dx,dy,stream); CHECK(hipMemcpy(y.data(),dy,y.size()*4,hipMemcpyDeviceToHost));
  FILE* out=std::fopen(argv[3],"wb"); if(!out)return 3;
  bool ok=std::fwrite(moe.ids(),4,8,out)==8 && std::fwrite(moe.weights(),4,8,out)==8 && std::fwrite(y.data(),4,y.size(),out)==y.size();
  std::fclose(out); CHECK(hipFree(dx)); CHECK(hipFree(dy)); CHECK(hipStreamDestroy(stream)); if(!ok)return 4;
  std::printf("layer %ld routed to",layer); for(int i=0;i<8;i++)std::printf(" %d",moe.ids()[i]); std::printf("\n");
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
