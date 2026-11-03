#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <exception>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess){std::fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));return 2;} } while(0)
int main(int argc,char** argv){
 if(argc!=4){std::fprintf(stderr,"usage: real_mmvq MODEL.gguf TENSOR OUTPUT.bin\n");return 2;}
 try { strata::GgufFile f(argv[1]);auto* t=f.find(argv[2]);constexpr int n=2560,rows=32;
  if(!t || t->type!=12 || (t->shape.size()!=2 && t->shape.size()!=3) || t->shape[0]!=n || t->shape[1]<rows){std::fprintf(stderr,"unexpected attn_q layout\n");return 3;}
  size_t bytes=strata::kernels::native_mmvq_weight_bytes(t->type,n,rows);if(t->offset>f.file_size()-f.data_start() || bytes>f.file_size()-f.data_start()-t->offset)return 4;
  std::vector<float> x(n),y(rows);for(int i=0;i<n;i++)x[i]=std::sin(float(i)*0.013f)*0.1f;
  void *dw,*dq;float *dx,*dy;hipStream_t stream;CHECK(hipStreamCreate(&stream));CHECK(hipMalloc(&dw,bytes));CHECK(hipMalloc(&dq,strata::kernels::native_q8_1_bytes(n)));CHECK(hipMalloc(&dx,n*4));CHECK(hipMalloc(&dy,rows*4));
  CHECK(hipMemcpy(dw,f.tensor_data(*t),bytes,hipMemcpyHostToDevice));CHECK(hipMemcpy(dx,x.data(),n*4,hipMemcpyHostToDevice));
  strata::kernels::native_quantize_q8_1(dx,dq,n,1,stream);strata::kernels::native_mmvq(t->type,dw,dq,dy,n,rows,1,stream);CHECK(hipStreamSynchronize(stream));CHECK(hipMemcpy(y.data(),dy,rows*4,hipMemcpyDeviceToHost));
  FILE* out=std::fopen(argv[3],"wb");if(!out)return 5;size_t wrote=std::fwrite(y.data(),4,y.size(),out);std::fclose(out);if(wrote!=y.size())return 6;
  CHECK(hipFree(dw));CHECK(hipFree(dq));CHECK(hipFree(dx));CHECK(hipFree(dy));CHECK(hipStreamDestroy(stream));std::printf("GPU computed %d rows of %s Q4_K (%zu bytes)\n",rows,t->name.c_str(),bytes);
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
