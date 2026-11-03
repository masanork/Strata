#include "../../src/llm_jp/attention.cuh"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess){std::fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));return 2;} } while(0)
int main(){
 constexpr int heads=40,kvheads=4,dim=128,maxlen=1024;std::vector<float> q(heads*dim),k(maxlen*kvheads*dim),v(k.size()),a(q.size()),b(q.size());
 for(size_t i=0;i<q.size();i++)q[i]=std::sin(float(i)*.019f);
 for(size_t i=0;i<k.size();i++){k[i]=std::cos(float(i)*.013f);v[i]=std::sin(float(i)*.007f);}
 float *dq,*dk,*dv,*da,*db;hipStream_t stream;hipEvent_t start,stop;
 CHECK(hipStreamCreate(&stream));CHECK(hipEventCreate(&start));CHECK(hipEventCreate(&stop));CHECK(hipMalloc(&dq,q.size()*4));CHECK(hipMalloc(&dk,k.size()*4));CHECK(hipMalloc(&dv,v.size()*4));CHECK(hipMalloc(&da,a.size()*4));CHECK(hipMalloc(&db,b.size()*4));
 CHECK(hipMemcpy(dq,q.data(),q.size()*4,hipMemcpyHostToDevice));CHECK(hipMemcpy(dk,k.data(),k.size()*4,hipMemcpyHostToDevice));CHECK(hipMemcpy(dv,v.data(),v.size()*4,hipMemcpyHostToDevice));
 for(int len:{17,256,1024}){
  auto run=[&](bool fast){if(fast)hipLaunchKernelGGL(strata::llm_jp::attention_decode,dim3(heads),dim3(dim),0,stream,dq,dk,dv,da,heads,kvheads,dim,len,1.f/std::sqrt(float(dim)));else hipLaunchKernelGGL(strata::llm_jp::attention_decode_reference,dim3(heads),dim3(dim),0,stream,dq,dk,dv,db,heads,kvheads,dim,len,1.f/std::sqrt(float(dim)));};
  float times[2];for(int mode=0;mode<2;mode++){run(mode);CHECK(hipStreamSynchronize(stream));CHECK(hipEventRecord(start,stream));for(int i=0;i<3;i++)run(mode);CHECK(hipEventRecord(stop,stream));CHECK(hipEventSynchronize(stop));CHECK(hipEventElapsedTime(&times[mode],start,stop));times[mode]/=3;}
  CHECK(hipMemcpy(a.data(),da,a.size()*4,hipMemcpyDeviceToHost));CHECK(hipMemcpy(b.data(),db,b.size()*4,hipMemcpyDeviceToHost));float err=0;for(size_t i=0;i<a.size();i++){if(!std::isfinite(a[i]))return 3;err=std::max(err,std::abs(a[i]-b[i]));}
  std::printf("length=%d reference_ms=%.3f optimized_ms=%.3f max_abs=%.9g\n",len,times[0],times[1],err);if(err>3e-5f)return 4;
 }
 CHECK(hipFree(dq));CHECK(hipFree(dk));CHECK(hipFree(dv));CHECK(hipFree(da));CHECK(hipFree(db));CHECK(hipEventDestroy(start));CHECK(hipEventDestroy(stop));CHECK(hipStreamDestroy(stream));
}
