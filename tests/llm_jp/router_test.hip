#include <hip/hip_runtime.h>
#include "strata/kernels/native_router.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess) { std::fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e)); return 2; } } while(0)
int test(int n, int k) {
 constexpr int rows=19;
 std::vector<float> logits(rows*n), weights(rows*k); std::vector<int32_t> ids(rows*k);
 for(int r=0;r<rows;r++) for(int i=0;i<n;i++) logits[r*n+i] = r==0 ? 1.f : float((i*71+r*13)%127)*.1f + float(i)*.00001f;
 float *dl,*dw; int32_t* di; hipStream_t stream;
 CHECK(hipStreamCreate(&stream)); CHECK(hipMalloc(&dl,logits.size()*4)); CHECK(hipMalloc(&dw,weights.size()*4)); CHECK(hipMalloc(&di,ids.size()*4));
 CHECK(hipMemcpy(dl,logits.data(),logits.size()*4,hipMemcpyHostToDevice));
 if(!strata::kernels::native_router_topk_supported(n,k)) return 3;
 strata::kernels::native_router_topk(dl,di,dw,n,k,rows,stream); CHECK(hipStreamSynchronize(stream));
 CHECK(hipMemcpy(ids.data(),di,ids.size()*4,hipMemcpyDeviceToHost)); CHECK(hipMemcpy(weights.data(),dw,weights.size()*4,hipMemcpyDeviceToHost));
 float maxerr=0;
 for(int r=0;r<rows;r++) { std::vector<int> order(n);std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](int a,int b){return logits[r*n+a]>logits[r*n+b];}); double sum=0; for(int j=0;j<k;j++)sum+=std::exp(double(logits[r*n+order[j]]-logits[r*n+order[0]]));
  for(int j=0;j<k;j++){if(ids[r*k+j]!=order[j]) { std::fprintf(stderr,"row %d rank %d: got %d expected %d\n",r,j,ids[r*k+j],order[j]);return 4; } double expected=std::exp(double(logits[r*n+order[j]]-logits[r*n+order[0]]))/sum; maxerr=std::max(maxerr,float(std::abs(weights[r*k+j]-expected)));}
 }
 CHECK(hipFree(dl));CHECK(hipFree(dw));CHECK(hipFree(di));CHECK(hipStreamDestroy(stream));std::printf("%d-expert top-%d: %d rows match, max weight error %.9g\n",n,k,rows,maxerr);return maxerr<1e-5f?0:5;
}

int main() { for (auto nk : {std::pair<int,int>{128,8}, {256,8}, {512,10}}) { int rc=test(nk.first,nk.second); if(rc) return rc; } }
