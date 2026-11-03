#include "../../src/llm_jp/attention.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); std::exit(1); } } while(0)
int main() {
    constexpr int heads=40, kvheads=4, dim=128, length=17;
    std::vector<float> q(heads*dim), k(length*kvheads*dim), v(k.size()), expected(q.size()), out(q.size());
    for (size_t i=0;i<q.size();++i) q[i] = std::sin(float(i)*.019f);
    for (size_t i=0;i<k.size();++i) { k[i]=std::cos(float(i)*.013f); v[i]=std::sin(float(i)*.007f); }
    const float scale=1/std::sqrt(float(dim));
    for (int h=0;h<heads;++h) {
        int kh=h/(heads/kvheads); std::vector<double> scores(length); double maximum=-INFINITY, den=0;
        for(int t=0;t<length;++t) { double dot=0; for(int d=0;d<dim;++d) dot+=double(q[h*dim+d])*k[(t*kvheads+kh)*dim+d]; scores[t]=dot*scale; maximum=std::max(maximum,scores[t]); }
        for(double& x:scores) { x=std::exp(x-maximum); den+=x; }
        for(int d=0;d<dim;++d) { double sum=0; for(int t=0;t<length;++t) sum+=scores[t]*v[(t*kvheads+kh)*dim+d]; expected[h*dim+d]=float(sum/den); }
    }
    float *dq,*dk,*dv,*dout; HIP_CHECK(hipMalloc(&dq,q.size()*4)); HIP_CHECK(hipMalloc(&dk,k.size()*4)); HIP_CHECK(hipMalloc(&dv,v.size()*4)); HIP_CHECK(hipMalloc(&dout,out.size()*4));
    HIP_CHECK(hipMemcpy(dq,q.data(),q.size()*4,hipMemcpyHostToDevice)); HIP_CHECK(hipMemcpy(dk,k.data(),k.size()*4,hipMemcpyHostToDevice)); HIP_CHECK(hipMemcpy(dv,v.data(),v.size()*4,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(strata::llm_jp::attention_decode,dim3(heads),dim3(dim),0,0,dq,dk,dv,dout,heads,kvheads,dim,length,scale);
    HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipMemcpy(out.data(),dout,out.size()*4,hipMemcpyDeviceToHost));
    float max_error=0; for(size_t i=0;i<out.size();++i) { if(!std::isfinite(out[i])) return 2; max_error=std::max(max_error,std::abs(out[i]-expected[i])); }
    HIP_CHECK(hipFree(dq)); HIP_CHECK(hipFree(dk)); HIP_CHECK(hipFree(dv)); HIP_CHECK(hipFree(dout));
    std::printf("Qwen3-MoE GQA decode max absolute error: %.9g\n",max_error); return max_error<1e-5f ? 0 : 3;
}
