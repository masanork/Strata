#include "../../src/llm_jp/norm_rope.cuh"
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess){std::fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));return 2;} } while(0)
int main(){constexpr int heads=40,dim=128,pos=123; constexpr float eps=1e-6f,theta=500000.f;
 std::vector<float> x(heads*dim),w(dim),y(x.size()),ref(x.size());
 for(size_t i=0;i<x.size();i++) x[i]=std::sin(float(i)*.013f);for(int i=0;i<dim;i++)w[i]=0.5f+float(i)/dim;
 for(int h=0;h<heads;h++){double sq=0;for(int d=0;d<dim;d++)sq+=double(x[h*dim+d])*x[h*dim+d];double inv=1/std::sqrt(sq/dim+eps);
  for(int d=0;d<dim;d++){int half=dim/2,i=d%half,o=d<half?d+half:d-half;double angle=pos*std::pow(double(theta),-2.*i/dim);double a=x[h*dim+d]*inv*w[d],b=x[h*dim+o]*inv*w[o];ref[h*dim+d]=float(d<half?a*std::cos(angle)-b*std::sin(angle):a*std::cos(angle)+b*std::sin(angle));}}
 float *dx,*dw,*dy;CHECK(hipMalloc(&dx,x.size()*4));CHECK(hipMalloc(&dw,w.size()*4));CHECK(hipMalloc(&dy,y.size()*4));CHECK(hipMemcpy(dx,x.data(),x.size()*4,hipMemcpyHostToDevice));CHECK(hipMemcpy(dw,w.data(),w.size()*4,hipMemcpyHostToDevice));
 hipLaunchKernelGGL(strata::llm_jp::norm_rope,dim3(heads),dim3(dim),0,0,dx,dw,dy,heads,dim,pos,eps,theta);CHECK(hipGetLastError());CHECK(hipDeviceSynchronize());CHECK(hipMemcpy(y.data(),dy,y.size()*4,hipMemcpyDeviceToHost));
 float maxerr=0;for(size_t i=0;i<y.size();i++){if(!std::isfinite(y[i]))return 3;maxerr=std::max(maxerr,std::abs(y[i]-ref[i]));}CHECK(hipFree(dx));CHECK(hipFree(dw));CHECK(hipFree(dy));std::printf("Q/K RMSNorm + RoPE max absolute error: %.9g\n",maxerr);return maxerr<1e-5f?0:4;
}
