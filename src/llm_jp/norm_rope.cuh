#pragma once
#include <hip/hip_runtime.h>
#include <cmath>
// Correctness-first weighted RMSNorm followed by NeoX-style RoPE per attention head.
// Input/output [heads, dim]; shared weight [dim]; caller checks even dim and 0 < dim <= max block size.
namespace strata::llm_jp {
__global__ void norm_rope(const float* input, const float* weight, float* output,
                          int heads, int dim, int position, float eps, float theta) {
    const int h=blockIdx.x, d=threadIdx.x;
    if(h>=heads || d>=dim) return;
    const float* row=input+h*dim;
    float sq=0.f; for(int i=0;i<dim;i++) sq+=row[i]*row[i];
    const float inv=rsqrtf(sq/dim+eps);
    const int half=dim/2, i=d%half, other=d<half?d+half:d-half;
    const float a=row[d]*inv*weight[d], b=row[other]*inv*weight[other];
    const float angle=float(position)*powf(theta,-2.f*float(i)/float(dim));
    output[h*dim+d]=d<half ? a*cosf(angle)-b*sinf(angle) : a*cosf(angle)+b*sinf(angle);
}
}
