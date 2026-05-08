#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifndef OSS_CUDA_KERNELS_ONLY
#include <torch/extension.h>
#endif

#define OSS_TILE_SIZE 16
#define OSS_PREPROCESS_BLOCK 256

__global__ void preprocess_gaussians(
    int N, int H, int W, int tile_size, int num_tiles_x, int num_tiles_y,
    const float2* __restrict__ xy,
    const float2* __restrict__ scale,
    const float*  __restrict__ rot,
    float3*       __restrict__ conic_out,
    int4*         __restrict__ aabb_out,
    int*          __restrict__ pair_count_out
);
