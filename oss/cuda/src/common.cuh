#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifndef OSS_CUDA_KERNELS_ONLY
#include <torch/extension.h>
#endif

#define OSS_TILE_SIZE 16
#define OSS_PREPROCESS_BLOCK 256
#define OSS_F_CHUNK 16
#define OSS_RASTER_BLOCK 256

constexpr float kMinScale = 1.0e-6f;

__global__ void preprocess_gaussians(
    int N, int H, int W, int tile_size, int num_tiles_x, int num_tiles_y,
    const float2* __restrict__ xy,
    const float2* __restrict__ scale,
    const float*  __restrict__ rot,
    float3*       __restrict__ conic_out,
    int4*         __restrict__ aabb_out,
    int*          __restrict__ pair_count_out
);

__global__ void build_tile_pairs(
    int N, int num_tiles_x,
    const int4* __restrict__ aabb,
    const int*  __restrict__ cum_pair_count,
    int64_t*    __restrict__ keys_out,
    int*        __restrict__ gid_out
);

__global__ __launch_bounds__(OSS_RASTER_BLOCK, 4)
void rasterize_sum(
    int H, int W, int num_tiles_x, int num_tiles_y,
    int F_chunk, int F_offset, int F_total,
    const int*    __restrict__ gaussian_idx_sorted,
    const int*    __restrict__ tile_offsets,
    const float2* __restrict__ xy,
    const float3* __restrict__ conic,
    const float*  __restrict__ feat,
    float*        __restrict__ out
);

__global__ __launch_bounds__(OSS_RASTER_BLOCK, 4)
void rasterize_backward(
    int H, int W, int num_tiles_x, int num_tiles_y,
    int F_chunk, int F_offset, int F_total,
    const int*    __restrict__ gaussian_idx_sorted,
    const int*    __restrict__ tile_offsets,
    const float2* __restrict__ xy,
    const float3* __restrict__ conic,
    const float*  __restrict__ feat,
    const float*  __restrict__ grad_out,
    float2*       __restrict__ d_xy,
    float3*       __restrict__ d_conic,
    float*        __restrict__ d_feat
);

__global__ void conic_to_scale_rot_grad(
    int N,
    const float2* __restrict__ scale,
    const float*  __restrict__ rot,
    const float3* __restrict__ d_conic,
    float2*       __restrict__ d_scale,
    float*        __restrict__ d_rot
);
