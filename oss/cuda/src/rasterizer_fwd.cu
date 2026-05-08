#include "common.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>

#ifndef OSS_CUDA_KERNELS_ONLY
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <limits>
#include <tuple>
#endif

namespace {

constexpr float kMinScale = 1.0e-6f;

__device__ __forceinline__ float clamp_min_preserve_nan(float v, float lo) {
    return v < lo ? lo : v;
}

__device__ __forceinline__ int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

#ifndef OSS_CUDA_KERNELS_ONLY
void check_preprocess_input(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be fp32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}
#endif

}  // namespace

__global__ void preprocess_gaussians(
    int N, int H, int W, int tile_size, int num_tiles_x, int num_tiles_y,
    const float2* __restrict__ xy,
    const float2* __restrict__ scale,
    const float*  __restrict__ rot,
    float3*       __restrict__ conic_out,
    int4*         __restrict__ aabb_out,
    int*          __restrict__ pair_count_out
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    const float2 center = xy[idx];
    const float2 raw_scale = scale[idx];
    const float sx = clamp_min_preserve_nan(raw_scale.x, kMinScale);
    const float sy = clamp_min_preserve_nan(raw_scale.y, kMinScale);
    const float angle = rot[idx];

    const float c = cosf(angle);
    const float s = sinf(angle);

    const float inv_sx2 = 1.0f / (sx * sx);
    const float inv_sy2 = 1.0f / (sy * sy);
    const float a = c * c * inv_sx2 + s * s * inv_sy2;
    const float b = c * s * (inv_sx2 - inv_sy2);
    const float d = s * s * inv_sx2 + c * c * inv_sy2;
    conic_out[idx] = make_float3(a, b, d);

    if (!isfinite(a) || !isfinite(b) || !isfinite(d)) {
        aabb_out[idx] = make_int4(0, 0, 0, 0);
        pair_count_out[idx] = 0;
        return;
    }

    const float radius = 3.0f * fmaxf(raw_scale.x, raw_scale.y);
    const int tx_min = clamp_int(static_cast<int>(floorf((center.x - radius) / tile_size)), 0, num_tiles_x);
    const int tx_max = clamp_int(static_cast<int>(ceilf((center.x + radius) / tile_size)), 0, num_tiles_x);
    const int ty_min = clamp_int(static_cast<int>(floorf((center.y - radius) / tile_size)), 0, num_tiles_y);
    const int ty_max = clamp_int(static_cast<int>(ceilf((center.y + radius) / tile_size)), 0, num_tiles_y);

    const int count_x = tx_max > tx_min ? tx_max - tx_min : 0;
    const int count_y = ty_max > ty_min ? ty_max - ty_min : 0;
    aabb_out[idx] = make_int4(tx_min, ty_min, tx_max, ty_max);
    pair_count_out[idx] = count_x * count_y;
}

#ifndef OSS_CUDA_KERNELS_ONLY
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> preprocess_gaussians_cuda(
    torch::Tensor xy,
    torch::Tensor scale,
    torch::Tensor rot,
    int64_t H,
    int64_t W,
    int64_t tile_size
) {
    check_preprocess_input(xy, "xy");
    check_preprocess_input(scale, "scale");
    check_preprocess_input(rot, "rot");
    TORCH_CHECK(xy.dim() == 2 && xy.size(1) == 2, "xy must be (N,2)");
    TORCH_CHECK(scale.dim() == 2 && scale.size(1) == 2, "scale must be (N,2)");
    TORCH_CHECK(rot.dim() == 1, "rot must be (N,)");
    TORCH_CHECK(xy.size(0) == scale.size(0), "xy/scale N mismatch");
    TORCH_CHECK(xy.size(0) == rot.size(0), "xy/rot N mismatch");
    TORCH_CHECK(H >= 0 && W >= 0, "H and W must be non-negative");
    TORCH_CHECK(tile_size == OSS_TILE_SIZE, "tile_size must be 16");

    const int64_t N64 = xy.size(0);
    TORCH_CHECK(N64 <= std::numeric_limits<int>::max(), "N exceeds int32 range");
    TORCH_CHECK(H <= std::numeric_limits<int>::max(), "H exceeds int32 range");
    TORCH_CHECK(W <= std::numeric_limits<int>::max(), "W exceeds int32 range");
    const int N = static_cast<int>(N64);
    const int h = static_cast<int>(H);
    const int w = static_cast<int>(W);
    const int ts = static_cast<int>(tile_size);
    const int num_tiles_x = (w + ts - 1) / ts;
    const int num_tiles_y = (h + ts - 1) / ts;

    auto f32_options = xy.options().dtype(torch::kFloat32);
    auto i32_options = xy.options().dtype(torch::kInt32);
    torch::Tensor conic = torch::empty({N, 3}, f32_options);
    torch::Tensor aabb = torch::empty({N, 4}, i32_options);
    torch::Tensor pair_count = torch::empty({N}, i32_options);

    if (N == 0) {
        return std::make_tuple(conic, aabb, pair_count);
    }

    const dim3 block(OSS_PREPROCESS_BLOCK);
    const dim3 grid((N + OSS_PREPROCESS_BLOCK - 1) / OSS_PREPROCESS_BLOCK);
    preprocess_gaussians<<<grid, block>>>(
        N, h, w, ts, num_tiles_x, num_tiles_y,
        reinterpret_cast<const float2*>(xy.data_ptr<float>()),
        reinterpret_cast<const float2*>(scale.data_ptr<float>()),
        rot.data_ptr<float>(),
        reinterpret_cast<float3*>(conic.data_ptr<float>()),
        reinterpret_cast<int4*>(aabb.data_ptr<int>()),
        pair_count.data_ptr<int>()
    );
    C10_CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(conic, aabb, pair_count);
}
#endif
