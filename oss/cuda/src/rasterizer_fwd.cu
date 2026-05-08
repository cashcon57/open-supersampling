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

__global__ void build_tile_pairs(
    int N, int num_tiles_x,
    const int4* __restrict__ aabb,
    const int*  __restrict__ cum_pair_count,
    int64_t*    __restrict__ keys_out,
    int*        __restrict__ gid_out
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    const int4 bounds = aabb[idx];
    int out_idx = cum_pair_count[idx];
    for (int ty = bounds.y; ty < bounds.w; ++ty) {
        for (int tx = bounds.x; tx < bounds.z; ++tx) {
            const int64_t tile_id = static_cast<int64_t>(ty) * num_tiles_x + tx;
            keys_out[out_idx] = (tile_id << 32) | static_cast<uint32_t>(idx);
            gid_out[out_idx] = idx;
            ++out_idx;
        }
    }
}

__global__ void build_full_tile_pairs(
    int N,
    int num_tiles,
    int* __restrict__ gid_out,
    int* __restrict__ tile_offsets_out
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pairs = N * num_tiles;
    if (idx < total_pairs) {
        gid_out[idx] = idx % N;
    }
    if (idx <= num_tiles) {
        tile_offsets_out[idx] = idx * N;
    }
}

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
) {
    const int tile_x = static_cast<int>(blockIdx.x);
    const int tile_y = static_cast<int>(blockIdx.y);
    if (tile_x >= num_tiles_x || tile_y >= num_tiles_y) {
        return;
    }

    const int lane = static_cast<int>(threadIdx.y) * OSS_TILE_SIZE + static_cast<int>(threadIdx.x);
    const int px = tile_x * OSS_TILE_SIZE + static_cast<int>(threadIdx.x);
    const int py = tile_y * OSS_TILE_SIZE + static_cast<int>(threadIdx.y);
    const bool inside = px < W && py < H;
    const int tile_id = tile_y * num_tiles_x + tile_x;

    __shared__ int id_batch[OSS_RASTER_BLOCK];
    __shared__ float2 xy_batch[OSS_RASTER_BLOCK];
    __shared__ float3 conic_batch[OSS_RASTER_BLOCK];

    float pix_out[OSS_F_CHUNK];
#pragma unroll
    for (int c = 0; c < OSS_F_CHUNK; ++c) {
        pix_out[c] = 0.0f;
    }

    const int start = tile_offsets[tile_id];
    const int end = tile_offsets[tile_id + 1];
    for (int batch_start = start; batch_start < end; batch_start += OSS_RASTER_BLOCK) {
        const int load_idx = batch_start + lane;
        if (load_idx < end) {
            const int gid = gaussian_idx_sorted[load_idx];
            id_batch[lane] = gid;
            xy_batch[lane] = xy[gid];
            conic_batch[lane] = conic[gid];
        }
        __syncthreads();

        const int remaining = end - batch_start;
        const int batch_count = remaining < OSS_RASTER_BLOCK ? remaining : OSS_RASTER_BLOCK;
        if (inside) {
            for (int t = 0; t < batch_count; ++t) {
                const int gid = id_batch[t];
                const float2 center = xy_batch[t];
                const float3 k = conic_batch[t];
                const float dx = static_cast<float>(px) - center.x;
                const float dy = static_cast<float>(py) - center.y;
                const float q = k.x * dx * dx + 2.0f * k.y * dx * dy + k.z * dy * dy;
                if (!isfinite(q) || q < 0.0f) {
                    continue;
                }
                const float weight = expf(-0.5f * q);
                const int feat_base = gid * F_total + F_offset;
#pragma unroll
                for (int c = 0; c < OSS_F_CHUNK; ++c) {
                    if (c < F_chunk) {
                        pix_out[c] += weight * feat[feat_base + c];
                    }
                }
            }
        }
        __syncthreads();
    }

    if (inside) {
        const int pix_id = py * W + px;
#pragma unroll
        for (int c = 0; c < OSS_F_CHUNK; ++c) {
            if (c < F_chunk) {
                out[(F_offset + c) * H * W + pix_id] = pix_out[c];
            }
        }
    }
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> pair_construction_cuda(
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
    const int64_t num_tiles = static_cast<int64_t>(num_tiles_x) * num_tiles_y;
    TORCH_CHECK(num_tiles <= std::numeric_limits<int>::max(), "number of tiles exceeds int32 range");

    auto f32_options = xy.options().dtype(torch::kFloat32);
    auto i32_options = xy.options().dtype(torch::kInt32);
    auto i64_options = xy.options().dtype(torch::kInt64);
    torch::Tensor conic = torch::empty({N, 3}, f32_options);
    torch::Tensor aabb = torch::empty({N, 4}, i32_options);
    torch::Tensor pair_count = torch::empty({N}, i32_options);

    if (N > 0) {
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
    }

    const int64_t total_pairs = N == 0 ? 0 : pair_count.to(torch::kInt64).sum().item<int64_t>();
    TORCH_CHECK(total_pairs >= 0, "total pair count must be non-negative");
    TORCH_CHECK(total_pairs <= std::numeric_limits<int>::max(), "total pair count exceeds int32 range");

    torch::Tensor cum_pair_count = torch::empty({N + 1}, i32_options);
    cum_pair_count.narrow(0, 0, 1).zero_();
    if (N > 0) {
        torch::Tensor cumsum = torch::cumsum(pair_count, 0, torch::kInt32);
        cum_pair_count.narrow(0, 1, N).copy_(cumsum);
    }

    torch::Tensor keys = torch::empty({total_pairs}, i64_options);
    torch::Tensor gid = torch::empty({total_pairs}, i32_options);
    if (N > 0 && total_pairs > 0) {
        const dim3 block(OSS_PREPROCESS_BLOCK);
        const dim3 grid((N + OSS_PREPROCESS_BLOCK - 1) / OSS_PREPROCESS_BLOCK);
        build_tile_pairs<<<grid, block>>>(
            N, num_tiles_x,
            reinterpret_cast<const int4*>(aabb.data_ptr<int>()),
            cum_pair_count.data_ptr<int>(),
            keys.data_ptr<int64_t>(),
            gid.data_ptr<int>()
        );
        C10_CUDA_CHECK(cudaGetLastError());
    }

    torch::Tensor gid_sorted = gid;
    torch::Tensor keys_sorted = keys;
    if (total_pairs > 0) {
        auto sort_result = keys.sort(0, false);
        keys_sorted = std::get<0>(sort_result);
        const torch::Tensor sort_indices = std::get<1>(sort_result);
        gid_sorted = gid.index_select(0, sort_indices);
    }

    torch::Tensor tile_boundaries = torch::arange(num_tiles + 1, i64_options).mul_(1LL << 32);
    torch::Tensor tile_offsets = torch::searchsorted(
        keys_sorted, tile_boundaries, /*out_int32=*/true
    );

    return std::make_tuple(gid_sorted, tile_offsets, conic, total_pairs);
}

torch::Tensor rasterize_forward_cuda(
    torch::Tensor xy,
    torch::Tensor scale,
    torch::Tensor rot,
    torch::Tensor feat,
    int64_t H,
    int64_t W,
    int64_t tile_size,
    bool topk_norm
) {
    (void)topk_norm;
    check_preprocess_input(xy, "xy");
    check_preprocess_input(scale, "scale");
    check_preprocess_input(rot, "rot");
    check_preprocess_input(feat, "feat");
    TORCH_CHECK(xy.dim() == 2 && xy.size(1) == 2, "xy must be (N,2)");
    TORCH_CHECK(scale.dim() == 2 && scale.size(1) == 2, "scale must be (N,2)");
    TORCH_CHECK(rot.dim() == 1, "rot must be (N,)");
    TORCH_CHECK(feat.dim() == 2, "feat must be (N,F)");
    TORCH_CHECK(xy.size(0) == scale.size(0), "xy/scale N mismatch");
    TORCH_CHECK(xy.size(0) == rot.size(0), "xy/rot N mismatch");
    TORCH_CHECK(xy.size(0) == feat.size(0), "xy/feat N mismatch");
    TORCH_CHECK(H >= 0 && W >= 0, "H and W must be non-negative");
    TORCH_CHECK(tile_size == OSS_TILE_SIZE, "tile_size must be 16");

    const int64_t N64 = xy.size(0);
    const int64_t F64 = feat.size(1);
    TORCH_CHECK(N64 <= std::numeric_limits<int>::max(), "N exceeds int32 range");
    TORCH_CHECK(F64 <= 64, "F must be <= 64");
    TORCH_CHECK(F64 >= 0, "F must be non-negative");
    TORCH_CHECK(H <= std::numeric_limits<int>::max(), "H exceeds int32 range");
    TORCH_CHECK(W <= std::numeric_limits<int>::max(), "W exceeds int32 range");

    const int N = static_cast<int>(N64);
    const int F_total = static_cast<int>(F64);
    const int h = static_cast<int>(H);
    const int w = static_cast<int>(W);
    const int num_tiles_x = (w + OSS_TILE_SIZE - 1) / OSS_TILE_SIZE;
    const int num_tiles_y = (h + OSS_TILE_SIZE - 1) / OSS_TILE_SIZE;
    const int64_t num_tiles = static_cast<int64_t>(num_tiles_x) * num_tiles_y;
    TORCH_CHECK(num_tiles <= std::numeric_limits<int>::max(), "number of tiles exceeds int32 range");
    TORCH_CHECK(
        N64 == 0 || num_tiles <= std::numeric_limits<int>::max() / N64,
        "full-frame raster pair count exceeds int32 range"
    );

    auto out = torch::zeros({F_total, h, w}, feat.options().dtype(torch::kFloat32));
    if (N == 0 || F_total == 0 || h == 0 || w == 0) {
        return out;
    }

    auto f32_options = xy.options().dtype(torch::kFloat32);
    auto i32_options = xy.options().dtype(torch::kInt32);
    torch::Tensor conic = torch::empty({N, 3}, f32_options);
    torch::Tensor aabb = torch::empty({N, 4}, i32_options);
    torch::Tensor pair_count = torch::empty({N}, i32_options);

    const dim3 preprocess_block(OSS_PREPROCESS_BLOCK);
    const dim3 preprocess_grid((N + OSS_PREPROCESS_BLOCK - 1) / OSS_PREPROCESS_BLOCK);
    preprocess_gaussians<<<preprocess_grid, preprocess_block>>>(
        N, h, w, OSS_TILE_SIZE, num_tiles_x, num_tiles_y,
        reinterpret_cast<const float2*>(xy.data_ptr<float>()),
        reinterpret_cast<const float2*>(scale.data_ptr<float>()),
        rot.data_ptr<float>(),
        reinterpret_cast<float3*>(conic.data_ptr<float>()),
        reinterpret_cast<int4*>(aabb.data_ptr<int>()),
        pair_count.data_ptr<int>()
    );
    C10_CUDA_CHECK(cudaGetLastError());

    // The Phase 2c reference sums the full Gaussian tail, while the Phase 2b
    // pair-construction helper keeps the 3-sigma tile AABB contract. Use a
    // full-frame tile map here so forward equivalence remains exact.
    const int64_t total_pairs = static_cast<int64_t>(N) * num_tiles;
    torch::Tensor gid_sorted = torch::empty({total_pairs}, i32_options);
    torch::Tensor tile_offsets = torch::empty({num_tiles + 1}, i32_options);
    const int full_pairs_threads = static_cast<int>(std::max(total_pairs, num_tiles + 1));
    const dim3 full_pairs_block(OSS_PREPROCESS_BLOCK);
    const dim3 full_pairs_grid((full_pairs_threads + OSS_PREPROCESS_BLOCK - 1) / OSS_PREPROCESS_BLOCK);
    build_full_tile_pairs<<<full_pairs_grid, full_pairs_block>>>(
        N,
        static_cast<int>(num_tiles),
        gid_sorted.data_ptr<int>(),
        tile_offsets.data_ptr<int>()
    );
    C10_CUDA_CHECK(cudaGetLastError());

    const dim3 grid(num_tiles_x, num_tiles_y);
    const dim3 block(OSS_TILE_SIZE, OSS_TILE_SIZE);
    const int F_chunks = (F_total + OSS_F_CHUNK - 1) / OSS_F_CHUNK;
    for (int chunk = 0; chunk < F_chunks; ++chunk) {
        const int F_offset = chunk * OSS_F_CHUNK;
        const int F_chunk = std::min(OSS_F_CHUNK, F_total - F_offset);
        rasterize_sum<<<grid, block>>>(
            h, w, num_tiles_x, num_tiles_y,
            F_chunk, F_offset, F_total,
            gid_sorted.data_ptr<int>(),
            tile_offsets.data_ptr<int>(),
            reinterpret_cast<const float2*>(xy.data_ptr<float>()),
            reinterpret_cast<const float3*>(conic.data_ptr<float>()),
            feat.data_ptr<float>(),
            out.data_ptr<float>()
        );
        C10_CUDA_CHECK(cudaGetLastError());
    }

    return out;
}
#endif
