#include "common.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>

#ifndef OSS_CUDA_KERNELS_ONLY
#include <c10/cuda/CUDAException.h>
#include <limits>
#include <torch/extension.h>
#include <tuple>
#endif

namespace {

#ifndef OSS_CUDA_KERNELS_ONLY
void check_float_cuda_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be fp32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_int_cuda_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}
#endif

}  // namespace

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
) {
    (void)feat;
    (void)d_xy;
    (void)d_conic;

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
            const int pix_id = py * W + px;
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
                        const float go = grad_out[(F_offset + c) * H * W + pix_id];
                        atomicAdd(&d_feat[feat_base + c], weight * go);
                    }
                }

                // TODO(Phase 3b): accumulate d_xy using dL/dq for this pixel/Gaussian.
                // TODO(Phase 3b): accumulate d_conic using dL/dq for this pixel/Gaussian.
            }
        }
        __syncthreads();
    }
}

#ifndef OSS_CUDA_KERNELS_ONLY
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rasterize_backward_cuda(
    torch::Tensor xy,
    torch::Tensor scale,
    torch::Tensor rot,
    torch::Tensor feat,
    torch::Tensor conic,
    torch::Tensor gaussian_idx_sorted,
    torch::Tensor tile_offsets,
    torch::Tensor grad_out,
    int64_t H,
    int64_t W,
    int64_t tile_size
) {
    check_float_cuda_contiguous(xy, "xy");
    check_float_cuda_contiguous(scale, "scale");
    check_float_cuda_contiguous(rot, "rot");
    check_float_cuda_contiguous(feat, "feat");
    check_float_cuda_contiguous(conic, "conic");
    check_float_cuda_contiguous(grad_out, "grad_out");
    check_int_cuda_contiguous(gaussian_idx_sorted, "gaussian_idx_sorted");
    check_int_cuda_contiguous(tile_offsets, "tile_offsets");

    TORCH_CHECK(xy.dim() == 2 && xy.size(1) == 2, "xy must be (N,2)");
    TORCH_CHECK(scale.dim() == 2 && scale.size(1) == 2, "scale must be (N,2)");
    TORCH_CHECK(rot.dim() == 1, "rot must be (N,)");
    TORCH_CHECK(feat.dim() == 2, "feat must be (N,F)");
    TORCH_CHECK(conic.dim() == 2 && conic.size(1) == 3, "conic must be (N,3)");
    TORCH_CHECK(gaussian_idx_sorted.dim() == 1, "gaussian_idx_sorted must be 1D");
    TORCH_CHECK(tile_offsets.dim() == 1, "tile_offsets must be 1D");
    TORCH_CHECK(grad_out.dim() == 3, "grad_out must be (F,H,W)");
    TORCH_CHECK(xy.size(0) == scale.size(0), "xy/scale N mismatch");
    TORCH_CHECK(xy.size(0) == rot.size(0), "xy/rot N mismatch");
    TORCH_CHECK(xy.size(0) == feat.size(0), "xy/feat N mismatch");
    TORCH_CHECK(xy.size(0) == conic.size(0), "xy/conic N mismatch");
    TORCH_CHECK(H >= 0 && W >= 0, "H and W must be non-negative");
    TORCH_CHECK(tile_size == OSS_TILE_SIZE, "tile_size must be 16");

    const int64_t N64 = xy.size(0);
    const int64_t F64 = feat.size(1);
    TORCH_CHECK(N64 <= std::numeric_limits<int>::max(), "N exceeds int32 range");
    TORCH_CHECK(F64 <= 64, "F must be <= 64");
    TORCH_CHECK(F64 >= 0, "F must be non-negative");
    TORCH_CHECK(H <= std::numeric_limits<int>::max(), "H exceeds int32 range");
    TORCH_CHECK(W <= std::numeric_limits<int>::max(), "W exceeds int32 range");
    TORCH_CHECK(grad_out.size(0) == F64, "grad_out F mismatch");
    TORCH_CHECK(grad_out.size(1) == H, "grad_out H mismatch");
    TORCH_CHECK(grad_out.size(2) == W, "grad_out W mismatch");

    const int N = static_cast<int>(N64);
    const int F_total = static_cast<int>(F64);
    const int h = static_cast<int>(H);
    const int w = static_cast<int>(W);
    const int num_tiles_x = (w + OSS_TILE_SIZE - 1) / OSS_TILE_SIZE;
    const int num_tiles_y = (h + OSS_TILE_SIZE - 1) / OSS_TILE_SIZE;
    const int64_t num_tiles = static_cast<int64_t>(num_tiles_x) * num_tiles_y;
    TORCH_CHECK(num_tiles <= std::numeric_limits<int>::max(), "number of tiles exceeds int32 range");
    TORCH_CHECK(tile_offsets.size(0) == num_tiles + 1, "tile_offsets length mismatch");

    auto f32_options = xy.options().dtype(torch::kFloat32);
    torch::Tensor d_xy = torch::zeros({N, 2}, f32_options);
    torch::Tensor d_conic = torch::zeros({N, 3}, f32_options);
    torch::Tensor d_feat = torch::zeros({N, F_total}, f32_options);

    if (N == 0 || F_total == 0 || h == 0 || w == 0 || num_tiles == 0) {
        return std::make_tuple(d_xy, d_conic, d_feat);
    }

    const dim3 grid(num_tiles_x, num_tiles_y);
    const dim3 block(OSS_TILE_SIZE, OSS_TILE_SIZE);
    const int F_chunks = (F_total + OSS_F_CHUNK - 1) / OSS_F_CHUNK;
    for (int chunk = 0; chunk < F_chunks; ++chunk) {
        const int F_offset = chunk * OSS_F_CHUNK;
        const int F_chunk = std::min(OSS_F_CHUNK, F_total - F_offset);
        rasterize_backward<<<grid, block>>>(
            h, w, num_tiles_x, num_tiles_y,
            F_chunk, F_offset, F_total,
            gaussian_idx_sorted.data_ptr<int>(),
            tile_offsets.data_ptr<int>(),
            reinterpret_cast<const float2*>(xy.data_ptr<float>()),
            reinterpret_cast<const float3*>(conic.data_ptr<float>()),
            feat.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            reinterpret_cast<float2*>(d_xy.data_ptr<float>()),
            reinterpret_cast<float3*>(d_conic.data_ptr<float>()),
            d_feat.data_ptr<float>()
        );
        C10_CUDA_CHECK(cudaGetLastError());
    }

    return std::make_tuple(d_xy, d_conic, d_feat);
}
#endif
