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
                float dL_dw = 0.0f;
#pragma unroll
                for (int c = 0; c < OSS_F_CHUNK; ++c) {
                    if (c < F_chunk) {
                        const float go = grad_out[(F_offset + c) * H * W + pix_id];
                        dL_dw += go * feat[feat_base + c];
                        atomicAdd(&d_feat[feat_base + c], weight * go);
                    }
                }

                const float dL_dq = -0.5f * weight * dL_dw;
                const float dL_ddx = dL_dq * (2.0f * k.x * dx + 2.0f * k.y * dy);
                const float dL_ddy = dL_dq * (2.0f * k.y * dx + 2.0f * k.z * dy);
                atomicAdd(&d_xy[gid].x, -dL_ddx);
                atomicAdd(&d_xy[gid].y, -dL_ddy);
                atomicAdd(&d_conic[gid].x, dL_dq * dx * dx);
                atomicAdd(&d_conic[gid].y, dL_dq * 2.0f * dx * dy);
                atomicAdd(&d_conic[gid].z, dL_dq * dy * dy);
            }
        }
        __syncthreads();
    }
}

__global__ void conic_to_scale_rot_grad(
    int N,
    const float2* __restrict__ scale,
    const float*  __restrict__ rot,
    const float3* __restrict__ d_conic,
    float2*       __restrict__ d_scale,
    float*        __restrict__ d_rot
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    const float2 sxy = scale[i];
    const float theta = rot[i];
    const float3 dc = d_conic[i];

    const float sx = fmaxf(sxy.x, kMinScale);
    const float sy = fmaxf(sxy.y, kMinScale);
    const float inv_sx2 = 1.0f / (sx * sx);
    const float inv_sy2 = 1.0f / (sy * sy);
    const float diff = inv_sx2 - inv_sy2;

    const float c = cosf(theta);
    const float s = sinf(theta);
    const float cc = c * c;
    const float ss = s * s;
    const float cs = c * s;

    const float d_inv_sx2 = cc * dc.x + cs * dc.y + ss * dc.z;
    const float d_inv_sy2 = ss * dc.x - cs * dc.y + cc * dc.z;
    const float d_sx = sxy.x > kMinScale ? d_inv_sx2 * (-2.0f / (sxy.x * sxy.x * sxy.x)) : 0.0f;
    const float d_sy = sxy.y > kMinScale ? d_inv_sy2 * (-2.0f / (sxy.y * sxy.y * sxy.y)) : 0.0f;

    d_scale[i] = make_float2(d_sx, d_sy);
    d_rot[i] =
        dc.x * (-2.0f * cs * diff) +
        dc.y * (diff * (cc - ss)) +
        dc.z * (2.0f * cs * diff);
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

std::tuple<torch::Tensor, torch::Tensor> conic_to_scale_rot_grad_cuda(
    torch::Tensor scale,
    torch::Tensor rot,
    torch::Tensor d_conic
) {
    check_float_cuda_contiguous(scale, "scale");
    check_float_cuda_contiguous(rot, "rot");
    check_float_cuda_contiguous(d_conic, "d_conic");

    TORCH_CHECK(scale.dim() == 2 && scale.size(1) == 2, "scale must be (N,2)");
    TORCH_CHECK(rot.dim() == 1, "rot must be (N,)");
    TORCH_CHECK(d_conic.dim() == 2 && d_conic.size(1) == 3, "d_conic must be (N,3)");
    TORCH_CHECK(scale.size(0) == rot.size(0), "scale/rot N mismatch");
    TORCH_CHECK(scale.size(0) == d_conic.size(0), "scale/d_conic N mismatch");

    const int64_t N64 = scale.size(0);
    TORCH_CHECK(N64 <= std::numeric_limits<int>::max(), "N exceeds int32 range");
    const int N = static_cast<int>(N64);

    auto f32_options = scale.options().dtype(torch::kFloat32);
    torch::Tensor d_scale = torch::zeros({N, 2}, f32_options);
    torch::Tensor d_rot = torch::zeros({N}, f32_options);

    if (N == 0) {
        return std::make_tuple(d_scale, d_rot);
    }

    const dim3 block(OSS_PREPROCESS_BLOCK);
    const dim3 grid((N + OSS_PREPROCESS_BLOCK - 1) / OSS_PREPROCESS_BLOCK);
    conic_to_scale_rot_grad<<<grid, block>>>(
        N,
        reinterpret_cast<const float2*>(scale.data_ptr<float>()),
        rot.data_ptr<float>(),
        reinterpret_cast<const float3*>(d_conic.data_ptr<float>()),
        reinterpret_cast<float2*>(d_scale.data_ptr<float>()),
        d_rot.data_ptr<float>()
    );
    C10_CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(d_scale, d_rot);
}
#endif
