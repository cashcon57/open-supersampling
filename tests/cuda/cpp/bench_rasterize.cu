#include "common.cuh"

#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

__global__ void init_gaussians(int N, int F, int H, int W, float2* xy, float3* conic, float* feat) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) {
        return;
    }
    const int safe_w = W > 0 ? W : 1;
    const int safe_h = H > 0 ? H : 1;
    xy[gid] = make_float2(static_cast<float>((gid * 13) % safe_w), static_cast<float>((gid * 17) % safe_h));
    conic[gid] = make_float3(1.0f / 9.0f, 0.0f, 1.0f / 9.0f);
    for (int c = 0; c < F; ++c) {
        feat[gid * F + c] = static_cast<float>((gid + c) % 17) * 0.01f;
    }
}

__global__ void init_full_pairs(int N, int num_tiles, int* gids, int* offsets) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pairs = N * num_tiles;
    if (idx < total_pairs) {
        gids[idx] = idx % N;
    }
    if (idx <= num_tiles) {
        offsets[idx] = idx * N;
    }
}

void bench_rasterize(nvbench::state& state) {
    const int N = static_cast<int>(state.get_int64("N"));
    const int H = static_cast<int>(state.get_int64("H"));
    const int W = static_cast<int>(state.get_int64("W"));
    const int F = static_cast<int>(state.get_int64("F"));
    const int num_tiles_x = (W + OSS_TILE_SIZE - 1) / OSS_TILE_SIZE;
    const int num_tiles_y = (H + OSS_TILE_SIZE - 1) / OSS_TILE_SIZE;
    const int num_tiles = num_tiles_x * num_tiles_y;
    const int total_pairs = N * num_tiles;

    float2* xy = nullptr;
    float3* conic = nullptr;
    float* feat = nullptr;
    float* out = nullptr;
    int* gids = nullptr;
    int* offsets = nullptr;

    cudaMalloc(&xy, static_cast<std::size_t>(N) * sizeof(float2));
    cudaMalloc(&conic, static_cast<std::size_t>(N) * sizeof(float3));
    cudaMalloc(&feat, static_cast<std::size_t>(N) * F * sizeof(float));
    cudaMalloc(&out, static_cast<std::size_t>(F) * H * W * sizeof(float));
    cudaMalloc(&gids, static_cast<std::size_t>(total_pairs) * sizeof(int));
    cudaMalloc(&offsets, static_cast<std::size_t>(num_tiles + 1) * sizeof(int));

    constexpr int init_block = 256;
    init_gaussians<<<(N + init_block - 1) / init_block, init_block>>>(N, F, H, W, xy, conic, feat);
    init_full_pairs<<<(std::max(total_pairs, num_tiles + 1) + init_block - 1) / init_block, init_block>>>(
        N, num_tiles, gids, offsets
    );
    cudaMemset(out, 0, static_cast<std::size_t>(F) * H * W * sizeof(float));
    cudaDeviceSynchronize();

    const dim3 grid(num_tiles_x, num_tiles_y);
    const dim3 block(OSS_TILE_SIZE, OSS_TILE_SIZE);
    state.add_element_count(static_cast<std::int64_t>(N) * H * W * F, "gaussian-channel-pixels");
    const std::int64_t global_read_bytes =
        static_cast<std::int64_t>(total_pairs) * (sizeof(int) + sizeof(float2) + sizeof(float3)) +
        static_cast<std::int64_t>(N) * F * sizeof(float);
    const std::int64_t global_write_bytes = static_cast<std::int64_t>(F) * H * W * sizeof(float);
    state.add_global_memory_reads<char>(global_read_bytes, "ReadBytes");
    state.add_global_memory_writes<char>(global_write_bytes, "WriteBytes");

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        for (int f_offset = 0; f_offset < F; f_offset += OSS_F_CHUNK) {
            const int f_chunk = std::min(OSS_F_CHUNK, F - f_offset);
            rasterize_sum<<<grid, block, 0, launch.get_stream()>>>(
                H, W, num_tiles_x, num_tiles_y,
                f_chunk, f_offset, F,
                gids, offsets, xy, conic, feat, out
            );
        }
    });

    cudaFree(xy);
    cudaFree(conic);
    cudaFree(feat);
    cudaFree(out);
    cudaFree(gids);
    cudaFree(offsets);
}

}  // namespace

NVBENCH_BENCH(bench_rasterize)
    .add_int64_axis("N", {16, 256, 4096})
    .add_int64_axis("H", {64, 270, 540})
    .add_int64_axis("W", {64, 480, 960})
    .add_int64_axis("F", {1, 12, 64});
