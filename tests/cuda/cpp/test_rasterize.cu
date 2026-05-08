#include "common.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

std::vector<int> FullTileGids(int n, int num_tiles) {
    std::vector<int> gids;
    gids.reserve(n * num_tiles);
    for (int tile = 0; tile < num_tiles; ++tile) {
        for (int gid = 0; gid < n; ++gid) {
            gids.push_back(gid);
        }
    }
    return gids;
}

std::vector<int> FullTileOffsets(int n, int num_tiles) {
    std::vector<int> offsets(num_tiles + 1);
    for (int tile = 0; tile <= num_tiles; ++tile) {
        offsets[tile] = tile * n;
    }
    return offsets;
}

void RunRasterize(
    int H,
    int W,
    int F,
    const std::vector<float2>& xy_h,
    const std::vector<float3>& conic_h,
    const std::vector<float>& feat_h,
    std::vector<float>* out_h
) {
    const int N = static_cast<int>(xy_h.size());
    const int num_tiles_x = (W + OSS_TILE_SIZE - 1) / OSS_TILE_SIZE;
    const int num_tiles_y = (H + OSS_TILE_SIZE - 1) / OSS_TILE_SIZE;
    const int num_tiles = num_tiles_x * num_tiles_y;
    const std::vector<int> gids_h = FullTileGids(N, num_tiles);
    const std::vector<int> offsets_h = FullTileOffsets(N, num_tiles);

    float2* xy_d = nullptr;
    float3* conic_d = nullptr;
    float* feat_d = nullptr;
    int* gids_d = nullptr;
    int* offsets_d = nullptr;
    float* out_d = nullptr;

    if (N > 0) {
        ASSERT_EQ(cudaMalloc(&xy_d, N * sizeof(float2)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&conic_d, N * sizeof(float3)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&feat_d, N * F * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(xy_d, xy_h.data(), N * sizeof(float2), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(conic_d, conic_h.data(), N * sizeof(float3), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(feat_d, feat_h.data(), N * F * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    }
    if (!gids_h.empty()) {
        ASSERT_EQ(cudaMalloc(&gids_d, gids_h.size() * sizeof(int)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(gids_d, gids_h.data(), gids_h.size() * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    }
    ASSERT_EQ(cudaMalloc(&offsets_d, offsets_h.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(offsets_d, offsets_h.data(), offsets_h.size() * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&out_d, F * H * W * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMemset(out_d, 0, F * H * W * sizeof(float)), cudaSuccess);

    const dim3 grid(num_tiles_x, num_tiles_y);
    const dim3 block(OSS_TILE_SIZE, OSS_TILE_SIZE);
    for (int chunk = 0; chunk < (F + OSS_F_CHUNK - 1) / OSS_F_CHUNK; ++chunk) {
        const int f_offset = chunk * OSS_F_CHUNK;
        const int f_chunk = std::min(OSS_F_CHUNK, F - f_offset);
        rasterize_sum<<<grid, block>>>(
            H, W, num_tiles_x, num_tiles_y,
            f_chunk, f_offset, F,
            gids_d, offsets_d, xy_d, conic_d, feat_d, out_d
        );
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    out_h->assign(F * H * W, 0.0f);
    ASSERT_EQ(cudaMemcpy(out_h->data(), out_d, out_h->size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(xy_d);
    cudaFree(conic_d);
    cudaFree(feat_d);
    cudaFree(gids_d);
    cudaFree(offsets_d);
    cudaFree(out_d);
}

float GaussianWeight(float px, float py, float cx, float cy, float sigma) {
    const float dx = px - cx;
    const float dy = py - cy;
    return std::exp(-0.5f * (dx * dx + dy * dy) / (sigma * sigma));
}

}  // namespace

TEST(Rasterize, SinglePointAtCenter) {
    std::vector<float> out;
    RunRasterize(
        32, 32, 1,
        {make_float2(16.0f, 16.0f)},
        {make_float3(1.0f / 9.0f, 0.0f, 1.0f / 9.0f)},
        {1.0f},
        &out
    );

    for (int py = 0; py < 32; ++py) {
        for (int px = 0; px < 32; ++px) {
            const float expected = GaussianWeight(static_cast<float>(px), static_cast<float>(py), 16.0f, 16.0f, 3.0f);
            EXPECT_NEAR(out[py * 32 + px], expected, 5.0e-3f);
        }
    }
}

TEST(Rasterize, ZeroGaussians) {
    std::vector<float> out;
    RunRasterize(32, 32, 1, {}, {}, {}, &out);

    for (float v : out) {
        EXPECT_EQ(v, 0.0f);
    }
}

TEST(Rasterize, OffScreenGaussian) {
    std::vector<float> out;
    RunRasterize(
        32, 32, 1,
        {make_float2(-100.0f, -100.0f)},
        {make_float3(0.25f, 0.0f, 0.25f)},
        {1.0f},
        &out
    );

    for (float v : out) {
        EXPECT_NEAR(v, 0.0f, 5.0e-3f);
    }
}

TEST(Rasterize, MultipleGaussiansAdditive) {
    std::vector<float> out;
    RunRasterize(
        32, 32, 1,
        {make_float2(16.0f, 16.0f), make_float2(8.0f, 8.0f)},
        {make_float3(1.0f / 9.0f, 0.0f, 1.0f / 9.0f), make_float3(1.0f / 4.0f, 0.0f, 1.0f / 4.0f)},
        {1.0f, 2.0f},
        &out
    );

    for (int py = 0; py < 32; ++py) {
        for (int px = 0; px < 32; ++px) {
            const float expected =
                GaussianWeight(static_cast<float>(px), static_cast<float>(py), 16.0f, 16.0f, 3.0f) +
                2.0f * GaussianWeight(static_cast<float>(px), static_cast<float>(py), 8.0f, 8.0f, 2.0f);
            EXPECT_NEAR(out[py * 32 + px], expected, 5.0e-3f);
        }
    }
}
