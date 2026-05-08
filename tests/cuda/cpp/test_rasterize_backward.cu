#include "common.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

float GaussianWeight(float px, float py, float cx, float cy, float sigma) {
    const float dx = px - cx;
    const float dy = py - cy;
    return std::exp(-0.5f * (dx * dx + dy * dy) / (sigma * sigma));
}

}  // namespace

TEST(RasterizeBackward, SingleGaussianDFeatMatchesWeightSum) {
    constexpr int H = 16;
    constexpr int W = 16;
    constexpr int F = 1;
    constexpr int N = 1;
    constexpr int num_tiles_x = 1;
    constexpr int num_tiles_y = 1;

    const std::vector<float2> xy_h{make_float2(8.0f, 8.0f)};
    const std::vector<float2> scale_h{make_float2(2.0f, 2.0f)};
    const std::vector<float> rot_h{0.0f};
    const std::vector<float> feat_h{1.0f};
    const std::vector<float> grad_out_h(F * H * W, 1.0f);
    const std::vector<int> gids_h{0};
    const std::vector<int> offsets_h{0, 1};

    float expected = 0.0f;
    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            expected += GaussianWeight(
                static_cast<float>(px),
                static_cast<float>(py),
                8.0f,
                8.0f,
                2.0f
            );
        }
    }

    float2* xy_d = nullptr;
    float2* scale_d = nullptr;
    float* rot_d = nullptr;
    float3* conic_d = nullptr;
    int4* aabb_d = nullptr;
    int* pair_count_d = nullptr;
    float* feat_d = nullptr;
    float* grad_out_d = nullptr;
    float2* d_xy_d = nullptr;
    float3* d_conic_d = nullptr;
    float* d_feat_d = nullptr;
    int* gids_d = nullptr;
    int* offsets_d = nullptr;

    ASSERT_EQ(cudaMalloc(&xy_d, N * sizeof(float2)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&scale_d, N * sizeof(float2)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&rot_d, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&conic_d, N * sizeof(float3)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&aabb_d, N * sizeof(int4)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&pair_count_d, N * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&feat_d, N * F * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&grad_out_d, F * H * W * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_xy_d, N * sizeof(float2)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_conic_d, N * sizeof(float3)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_feat_d, N * F * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&gids_d, gids_h.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&offsets_d, offsets_h.size() * sizeof(int)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(xy_d, xy_h.data(), N * sizeof(float2), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(scale_d, scale_h.data(), N * sizeof(float2), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(rot_d, rot_h.data(), N * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(feat_d, feat_h.data(), N * F * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(grad_out_d, grad_out_h.data(), F * H * W * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(gids_d, gids_h.data(), gids_h.size() * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(offsets_d, offsets_h.data(), offsets_h.size() * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_xy_d, 0, N * sizeof(float2)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_conic_d, 0, N * sizeof(float3)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_feat_d, 0, N * F * sizeof(float)), cudaSuccess);

    preprocess_gaussians<<<1, OSS_PREPROCESS_BLOCK>>>(
        N, H, W, OSS_TILE_SIZE, num_tiles_x, num_tiles_y,
        xy_d, scale_d, rot_d, conic_d, aabb_d, pair_count_d
    );
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    rasterize_backward<<<dim3(num_tiles_x, num_tiles_y), dim3(OSS_TILE_SIZE, OSS_TILE_SIZE)>>>(
        H, W, num_tiles_x, num_tiles_y,
        F, 0, F,
        gids_d, offsets_d, xy_d, conic_d, feat_d, grad_out_d,
        d_xy_d, d_conic_d, d_feat_d
    );
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float d_feat_h = 0.0f;
    ASSERT_EQ(cudaMemcpy(&d_feat_h, d_feat_d, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_NEAR(d_feat_h, expected, 0.5f);

    cudaFree(xy_d);
    cudaFree(scale_d);
    cudaFree(rot_d);
    cudaFree(conic_d);
    cudaFree(aabb_d);
    cudaFree(pair_count_d);
    cudaFree(feat_d);
    cudaFree(grad_out_d);
    cudaFree(d_xy_d);
    cudaFree(d_conic_d);
    cudaFree(d_feat_d);
    cudaFree(gids_d);
    cudaFree(offsets_d);
}

TEST(RasterizeBackward, DxyAnalytic) {
    constexpr int H = 16;
    constexpr int W = 16;
    constexpr int F = 1;
    constexpr int N = 1;
    constexpr int num_tiles_x = 1;
    constexpr int num_tiles_y = 1;
    constexpr int px = 9;
    constexpr int py = 8;

    const std::vector<float2> xy_h{make_float2(8.0f, 8.0f)};
    const std::vector<float3> conic_h{make_float3(0.25f, 0.0f, 0.25f)};
    const std::vector<float> feat_h{3.0f};
    std::vector<float> grad_out_h(F * H * W, 0.0f);
    grad_out_h[py * W + px] = 2.0f;
    const std::vector<int> gids_h{0};
    const std::vector<int> offsets_h{0, 1};

    const float weight = std::exp(-0.125f);
    const float expected_d_feat = 2.0f * weight;
    const float expected_dxy_x = 1.5f * weight;
    const float expected_dconic_x = -3.0f * weight;

    float2* xy_d = nullptr;
    float3* conic_d = nullptr;
    float* feat_d = nullptr;
    float* grad_out_d = nullptr;
    float2* d_xy_d = nullptr;
    float3* d_conic_d = nullptr;
    float* d_feat_d = nullptr;
    int* gids_d = nullptr;
    int* offsets_d = nullptr;

    ASSERT_EQ(cudaMalloc(&xy_d, N * sizeof(float2)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&conic_d, N * sizeof(float3)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&feat_d, N * F * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&grad_out_d, F * H * W * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_xy_d, N * sizeof(float2)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_conic_d, N * sizeof(float3)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_feat_d, N * F * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&gids_d, gids_h.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&offsets_d, offsets_h.size() * sizeof(int)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(xy_d, xy_h.data(), N * sizeof(float2), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(conic_d, conic_h.data(), N * sizeof(float3), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(feat_d, feat_h.data(), N * F * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(grad_out_d, grad_out_h.data(), F * H * W * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(gids_d, gids_h.data(), gids_h.size() * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(offsets_d, offsets_h.data(), offsets_h.size() * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_xy_d, 0, N * sizeof(float2)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_conic_d, 0, N * sizeof(float3)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_feat_d, 0, N * F * sizeof(float)), cudaSuccess);

    rasterize_backward<<<dim3(num_tiles_x, num_tiles_y), dim3(OSS_TILE_SIZE, OSS_TILE_SIZE)>>>(
        H, W, num_tiles_x, num_tiles_y,
        F, 0, F,
        gids_d, offsets_d, xy_d, conic_d, feat_d, grad_out_d,
        d_xy_d, d_conic_d, d_feat_d
    );
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    float2 d_xy_h{};
    float3 d_conic_h{};
    float d_feat_h = 0.0f;
    ASSERT_EQ(cudaMemcpy(&d_xy_h, d_xy_d, sizeof(float2), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&d_conic_h, d_conic_d, sizeof(float3), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&d_feat_h, d_feat_d, sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    EXPECT_NEAR(d_feat_h, expected_d_feat, 1e-5f);
    EXPECT_NEAR(d_xy_h.x, expected_dxy_x, 1e-5f);
    EXPECT_NEAR(d_xy_h.y, 0.0f, 1e-5f);
    EXPECT_NEAR(d_conic_h.x, expected_dconic_x, 1e-5f);
    EXPECT_NEAR(d_conic_h.y, 0.0f, 1e-5f);
    EXPECT_NEAR(d_conic_h.z, 0.0f, 1e-5f);

    cudaFree(xy_d);
    cudaFree(conic_d);
    cudaFree(feat_d);
    cudaFree(grad_out_d);
    cudaFree(d_xy_d);
    cudaFree(d_conic_d);
    cudaFree(d_feat_d);
    cudaFree(gids_d);
    cudaFree(offsets_d);
}
