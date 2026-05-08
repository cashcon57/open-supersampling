#include "common.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

void RunPreprocess(
    const std::vector<float2>& xy_h,
    const std::vector<float2>& scale_h,
    const std::vector<float>& rot_h,
    int H,
    int W,
    std::vector<float3>* conic_h,
    std::vector<int4>* aabb_h,
    std::vector<int>* pair_count_h
) {
    const int N = static_cast<int>(xy_h.size());
    float2* xy_d = nullptr;
    float2* scale_d = nullptr;
    float* rot_d = nullptr;
    float3* conic_d = nullptr;
    int4* aabb_d = nullptr;
    int* pair_count_d = nullptr;

    ASSERT_EQ(cudaMalloc(&xy_d, N * sizeof(float2)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&scale_d, N * sizeof(float2)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&rot_d, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&conic_d, N * sizeof(float3)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&aabb_d, N * sizeof(int4)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&pair_count_d, N * sizeof(int)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(xy_d, xy_h.data(), N * sizeof(float2), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(scale_d, scale_h.data(), N * sizeof(float2), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(rot_d, rot_h.data(), N * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    constexpr int tile_size = OSS_TILE_SIZE;
    const int num_tiles_x = (W + tile_size - 1) / tile_size;
    const int num_tiles_y = (H + tile_size - 1) / tile_size;
    preprocess_gaussians<<<1, OSS_PREPROCESS_BLOCK>>>(
        N, H, W, tile_size, num_tiles_x, num_tiles_y,
        xy_d, scale_d, rot_d, conic_d, aabb_d, pair_count_d
    );
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    conic_h->resize(N);
    aabb_h->resize(N);
    pair_count_h->resize(N);
    ASSERT_EQ(cudaMemcpy(conic_h->data(), conic_d, N * sizeof(float3), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(aabb_h->data(), aabb_d, N * sizeof(int4), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(pair_count_h->data(), pair_count_d, N * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(xy_d);
    cudaFree(scale_d);
    cudaFree(rot_d);
    cudaFree(conic_d);
    cudaFree(aabb_d);
    cudaFree(pair_count_d);
}

}  // namespace

TEST(Preprocess, SinglePointAtCenter) {
    std::vector<float3> conic;
    std::vector<int4> aabb;
    std::vector<int> pair_count;
    RunPreprocess({make_float2(64.0f, 64.0f)}, {make_float2(3.0f, 3.0f)}, {0.0f}, 128, 128, &conic, &aabb, &pair_count);

    EXPECT_NEAR(conic[0].x, 1.0f / 9.0f, 1e-7f);
    EXPECT_NEAR(conic[0].y, 0.0f, 1e-7f);
    EXPECT_NEAR(conic[0].z, 1.0f / 9.0f, 1e-7f);
    EXPECT_EQ(aabb[0].x, 3);
    EXPECT_EQ(aabb[0].y, 3);
    EXPECT_EQ(aabb[0].z, 5);
    EXPECT_EQ(aabb[0].w, 5);
    EXPECT_EQ(pair_count[0], 4);
}

TEST(Preprocess, OffScreenGaussian) {
    std::vector<float3> conic;
    std::vector<int4> aabb;
    std::vector<int> pair_count;
    RunPreprocess({make_float2(-100.0f, -100.0f)}, {make_float2(2.0f, 2.0f)}, {0.0f}, 128, 128, &conic, &aabb, &pair_count);

    EXPECT_EQ(aabb[0].x, 0);
    EXPECT_EQ(aabb[0].y, 0);
    EXPECT_EQ(aabb[0].z, 0);
    EXPECT_EQ(aabb[0].w, 0);
    EXPECT_EQ(pair_count[0], 0);
}

TEST(Preprocess, NaNRot) {
    std::vector<float3> conic;
    std::vector<int4> aabb;
    std::vector<int> pair_count;
    RunPreprocess({make_float2(10.0f, 10.0f)}, {make_float2(3.0f, 3.0f)}, {NAN}, 128, 128, &conic, &aabb, &pair_count);

    EXPECT_EQ(aabb[0].x, 0);
    EXPECT_EQ(aabb[0].y, 0);
    EXPECT_EQ(aabb[0].z, 0);
    EXPECT_EQ(aabb[0].w, 0);
    EXPECT_EQ(pair_count[0], 0);
}
