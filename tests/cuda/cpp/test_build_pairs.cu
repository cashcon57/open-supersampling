#include "common.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

struct BuildPairsResult {
    std::vector<int64_t> keys;
    std::vector<int> gids;
};

int64_t MakeKey(int tile_id, int gid) {
    return (static_cast<int64_t>(tile_id) << 32) | static_cast<uint32_t>(gid);
}

int TileIdFromKey(int64_t key) {
    return static_cast<int>(key >> 32);
}

int GidFromKey(int64_t key) {
    return static_cast<int>(key & 0xffffffff);
}

void RunBuildPairs(
    const std::vector<int4>& aabb_h,
    const std::vector<int>& cum_pair_count_h,
    int num_tiles_x,
    int output_capacity,
    BuildPairsResult* result
) {
    const int N = static_cast<int>(aabb_h.size());
    int4* aabb_d = nullptr;
    int* cum_pair_count_d = nullptr;
    int64_t* keys_d = nullptr;
    int* gid_d = nullptr;

    ASSERT_GT(output_capacity, 0);
    ASSERT_EQ(cudaMalloc(&aabb_d, N * sizeof(int4)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&cum_pair_count_d, cum_pair_count_h.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&keys_d, output_capacity * sizeof(int64_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&gid_d, output_capacity * sizeof(int)), cudaSuccess);

    std::vector<int64_t> keys_h(output_capacity, MakeKey(12345, 67890));
    std::vector<int> gid_h(output_capacity, -777);
    ASSERT_EQ(cudaMemcpy(aabb_d, aabb_h.data(), N * sizeof(int4), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(
                  cum_pair_count_d,
                  cum_pair_count_h.data(),
                  cum_pair_count_h.size() * sizeof(int),
                  cudaMemcpyHostToDevice
              ),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(keys_d, keys_h.data(), output_capacity * sizeof(int64_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(gid_d, gid_h.data(), output_capacity * sizeof(int), cudaMemcpyHostToDevice), cudaSuccess);

    constexpr int block_size = 256;
    build_tile_pairs<<<(N + block_size - 1) / block_size, block_size>>>(
        N, num_tiles_x, aabb_d, cum_pair_count_d, keys_d, gid_d
    );
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(keys_h.data(), keys_d, output_capacity * sizeof(int64_t), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(gid_h.data(), gid_d, output_capacity * sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(aabb_d);
    cudaFree(cum_pair_count_d);
    cudaFree(keys_d);
    cudaFree(gid_d);

    result->keys = keys_h;
    result->gids = gid_h;
}

}  // namespace

TEST(BuildPairs, Single1x1Tile) {
    BuildPairsResult result;
    RunBuildPairs(
        {make_int4(2, 2, 3, 3)},
        {0, 1},
        4,
        1,
        &result
    );

    ASSERT_EQ(result.keys.size(), 1);
    ASSERT_EQ(result.gids.size(), 1);
    EXPECT_EQ(TileIdFromKey(result.keys[0]), 10);
    EXPECT_EQ(GidFromKey(result.keys[0]), 0);
    EXPECT_EQ(result.gids[0], 0);
}

TEST(BuildPairs, Single2x2Block) {
    BuildPairsResult result;
    RunBuildPairs(
        {make_int4(0, 0, 2, 2)},
        {0, 4},
        4,
        4,
        &result
    );

    const std::vector<int> expected_tile_ids = {0, 1, 4, 5};
    ASSERT_EQ(result.keys.size(), expected_tile_ids.size());
    ASSERT_EQ(result.gids.size(), expected_tile_ids.size());
    for (std::size_t i = 0; i < expected_tile_ids.size(); ++i) {
        EXPECT_EQ(TileIdFromKey(result.keys[i]), expected_tile_ids[i]);
        EXPECT_EQ(GidFromKey(result.keys[i]), 0);
        EXPECT_EQ(result.gids[i], 0);
    }
}

TEST(BuildPairs, ZeroPairs) {
    const int64_t sentinel_key = MakeKey(12345, 67890);
    const int sentinel_gid = -777;
    BuildPairsResult result;
    RunBuildPairs(
        {make_int4(0, 0, 0, 0)},
        {0, 0},
        4,
        1,
        &result
    );

    ASSERT_EQ(result.keys.size(), 1);
    ASSERT_EQ(result.gids.size(), 1);
    EXPECT_EQ(result.keys[0], sentinel_key);
    EXPECT_EQ(result.gids[0], sentinel_gid);
}
