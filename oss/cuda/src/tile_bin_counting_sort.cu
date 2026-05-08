#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <climits>
#include <tuple>

namespace {

constexpr int kBlockSize = 256;

void check_int32_cuda_1d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 1, name, " must be 1D");
}

__global__ void count_tiles_kernel(
    int n,
    int num_tiles,
    const int* __restrict__ tile_id,
    int* __restrict__ counts
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int tile = tile_id[idx];
    if (tile >= 0 && tile < num_tiles) {
        atomicAdd(counts + tile, 1);
    }
}

__global__ void set_last_offset_kernel(
    int n,
    int num_tiles,
    int* __restrict__ tile_offsets
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        tile_offsets[num_tiles] = n;
    }
}

__global__ void scatter_gids_kernel(
    int n,
    int num_tiles,
    const int* __restrict__ tile_id,
    const int* __restrict__ gid,
    int* __restrict__ write_cursor,
    int* __restrict__ sorted_gid
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int tile = tile_id[idx];
    if (tile >= 0 && tile < num_tiles) {
        const int out_idx = atomicAdd(write_cursor + tile, 1);
        sorted_gid[out_idx] = gid[idx];
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> tile_bin_counting_sort_cuda(
    torch::Tensor tile_id,
    torch::Tensor gid,
    int64_t num_tiles
) {
    check_int32_cuda_1d(tile_id, "tile_id");
    check_int32_cuda_1d(gid, "gid");
    TORCH_CHECK(tile_id.numel() == gid.numel(), "tile_id and gid must have the same length");
    TORCH_CHECK(num_tiles > 0, "num_tiles must be positive");
    TORCH_CHECK(num_tiles <= static_cast<int64_t>(INT_MAX), "num_tiles exceeds int32 range");
    TORCH_CHECK(tile_id.numel() <= static_cast<int64_t>(INT_MAX), "N exceeds int32 range");

    const int n = static_cast<int>(tile_id.numel());
    const int tiles = static_cast<int>(num_tiles);
    auto int_opts = tile_id.options().dtype(torch::kInt32);
    auto counts = torch::zeros({tiles}, int_opts);
    auto tile_offsets = torch::empty({tiles + 1}, int_opts);
    auto sorted_gid = torch::empty({n}, int_opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (n + kBlockSize - 1) / kBlockSize;
    if (n > 0) {
        count_tiles_kernel<<<blocks, kBlockSize, 0, stream>>>(
            n,
            tiles,
            tile_id.data_ptr<int>(),
            counts.data_ptr<int>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    thrust::device_ptr<int> counts_ptr(counts.data_ptr<int>());
    thrust::device_ptr<int> offsets_ptr(tile_offsets.data_ptr<int>());
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        counts_ptr,
        counts_ptr + tiles,
        offsets_ptr
    );

    set_last_offset_kernel<<<1, 1, 0, stream>>>(n, tiles, tile_offsets.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto write_cursor = tile_offsets.slice(/*dim=*/0, /*start=*/0, /*end=*/tiles).clone();
    if (n > 0) {
        scatter_gids_kernel<<<blocks, kBlockSize, 0, stream>>>(
            n,
            tiles,
            tile_id.data_ptr<int>(),
            gid.data_ptr<int>(),
            write_cursor.data_ptr<int>(),
            sorted_gid.data_ptr<int>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return std::make_tuple(sorted_gid, tile_offsets);
}
