#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> preprocess_gaussians_cuda(
    torch::Tensor xy, torch::Tensor scale, torch::Tensor rot,
    int64_t h, int64_t w, int64_t tile_size
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> pair_construction_cuda(
    torch::Tensor xy, torch::Tensor scale, torch::Tensor rot,
    int64_t h, int64_t w, int64_t tile_size
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_forward_cuda(
    torch::Tensor xy, torch::Tensor scale, torch::Tensor rot, torch::Tensor feat,
    int64_t h, int64_t w, int64_t tile_size, bool topk_norm
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize_forward(
    torch::Tensor xy, torch::Tensor scale, torch::Tensor rot, torch::Tensor feat,
    int64_t h, int64_t w, int64_t tile_size, bool topk_norm
) {
    return rasterize_forward_cuda(xy, scale, rot, feat, h, w, tile_size, topk_norm);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rasterize_backward_cuda(
    torch::Tensor xy, torch::Tensor scale, torch::Tensor rot, torch::Tensor feat,
    torch::Tensor conic, torch::Tensor gaussian_idx_sorted,
    torch::Tensor tile_offsets, torch::Tensor grad_out,
    int64_t h, int64_t w, int64_t tile_size
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rasterize_backward(
    torch::Tensor xy, torch::Tensor scale, torch::Tensor rot, torch::Tensor feat,
    torch::Tensor conic, torch::Tensor gaussian_idx_sorted,
    torch::Tensor tile_offsets, torch::Tensor grad_out,
    int64_t h, int64_t w, int64_t tile_size
) {
    return rasterize_backward_cuda(
        xy, scale, rot, feat, conic, gaussian_idx_sorted, tile_offsets, grad_out,
        h, w, tile_size
    );
}

std::tuple<torch::Tensor, torch::Tensor> conic_to_scale_rot_grad_cuda(
    torch::Tensor scale, torch::Tensor rot, torch::Tensor d_conic
);

std::tuple<torch::Tensor, torch::Tensor> conic_to_scale_rot_grad(
    torch::Tensor scale, torch::Tensor rot, torch::Tensor d_conic
) {
    return conic_to_scale_rot_grad_cuda(scale, rot, d_conic);
}

std::tuple<torch::Tensor, torch::Tensor> tile_bin_counting_sort_cuda(
    torch::Tensor tile_id, torch::Tensor gid, int64_t num_tiles
);

std::tuple<torch::Tensor, torch::Tensor> tile_bin_counting_sort(
    torch::Tensor tile_id, torch::Tensor gid, int64_t num_tiles
) {
    return tile_bin_counting_sort_cuda(tile_id, gid, num_tiles);
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "OSS custom CUDA extension (Phase 3c rasterizer)";
    m.def("rasterize_forward", &rasterize_forward,
          "Rasterize Gaussians (Phase 3c CUDA forward)");
    m.def("rasterize_backward", &rasterize_backward,
          "Rasterize Gaussians backward (Phase 3c dxy+dconic+dfeat)");
    m.def("conic_to_scale_rot_grad", &conic_to_scale_rot_grad,
          "Convert d_conic to d_scale and d_rot (Phase 3c)");
    m.def("preprocess_only", &preprocess_gaussians_cuda,
          "Preprocess Gaussian conics and tile AABBs (Phase 2a)");
    m.def("pair_construction_only", &pair_construction_cuda,
          "Build sorted Gaussian/tile pairs and tile offsets (Phase 2b)");
    m.def("tile_bin_counting_sort", &tile_bin_counting_sort,
          "Counting-sort tile ids into grouped gid bins");
}
