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

torch::Tensor rasterize_forward_cuda(
    torch::Tensor xy, torch::Tensor scale, torch::Tensor rot, torch::Tensor feat,
    int64_t h, int64_t w, int64_t tile_size, bool topk_norm
);

torch::Tensor rasterize_forward(
    torch::Tensor xy, torch::Tensor scale, torch::Tensor rot, torch::Tensor feat,
    int64_t h, int64_t w, int64_t tile_size, bool topk_norm
) {
    return rasterize_forward_cuda(xy, scale, rot, feat, h, w, tile_size, topk_norm);
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "OSS custom CUDA extension (Phase 2 rasterize_sum)";
    m.def("rasterize_forward", &rasterize_forward,
          "Rasterize Gaussians (Phase 2 CUDA forward)");
    m.def("preprocess_only", &preprocess_gaussians_cuda,
          "Preprocess Gaussian conics and tile AABBs (Phase 2a)");
    m.def("pair_construction_only", &pair_construction_cuda,
          "Build sorted Gaussian/tile pairs and tile offsets (Phase 2b)");
}
