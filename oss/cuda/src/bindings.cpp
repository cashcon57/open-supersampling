#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdlib>
#include <string>
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
    const char* debug = std::getenv("OSS_CUDA_RASTER_DEBUG");
    if (debug != nullptr && std::string(debug) == "1") {
        // Temporary Phase 2c A/B hatch. Remove with _phase1_ref_forward in 2d.
        pybind11::gil_scoped_acquire gil;
        auto rasterizer = pybind11::module_::import("oss.cuda.oss_cuda.rasterizer");
        TORCH_CHECK(
            pybind11::hasattr(rasterizer, "_phase1_ref_forward"),
            "OSS_CUDA_RASTER_DEBUG=1 requested, but _phase1_ref_forward is missing"
        );
        auto ref = rasterizer.attr("_phase1_ref_forward");
        return ref(xy, scale, rot, feat, h, w, tile_size, topk_norm)
            .cast<torch::Tensor>();
    }

    return rasterize_forward_cuda(xy, scale, rot, feat, h, w, tile_size, topk_norm);
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "OSS custom CUDA extension (Phase 2c rasterize_sum)";
    m.def("rasterize_forward", &rasterize_forward,
          "Rasterize Gaussians (Phase 2c CUDA forward)");
    m.def("preprocess_only", &preprocess_gaussians_cuda,
          "Preprocess Gaussian conics and tile AABBs (Phase 2a)");
    m.def("pair_construction_only", &pair_construction_cuda,
          "Build sorted Gaussian/tile pairs and tile offsets (Phase 2b)");
}
