#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> preprocess_gaussians_cuda(
    torch::Tensor xy, torch::Tensor scale, torch::Tensor rot,
    int64_t h, int64_t w, int64_t tile_size
);

// Phase 1 stub: takes torch tensors, returns a passthrough.
// Phase 2 will replace this with a real CUDA fwd kernel.
torch::Tensor rasterize_forward_stub(
    torch::Tensor xy, torch::Tensor scale, torch::Tensor rot, torch::Tensor feat,
    int64_t h, int64_t w, int64_t tile_size, bool topk_norm
) {
    TORCH_CHECK(xy.dim() == 2 && xy.size(1) == 2, "xy must be (N,2)");
    TORCH_CHECK(scale.dim() == 2 && scale.size(1) == 2, "scale must be (N,2)");
    TORCH_CHECK(rot.dim() == 1, "rot must be (N,)");
    TORCH_CHECK(feat.dim() == 2, "feat must be (N,F)");
    TORCH_CHECK(xy.size(0) == scale.size(0), "xy/scale N mismatch");
    TORCH_CHECK(xy.size(0) == rot.size(0), "xy/rot N mismatch");
    TORCH_CHECK(xy.size(0) == feat.size(0), "xy/feat N mismatch");

    // Phase 1: defer to the Python-side reference renderer via a callback.
    // Phase 2 will replace this with a real CUDA implementation.
    auto ref = pybind11::module_::import("oss.cuda.oss_cuda.rasterizer")
                   .attr("_phase1_ref_forward");
    auto out = ref(xy, scale, rot, feat, h, w, tile_size, topk_norm)
                   .cast<torch::Tensor>();
    return out;
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "OSS custom CUDA extension (Phase 2a preprocess)";
    m.def("rasterize_forward", &rasterize_forward_stub,
          "Rasterize Gaussians (Phase 1: delegates to PyTorch reference)");
    m.def("preprocess_only", &preprocess_gaussians_cuda,
          "Preprocess Gaussian conics and tile AABBs (Phase 2a)");
}
