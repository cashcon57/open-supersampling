// OSS-Gaussian Vulkan rasterizer C++ host harness — Sprint 7 / T7.V.1 skeleton.
//
// Loads `rasterizer.spv`, creates a compute pipeline, and exposes a single
// `dispatch(...)` entry point. Used by the Sprint 7 unit harness for kernel
// parity testing against the Python reference rasterizer.
//
// Sprint 7 prep ships the dispatch skeleton only. T7.V.2 fills out the
// staging-buffer marshalling once the kernel body lands.
//
// Build via CMakeLists.txt:
//     cmake -B build && cmake --build build
//
// Pure validation harness; production driver is Python (`run_sintel.py`)
// calling into this via a small CFFI wrapper.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace oss_gaussian {

struct DispatchParams {
    uint32_t num_gaussians;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t feat_dim;
    uint32_t topk;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

class GaussianRasterizer {
public:
    // Construct the Vulkan instance + device + load `rasterizer.spv` from
    // `spv_path`. Throws std::runtime_error on any setup failure.
    explicit GaussianRasterizer(const std::string& spv_path);
    ~GaussianRasterizer();

    GaussianRasterizer(const GaussianRasterizer&) = delete;
    GaussianRasterizer& operator=(const GaussianRasterizer&) = delete;

    // Dispatch a single tile rasterization pass. Buffers must be pre-populated
    // by the caller with the layout documented in `rasterizer.comp`.
    // TODO(T7.V.2): wire the buffer marshalling once the kernel lands.
    void dispatch(VkBuffer gaussians,
                  VkBuffer tile_index,
                  VkBuffer tile_starts,
                  VkBuffer out_image,
                  const DispatchParams& params);

private:
    VkInstance       instance_       = VK_NULL_HANDLE;
    VkPhysicalDevice physical_       = VK_NULL_HANDLE;
    VkDevice         device_         = VK_NULL_HANDLE;
    VkQueue          queue_          = VK_NULL_HANDLE;
    uint32_t         queue_family_   = 0;
    VkShaderModule   shader_         = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_= VK_NULL_HANDLE;
    VkPipeline       pipeline_       = VK_NULL_HANDLE;
};

GaussianRasterizer::GaussianRasterizer(const std::string& spv_path) {
    // TODO(T7.V.1): Vulkan boilerplate — instance, physical device pick
    // (prefer discrete GPU, fall back to integrated for Deck APU), device
    // creation with VK_KHR_shader_float16_int8 enabled, compute queue
    // acquisition, shader module load from `spv_path`, descriptor set layout
    // (4 storage buffers + push constants), pipeline creation.
    //
    // Sprint 7 prep stops at the signature so the rest of the codebase can
    // reference the type. The actual Vulkan dance lands in T7.V.1.
    (void)spv_path;
}

GaussianRasterizer::~GaussianRasterizer() {
    if (pipeline_)        vkDestroyPipeline(device_, pipeline_, nullptr);
    if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    if (shader_)          vkDestroyShaderModule(device_, shader_, nullptr);
    if (device_)          vkDestroyDevice(device_, nullptr);
    if (instance_)        vkDestroyInstance(instance_, nullptr);
}

void GaussianRasterizer::dispatch(VkBuffer gaussians,
                                  VkBuffer tile_index,
                                  VkBuffer tile_starts,
                                  VkBuffer out_image,
                                  const DispatchParams& params) {
    // TODO(T7.V.2): allocate command buffer, bind descriptor set referencing
    // (gaussians, tile_index, tile_starts, out_image), push DispatchParams,
    // vkCmdDispatch with workgroup count = ceil_div(params.out_w, 16) ×
    // ceil_div(params.out_h, 16) × 1, submit + wait.
    (void)gaussians; (void)tile_index; (void)tile_starts;
    (void)out_image; (void)params;
}

}  // namespace oss_gaussian
