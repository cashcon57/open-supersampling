// =============================================================================
//  ffx_fsr3_proxy.cpp
//
//  Pass-through proxy for games that import AMD FidelityFX FSR3 directly as
//  ffx_fsr3_x64.dll or ffx_fsr3upscaler_x64.dll. The wrapper keeps the
//  game-facing ABI narrow and forwards to the original DLL renamed by the
//  installer/smoke harness.
// =============================================================================
#define OSS_GAUSSIAN_BUILDING_DLL 1

#include "ffx_fsr3_proxy.h"

#include "log.h"
#include "ngx_frame_capture.h"

#include <Windows.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace oss_gaussian {
void OssGaussianEnsureInitializedFromProxyExport();
}

namespace oss_gaussian::ffx_fsr3 {

using FfxErrorCode = int32_t;
constexpr FfxErrorCode kFfxOk = 0;
constexpr FfxErrorCode kFfxErrorInvalidPointer = static_cast<FfxErrorCode>(0x80000000u);

struct FfxDimensions2DAbi {
    uint32_t width;
    uint32_t height;
};

struct FfxFloatCoords2DAbi {
    float x;
    float y;
};

struct FfxResourceDescriptionAbi {
    uint32_t type;
    uint32_t format;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t mip_count;
    uint32_t flags;
    uint32_t usage;
};

struct FfxResourceAbi {
    void* resource;
    FfxResourceDescriptionAbi description;
    uint32_t state;
};

struct FfxConstantBufferAbi {
    uint32_t num32_bit_entries;
    uint32_t* data;
};

struct FfxInterfaceAbi;

using FfxRegisterResourceFn = FfxErrorCode (*)(
    FfxInterfaceAbi* backend_interface,
    const FfxResourceAbi* in_resource,
    uint32_t effect_context_id,
    int32_t* out_resource);
using FfxGetResourceFn = FfxResourceAbi (*)(
    FfxInterfaceAbi* backend_interface,
    int32_t resource);
using FfxStageConstantBufferDataFn = FfxErrorCode (*)(
    FfxInterfaceAbi* backend_interface,
    void* data,
    uint32_t size,
    FfxConstantBufferAbi* constant_buffer);

struct FfxInterfaceAbi {
    void* fp_get_sdk_version;
    void* fp_get_effect_gpu_memory_usage;
    void* fp_create_backend_context;
    void* fp_get_device_capabilities;
    void* fp_destroy_backend_context;
    void* fp_create_resource;
    FfxRegisterResourceFn fp_register_resource;
    FfxGetResourceFn fp_get_resource;
    void* fp_unregister_resources;
    void* fp_register_static_resource;
    void* fp_get_resource_description;
    void* fp_destroy_resource;
    void* fp_map_resource;
    void* fp_unmap_resource;
    FfxStageConstantBufferDataFn fp_stage_constant_buffer_data;
    void* fp_create_pipeline;
    void* fp_destroy_pipeline;
    void* fp_legacy17;
    void* fp_legacy18;
};

struct FfxInterfaceCallbacks {
    FfxRegisterResourceFn register_resource = nullptr;
    FfxGetResourceFn get_resource = nullptr;
    FfxStageConstantBufferDataFn stage_constant_buffer_data = nullptr;
};

struct FfxFsr3UpscalerContextDescriptionAbi {
    uint32_t flags;
    FfxDimensions2DAbi max_render_size;
    FfxDimensions2DAbi max_upscale_size;
    void* fp_message;
    FfxInterfaceAbi backend_interface;
};

struct FfxFsr3DispatchUpscaleDescriptionAbi {
    void* command_list;
    FfxResourceAbi color;
    FfxResourceAbi depth;
    FfxResourceAbi motion_vectors;
    FfxResourceAbi exposure;
    FfxResourceAbi reactive;
    FfxResourceAbi transparency_and_composition;
    FfxResourceAbi dilated_depth;
    FfxResourceAbi dilated_motion_vectors;
    FfxResourceAbi reconstructed_prev_nearest_depth;
    FfxResourceAbi upscale_output;
    FfxFloatCoords2DAbi jitter_offset;
    FfxFloatCoords2DAbi motion_vector_scale;
    FfxDimensions2DAbi render_size;
    FfxDimensions2DAbi upscale_size;
    bool enable_sharpening;
    float sharpness;
    float frame_time_delta;
    float pre_exposure;
    bool reset;
    float camera_near;
    float camera_far;
    float camera_fov_angle_vertical;
    float view_space_to_meters_factor;
    uint32_t flags;
};

struct FfxFsr3GenerateReactiveDescriptionAbi {
    void* command_list;
    FfxResourceAbi color_opaque_only;
    FfxResourceAbi color_pre_upscale;
    FfxResourceAbi out_reactive;
    FfxDimensions2DAbi render_size;
    float scale;
    float cutoff_threshold;
    float binary_value;
    uint32_t flags;
};

namespace {

constexpr wchar_t kRealFsr3DllName[] = L"oss_ffx_fsr3_real.dll";
constexpr wchar_t kBackupFsr3DllName[] = L"ffx_fsr3_x64.dll.oss-backup";
constexpr wchar_t kBackupFsr3UpscalerDllName[] = L"ffx_fsr3upscaler_x64.dll.oss-backup";

std::once_flag g_load_once;
std::atomic<uint64_t> g_dispatch_frame_index{1};
std::mutex g_embedded_interface_mutex;
std::unordered_map<FfxInterfaceAbi*, FfxInterfaceCallbacks> g_embedded_interface_callbacks;
FfxInterfaceCallbacks g_last_embedded_interface_callbacks;
bool g_has_last_embedded_interface_callbacks = false;

// Defined below; declared here for the embedded-callback wrappers.
void LogResourceSummary(const char* name, const FfxResourceAbi& resource);

enum class RealFsr3Kind {
    kFsr3,
    kUpscaler,
};

struct RealFsr3Candidate {
    std::wstring path;
    RealFsr3Kind kind;
};

struct RealFsr3Module {
    HMODULE module = nullptr;
    RealFsr3Kind kind = RealFsr3Kind::kFsr3;
};

std::vector<RealFsr3Module> g_real_fsr3_modules;

std::wstring DirectoryOf(const std::wstring& path) {
    const size_t slash = path.find_last_of(L"\\/");
    if (slash == std::wstring::npos) return std::wstring();
    return path.substr(0, slash);
}

std::wstring JoinPath(const std::wstring& dir, const wchar_t* name) {
    if (dir.empty()) return std::wstring(name);
    if (dir.back() == L'\\' || dir.back() == L'/') return dir + name;
    return dir + L"\\" + name;
}

std::wstring FullPath(const std::wstring& path) {
    wchar_t full[MAX_PATH] = {};
    const DWORD len = GetFullPathNameW(path.c_str(), MAX_PATH, full, nullptr);
    if (len == 0 || len >= MAX_PATH) return path;
    return std::wstring(full);
}

bool SamePath(const std::wstring& a, const std::wstring& b) {
    const std::wstring fa = FullPath(a);
    const std::wstring fb = FullPath(b);
    return _wcsicmp(fa.c_str(), fb.c_str()) == 0;
}

void SelfModuleAnchor() {}

std::wstring ModulePath(HMODULE module) {
    wchar_t path[MAX_PATH] = {};
    const DWORD len = GetModuleFileNameW(module, path, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) return std::wstring();
    return std::wstring(path);
}

std::wstring SelfModulePath() {
    HMODULE self = nullptr;
    const DWORD flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if (!GetModuleHandleExW(flags,
                            reinterpret_cast<LPCWSTR>(&SelfModuleAnchor),
                            &self)) {
        return std::wstring();
    }
    return ModulePath(self);
}

void PushEnvOverride(std::vector<RealFsr3Candidate>& candidates,
                     const wchar_t* env_name,
                     RealFsr3Kind kind) {
    wchar_t value[MAX_PATH] = {};
    const DWORD len = GetEnvironmentVariableW(env_name, value, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        candidates.push_back({value, kind});
    }
}

void PushStandardCandidates(std::vector<RealFsr3Candidate>& candidates,
                            const std::wstring& dir) {
    candidates.push_back({JoinPath(dir, kRealFsr3DllName), RealFsr3Kind::kFsr3});
    candidates.push_back({JoinPath(dir, kBackupFsr3DllName), RealFsr3Kind::kFsr3});
    candidates.push_back({JoinPath(dir, kBackupFsr3UpscalerDllName), RealFsr3Kind::kUpscaler});
}

std::vector<RealFsr3Candidate> BuildCandidatePaths() {
    std::vector<RealFsr3Candidate> candidates;
    PushEnvOverride(candidates, L"OSS_FFX_FSR3_REAL_DLL", RealFsr3Kind::kFsr3);
    PushEnvOverride(candidates, L"OSS_FFX_FSR3_UPSCALER_REAL_DLL", RealFsr3Kind::kUpscaler);

    wchar_t exe_path[MAX_PATH] = {};
    const DWORD exe_len = GetModuleFileNameW(nullptr, exe_path, MAX_PATH);
    if (exe_len > 0 && exe_len < MAX_PATH) {
        PushStandardCandidates(candidates, DirectoryOf(exe_path));
    }

    wchar_t cwd[MAX_PATH] = {};
    const DWORD cwd_len = GetCurrentDirectoryW(MAX_PATH, cwd);
    if (cwd_len > 0 && cwd_len < MAX_PATH) {
        PushStandardCandidates(candidates, cwd);
    }

    return candidates;
}

HMODULE TryLoadCandidate(const std::wstring& candidate, const std::wstring& self_path) {
    if (candidate.empty()) return nullptr;
    if (!self_path.empty() && SamePath(candidate, self_path)) return nullptr;

    const DWORD attrs = GetFileAttributesW(candidate.c_str());
    if (attrs == INVALID_FILE_ATTRIBUTES || (attrs & FILE_ATTRIBUTE_DIRECTORY)) {
        return nullptr;
    }

    HMODULE module = LoadLibraryExW(candidate.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) {
        OSSG_LOG_WARN("ffx_fsr3", "failed to load real FSR3 candidate err=%lu", GetLastError());
        return nullptr;
    }

    const std::wstring loaded_path = ModulePath(module);
    if (!self_path.empty() && !loaded_path.empty() && SamePath(loaded_path, self_path)) {
        FreeLibrary(module);
        return nullptr;
    }

    return module;
}

void LoadRealFsr3Once() {
    LogInit();
    const std::wstring self_path = SelfModulePath();
    for (const RealFsr3Candidate& candidate : BuildCandidatePaths()) {
        const std::wstring candidate_path = FullPath(candidate.path);
        bool already_loaded = false;
        for (const RealFsr3Module& existing : g_real_fsr3_modules) {
            const std::wstring loaded_path = FullPath(ModulePath(existing.module));
            if (!loaded_path.empty() && SamePath(loaded_path, candidate_path)) {
                already_loaded = true;
                break;
            }
        }
        if (already_loaded) continue;

        if (HMODULE module = TryLoadCandidate(candidate.path, self_path)) {
            g_real_fsr3_modules.push_back({module, candidate.kind});
            OSSG_LOG_INFO("ffx_fsr3", "loaded real FSR3 DLL candidate kind=%u",
                          candidate.kind == RealFsr3Kind::kUpscaler ? 1u : 0u);
        }
    }
    if (g_real_fsr3_modules.empty()) {
        OSSG_LOG_ERROR("ffx_fsr3", "real FSR3 DLL not found");
    }
}

const std::vector<RealFsr3Module>& RealFsr3Modules() {
    std::call_once(g_load_once, LoadRealFsr3Once);
    return g_real_fsr3_modules;
}

RealFsr3Kind PreferredKindForExport(const char* name) {
    constexpr const char kUpscalerPrefix[] = "ffxFsr3Upscaler";
    if (name && std::strncmp(name, kUpscalerPrefix, sizeof(kUpscalerPrefix) - 1) == 0) {
        return RealFsr3Kind::kUpscaler;
    }
    return RealFsr3Kind::kFsr3;
}

FARPROC TryResolveFromKind(const std::vector<RealFsr3Module>& modules,
                           RealFsr3Kind kind,
                           const char* name) {
    for (const RealFsr3Module& loaded : modules) {
        if (loaded.kind != kind) continue;
        if (FARPROC proc = GetProcAddress(loaded.module, name)) return proc;
    }
    return nullptr;
}

FARPROC TryResolveAny(const std::vector<RealFsr3Module>& modules, const char* name) {
    for (const RealFsr3Module& loaded : modules) {
        if (FARPROC proc = GetProcAddress(loaded.module, name)) return proc;
    }
    return nullptr;
}

bool SafeCopyDispatch(const FfxFsr3DispatchUpscaleDescriptionAbi* src,
                      FfxFsr3DispatchUpscaleDescriptionAbi* dst) {
    if (!src || !dst) return false;
    __try {
        *dst = *src;
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

bool SafeCopyReactive(const FfxFsr3GenerateReactiveDescriptionAbi* src,
                      FfxFsr3GenerateReactiveDescriptionAbi* dst) {
    if (!src || !dst) return false;
    __try {
        *dst = *src;
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

bool SafeCopyUpscalerContextDescription(
    const void* src,
    FfxFsr3UpscalerContextDescriptionAbi* dst) {
    if (!src || !dst) return false;
    __try {
        *dst = *static_cast<const FfxFsr3UpscalerContextDescriptionAbi*>(src);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

bool LoadEmbeddedCallbacks(FfxInterfaceAbi* backend_interface, FfxInterfaceCallbacks* out) {
    if (!out) return false;
    std::lock_guard<std::mutex> lock(g_embedded_interface_mutex);
    const auto it = g_embedded_interface_callbacks.find(backend_interface);
    if (it != g_embedded_interface_callbacks.end()) {
        *out = it->second;
        return true;
    }
    if (g_has_last_embedded_interface_callbacks) {
        *out = g_last_embedded_interface_callbacks;
        return true;
    }
    return false;
}

FfxErrorCode WrappedEmbeddedRegisterResource(
    FfxInterfaceAbi* backend_interface,
    const FfxResourceAbi* in_resource,
    uint32_t effect_context_id,
    int32_t* out_resource) {
    LogInit();
    if (in_resource) {
        OSSG_LOG_INFO("ffx_fsr3",
                      "embedded fpRegisterResource iface=%p effect=%u resource=%p state=%u",
                      backend_interface,
                      effect_context_id,
                      in_resource->resource,
                      in_resource->state);
        LogResourceSummary("embedded fpRegisterResource", *in_resource);
    } else {
        OSSG_LOG_INFO("ffx_fsr3",
                      "embedded fpRegisterResource iface=%p effect=%u resource=<null>",
                      backend_interface,
                      effect_context_id);
    }
    FfxInterfaceCallbacks callbacks;
    if (!LoadEmbeddedCallbacks(backend_interface, &callbacks) || !callbacks.register_resource) {
        return kFfxErrorInvalidPointer;
    }
    return callbacks.register_resource(
        backend_interface, in_resource, effect_context_id, out_resource);
}

FfxResourceAbi WrappedEmbeddedGetResource(FfxInterfaceAbi* backend_interface, int32_t resource) {
    LogInit();
    OSSG_LOG_INFO("ffx_fsr3",
                  "embedded fpGetResource iface=%p internal=%d",
                  backend_interface,
                  resource);
    FfxInterfaceCallbacks callbacks;
    if (!LoadEmbeddedCallbacks(backend_interface, &callbacks) || !callbacks.get_resource) {
        return {};
    }
    const FfxResourceAbi out = callbacks.get_resource(backend_interface, resource);
    LogResourceSummary("embedded fpGetResource result", out);
    return out;
}

FfxErrorCode WrappedEmbeddedStageConstantBufferData(
    FfxInterfaceAbi* backend_interface,
    void* data,
    uint32_t size,
    FfxConstantBufferAbi* constant_buffer) {
    LogInit();
    float first_values[8] = {};
    uint32_t copied_floats = 0;
    if (data && size >= sizeof(float)) {
        __try {
            copied_floats = (std::min)(8u, size / static_cast<uint32_t>(sizeof(float)));
            std::memcpy(first_values, data, copied_floats * sizeof(float));
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            copied_floats = 0;
        }
    }
    OSSG_LOG_INFO("ffx_fsr3",
                  "embedded fpStageConstantBufferData iface=%p data=%p size=%u first_floats=%u [%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f]",
                  backend_interface,
                  data,
                  size,
                  copied_floats,
                  first_values[0],
                  first_values[1],
                  first_values[2],
                  first_values[3],
                  first_values[4],
                  first_values[5],
                  first_values[6],
                  first_values[7]);
    FfxInterfaceCallbacks callbacks;
    if (!LoadEmbeddedCallbacks(backend_interface, &callbacks) ||
        !callbacks.stage_constant_buffer_data) {
        return kFfxErrorInvalidPointer;
    }
    return callbacks.stage_constant_buffer_data(
        backend_interface, data, size, constant_buffer);
}

bool WrapEmbeddedInterface(FfxFsr3UpscalerContextDescriptionAbi* desc) {
    if (!desc) return false;

    FfxInterfaceCallbacks callbacks;
    callbacks.register_resource = desc->backend_interface.fp_register_resource;
    callbacks.get_resource = desc->backend_interface.fp_get_resource;
    callbacks.stage_constant_buffer_data =
        desc->backend_interface.fp_stage_constant_buffer_data;

    LogInit();
    OSSG_LOG_INFO("ffx_fsr3",
                  "Fsr3UpscalerContextDescription flags=0x%x max_render=%ux%u max_upscale=%ux%u device=%p scratch=%p scratch_size_or_legacy=%p register=%p get=%p stage_cb=%p legacy17=%p legacy18=%p",
                  desc->flags,
                  desc->max_render_size.width,
                  desc->max_render_size.height,
                  desc->max_upscale_size.width,
                  desc->max_upscale_size.height,
                  desc->backend_interface.fp_legacy18,
                  desc->backend_interface.fp_legacy17,
                  desc->backend_interface.fp_legacy18,
                  reinterpret_cast<void*>(callbacks.register_resource),
                  reinterpret_cast<void*>(callbacks.get_resource),
                  reinterpret_cast<void*>(callbacks.stage_constant_buffer_data),
                  desc->backend_interface.fp_legacy17,
                  desc->backend_interface.fp_legacy18);

    {
        std::lock_guard<std::mutex> lock(g_embedded_interface_mutex);
        g_embedded_interface_callbacks[&desc->backend_interface] = callbacks;
        g_last_embedded_interface_callbacks = callbacks;
        g_has_last_embedded_interface_callbacks = true;
    }

    if (callbacks.register_resource) {
        desc->backend_interface.fp_register_resource = WrappedEmbeddedRegisterResource;
    }
    if (callbacks.get_resource) {
        desc->backend_interface.fp_get_resource = WrappedEmbeddedGetResource;
    }
    if (callbacks.stage_constant_buffer_data) {
        desc->backend_interface.fp_stage_constant_buffer_data =
            WrappedEmbeddedStageConstantBufferData;
    }
    return callbacks.register_resource || callbacks.get_resource ||
           callbacks.stage_constant_buffer_data;
}

void LogResourceSummary(const char* name, const FfxResourceAbi& resource) {
    OSSG_LOG_INFO("ffx_fsr3",
                  "%s resource=%p type=%u format=%u size=%ux%u depth=%u mips=%u flags=%u usage=%u state=%u",
                  name,
                  resource.resource,
                  resource.description.type,
                  resource.description.format,
                  resource.description.width,
                  resource.description.height,
                  resource.description.depth,
                  resource.description.mip_count,
                  resource.description.flags,
                  resource.description.usage,
                  resource.state);
}

void LogDispatchSummary(const FfxFsr3DispatchUpscaleDescriptionAbi* desc) {
    FfxFsr3DispatchUpscaleDescriptionAbi copy = {};
    if (!SafeCopyDispatch(desc, &copy)) {
        OSSG_LOG_INFO("ffx_fsr3", "ffxFsr3ContextDispatchUpscale desc=<unreadable>");
        return;
    }

    OSSG_LOG_INFO("ffx_fsr3",
                  "ffxFsr3ContextDispatchUpscale cmd=%p render=%ux%u upscale=%ux%u jitter=(%.4f,%.4f) mv_scale=(%.4f,%.4f) sharpen=%u sharpness=%.4f dt=%.4f pre_exp=%.4f reset=%u flags=%u",
                  copy.command_list,
                  copy.render_size.width,
                  copy.render_size.height,
                  copy.upscale_size.width,
                  copy.upscale_size.height,
                  copy.jitter_offset.x,
                  copy.jitter_offset.y,
                  copy.motion_vector_scale.x,
                  copy.motion_vector_scale.y,
                  copy.enable_sharpening ? 1u : 0u,
                  copy.sharpness,
                  copy.frame_time_delta,
                  copy.pre_exposure,
                  copy.reset ? 1u : 0u,
                  copy.flags);
    LogResourceSummary("color", copy.color);
    LogResourceSummary("depth", copy.depth);
    LogResourceSummary("motion_vectors", copy.motion_vectors);
    LogResourceSummary("exposure", copy.exposure);
    LogResourceSummary("reactive", copy.reactive);
    LogResourceSummary("transparency", copy.transparency_and_composition);
    LogResourceSummary("dilated_depth", copy.dilated_depth);
    LogResourceSummary("dilated_motion_vectors", copy.dilated_motion_vectors);
    LogResourceSummary("reconstructed_prev_nearest_depth", copy.reconstructed_prev_nearest_depth);
    LogResourceSummary("upscale_output", copy.upscale_output);
}

void LogReactiveSummary(const FfxFsr3GenerateReactiveDescriptionAbi* desc) {
    FfxFsr3GenerateReactiveDescriptionAbi copy = {};
    if (!SafeCopyReactive(desc, &copy)) {
        OSSG_LOG_INFO("ffx_fsr3", "ffxFsr3ContextGenerateReactiveMask desc=<unreadable>");
        return;
    }
    OSSG_LOG_INFO("ffx_fsr3",
                  "ffxFsr3ContextGenerateReactiveMask cmd=%p render=%ux%u scale=%.4f cutoff=%.4f binary=%.4f flags=%u",
                  copy.command_list,
                  copy.render_size.width,
                  copy.render_size.height,
                  copy.scale,
                  copy.cutoff_threshold,
                  copy.binary_value,
                  copy.flags);
    LogResourceSummary("opaque_only", copy.color_opaque_only);
    LogResourceSummary("pre_upscale", copy.color_pre_upscale);
    LogResourceSummary("out_reactive", copy.out_reactive);
}

void EnsureNativeInit() {
    oss_gaussian::OssGaussianEnsureInitializedFromProxyExport();
}

uint32_t ResourceStateMask(const FfxFsr3DispatchUpscaleDescriptionAbi& desc) {
    uint32_t mask = 0u;
    if (desc.color.resource) mask |= 1u;
    if (desc.upscale_output.resource) mask |= 2u;
    if (desc.depth.resource) mask |= 4u;
    if (desc.motion_vectors.resource) mask |= 8u;
    if (desc.exposure.resource) mask |= 16u;
    return mask;
}

OssGaussianFrame MakeCaptureFrame(const FfxFsr3DispatchUpscaleDescriptionAbi& desc) {
    OssGaussianFrame frame{};
    frame.color = desc.color.resource;
    frame.output = desc.upscale_output.resource;
    frame.depth = desc.depth.resource;
    frame.motion_vectors = desc.motion_vectors.resource;
    frame.exposure_texture = desc.exposure.resource;
    frame.output_width = desc.upscale_size.width;
    frame.output_height = desc.upscale_size.height;
    frame.subrect_render_width = desc.render_size.width;
    frame.subrect_render_height = desc.render_size.height;
    frame.jitter_offset_x = desc.jitter_offset.x;
    frame.jitter_offset_y = desc.jitter_offset.y;
    frame.mv_scale_x = desc.motion_vector_scale.x;
    frame.mv_scale_y = desc.motion_vector_scale.y;
    frame.exposure_scale = desc.pre_exposure;
    frame.reset = desc.reset ? 1u : 0u;
    frame.feature_create_flags = desc.flags;
    frame.frame_index = g_dispatch_frame_index.fetch_add(1, std::memory_order_relaxed);
    frame.resource_states_valid = ResourceStateMask(desc);
    frame.color_state = desc.color.state;
    frame.output_state = desc.upscale_output.state;
    frame.depth_state = desc.depth.state;
    frame.motion_vectors_state = desc.motion_vectors.state;
    frame.exposure_texture_state = desc.exposure.state;
    return frame;
}

FfxErrorCode MissingRealFsr3(const char* fn) {
    LogInit();
    OSSG_LOG_ERROR("ffx_fsr3", "%s: real FSR3 export unavailable", fn);
    return kFfxErrorInvalidPointer;
}

} // namespace

void* ResolveExport(const char* name) {
    if (!name || !*name) return nullptr;
    const std::vector<RealFsr3Module>& modules = RealFsr3Modules();
    if (modules.empty()) return nullptr;
    const RealFsr3Kind preferred_kind = PreferredKindForExport(name);
    FARPROC proc = TryResolveFromKind(modules, preferred_kind, name);
    if (!proc) proc = TryResolveAny(modules, name);
    if (!proc) {
        LogInit();
        OSSG_LOG_ERROR("ffx_fsr3", "real FSR3 missing export: %s", name);
        return nullptr;
    }
    return reinterpret_cast<void*>(proc);
}

} // namespace oss_gaussian::ffx_fsr3

extern "C" {

#define OSSG_FFX_FWD_RET(RET, IMPL, NAME, SIG, ARGS, FAIL_VALUE)                 \
    RET IMPL SIG {                                                               \
        oss_gaussian::ffx_fsr3::EnsureNativeInit();                              \
        using Fn = RET (*) SIG;                                                   \
        auto real = oss_gaussian::ffx_fsr3::ResolveTyped<Fn>(NAME);              \
        if (!real) return FAIL_VALUE;                                             \
        return real ARGS;                                                         \
    }

#define OSSG_FFX_FWD_VOID(IMPL, NAME, SIG, ARGS)                                 \
    void IMPL SIG {                                                              \
        oss_gaussian::ffx_fsr3::EnsureNativeInit();                              \
        using Fn = void (*) SIG;                                                  \
        auto real = oss_gaussian::ffx_fsr3::ResolveTyped<Fn>(NAME);              \
        if (!real) {                                                             \
            oss_gaussian::ffx_fsr3::MissingRealFsr3(NAME);                       \
            return;                                                              \
        }                                                                        \
        real ARGS;                                                               \
    }

bool
OssgFfxAssertReport(const char* file, int32_t line, const char* condition, const char* msg) {
    using Fn = bool (*)(const char*, int32_t, const char*, const char*);
    auto real = oss_gaussian::ffx_fsr3::ResolveTyped<Fn>("ffxAssertReport");
    if (!real) return false;
    return real(file, line, condition, msg);
}

OSSG_FFX_FWD_VOID(OssgFfxAssertSetPrintingCallback, "ffxAssertSetPrintingCallback",
                  (void* callback), (callback))

oss_gaussian::ffx_fsr3::FfxErrorCode
OssgFfxFsr3ContextCreate(void* context, const void* context_description) {
    oss_gaussian::ffx_fsr3::EnsureNativeInit();
    oss_gaussian::LogInit();
    OSSG_LOG_INFO("ffx_fsr3", "ffxFsr3ContextCreate context=%p desc=%p",
                  context, context_description);
    using Fn = oss_gaussian::ffx_fsr3::FfxErrorCode (*)(void*, const void*);
    auto real = oss_gaussian::ffx_fsr3::ResolveTyped<Fn>("ffxFsr3ContextCreate");
    if (!real) return oss_gaussian::ffx_fsr3::MissingRealFsr3("ffxFsr3ContextCreate");
    const auto result = real(context, context_description);
    OSSG_LOG_INFO("ffx_fsr3", "ffxFsr3ContextCreate result=%d", result);
    return result;
}

oss_gaussian::ffx_fsr3::FfxErrorCode
OssgFfxFsr3ContextDestroy(void* context) {
    oss_gaussian::ffx_fsr3::EnsureNativeInit();
    oss_gaussian::LogInit();
    OSSG_LOG_INFO("ffx_fsr3", "ffxFsr3ContextDestroy context=%p", context);
    using Fn = oss_gaussian::ffx_fsr3::FfxErrorCode (*)(void*);
    auto real = oss_gaussian::ffx_fsr3::ResolveTyped<Fn>("ffxFsr3ContextDestroy");
    if (!real) return oss_gaussian::ffx_fsr3::MissingRealFsr3("ffxFsr3ContextDestroy");
    const auto result = real(context);
    OSSG_LOG_INFO("ffx_fsr3", "ffxFsr3ContextDestroy result=%d", result);
    return result;
}

oss_gaussian::ffx_fsr3::FfxErrorCode
OssgFfxFsr3ContextDispatchUpscale(
    void* context,
    const oss_gaussian::ffx_fsr3::FfxFsr3DispatchUpscaleDescriptionAbi* desc) {
    oss_gaussian::ffx_fsr3::EnsureNativeInit();
    oss_gaussian::LogInit();
    oss_gaussian::ffx_fsr3::LogDispatchSummary(desc);
    using Fn = oss_gaussian::ffx_fsr3::FfxErrorCode (*)(
        void*,
        const oss_gaussian::ffx_fsr3::FfxFsr3DispatchUpscaleDescriptionAbi*);
    auto real = oss_gaussian::ffx_fsr3::ResolveTyped<Fn>("ffxFsr3ContextDispatchUpscale");
    if (!real) {
        return oss_gaussian::ffx_fsr3::MissingRealFsr3("ffxFsr3ContextDispatchUpscale");
    }
    oss_gaussian::ffx_fsr3::FfxFsr3DispatchUpscaleDescriptionAbi copy = {};
    void* ticket = nullptr;
    if (oss_gaussian::ffx_fsr3::SafeCopyDispatch(desc, &copy)) {
        ticket = oss_gaussian::BeginUpscalerFrameCapture(
            "fsr3",
            reinterpret_cast<ID3D12GraphicsCommandList*>(copy.command_list),
            oss_gaussian::ffx_fsr3::MakeCaptureFrame(copy));
    }
    const auto result = real(context, desc);
    oss_gaussian::EndUpscalerFrameCapture(ticket, result == oss_gaussian::ffx_fsr3::kFfxOk, result);
    OSSG_LOG_INFO("ffx_fsr3", "ffxFsr3ContextDispatchUpscale result=%d", result);
    return result;
}

oss_gaussian::ffx_fsr3::FfxErrorCode
OssgFfxFsr3ContextGenerateReactiveMask(
    void* context,
    const oss_gaussian::ffx_fsr3::FfxFsr3GenerateReactiveDescriptionAbi* desc) {
    oss_gaussian::ffx_fsr3::EnsureNativeInit();
    oss_gaussian::LogInit();
    oss_gaussian::ffx_fsr3::LogReactiveSummary(desc);
    using Fn = oss_gaussian::ffx_fsr3::FfxErrorCode (*)(
        void*,
        const oss_gaussian::ffx_fsr3::FfxFsr3GenerateReactiveDescriptionAbi*);
    auto real = oss_gaussian::ffx_fsr3::ResolveTyped<Fn>("ffxFsr3ContextGenerateReactiveMask");
    if (!real) {
        return oss_gaussian::ffx_fsr3::MissingRealFsr3("ffxFsr3ContextGenerateReactiveMask");
    }
    return real(context, desc);
}

oss_gaussian::ffx_fsr3::FfxErrorCode
OssgFfxFsr3UpscalerContextCreate(void* context, const void* context_description) {
    oss_gaussian::ffx_fsr3::EnsureNativeInit();
    oss_gaussian::LogInit();
    OSSG_LOG_INFO("ffx_fsr3", "ffxFsr3UpscalerContextCreate context=%p desc=%p",
                  context, context_description);
    using Fn = oss_gaussian::ffx_fsr3::FfxErrorCode (*)(void*, const void*);
    auto real = oss_gaussian::ffx_fsr3::ResolveTyped<Fn>("ffxFsr3UpscalerContextCreate");
    if (!real) return oss_gaussian::ffx_fsr3::MissingRealFsr3("ffxFsr3UpscalerContextCreate");
    const auto result = real(context, context_description);
    OSSG_LOG_INFO("ffx_fsr3", "ffxFsr3UpscalerContextCreate result=%d", result);
    return result;
}

oss_gaussian::ffx_fsr3::FfxErrorCode
OssgFfxFsr3UpscalerContextDestroy(void* context) {
    oss_gaussian::ffx_fsr3::EnsureNativeInit();
    oss_gaussian::LogInit();
    OSSG_LOG_INFO("ffx_fsr3", "ffxFsr3UpscalerContextDestroy context=%p", context);
    using Fn = oss_gaussian::ffx_fsr3::FfxErrorCode (*)(void*);
    auto real = oss_gaussian::ffx_fsr3::ResolveTyped<Fn>("ffxFsr3UpscalerContextDestroy");
    if (!real) return oss_gaussian::ffx_fsr3::MissingRealFsr3("ffxFsr3UpscalerContextDestroy");
    const auto result = real(context);
    OSSG_LOG_INFO("ffx_fsr3", "ffxFsr3UpscalerContextDestroy result=%d", result);
    return result;
}

oss_gaussian::ffx_fsr3::FfxErrorCode
OssgFfxFsr3UpscalerContextDispatch(
    void* context,
    const oss_gaussian::ffx_fsr3::FfxFsr3DispatchUpscaleDescriptionAbi* desc) {
    oss_gaussian::ffx_fsr3::EnsureNativeInit();
    oss_gaussian::LogInit();
    oss_gaussian::ffx_fsr3::LogDispatchSummary(desc);
    using Fn = oss_gaussian::ffx_fsr3::FfxErrorCode (*)(
        void*,
        const oss_gaussian::ffx_fsr3::FfxFsr3DispatchUpscaleDescriptionAbi*);
    auto real = oss_gaussian::ffx_fsr3::ResolveTyped<Fn>("ffxFsr3UpscalerContextDispatch");
    if (!real) {
        return oss_gaussian::ffx_fsr3::MissingRealFsr3("ffxFsr3UpscalerContextDispatch");
    }
    oss_gaussian::ffx_fsr3::FfxFsr3DispatchUpscaleDescriptionAbi copy = {};
    void* ticket = nullptr;
    if (oss_gaussian::ffx_fsr3::SafeCopyDispatch(desc, &copy)) {
        ticket = oss_gaussian::BeginUpscalerFrameCapture(
            "fsr3",
            reinterpret_cast<ID3D12GraphicsCommandList*>(copy.command_list),
            oss_gaussian::ffx_fsr3::MakeCaptureFrame(copy));
    }
    const auto result = real(context, desc);
    oss_gaussian::EndUpscalerFrameCapture(ticket, result == oss_gaussian::ffx_fsr3::kFfxOk, result);
    OSSG_LOG_INFO("ffx_fsr3", "ffxFsr3UpscalerContextDispatch result=%d", result);
    return result;
}

oss_gaussian::ffx_fsr3::FfxErrorCode
OssgFfxFsr3UpscalerContextGenerateReactiveMask(
    void* context,
    const oss_gaussian::ffx_fsr3::FfxFsr3GenerateReactiveDescriptionAbi* desc) {
    oss_gaussian::ffx_fsr3::EnsureNativeInit();
    oss_gaussian::LogInit();
    oss_gaussian::ffx_fsr3::LogReactiveSummary(desc);
    using Fn = oss_gaussian::ffx_fsr3::FfxErrorCode (*)(
        void*,
        const oss_gaussian::ffx_fsr3::FfxFsr3GenerateReactiveDescriptionAbi*);
    auto real = oss_gaussian::ffx_fsr3::ResolveTyped<Fn>("ffxFsr3UpscalerContextGenerateReactiveMask");
    if (!real) {
        return oss_gaussian::ffx_fsr3::MissingRealFsr3("ffxFsr3UpscalerContextGenerateReactiveMask");
    }
    return real(context, desc);
}

OSSG_FFX_FWD_RET(float, OssgFfxFsr3GetUpscaleRatioFromQualityMode,
                 "ffxFsr3GetUpscaleRatioFromQualityMode",
                 (int32_t quality_mode), (quality_mode), 0.0f)

OSSG_FFX_FWD_RET(float, OssgFfxFsr3UpscalerGetUpscaleRatioFromQualityMode,
                 "ffxFsr3UpscalerGetUpscaleRatioFromQualityMode",
                 (int32_t quality_mode), (quality_mode), 0.0f)

OSSG_FFX_FWD_RET(oss_gaussian::ffx_fsr3::FfxErrorCode,
                 OssgFfxFsr3GetRenderResolutionFromQualityMode,
                 "ffxFsr3GetRenderResolutionFromQualityMode",
                 (uint32_t* render_width,
                  uint32_t* render_height,
                  uint32_t display_width,
                  uint32_t display_height,
                  int32_t quality_mode),
                 (render_width, render_height, display_width, display_height, quality_mode),
                 oss_gaussian::ffx_fsr3::kFfxErrorInvalidPointer)

OSSG_FFX_FWD_RET(oss_gaussian::ffx_fsr3::FfxErrorCode,
                 OssgFfxFsr3UpscalerGetRenderResolutionFromQualityMode,
                 "ffxFsr3UpscalerGetRenderResolutionFromQualityMode",
                 (uint32_t* render_width,
                  uint32_t* render_height,
                  uint32_t display_width,
                  uint32_t display_height,
                  int32_t quality_mode),
                 (render_width, render_height, display_width, display_height, quality_mode),
                 oss_gaussian::ffx_fsr3::kFfxErrorInvalidPointer)

OSSG_FFX_FWD_RET(int32_t, OssgFfxFsr3GetJitterPhaseCount,
                 "ffxFsr3GetJitterPhaseCount",
                 (int32_t render_width, int32_t display_width),
                 (render_width, display_width), 0)

OSSG_FFX_FWD_RET(int32_t, OssgFfxFsr3UpscalerGetJitterPhaseCount,
                 "ffxFsr3UpscalerGetJitterPhaseCount",
                 (int32_t render_width, int32_t display_width),
                 (render_width, display_width), 0)

OSSG_FFX_FWD_RET(oss_gaussian::ffx_fsr3::FfxErrorCode, OssgFfxFsr3GetJitterOffset,
                 "ffxFsr3GetJitterOffset",
                 (float* out_x, float* out_y, int32_t index, int32_t phase_count),
                 (out_x, out_y, index, phase_count),
                 oss_gaussian::ffx_fsr3::kFfxErrorInvalidPointer)

OSSG_FFX_FWD_RET(oss_gaussian::ffx_fsr3::FfxErrorCode, OssgFfxFsr3UpscalerGetJitterOffset,
                 "ffxFsr3UpscalerGetJitterOffset",
                 (float* out_x, float* out_y, int32_t index, int32_t phase_count),
                 (out_x, out_y, index, phase_count),
                 oss_gaussian::ffx_fsr3::kFfxErrorInvalidPointer)

OSSG_FFX_FWD_RET(bool, OssgFfxFsr3ResourceIsNull, "ffxFsr3ResourceIsNull",
                 (const oss_gaussian::ffx_fsr3::FfxResourceAbi* resource),
                 (resource), true)

OSSG_FFX_FWD_RET(bool, OssgFfxFsr3UpscalerResourceIsNull, "ffxFsr3UpscalerResourceIsNull",
                 (const oss_gaussian::ffx_fsr3::FfxResourceAbi* resource),
                 (resource), true)

OSSG_FFX_FWD_RET(oss_gaussian::ffx_fsr3::FfxErrorCode,
                 OssgFfxFsr3UpscalerGetSharedResourceDescriptions,
                 "ffxFsr3UpscalerGetSharedResourceDescriptions",
                 (void* context, void* descriptions),
                 (context, descriptions),
                 oss_gaussian::ffx_fsr3::kFfxErrorInvalidPointer)

OSSG_FFX_FWD_RET(oss_gaussian::ffx_fsr3::FfxErrorCode,
                 OssgFfxFsr3ConfigureFrameGeneration,
                 "ffxFsr3ConfigureFrameGeneration",
                 (void* context, const void* config),
                 (context, config),
                 oss_gaussian::ffx_fsr3::kFfxErrorInvalidPointer)

OSSG_FFX_FWD_RET(oss_gaussian::ffx_fsr3::FfxErrorCode,
                 OssgFfxFsr3DispatchFrameGeneration,
                 "ffxFsr3DispatchFrameGeneration",
                 (const void* desc),
                 (desc),
                 oss_gaussian::ffx_fsr3::kFfxErrorInvalidPointer)

OSSG_FFX_FWD_RET(bool, OssgFfxFsr3SkipPresent, "ffxFsr3SkipPresent",
                 (void* swapchain), (swapchain), false)

OSSG_FFX_FWD_VOID(OssgFfxSafeReleaseCopyResource, "ffxSafeReleaseCopyResource",
                  (void* backend_interface, uint32_t resource_id, uint32_t effect_context_id),
                  (backend_interface, resource_id, effect_context_id))

OSSG_FFX_FWD_VOID(OssgFfxSafeReleasePipeline, "ffxSafeReleasePipeline",
                  (void* backend_interface, void* pipeline, uint32_t effect_context_id),
                  (backend_interface, pipeline, effect_context_id))

OSSG_FFX_FWD_VOID(OssgFfxSafeReleaseResource, "ffxSafeReleaseResource",
                  (void* backend_interface, uint32_t resource_id, uint32_t effect_context_id),
                  (backend_interface, resource_id, effect_context_id))

} // extern "C"
