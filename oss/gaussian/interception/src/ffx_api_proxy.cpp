// =============================================================================
//  ffx_api_proxy.cpp
//
//  Pass-through proxy for AMD's generic FidelityFX SDK loader API.
// =============================================================================
#define OSS_GAUSSIAN_BUILDING_DLL 1

#include "ffx_api_proxy.h"

#include "log.h"
#include "ngx_frame_capture.h"

#include <Windows.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace oss_gaussian {
void OssGaussianEnsureInitializedFromProxyExport();
}

namespace oss_gaussian::ffx_api {

using ffxContext = void*;
using ffxReturnCode_t = uint32_t;
constexpr ffxReturnCode_t kFfxApiReturnErrorParameter = 6u;
constexpr ffxReturnCode_t kFfxApiReturnOk = 0u;
constexpr uint64_t kFfxApiCreateContextDescTypeUpscale = 0x00010000u;
constexpr uint64_t kFfxApiDispatchDescTypeUpscale = 0x00010001u;

struct ffxApiHeader {
    uint64_t type;
    ffxApiHeader* p_next;
};

struct ffxAllocationCallbacks {
    void* user_data;
    void* alloc;
    void* dealloc;
};

struct FfxApiDimensions2D {
    uint32_t width;
    uint32_t height;
};

struct FfxApiFloatCoords2D {
    float x;
    float y;
};

struct FfxApiResourceDescription {
    uint32_t type;
    uint32_t format;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t mip_count;
    uint32_t flags;
    uint32_t usage;
};

struct FfxApiResource {
    void* resource;
    FfxApiResourceDescription description;
    uint32_t state;
};

struct ffxCreateContextDescUpscale {
    ffxApiHeader header;
    uint32_t flags;
    FfxApiDimensions2D max_render_size;
    FfxApiDimensions2D max_upscale_size;
    void* message_callback;
};

struct ffxDispatchDescUpscale {
    ffxApiHeader header;
    void* command_list;
    FfxApiResource color;
    FfxApiResource depth;
    FfxApiResource motion_vectors;
    FfxApiResource exposure;
    FfxApiResource reactive;
    FfxApiResource transparency_and_composition;
    FfxApiResource output;
    FfxApiFloatCoords2D jitter_offset;
    FfxApiFloatCoords2D motion_vector_scale;
    FfxApiDimensions2D render_size;
    FfxApiDimensions2D upscale_size;
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

static_assert(sizeof(FfxApiResourceDescription) == 32);
static_assert(sizeof(FfxApiResource) == 48);
static_assert(sizeof(ffxDispatchDescUpscale) == 432);
static_assert(offsetof(ffxDispatchDescUpscale, command_list) == 16);
static_assert(offsetof(ffxDispatchDescUpscale, color) == 24);
static_assert(offsetof(ffxDispatchDescUpscale, output) == 312);
static_assert(offsetof(ffxDispatchDescUpscale, flags) == 428);

namespace {

constexpr wchar_t kRealDllName[] = L"oss_amd_fidelityfx_dx12_real.dll";
constexpr wchar_t kBackupDllName[] = L"amd_fidelityfx_dx12.DLL.oss-backup";
constexpr wchar_t kBackupDllNameLower[] = L"amd_fidelityfx_dx12.dll.oss-backup";
constexpr wchar_t kOriginalDllName[] = L"amd_fidelityfx_dx12.DLL";
constexpr wchar_t kOriginalDllNameLower[] = L"amd_fidelityfx_dx12.dll";

std::once_flag g_load_once;
HMODULE g_real_api = nullptr;
std::atomic<uint64_t> g_create_calls{0};
std::atomic<uint64_t> g_configure_calls{0};
std::atomic<uint64_t> g_query_calls{0};
std::atomic<uint64_t> g_dispatch_calls{0};
std::atomic<uint64_t> g_destroy_calls{0};
std::atomic<uint64_t> g_upscale_frame_index{1};
std::mutex g_context_mu;
std::unordered_map<void*, uint32_t> g_context_create_flags;

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

void PushEnvOverride(std::vector<std::wstring>& candidates) {
    wchar_t value[MAX_PATH] = {};
    const DWORD len = GetEnvironmentVariableW(L"OSS_AMD_FIDELITYFX_DX12_REAL_DLL",
                                              value,
                                              MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        candidates.emplace_back(value);
    }
}

void PushDirectoryCandidates(std::vector<std::wstring>& candidates, const std::wstring& dir) {
    candidates.push_back(JoinPath(dir, kRealDllName));
    candidates.push_back(JoinPath(dir, kBackupDllName));
    candidates.push_back(JoinPath(dir, kBackupDllNameLower));
    candidates.push_back(JoinPath(dir, kOriginalDllName));
    candidates.push_back(JoinPath(dir, kOriginalDllNameLower));
}

std::vector<std::wstring> BuildCandidatePaths() {
    std::vector<std::wstring> candidates;
    PushEnvOverride(candidates);

    wchar_t exe_path[MAX_PATH] = {};
    const DWORD exe_len = GetModuleFileNameW(nullptr, exe_path, MAX_PATH);
    if (exe_len > 0 && exe_len < MAX_PATH) {
        PushDirectoryCandidates(candidates, DirectoryOf(exe_path));
    }

    wchar_t cwd[MAX_PATH] = {};
    const DWORD cwd_len = GetCurrentDirectoryW(MAX_PATH, cwd);
    if (cwd_len > 0 && cwd_len < MAX_PATH) {
        PushDirectoryCandidates(candidates, cwd);
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
        OSSG_LOG_WARN("ffx_api", "failed to load real amd_fidelityfx_dx12 candidate err=%lu",
                      GetLastError());
        return nullptr;
    }

    const std::wstring loaded_path = ModulePath(module);
    if (!self_path.empty() && !loaded_path.empty() && SamePath(loaded_path, self_path)) {
        FreeLibrary(module);
        return nullptr;
    }
    return module;
}

void LoadRealApiOnce() {
    LogInit();
    const std::wstring self_path = SelfModulePath();
    for (const std::wstring& candidate : BuildCandidatePaths()) {
        if (HMODULE module = TryLoadCandidate(candidate, self_path)) {
            g_real_api = module;
            OSSG_LOG_INFO("ffx_api", "loaded real amd_fidelityfx_dx12.dll");
            return;
        }
    }
    OSSG_LOG_ERROR("ffx_api", "real amd_fidelityfx_dx12.dll not found");
}

HMODULE RealApiModule() {
    std::call_once(g_load_once, LoadRealApiOnce);
    return g_real_api;
}

bool SafeCopyHeader(const ffxApiHeader* src, ffxApiHeader* dst) {
    if (!src || !dst) return false;
    __try {
        *dst = *src;
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

bool SafeCopyUpscale(const ffxDispatchDescUpscale* src, ffxDispatchDescUpscale* dst) {
    if (!src || !dst) return false;
    __try {
        *dst = *src;
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

bool SafeCopyCreateUpscale(const ffxCreateContextDescUpscale* src,
                           ffxCreateContextDescUpscale* dst) {
    if (!src || !dst) return false;
    __try {
        *dst = *src;
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

bool ShouldLog(uint64_t index) {
    return index <= 64 || (index % 120) == 0;
}

void LogResourceSummary(const char* name, const FfxApiResource& resource) {
    OSSG_LOG_INFO("ffx_api",
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

void LogHeaderChain(const char* fn, const ffxApiHeader* header) {
    if (!header) {
        OSSG_LOG_INFO("ffx_api", "%s header=<null>", fn);
        return;
    }

    const ffxApiHeader* cursor = header;
    for (uint32_t depth = 0; depth < 8 && cursor; ++depth) {
        ffxApiHeader copy = {};
        if (!SafeCopyHeader(cursor, &copy)) {
            OSSG_LOG_WARN("ffx_api", "%s header[%u]=<unreadable:%p>", fn, depth, cursor);
            return;
        }
        OSSG_LOG_INFO("ffx_api",
                      "%s header[%u] ptr=%p type=%llu p_next=%p",
                      fn,
                      depth,
                      cursor,
                      static_cast<unsigned long long>(copy.type),
                      copy.p_next);
        cursor = copy.p_next;
    }
}

bool CopyUpscaleDispatch(const ffxApiHeader* desc, ffxDispatchDescUpscale* out) {
    if (!desc || !out) return false;
    ffxApiHeader header = {};
    if (!SafeCopyHeader(desc, &header)) return false;
    if (header.type != kFfxApiDispatchDescTypeUpscale) return false;
    return SafeCopyUpscale(reinterpret_cast<const ffxDispatchDescUpscale*>(desc), out);
}

bool CopyUpscaleCreateDesc(const ffxApiHeader* desc, ffxCreateContextDescUpscale* out) {
    if (!desc || !out) return false;
    const ffxApiHeader* cursor = desc;
    for (uint32_t depth = 0; depth < 8 && cursor; ++depth) {
        ffxApiHeader header = {};
        if (!SafeCopyHeader(cursor, &header)) return false;
        if (header.type == kFfxApiCreateContextDescTypeUpscale) {
            return SafeCopyCreateUpscale(
                reinterpret_cast<const ffxCreateContextDescUpscale*>(cursor), out);
        }
        cursor = header.p_next;
    }
    return false;
}

void RememberContextCreateFlags(ffxContext* context,
                                const ffxCreateContextDescUpscale& desc) {
    if (!context || !*context) return;
    std::lock_guard<std::mutex> lock(g_context_mu);
    g_context_create_flags[*context] = desc.flags;
}

uint32_t ContextCreateFlags(ffxContext* context) {
    if (!context || !*context) return 0u;
    std::lock_guard<std::mutex> lock(g_context_mu);
    auto it = g_context_create_flags.find(*context);
    return it == g_context_create_flags.end() ? 0u : it->second;
}

void ForgetContext(ffxContext* context) {
    if (!context || !*context) return;
    std::lock_guard<std::mutex> lock(g_context_mu);
    g_context_create_flags.erase(*context);
}

void LogUpscaleDispatchSummary(const ffxDispatchDescUpscale& desc) {
    OSSG_LOG_INFO("ffx_api",
                  "ffxDispatch upscale cmd=%p render=%ux%u upscale=%ux%u jitter=(%.4f,%.4f) mv_scale=(%.4f,%.4f) sharpen=%u sharpness=%.4f dt=%.4f pre_exp=%.4f reset=%u flags=%u",
                  desc.command_list,
                  desc.render_size.width,
                  desc.render_size.height,
                  desc.upscale_size.width,
                  desc.upscale_size.height,
                  desc.jitter_offset.x,
                  desc.jitter_offset.y,
                  desc.motion_vector_scale.x,
                  desc.motion_vector_scale.y,
                  desc.enable_sharpening ? 1u : 0u,
                  desc.sharpness,
                  desc.frame_time_delta,
                  desc.pre_exposure,
                  desc.reset ? 1u : 0u,
                  desc.flags);
    LogResourceSummary("color", desc.color);
    LogResourceSummary("depth", desc.depth);
    LogResourceSummary("motion_vectors", desc.motion_vectors);
    LogResourceSummary("exposure", desc.exposure);
    LogResourceSummary("reactive", desc.reactive);
    LogResourceSummary("transparency", desc.transparency_and_composition);
    LogResourceSummary("output", desc.output);
}

uint32_t ResourceStateMask(const ffxDispatchDescUpscale& desc) {
    uint32_t mask = 0u;
    if (desc.color.resource) mask |= 1u;
    if (desc.output.resource) mask |= 2u;
    if (desc.depth.resource) mask |= 4u;
    if (desc.motion_vectors.resource) mask |= 8u;
    if (desc.exposure.resource) mask |= 16u;
    return mask;
}

OssGaussianFrame MakeCaptureFrame(const ffxDispatchDescUpscale& desc, uint32_t create_flags) {
    OssGaussianFrame frame{};
    frame.frame_index = g_upscale_frame_index.fetch_add(1, std::memory_order_relaxed);
    frame.color = desc.color.resource;
    frame.output = desc.output.resource;
    frame.depth = desc.depth.resource;
    frame.motion_vectors = desc.motion_vectors.resource;
    frame.exposure_texture = desc.exposure.resource;
    frame.jitter_offset_x = desc.jitter_offset.x;
    frame.jitter_offset_y = desc.jitter_offset.y;
    frame.mv_scale_x = desc.motion_vector_scale.x;
    frame.mv_scale_y = desc.motion_vector_scale.y;
    frame.exposure_scale = desc.pre_exposure;
    frame.reset = desc.reset ? 1u : 0u;
    frame.feature_create_flags = create_flags;
    frame.subrect_render_width = desc.render_size.width ? desc.render_size.width
                                                        : desc.color.description.width;
    frame.subrect_render_height = desc.render_size.height ? desc.render_size.height
                                                          : desc.color.description.height;
    frame.output_width = desc.upscale_size.width ? desc.upscale_size.width
                                                 : desc.output.description.width;
    frame.output_height = desc.upscale_size.height ? desc.upscale_size.height
                                                   : desc.output.description.height;
    frame.resource_states_valid = ResourceStateMask(desc);
    frame.color_state = desc.color.state;
    frame.output_state = desc.output.state;
    frame.depth_state = desc.depth.state;
    frame.motion_vectors_state = desc.motion_vectors.state;
    frame.exposure_texture_state = desc.exposure.state;
    return frame;
}

void EnsureNativeInit() {
    oss_gaussian::OssGaussianEnsureInitializedFromProxyExport();
}

ffxReturnCode_t MissingReal(const char* fn) {
    LogInit();
    OSSG_LOG_ERROR("ffx_api", "%s: real amd_fidelityfx_dx12 export unavailable", fn);
    return kFfxApiReturnErrorParameter;
}

} // namespace

void* ResolveExport(const char* name) {
    if (!name || !*name) return nullptr;
    HMODULE module = RealApiModule();
    if (!module) return nullptr;
    FARPROC proc = GetProcAddress(module, name);
    if (!proc) {
        LogInit();
        OSSG_LOG_ERROR("ffx_api", "real amd_fidelityfx_dx12 missing export: %s", name);
        return nullptr;
    }
    return reinterpret_cast<void*>(proc);
}

} // namespace oss_gaussian::ffx_api

extern "C" {

oss_gaussian::ffx_api::ffxReturnCode_t
OssgFfxApiCreateContext(oss_gaussian::ffx_api::ffxContext* context,
                        oss_gaussian::ffx_api::ffxApiHeader* desc,
                        const oss_gaussian::ffx_api::ffxAllocationCallbacks* mem_cb) {
    oss_gaussian::LogInit();
    const uint64_t call = ++oss_gaussian::ffx_api::g_create_calls;
    OSSG_LOG_INFO("ffx_api",
                  "ffxCreateContext call=%llu context=%p mem_cb=%p",
                  static_cast<unsigned long long>(call),
                  context,
                  mem_cb);
    oss_gaussian::ffx_api::LogHeaderChain("ffxCreateContext", desc);
    oss_gaussian::ffx_api::ffxCreateContextDescUpscale create_upscale = {};
    const bool has_upscale_create =
        oss_gaussian::ffx_api::CopyUpscaleCreateDesc(desc, &create_upscale);
    if (has_upscale_create) {
        OSSG_LOG_INFO("ffx_api",
                      "ffxCreateContext upscale flags=%u max_render=%ux%u max_upscale=%ux%u",
                      create_upscale.flags,
                      create_upscale.max_render_size.width,
                      create_upscale.max_render_size.height,
                      create_upscale.max_upscale_size.width,
                      create_upscale.max_upscale_size.height);
    }
    using Fn = oss_gaussian::ffx_api::ffxReturnCode_t (*)(
        oss_gaussian::ffx_api::ffxContext*,
        oss_gaussian::ffx_api::ffxApiHeader*,
        const oss_gaussian::ffx_api::ffxAllocationCallbacks*);
    auto real = oss_gaussian::ffx_api::ResolveTyped<Fn>("ffxCreateContext");
    if (!real) return oss_gaussian::ffx_api::MissingReal("ffxCreateContext");
    const auto result = real(context, desc, mem_cb);
    if (result == oss_gaussian::ffx_api::kFfxApiReturnOk && has_upscale_create) {
        oss_gaussian::ffx_api::RememberContextCreateFlags(context, create_upscale);
    }
    return result;
}

oss_gaussian::ffx_api::ffxReturnCode_t
OssgFfxApiDestroyContext(oss_gaussian::ffx_api::ffxContext* context,
                         const oss_gaussian::ffx_api::ffxAllocationCallbacks* mem_cb) {
    oss_gaussian::LogInit();
    const uint64_t call = ++oss_gaussian::ffx_api::g_destroy_calls;
    OSSG_LOG_INFO("ffx_api",
                  "ffxDestroyContext call=%llu context=%p mem_cb=%p",
                  static_cast<unsigned long long>(call),
                  context,
                  mem_cb);
    using Fn = oss_gaussian::ffx_api::ffxReturnCode_t (*)(
        oss_gaussian::ffx_api::ffxContext*,
        const oss_gaussian::ffx_api::ffxAllocationCallbacks*);
    auto real = oss_gaussian::ffx_api::ResolveTyped<Fn>("ffxDestroyContext");
    if (!real) return oss_gaussian::ffx_api::MissingReal("ffxDestroyContext");
    const auto result = real(context, mem_cb);
    if (result == oss_gaussian::ffx_api::kFfxApiReturnOk) {
        oss_gaussian::ffx_api::ForgetContext(context);
    }
    return result;
}

oss_gaussian::ffx_api::ffxReturnCode_t
OssgFfxApiConfigure(oss_gaussian::ffx_api::ffxContext* context,
                    const oss_gaussian::ffx_api::ffxApiHeader* desc) {
    oss_gaussian::LogInit();
    const uint64_t call = ++oss_gaussian::ffx_api::g_configure_calls;
    if (oss_gaussian::ffx_api::ShouldLog(call)) {
        OSSG_LOG_INFO("ffx_api",
                      "ffxConfigure call=%llu context=%p",
                      static_cast<unsigned long long>(call),
                      context);
        oss_gaussian::ffx_api::LogHeaderChain("ffxConfigure", desc);
    }
    using Fn = oss_gaussian::ffx_api::ffxReturnCode_t (*)(
        oss_gaussian::ffx_api::ffxContext*,
        const oss_gaussian::ffx_api::ffxApiHeader*);
    auto real = oss_gaussian::ffx_api::ResolveTyped<Fn>("ffxConfigure");
    if (!real) return oss_gaussian::ffx_api::MissingReal("ffxConfigure");
    return real(context, desc);
}

oss_gaussian::ffx_api::ffxReturnCode_t
OssgFfxApiQuery(oss_gaussian::ffx_api::ffxContext* context,
                oss_gaussian::ffx_api::ffxApiHeader* desc) {
    oss_gaussian::LogInit();
    const uint64_t call = ++oss_gaussian::ffx_api::g_query_calls;
    if (oss_gaussian::ffx_api::ShouldLog(call)) {
        OSSG_LOG_INFO("ffx_api",
                      "ffxQuery call=%llu context=%p",
                      static_cast<unsigned long long>(call),
                      context);
        oss_gaussian::ffx_api::LogHeaderChain("ffxQuery", desc);
    }
    using Fn = oss_gaussian::ffx_api::ffxReturnCode_t (*)(
        oss_gaussian::ffx_api::ffxContext*,
        oss_gaussian::ffx_api::ffxApiHeader*);
    auto real = oss_gaussian::ffx_api::ResolveTyped<Fn>("ffxQuery");
    if (!real) return oss_gaussian::ffx_api::MissingReal("ffxQuery");
    return real(context, desc);
}

oss_gaussian::ffx_api::ffxReturnCode_t
OssgFfxApiDispatch(oss_gaussian::ffx_api::ffxContext* context,
                   const oss_gaussian::ffx_api::ffxApiHeader* desc) {
    oss_gaussian::ffx_api::EnsureNativeInit();
    oss_gaussian::LogInit();
    const uint64_t call = ++oss_gaussian::ffx_api::g_dispatch_calls;
    oss_gaussian::ffx_api::ffxDispatchDescUpscale upscale = {};
    const bool has_upscale =
        oss_gaussian::ffx_api::CopyUpscaleDispatch(desc, &upscale);
    if (oss_gaussian::ffx_api::ShouldLog(call)) {
        OSSG_LOG_INFO("ffx_api",
                      "ffxDispatch call=%llu context=%p",
                      static_cast<unsigned long long>(call),
                      context);
        oss_gaussian::ffx_api::LogHeaderChain("ffxDispatch", desc);
        if (has_upscale) {
            oss_gaussian::ffx_api::LogUpscaleDispatchSummary(upscale);
        }
    }
    using Fn = oss_gaussian::ffx_api::ffxReturnCode_t (*)(
        oss_gaussian::ffx_api::ffxContext*,
        const oss_gaussian::ffx_api::ffxApiHeader*);
    auto real = oss_gaussian::ffx_api::ResolveTyped<Fn>("ffxDispatch");
    if (!real) return oss_gaussian::ffx_api::MissingReal("ffxDispatch");
    void* ticket = nullptr;
    if (has_upscale) {
        ticket = oss_gaussian::BeginUpscalerFrameCapture(
            "ffx-api",
            reinterpret_cast<ID3D12GraphicsCommandList*>(upscale.command_list),
            oss_gaussian::ffx_api::MakeCaptureFrame(
                upscale,
                oss_gaussian::ffx_api::ContextCreateFlags(context)));
    }
    const auto result = real(context, desc);
    oss_gaussian::EndUpscalerFrameCapture(
        ticket,
        result == oss_gaussian::ffx_api::kFfxApiReturnOk,
        static_cast<int>(result));
    if (has_upscale) {
        OSSG_LOG_INFO("ffx_api", "ffxDispatch upscale result=%u", result);
    }
    return result;
}

} // extern "C"
