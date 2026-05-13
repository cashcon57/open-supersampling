// =============================================================================
//  sl_interposer_proxy.cpp
//
//  Minimal Streamline proxy for DLSS-era titles that import sl.interposer.dll.
//  Cyberpunk 2077 imports Streamline directly, so replacing dxgi.dll is the
//  wrong first loader surface on modded installs that already use ReShade.
// =============================================================================
#define OSS_GAUSSIAN_BUILDING_DLL 1

#include "sl_interposer_proxy.h"

#include "log.h"
#include "ngx_frame_capture.h"

#include <Windows.h>

#include <cstring>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include <atomic>

namespace oss_gaussian {
void OssGaussianEnsureInitializedFromProxyExport();
}

namespace oss_gaussian::sl_proxy {

struct SlBaseStructureAbi {
    void* next;
    uint32_t type_data1;
    uint16_t type_data2;
    uint16_t type_data3;
    uint8_t type_data4[8];
    size_t struct_version;
};

struct SlExtentAbi {
    uint32_t top;
    uint32_t left;
    uint32_t width;
    uint32_t height;
};

struct SlResourceAbi {
    SlBaseStructureAbi base;
    uint32_t type;
    uint32_t _pad0;
    void* native;
    void* memory;
    void* view;
    uint32_t state;
    uint32_t width;
    uint32_t height;
    uint32_t native_format;
};

struct SlResourceTagAbi {
    SlBaseStructureAbi base;
    SlResourceAbi* resource;
    uint32_t type;
    uint32_t lifecycle;
    SlExtentAbi extent;
};

static_assert(sizeof(SlBaseStructureAbi) == 32, "unexpected Streamline BaseStructure ABI");

struct SlFloat2Abi {
    float x;
    float y;
};

struct SlFloat3Abi {
    float x;
    float y;
    float z;
};

struct SlConstantsAbi {
    SlBaseStructureAbi base;
    float camera_view_to_clip[16];
    float clip_to_camera_view[16];
    float clip_to_lens_clip[16];
    float clip_to_prev_clip[16];
    float prev_clip_to_clip[16];
    SlFloat2Abi jitter_offset;
    SlFloat2Abi mvec_scale;
    SlFloat2Abi camera_pinhole_offset;
    SlFloat3Abi camera_pos;
    SlFloat3Abi camera_up;
    SlFloat3Abi camera_right;
    SlFloat3Abi camera_fwd;
    float camera_near;
    float camera_far;
    float camera_fov;
    float camera_aspect_ratio;
    float motion_vectors_invalid_value;
    uint8_t depth_inverted;
    uint8_t camera_motion_included;
    uint8_t motion_vectors_3d;
    uint8_t reset;
    uint8_t orthographic_projection;
    uint8_t motion_vectors_dilated;
    uint8_t motion_vectors_jittered;
};

namespace {

constexpr wchar_t kRealSlDllName[] = L"oss_sl_real.dll";
constexpr wchar_t kOriginalSlDllName[] = L"sl.interposer.dll";
constexpr uint32_t kFeatureDLSS = 0;
constexpr uint32_t kBufferTypeDepth = 0;
constexpr uint32_t kBufferTypeMotionVectors = 1;
constexpr uint32_t kBufferTypeScalingInputColor = 3;
constexpr uint32_t kBufferTypeScalingOutputColor = 4;
constexpr uint32_t kBufferTypeExposure = 13;
constexpr uint32_t kStateColor = 1u << 0u;
constexpr uint32_t kStateOutput = 1u << 1u;
constexpr uint32_t kStateDepth = 1u << 2u;
constexpr uint32_t kStateMotionVectors = 1u << 3u;
constexpr uint32_t kStateExposure = 1u << 4u;
constexpr oss_gaussian::sl_proxy::SlBaseStructureAbi kResourceTagType = {
    nullptr, 0x4c6a5aad, 0xb445, 0x496c,
    {0x87, 0xff, 0x1a, 0xf3, 0x84, 0x5b, 0xe6, 0x53}, 1};
constexpr oss_gaussian::sl_proxy::SlBaseStructureAbi kViewportHandleType = {
    nullptr, 0x171b6435, 0x9b3c, 0x4fc8,
    {0x99, 0x94, 0xfb, 0xe5, 0x25, 0x69, 0xaa, 0xa4}, 1};
std::once_flag g_load_once;
HMODULE g_real_sl = nullptr;
std::atomic<uint64_t> g_sl_frame_counter{0};

struct SlTagSnapshot {
    const void* frame = nullptr;
    const void* viewport = nullptr;
    std::vector<oss_gaussian::sl_proxy::SlResourceTagAbi> tags;
    oss_gaussian::sl_proxy::SlConstantsAbi constants{};
    bool has_constants = false;
};

std::mutex g_tag_mu;
std::vector<SlTagSnapshot> g_tag_snapshots;

bool SameStructType(const oss_gaussian::sl_proxy::SlBaseStructureAbi* value,
                    const oss_gaussian::sl_proxy::SlBaseStructureAbi& type) {
    if (!value) return false;
    return value->type_data1 == type.type_data1 &&
           value->type_data2 == type.type_data2 &&
           value->type_data3 == type.type_data3 &&
           std::memcmp(value->type_data4, type.type_data4, sizeof(type.type_data4)) == 0;
}

SlTagSnapshot& FindOrCreateSnapshotLocked(const void* frame, const void* viewport) {
    for (SlTagSnapshot& snapshot : g_tag_snapshots) {
        if (snapshot.frame == frame && snapshot.viewport == viewport) {
            return snapshot;
        }
    }
    if (g_tag_snapshots.size() >= 64) {
        g_tag_snapshots.erase(g_tag_snapshots.begin());
    }
    g_tag_snapshots.push_back(SlTagSnapshot{});
    SlTagSnapshot& snapshot = g_tag_snapshots.back();
    snapshot.frame = frame;
    snapshot.viewport = viewport;
    return snapshot;
}

void StoreTags(const void* frame,
               const void* viewport,
               const oss_gaussian::sl_proxy::SlResourceTagAbi* tags,
               uint32_t num_tags) {
    if (!tags && num_tags != 0) return;
    std::lock_guard<std::mutex> lk(g_tag_mu);
    SlTagSnapshot& snapshot = FindOrCreateSnapshotLocked(frame, viewport);
    snapshot.tags.clear();
    for (uint32_t i = 0; i < num_tags; ++i) {
        snapshot.tags.push_back(tags[i]);
    }
    OSSG_LOG_INFO("sl_proxy", "stored %u Streamline tags frame=%p viewport=%p",
                  num_tags, frame, viewport);
}

void StoreConstants(const void* frame, const void* viewport, const void* constants) {
    if (!constants) return;
    std::lock_guard<std::mutex> lk(g_tag_mu);
    SlTagSnapshot& snapshot = FindOrCreateSnapshotLocked(frame, viewport);
    snapshot.constants =
        *reinterpret_cast<const oss_gaussian::sl_proxy::SlConstantsAbi*>(constants);
    snapshot.has_constants = true;
}

void ApplyTagToFrame(const oss_gaussian::sl_proxy::SlResourceTagAbi& tag,
                     OssGaussianFrame& frame) {
    if (!tag.resource) return;
    void* native = tag.resource->native;
    switch (tag.type) {
        case kBufferTypeScalingInputColor:
            frame.color = native;
            frame.color_state = tag.resource->state;
            frame.resource_states_valid |= kStateColor;
            frame.subrect_base_x = tag.extent.left;
            frame.subrect_base_y = tag.extent.top;
            frame.subrect_render_width =
                tag.extent.width ? tag.extent.width : tag.resource->width;
            frame.subrect_render_height =
                tag.extent.height ? tag.extent.height : tag.resource->height;
            break;
        case kBufferTypeScalingOutputColor:
            frame.output = native;
            frame.output_state = tag.resource->state;
            frame.resource_states_valid |= kStateOutput;
            frame.output_width = tag.extent.width ? tag.extent.width : tag.resource->width;
            frame.output_height = tag.extent.height ? tag.extent.height : tag.resource->height;
            break;
        case kBufferTypeDepth:
            frame.depth = native;
            frame.depth_state = tag.resource->state;
            frame.resource_states_valid |= kStateDepth;
            break;
        case kBufferTypeMotionVectors:
            frame.motion_vectors = native;
            frame.motion_vectors_state = tag.resource->state;
            frame.resource_states_valid |= kStateMotionVectors;
            break;
        case kBufferTypeExposure:
            frame.exposure_texture = native;
            frame.exposure_texture_state = tag.resource->state;
            frame.resource_states_valid |= kStateExposure;
            break;
        default:
            break;
    }
}

bool BuildStreamlineFrame(uint32_t feature,
                          const void* frame_token,
                          const void** inputs,
                          uint32_t num_inputs,
                          OssGaussianFrame* out) {
    if (feature != kFeatureDLSS || !out) return false;

    const void* viewport = nullptr;
    std::vector<oss_gaussian::sl_proxy::SlResourceTagAbi> local_tags;
    for (uint32_t i = 0; inputs && i < num_inputs; ++i) {
        const auto* base =
            reinterpret_cast<const oss_gaussian::sl_proxy::SlBaseStructureAbi*>(inputs[i]);
        if (SameStructType(base, kViewportHandleType)) {
            viewport = inputs[i];
        } else if (SameStructType(base, kResourceTagType)) {
            local_tags.push_back(
                *reinterpret_cast<const oss_gaussian::sl_proxy::SlResourceTagAbi*>(inputs[i]));
        }
    }

    SlTagSnapshot snapshot{};
    bool found_snapshot = false;
    {
        std::lock_guard<std::mutex> lk(g_tag_mu);
        for (auto it = g_tag_snapshots.rbegin(); it != g_tag_snapshots.rend(); ++it) {
            const bool frame_matches = it->frame == frame_token || it->frame == nullptr;
            const bool viewport_matches = !viewport || it->viewport == viewport;
            if (frame_matches && viewport_matches) {
                snapshot = *it;
                found_snapshot = true;
                break;
            }
        }
    }

    *out = OssGaussianFrame{};
    out->frame_index = g_sl_frame_counter.fetch_add(1, std::memory_order_relaxed);
    out->exposure_scale = 1.0f;
    out->mv_scale_x = 1.0f;
    out->mv_scale_y = 1.0f;
    if (found_snapshot && snapshot.has_constants) {
        out->jitter_offset_x = snapshot.constants.jitter_offset.x;
        out->jitter_offset_y = snapshot.constants.jitter_offset.y;
        out->mv_scale_x = snapshot.constants.mvec_scale.x;
        out->mv_scale_y = snapshot.constants.mvec_scale.y;
        out->reset = snapshot.constants.reset ? 1u : 0u;
    }
    for (const auto& tag : snapshot.tags) {
        ApplyTagToFrame(tag, *out);
    }
    for (const auto& tag : local_tags) {
        ApplyTagToFrame(tag, *out);
    }

    const bool complete = out->color && out->output && out->depth && out->motion_vectors;
    OSSG_LOG_INFO(
        "sl_proxy",
        "Streamline DLSS frame candidate complete=%u color=%p depth=%p mv=%p output=%p "
        "render=%ux%u out=%ux%u",
        complete ? 1u : 0u,
        out->color, out->depth, out->motion_vectors, out->output,
        out->subrect_render_width, out->subrect_render_height,
        out->output_width, out->output_height);
    return complete;
}

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
    if (!GetModuleHandleExW(flags, reinterpret_cast<LPCWSTR>(&SelfModuleAnchor), &self)) {
        return std::wstring();
    }
    return ModulePath(self);
}

void PushEnvOverride(std::vector<std::wstring>& candidates) {
    wchar_t value[MAX_PATH] = {};
    const DWORD len = GetEnvironmentVariableW(L"OSS_SL_REAL_DLL", value, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        candidates.emplace_back(value);
    }
}

std::vector<std::wstring> BuildCandidatePaths() {
    std::vector<std::wstring> candidates;
    PushEnvOverride(candidates);

    wchar_t exe_path[MAX_PATH] = {};
    const DWORD exe_len = GetModuleFileNameW(nullptr, exe_path, MAX_PATH);
    if (exe_len > 0 && exe_len < MAX_PATH) {
        const std::wstring exe_dir = DirectoryOf(exe_path);
        candidates.push_back(JoinPath(exe_dir, kRealSlDllName));
    }

    wchar_t cwd[MAX_PATH] = {};
    const DWORD cwd_len = GetCurrentDirectoryW(MAX_PATH, cwd);
    if (cwd_len > 0 && cwd_len < MAX_PATH) {
        candidates.push_back(JoinPath(cwd, kRealSlDllName));
    }

    // Last-resort self-name lookup is useful for unit tests where OSS_SL_REAL_DLL
    // points elsewhere; it is skipped if it resolves back to this proxy.
    if (exe_len > 0 && exe_len < MAX_PATH) {
        candidates.push_back(JoinPath(DirectoryOf(exe_path), kOriginalSlDllName));
    }
    if (cwd_len > 0 && cwd_len < MAX_PATH) {
        candidates.push_back(JoinPath(cwd, kOriginalSlDllName));
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
        OSSG_LOG_WARN("sl_proxy", "failed to load real Streamline candidate err=%lu",
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

void LoadRealSlOnce() {
    LogInit();
    const std::wstring self_path = SelfModulePath();
    for (const std::wstring& candidate : BuildCandidatePaths()) {
        if (HMODULE module = TryLoadCandidate(candidate, self_path)) {
            g_real_sl = module;
            OSSG_LOG_INFO("sl_proxy", "loaded real Streamline interposer");
            return;
        }
    }
    OSSG_LOG_ERROR("sl_proxy", "real Streamline interposer not found");
}

HMODULE RealSlModule() {
    std::call_once(g_load_once, LoadRealSlOnce);
    return g_real_sl;
}

} // namespace

void EnsureNativeInit() {
    oss_gaussian::OssGaussianEnsureInitializedFromProxyExport();
}

using SlResult = int32_t;
constexpr SlResult kSlErrorInitNotCalled = 24;
constexpr long kForwardENotImpl = 0x80004001L;

SlResult MissingRealSl(const char* fn) {
    LogInit();
    OSSG_LOG_ERROR("sl_proxy", "%s: real Streamline export unavailable", fn);
    return kSlErrorInitNotCalled;
}

void LogTagSummary(const SlResourceTagAbi* tags, uint32_t num_tags) {
    if (!tags || num_tags == 0) {
        OSSG_LOG_INFO("sl_proxy", "slSetTag: no tags");
        return;
    }
    for (uint32_t i = 0; i < num_tags && i < 8; ++i) {
        const SlResourceTagAbi& item = tags[i];
        const SlResourceAbi* item_resource = item.resource;
        OSSG_LOG_INFO(
            "sl_proxy",
            "slSetTag[%u/%u]: type=%u lifecycle=%u resource=%p native=%p "
            "extent=%u,%u %ux%u size=%ux%u format=%u state=%u",
            i,
            num_tags,
            item.type,
            item.lifecycle,
            static_cast<const void*>(item_resource),
            item_resource ? item_resource->native : nullptr,
            item.extent.left,
            item.extent.top,
            item.extent.width,
            item.extent.height,
            item_resource ? item_resource->width : 0u,
            item_resource ? item_resource->height : 0u,
            item_resource ? item_resource->native_format : 0u,
            item_resource ? item_resource->state : 0u);
    }
}

void LogEvaluateInputs(const void** inputs, uint32_t num_inputs) {
    if (!inputs || num_inputs == 0) {
        OSSG_LOG_INFO("sl_proxy", "slEvaluateFeature: no local inputs");
        return;
    }
    for (uint32_t i = 0; i < num_inputs && i < 8; ++i) {
        const auto* base = reinterpret_cast<const SlBaseStructureAbi*>(inputs[i]);
        if (!base) {
            OSSG_LOG_INFO("sl_proxy", "slEvaluateFeature input[%u]=null", i);
            continue;
        }
        OSSG_LOG_INFO(
            "sl_proxy",
            "slEvaluateFeature input[%u]=%p type=%08x-%04x-%04x version=%zu next=%p",
            i,
            static_cast<const void*>(base),
            base->type_data1,
            base->type_data2,
            base->type_data3,
            base->struct_version,
            base->next);
    }
}

void* ResolveExport(const char* name) {
    if (!name || !*name) return nullptr;
    HMODULE module = RealSlModule();
    if (!module) return nullptr;
    FARPROC proc = GetProcAddress(module, name);
    if (!proc) {
        LogInit();
        OSSG_LOG_ERROR("sl_proxy", "real Streamline missing export: %s", name);
        return nullptr;
    }
    return reinterpret_cast<void*>(proc);
}

} // namespace oss_gaussian::sl_proxy

extern "C" {

__declspec(dllexport) int32_t
OssgSlEvaluateFeature(uint32_t feature,
                      const void* frame,
                      const void** inputs,
                      uint32_t num_inputs,
                      void* command_buffer);
__declspec(dllexport) int32_t
OssgSlSetConstants(const void* constants, const void* frame, const void* viewport);
__declspec(dllexport) int32_t
OssgSlSetTag(const void* viewport,
             const oss_gaussian::sl_proxy::SlResourceTagAbi* tags,
             uint32_t num_tags,
             void* command_buffer);
__declspec(dllexport) int32_t
OssgSlSetTagForFrame(const void* frame,
                     const void* viewport,
                     const oss_gaussian::sl_proxy::SlResourceTagAbi* tags,
                     uint32_t num_tags,
                     void* command_buffer);

	#define OSSG_SL_FWD(RET, IMPL, NAME, SIG, ARGS, FAIL_VALUE)                  \
	    __declspec(dllexport) RET IMPL SIG {                                      \
	        oss_gaussian::sl_proxy::EnsureNativeInit();                          \
	        using Fn = RET (*) SIG;                                               \
	        auto real = oss_gaussian::sl_proxy::ResolveTyped<Fn>(NAME);          \
	        if (!real) return FAIL_VALUE;                                         \
	        return real ARGS;                                                     \
	    }

	#define OSSG_SL_FWD_WINAPI(RET, IMPL, NAME, SIG, ARGS, FAIL_VALUE)           \
	    __declspec(dllexport) RET WINAPI IMPL SIG {                               \
	        oss_gaussian::LogInit();                                             \
	        OSSG_LOG_TRACE("sl_proxy", "forward %s", NAME);                     \
	        oss_gaussian::sl_proxy::EnsureNativeInit();                          \
	        using Fn = RET (WINAPI*) SIG;                                         \
	        auto real = oss_gaussian::sl_proxy::ResolveTyped<Fn>(NAME);          \
	        if (!real) return FAIL_VALUE;                                         \
        return real ARGS;                                                     \
    }

OSSG_SL_FWD_WINAPI(long, OssgSlCreateDXGIFactory, "CreateDXGIFactory",
                   (REFIID riid, void** pp_factory), (riid, pp_factory), oss_gaussian::sl_proxy::kForwardENotImpl)
OSSG_SL_FWD_WINAPI(long, OssgSlCreateDXGIFactory1, "CreateDXGIFactory1",
                   (REFIID riid, void** pp_factory), (riid, pp_factory), oss_gaussian::sl_proxy::kForwardENotImpl)
OSSG_SL_FWD_WINAPI(long, OssgSlCreateDXGIFactory2, "CreateDXGIFactory2",
                   (uint32_t flags, REFIID riid, void** pp_factory), (flags, riid, pp_factory), oss_gaussian::sl_proxy::kForwardENotImpl)
OSSG_SL_FWD_WINAPI(long, OssgSlDXGIGetDebugInterface1, "DXGIGetDebugInterface1",
                   (uint32_t flags, REFIID riid, void** pp_debug), (flags, riid, pp_debug), oss_gaussian::sl_proxy::kForwardENotImpl)

OSSG_SL_FWD_WINAPI(long, OssgSlD3D11CreateDevice, "D3D11CreateDevice",
                   (void* adapter, int driver_type, HMODULE software, uint32_t flags,
                    const void* feature_levels, uint32_t feature_level_count,
                    uint32_t sdk_version, void** device, void* feature_level,
                    void** immediate_context),
                   (adapter, driver_type, software, flags, feature_levels, feature_level_count,
                    sdk_version, device, feature_level, immediate_context),
                   oss_gaussian::sl_proxy::kForwardENotImpl)
OSSG_SL_FWD_WINAPI(long, OssgSlD3D11CreateDeviceAndSwapChain, "D3D11CreateDeviceAndSwapChain",
                   (void* adapter, int driver_type, HMODULE software, uint32_t flags,
                    const void* feature_levels, uint32_t feature_level_count,
                    uint32_t sdk_version, const void* swap_chain_desc,
                    void** swap_chain, void** device, void* feature_level,
                    void** immediate_context),
                   (adapter, driver_type, software, flags, feature_levels, feature_level_count,
                    sdk_version, swap_chain_desc, swap_chain, device, feature_level,
                    immediate_context),
                   oss_gaussian::sl_proxy::kForwardENotImpl)
	__declspec(dllexport) long WINAPI
	OssgSlD3D12CreateDevice(void* adapter, int minimum_feature_level, REFIID riid, void** device) {
	    oss_gaussian::LogInit();
	    OSSG_LOG_INFO("sl_proxy",
	                  "D3D12CreateDevice adapter=%p min_feature_level=%d out=%p",
	                  adapter,
	                  minimum_feature_level,
	                  static_cast<void*>(device));
	    using Fn = long (WINAPI*)(void*, int, REFIID, void**);
	    auto real = oss_gaussian::sl_proxy::ResolveTyped<Fn>("D3D12CreateDevice");
	    if (!real) return oss_gaussian::sl_proxy::kForwardENotImpl;
	    const long hr = real(adapter, minimum_feature_level, riid, device);
	    OSSG_LOG_INFO("sl_proxy", "D3D12CreateDevice result=0x%08lx device=%p",
	                  hr,
	                  device ? *device : nullptr);
	    if (hr >= 0) {
	        oss_gaussian::sl_proxy::EnsureNativeInit();
	    }
	    return hr;
	}
OSSG_SL_FWD_WINAPI(long, OssgSlD3D12CreateRootSignatureDeserializer, "D3D12CreateRootSignatureDeserializer",
                   (const void* src_data, size_t src_data_size, REFIID riid, void** deserializer),
                   (src_data, src_data_size, riid, deserializer), oss_gaussian::sl_proxy::kForwardENotImpl)
OSSG_SL_FWD_WINAPI(long, OssgSlD3D12CreateVersionedRootSignatureDeserializer, "D3D12CreateVersionedRootSignatureDeserializer",
                   (const void* src_data, size_t src_data_size, REFIID riid, void** deserializer),
                   (src_data, src_data_size, riid, deserializer), oss_gaussian::sl_proxy::kForwardENotImpl)
OSSG_SL_FWD_WINAPI(long, OssgSlD3D12EnableExperimentalFeatures, "D3D12EnableExperimentalFeatures",
                   (uint32_t num_features, const void* feature_iids, void* configuration_structs, uint32_t* configuration_sizes),
                   (num_features, feature_iids, configuration_structs, configuration_sizes), oss_gaussian::sl_proxy::kForwardENotImpl)
OSSG_SL_FWD_WINAPI(long, OssgSlD3D12GetDebugInterface, "D3D12GetDebugInterface",
                   (REFIID riid, void** debug), (riid, debug), oss_gaussian::sl_proxy::kForwardENotImpl)
OSSG_SL_FWD_WINAPI(long, OssgSlD3D12GetInterface, "D3D12GetInterface",
                   (REFCLSID clsid, REFIID riid, void** object), (clsid, riid, object), oss_gaussian::sl_proxy::kForwardENotImpl)
OSSG_SL_FWD_WINAPI(long, OssgSlD3D12SerializeRootSignature, "D3D12SerializeRootSignature",
                   (const void* root_signature, int version, void** blob, void** error_blob),
                   (root_signature, version, blob, error_blob), oss_gaussian::sl_proxy::kForwardENotImpl)
OSSG_SL_FWD_WINAPI(long, OssgSlD3D12SerializeVersionedRootSignature, "D3D12SerializeVersionedRootSignature",
                   (const void* root_signature, void** blob, void** error_blob),
                   (root_signature, blob, error_blob), oss_gaussian::sl_proxy::kForwardENotImpl)

OSSG_SL_FWD(int32_t, OssgSlAllocateResources, "slAllocateResources",
            (void* command_buffer, uint32_t feature, const void* viewport),
            (command_buffer, feature, viewport), oss_gaussian::sl_proxy::kSlErrorInitNotCalled)
	OSSG_SL_FWD(int32_t, OssgSlFreeResources, "slFreeResources",
	            (uint32_t feature, const void* viewport), (feature, viewport),
	            oss_gaussian::sl_proxy::kSlErrorInitNotCalled)
	OSSG_SL_FWD(int32_t, OssgSlGetFeatureRequirements, "slGetFeatureRequirements",
	            (uint32_t feature, void* requirements), (feature, requirements),
	            oss_gaussian::sl_proxy::kSlErrorInitNotCalled)
OSSG_SL_FWD(int32_t, OssgSlGetFeatureVersion, "slGetFeatureVersion",
            (uint32_t feature, void* version), (feature, version),
            oss_gaussian::sl_proxy::kSlErrorInitNotCalled)
OSSG_SL_FWD(int32_t, OssgSlGetNativeInterface, "slGetNativeInterface",
            (void* proxy_interface, void** base_interface), (proxy_interface, base_interface),
            oss_gaussian::sl_proxy::kSlErrorInitNotCalled)
OSSG_SL_FWD(int32_t, OssgSlGetNewFrameToken, "slGetNewFrameToken",
            (void** token, const uint32_t* frame_index), (token, frame_index),
            oss_gaussian::sl_proxy::kSlErrorInitNotCalled)
OSSG_SL_FWD(int32_t, OssgSlIsFeatureLoaded, "slIsFeatureLoaded",
            (uint32_t feature, bool* loaded), (feature, loaded),
            oss_gaussian::sl_proxy::kSlErrorInitNotCalled)
OSSG_SL_FWD(int32_t, OssgSlIsFeatureSupported, "slIsFeatureSupported",
            (uint32_t feature, const void* adapter_info), (feature, adapter_info),
            oss_gaussian::sl_proxy::kSlErrorInitNotCalled)
OSSG_SL_FWD(int32_t, OssgSlSetFeatureLoaded, "slSetFeatureLoaded",
            (uint32_t feature, bool loaded), (feature, loaded),
            oss_gaussian::sl_proxy::kSlErrorInitNotCalled)
OSSG_SL_FWD(int32_t, OssgSlSetVulkanInfo, "slSetVulkanInfo",
            (const void* vulkan_info), (vulkan_info),
            oss_gaussian::sl_proxy::kSlErrorInitNotCalled)
OSSG_SL_FWD(int32_t, OssgSlShutdown, "slShutdown",
            (void), (), oss_gaussian::sl_proxy::kSlErrorInitNotCalled)
OSSG_SL_FWD(int32_t, OssgSlUpgradeInterface, "slUpgradeInterface",
            (void** base_interface), (base_interface),
            oss_gaussian::sl_proxy::kSlErrorInitNotCalled)

	__declspec(dllexport) int32_t
	OssgSlInit(const void* preferences, uint64_t sdk_version) {
	    oss_gaussian::sl_proxy::EnsureNativeInit();
	    oss_gaussian::LogInit();
    OSSG_LOG_INFO("sl_proxy", "slInit sdk_version=%llu",
                  static_cast<unsigned long long>(sdk_version));
    using Fn = int32_t (*)(const void*, uint64_t);
    auto real = oss_gaussian::sl_proxy::ResolveTyped<Fn>("slInit");
    if (!real) return oss_gaussian::sl_proxy::MissingRealSl("slInit");
    return real(preferences, sdk_version);
}

	__declspec(dllexport) int32_t
	OssgSlSetD3DDevice(void* d3d_device) {
	    oss_gaussian::sl_proxy::EnsureNativeInit();
	    oss_gaussian::LogInit();
    OSSG_LOG_INFO("sl_proxy", "slSetD3DDevice device=%p", d3d_device);
    using Fn = int32_t (*)(void*);
    auto real = oss_gaussian::sl_proxy::ResolveTyped<Fn>("slSetD3DDevice");
    if (!real) return oss_gaussian::sl_proxy::MissingRealSl("slSetD3DDevice");
    return real(d3d_device);
}

	__declspec(dllexport) int32_t
	OssgSlSetConstants(const void* constants, const void* frame, const void* viewport) {
	    oss_gaussian::sl_proxy::EnsureNativeInit();
	    oss_gaussian::LogInit();
    OSSG_LOG_INFO("sl_proxy", "slSetConstants constants=%p frame=%p viewport=%p",
                  constants, frame, viewport);
    oss_gaussian::sl_proxy::StoreConstants(frame, viewport, constants);
    using Fn = int32_t (*)(const void*, const void*, const void*);
    auto real = oss_gaussian::sl_proxy::ResolveTyped<Fn>("slSetConstants");
    if (!real) return oss_gaussian::sl_proxy::MissingRealSl("slSetConstants");
    return real(constants, frame, viewport);
}

	__declspec(dllexport) int32_t
	OssgSlSetTag(const void* viewport,
	             const oss_gaussian::sl_proxy::SlResourceTagAbi* tags,
	             uint32_t num_tags,
	             void* command_buffer) {
	    oss_gaussian::sl_proxy::EnsureNativeInit();
	    oss_gaussian::LogInit();
    OSSG_LOG_INFO("sl_proxy", "slSetTag viewport=%p cmd=%p", viewport, command_buffer);
    oss_gaussian::sl_proxy::LogTagSummary(tags, num_tags);
    oss_gaussian::sl_proxy::StoreTags(nullptr, viewport, tags, num_tags);
    using Fn = int32_t (*)(const void*, const oss_gaussian::sl_proxy::SlResourceTagAbi*, uint32_t, void*);
    auto real = oss_gaussian::sl_proxy::ResolveTyped<Fn>("slSetTag");
    if (!real) return oss_gaussian::sl_proxy::MissingRealSl("slSetTag");
	    return real(viewport, tags, num_tags, command_buffer);
	}

	__declspec(dllexport) int32_t
	OssgSlSetTagForFrame(const void* frame,
	                     const void* viewport,
	                     const oss_gaussian::sl_proxy::SlResourceTagAbi* tags,
	                     uint32_t num_tags,
	                     void* command_buffer) {
	    oss_gaussian::sl_proxy::EnsureNativeInit();
	    oss_gaussian::LogInit();
	    OSSG_LOG_INFO("sl_proxy", "slSetTagForFrame frame=%p viewport=%p cmd=%p",
	                  frame, viewport, command_buffer);
	    oss_gaussian::sl_proxy::LogTagSummary(tags, num_tags);
	    oss_gaussian::sl_proxy::StoreTags(frame, viewport, tags, num_tags);
	    using Fn = int32_t (*)(const void*, const void*, const oss_gaussian::sl_proxy::SlResourceTagAbi*, uint32_t, void*);
	    auto real = oss_gaussian::sl_proxy::ResolveTyped<Fn>("slSetTagForFrame");
	    if (!real) return oss_gaussian::sl_proxy::MissingRealSl("slSetTagForFrame");
	    return real(frame, viewport, tags, num_tags, command_buffer);
	}

	__declspec(dllexport) int32_t
	OssgSlGetFeatureFunction(uint32_t feature, const char* function_name, void** function) {
	    oss_gaussian::sl_proxy::EnsureNativeInit();
	    oss_gaussian::LogInit();
	    using Fn = int32_t (*)(uint32_t, const char*, void**);
	    auto real = oss_gaussian::sl_proxy::ResolveTyped<Fn>("slGetFeatureFunction");
	    if (!real) return oss_gaussian::sl_proxy::MissingRealSl("slGetFeatureFunction");
	    const int32_t result = real(feature, function_name, function);
	    OSSG_LOG_INFO("sl_proxy", "slGetFeatureFunction feature=%u name=%s result=%d ptr=%p",
	                  feature,
	                  function_name ? function_name : "<null>",
	                  result,
	                  function ? *function : nullptr);
	    if (result == 0 && function && function_name) {
	        if (std::strcmp(function_name, "slEvaluateFeature") == 0) {
	            *function = reinterpret_cast<void*>(&OssgSlEvaluateFeature);
	            OSSG_LOG_INFO("sl_proxy", "slGetFeatureFunction wrapped slEvaluateFeature");
	        } else if (std::strcmp(function_name, "slSetTag") == 0) {
	            *function = reinterpret_cast<void*>(&OssgSlSetTag);
	            OSSG_LOG_INFO("sl_proxy", "slGetFeatureFunction wrapped slSetTag");
	        } else if (std::strcmp(function_name, "slSetTagForFrame") == 0) {
	            *function = reinterpret_cast<void*>(&OssgSlSetTagForFrame);
	            OSSG_LOG_INFO("sl_proxy", "slGetFeatureFunction wrapped slSetTagForFrame");
	        } else if (std::strcmp(function_name, "slSetConstants") == 0) {
	            *function = reinterpret_cast<void*>(&OssgSlSetConstants);
	            OSSG_LOG_INFO("sl_proxy", "slGetFeatureFunction wrapped slSetConstants");
	        }
	    }
	    return result;
	}

	__declspec(dllexport) int32_t
	OssgSlEvaluateFeature(uint32_t feature,
                      const void* frame,
	                      const void** inputs,
	                      uint32_t num_inputs,
	                      void* command_buffer) {
	    oss_gaussian::sl_proxy::EnsureNativeInit();
	    oss_gaussian::LogInit();
    OSSG_LOG_INFO("sl_proxy",
                  "slEvaluateFeature feature=%u frame=%p inputs=%p num_inputs=%u cmd=%p",
                  feature,
                  frame,
                  static_cast<const void*>(inputs),
	                  num_inputs,
	                  command_buffer);
	    oss_gaussian::sl_proxy::LogEvaluateInputs(inputs, num_inputs);
    using Fn = int32_t (*)(uint32_t, const void*, const void**, uint32_t, void*);
    auto real = oss_gaussian::sl_proxy::ResolveTyped<Fn>("slEvaluateFeature");
    if (!real) return oss_gaussian::sl_proxy::MissingRealSl("slEvaluateFeature");
    OssGaussianFrame captured_frame{};
    void* capture_ticket = nullptr;
    if (oss_gaussian::sl_proxy::BuildStreamlineFrame(
            feature, frame, inputs, num_inputs, &captured_frame)) {
        capture_ticket = oss_gaussian::BeginNgxFrameCapture(
            static_cast<ID3D12GraphicsCommandList*>(command_buffer),
            captured_frame);
    }
    const int32_t result = real(feature, frame, inputs, num_inputs, command_buffer);
    if (capture_ticket) {
        oss_gaussian::EndNgxFrameCapture(capture_ticket, result == 0 ? 0x1 : result);
    }
    OSSG_LOG_INFO("sl_proxy", "slEvaluateFeature result=%d", result);
    return result;
}

} // extern "C"
