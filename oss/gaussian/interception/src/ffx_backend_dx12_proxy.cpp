// =============================================================================
//  ffx_backend_dx12_proxy.cpp
//
//  Pass-through proxy for AMD FidelityFX DX12 backend imports.
// =============================================================================
#define OSS_GAUSSIAN_BUILDING_DLL 1

#include "ffx_backend_dx12_proxy.h"

#include "log.h"

#include <Windows.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace oss_gaussian::ffx_backend_dx12 {

using FfxErrorCode = int32_t;
constexpr FfxErrorCode kFfxErrorInvalidPointer = static_cast<FfxErrorCode>(0x80000000u);

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

namespace {

constexpr wchar_t kRealDllName[] = L"oss_ffx_backend_dx12_real.dll";
constexpr wchar_t kBackupDllName[] = L"ffx_backend_dx12_x64.dll.oss-backup";
constexpr wchar_t kOriginalDllName[] = L"ffx_backend_dx12_x64.dll";

std::once_flag g_load_once;
HMODULE g_real_backend = nullptr;

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
    const DWORD len = GetEnvironmentVariableW(L"OSS_FFX_BACKEND_DX12_REAL_DLL", value, MAX_PATH);
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
        candidates.push_back(JoinPath(exe_dir, kRealDllName));
        candidates.push_back(JoinPath(exe_dir, kBackupDllName));
        candidates.push_back(JoinPath(exe_dir, kOriginalDllName));
    }

    wchar_t cwd[MAX_PATH] = {};
    const DWORD cwd_len = GetCurrentDirectoryW(MAX_PATH, cwd);
    if (cwd_len > 0 && cwd_len < MAX_PATH) {
        candidates.push_back(JoinPath(cwd, kRealDllName));
        candidates.push_back(JoinPath(cwd, kBackupDllName));
        candidates.push_back(JoinPath(cwd, kOriginalDllName));
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
        OSSG_LOG_WARN("ffx_backend", "failed to load real DX12 backend candidate err=%lu",
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

void LoadRealBackendOnce() {
    LogInit();
    const std::wstring self_path = SelfModulePath();
    for (const std::wstring& candidate : BuildCandidatePaths()) {
        if (HMODULE module = TryLoadCandidate(candidate, self_path)) {
            g_real_backend = module;
            OSSG_LOG_INFO("ffx_backend", "loaded real ffx_backend_dx12_x64.dll");
            return;
        }
    }
    OSSG_LOG_ERROR("ffx_backend", "real ffx_backend_dx12_x64.dll not found");
}

HMODULE RealBackendModule() {
    std::call_once(g_load_once, LoadRealBackendOnce);
    return g_real_backend;
}

FfxErrorCode MissingReal(const char* fn) {
    LogInit();
    OSSG_LOG_ERROR("ffx_backend", "%s: real DX12 backend export unavailable", fn);
    return kFfxErrorInvalidPointer;
}

void LogResourceDescription(const char* fn, const FfxResourceDescriptionAbi& desc) {
    OSSG_LOG_INFO("ffx_backend",
                  "%s desc type=%u format=%u size=%ux%u depth=%u mips=%u flags=%u usage=%u",
                  fn,
                  desc.type,
                  desc.format,
                  desc.width,
                  desc.height,
                  desc.depth,
                  desc.mip_count,
                  desc.flags,
                  desc.usage);
}

} // namespace

void* ResolveExport(const char* name) {
    if (!name || !*name) return nullptr;
    HMODULE module = RealBackendModule();
    if (!module) return nullptr;
    FARPROC proc = GetProcAddress(module, name);
    if (!proc) {
        LogInit();
        OSSG_LOG_ERROR("ffx_backend", "real DX12 backend missing export: %s", name);
        return nullptr;
    }
    return reinterpret_cast<void*>(proc);
}

} // namespace oss_gaussian::ffx_backend_dx12

extern "C" {

#define OSSG_BACKEND_FWD_RET(RET, IMPL, NAME, SIG, ARGS, FAIL_VALUE)             \
    RET IMPL SIG {                                                               \
        using Fn = RET (*) SIG;                                                   \
        auto real = oss_gaussian::ffx_backend_dx12::ResolveTyped<Fn>(NAME);      \
        if (!real) return FAIL_VALUE;                                             \
        return real ARGS;                                                         \
    }

#define OSSG_BACKEND_FWD_VOID(IMPL, NAME, SIG, ARGS)                             \
    void IMPL SIG {                                                              \
        using Fn = void (*) SIG;                                                  \
        auto real = oss_gaussian::ffx_backend_dx12::ResolveTyped<Fn>(NAME);      \
        if (!real) {                                                             \
            oss_gaussian::ffx_backend_dx12::MissingReal(NAME);                   \
            return;                                                              \
        }                                                                        \
        real ARGS;                                                               \
    }

bool OssgBackendAssertReport(const char* file, int32_t line, const char* condition, const char* msg) {
    using Fn = bool (*)(const char*, int32_t, const char*, const char*);
    auto real = oss_gaussian::ffx_backend_dx12::ResolveTyped<Fn>("ffxAssertReport");
    if (!real) return false;
    return real(file, line, condition, msg);
}

OSSG_BACKEND_FWD_VOID(OssgBackendAssertSetPrintingCallback, "ffxAssertSetPrintingCallback",
                      (void* callback), (callback))

size_t OssgBackendGetScratchMemorySizeDX12(size_t max_contexts) {
    oss_gaussian::LogInit();
    OSSG_LOG_INFO("ffx_backend", "ffxGetScratchMemorySizeDX12 max_contexts=%zu",
                  max_contexts);
    using Fn = size_t (*)(size_t);
    auto real = oss_gaussian::ffx_backend_dx12::ResolveTyped<Fn>("ffxGetScratchMemorySizeDX12");
    if (!real) return 0;
    return real(max_contexts);
}

void* OssgBackendGetDeviceDX12(void* device) {
    oss_gaussian::LogInit();
    OSSG_LOG_INFO("ffx_backend", "ffxGetDeviceDX12 device=%p", device);
    using Fn = void* (*)(void*);
    auto real = oss_gaussian::ffx_backend_dx12::ResolveTyped<Fn>("ffxGetDeviceDX12");
    if (!real) return nullptr;
    return real(device);
}

oss_gaussian::ffx_backend_dx12::FfxErrorCode
OssgBackendGetInterfaceDX12(void* backend_interface,
                            void* device,
                            void* scratch_buffer,
                            size_t scratch_buffer_size,
                            size_t max_contexts) {
    oss_gaussian::LogInit();
    OSSG_LOG_INFO("ffx_backend",
                  "ffxGetInterfaceDX12 iface=%p device=%p scratch=%p scratch_size=%zu max_contexts=%zu",
                  backend_interface,
                  device,
                  scratch_buffer,
                  scratch_buffer_size,
                  max_contexts);
    using Fn = oss_gaussian::ffx_backend_dx12::FfxErrorCode (*)(
        void*, void*, void*, size_t, size_t);
    auto real = oss_gaussian::ffx_backend_dx12::ResolveTyped<Fn>("ffxGetInterfaceDX12");
    if (!real) {
        return oss_gaussian::ffx_backend_dx12::MissingReal("ffxGetInterfaceDX12");
    }
    return real(backend_interface, device, scratch_buffer, scratch_buffer_size, max_contexts);
}

void* OssgBackendGetCommandListDX12(void* command_list) {
    oss_gaussian::LogInit();
    OSSG_LOG_INFO("ffx_backend", "ffxGetCommandListDX12 command_list=%p", command_list);
    using Fn = void* (*)(void*);
    auto real = oss_gaussian::ffx_backend_dx12::ResolveTyped<Fn>("ffxGetCommandListDX12");
    if (!real) return nullptr;
    return real(command_list);
}

void* OssgBackendGetCommandQueueDX12(void* command_queue) {
    using Fn = void* (*)(void*);
    auto real = oss_gaussian::ffx_backend_dx12::ResolveTyped<Fn>("ffxGetCommandQueueDX12");
    if (!real) return nullptr;
    return real(command_queue);
}

void* OssgBackendGetSwapchainDX12(void* swapchain) {
    using Fn = void* (*)(void*);
    auto real = oss_gaussian::ffx_backend_dx12::ResolveTyped<Fn>("ffxGetSwapchainDX12");
    if (!real) return nullptr;
    return real(swapchain);
}

void* OssgBackendGetDX12SwapchainPtr(void* ffx_swapchain) {
    using Fn = void* (*)(void*);
    auto real = oss_gaussian::ffx_backend_dx12::ResolveTyped<Fn>("ffxGetDX12SwapchainPtr");
    if (!real) return nullptr;
    return real(ffx_swapchain);
}

void* OssgBackendGetFrameinterpolationCommandlistDX12(void* command_list) {
    using Fn = void* (*)(void*);
    auto real = oss_gaussian::ffx_backend_dx12::ResolveTyped<Fn>(
        "ffxGetFrameinterpolationCommandlistDX12");
    if (!real) return nullptr;
    return real(command_list);
}

void* OssgBackendGetFrameinterpolationTextureDX12(void* resource) {
    using Fn = void* (*)(void*);
    auto real = oss_gaussian::ffx_backend_dx12::ResolveTyped<Fn>(
        "ffxGetFrameinterpolationTextureDX12");
    if (!real) return nullptr;
    return real(resource);
}

oss_gaussian::ffx_backend_dx12::FfxResourceAbi
OssgBackendGetResourceDX12(
    const void* dx12_resource,
    oss_gaussian::ffx_backend_dx12::FfxResourceDescriptionAbi description,
    const wchar_t* resource_name,
    uint32_t state) {
    oss_gaussian::LogInit();
    OSSG_LOG_INFO("ffx_backend", "ffxGetResourceDX12 resource=%p name=%ls state=%u",
                  dx12_resource,
                  resource_name ? resource_name : L"<null>",
                  state);
    oss_gaussian::ffx_backend_dx12::LogResourceDescription("ffxGetResourceDX12", description);
    using Fn = oss_gaussian::ffx_backend_dx12::FfxResourceAbi (*)(
        const void*,
        oss_gaussian::ffx_backend_dx12::FfxResourceDescriptionAbi,
        const wchar_t*,
        uint32_t);
    auto real = oss_gaussian::ffx_backend_dx12::ResolveTyped<Fn>("ffxGetResourceDX12");
    if (!real) return {};
    return real(dx12_resource, description, resource_name, state);
}

oss_gaussian::ffx_backend_dx12::FfxResourceDescriptionAbi
OssgBackendGetFfxResourceDescriptionDX12(const void* resource, uint32_t additional_usages) {
    oss_gaussian::LogInit();
    OSSG_LOG_INFO("ffx_backend", "GetFfxResourceDescriptionDX12 resource=%p usage=%u",
                  resource, additional_usages);
    using Fn = oss_gaussian::ffx_backend_dx12::FfxResourceDescriptionAbi (*)(const void*, uint32_t);
    auto real = oss_gaussian::ffx_backend_dx12::ResolveTyped<Fn>("GetFfxResourceDescriptionDX12");
    if (!real) return {};
    const auto desc = real(resource, additional_usages);
    oss_gaussian::ffx_backend_dx12::LogResourceDescription(
        "GetFfxResourceDescriptionDX12", desc);
    return desc;
}

OSSG_BACKEND_FWD_RET(uint32_t, OssgBackendGetSurfaceFormatDX12, "ffxGetSurfaceFormatDX12",
                     (int32_t format), (format), 0u)

OSSG_BACKEND_FWD_RET(int32_t, OssgBackendCreateFrameinterpolationSwapchainDX12,
                     "ffxCreateFrameinterpolationSwapchainDX12",
                     (void* desc, void** out_swapchain),
                     (desc, out_swapchain),
                     oss_gaussian::ffx_backend_dx12::kFfxErrorInvalidPointer)

OSSG_BACKEND_FWD_RET(int32_t, OssgBackendCreateFrameinterpolationSwapchainForHwndDX12,
                     "ffxCreateFrameinterpolationSwapchainForHwndDX12",
                     (void* hwnd, void* desc, void** out_swapchain),
                     (hwnd, desc, out_swapchain),
                     oss_gaussian::ffx_backend_dx12::kFfxErrorInvalidPointer)

OSSG_BACKEND_FWD_RET(int32_t, OssgBackendRegisterFrameinterpolationUiResourceDX12,
                     "ffxRegisterFrameinterpolationUiResourceDX12",
                     (void* swapchain, void* resource),
                     (swapchain, resource),
                     oss_gaussian::ffx_backend_dx12::kFfxErrorInvalidPointer)

OSSG_BACKEND_FWD_RET(int32_t, OssgBackendReplaceSwapchainForFrameinterpolationDX12,
                     "ffxReplaceSwapchainForFrameinterpolationDX12",
                     (void* command_queue, void* swapchain),
                     (command_queue, swapchain),
                     oss_gaussian::ffx_backend_dx12::kFfxErrorInvalidPointer)

OSSG_BACKEND_FWD_RET(int32_t, OssgBackendSetFrameGenerationConfigToSwapchainDX12,
                     "ffxSetFrameGenerationConfigToSwapchainDX12",
                     (void* swapchain, const void* config),
                     (swapchain, config),
                     oss_gaussian::ffx_backend_dx12::kFfxErrorInvalidPointer)

OSSG_BACKEND_FWD_RET(int32_t, OssgBackendWaitForPresents, "ffxWaitForPresents",
                     (void* swapchain), (swapchain),
                     oss_gaussian::ffx_backend_dx12::kFfxErrorInvalidPointer)

OSSG_BACKEND_FWD_RET(int32_t, OssgBackendFrameInterpolationUiComposition,
                     "?ffxFrameInterpolationUiComposition@@YAHPEBUFfxPresentCallbackDescription@@@Z",
                     (const void* desc), (desc),
                     oss_gaussian::ffx_backend_dx12::kFfxErrorInvalidPointer)

} // extern "C"
