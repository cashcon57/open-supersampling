// =============================================================================
//  ffx_api_proxy.cpp
//
//  Pass-through proxy for AMD's generic FidelityFX SDK loader API.
// =============================================================================
#define OSS_GAUSSIAN_BUILDING_DLL 1

#include "ffx_api_proxy.h"

#include "log.h"

#include <Windows.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace oss_gaussian::ffx_api {

using ffxContext = void*;
using ffxReturnCode_t = uint32_t;
constexpr ffxReturnCode_t kFfxApiReturnErrorParameter = 6u;

struct ffxApiHeader {
    uint64_t type;
    ffxApiHeader* p_next;
};

struct ffxAllocationCallbacks {
    void* user_data;
    void* alloc;
    void* dealloc;
};

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

bool ShouldLog(uint64_t index) {
    return index <= 64 || (index % 120) == 0;
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
    using Fn = oss_gaussian::ffx_api::ffxReturnCode_t (*)(
        oss_gaussian::ffx_api::ffxContext*,
        oss_gaussian::ffx_api::ffxApiHeader*,
        const oss_gaussian::ffx_api::ffxAllocationCallbacks*);
    auto real = oss_gaussian::ffx_api::ResolveTyped<Fn>("ffxCreateContext");
    if (!real) return oss_gaussian::ffx_api::MissingReal("ffxCreateContext");
    return real(context, desc, mem_cb);
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
    return real(context, mem_cb);
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
    oss_gaussian::LogInit();
    const uint64_t call = ++oss_gaussian::ffx_api::g_dispatch_calls;
    if (oss_gaussian::ffx_api::ShouldLog(call)) {
        OSSG_LOG_INFO("ffx_api",
                      "ffxDispatch call=%llu context=%p",
                      static_cast<unsigned long long>(call),
                      context);
        oss_gaussian::ffx_api::LogHeaderChain("ffxDispatch", desc);
    }
    using Fn = oss_gaussian::ffx_api::ffxReturnCode_t (*)(
        oss_gaussian::ffx_api::ffxContext*,
        const oss_gaussian::ffx_api::ffxApiHeader*);
    auto real = oss_gaussian::ffx_api::ResolveTyped<Fn>("ffxDispatch");
    if (!real) return oss_gaussian::ffx_api::MissingReal("ffxDispatch");
    return real(context, desc);
}

} // extern "C"
