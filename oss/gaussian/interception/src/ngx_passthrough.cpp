// =============================================================================
//  ngx_passthrough.cpp
//
//  Loads the real nvngx_dlss.dll from a game/system location and resolves NGX
//  exports for the wrapper in ngx_exports.cpp.
// =============================================================================
#include "ngx_passthrough.h"

#include "log.h"

#include <Windows.h>

#include <mutex>
#include <string>
#include <vector>

namespace oss_gaussian::ngx {
namespace {

constexpr wchar_t kRealNgxDllName[] = L"nvngx_dlss.dll";
constexpr wchar_t kBackupNgxDllName[] = L"nvngx_dlss.dll.oss-backup";

std::once_flag g_load_once;
HMODULE        g_real_ngx = nullptr;

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
    const DWORD len = GetEnvironmentVariableW(L"OSS_NGX_REAL_DLL", value, MAX_PATH);
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
        candidates.push_back(JoinPath(exe_dir, kBackupNgxDllName));
        candidates.push_back(JoinPath(exe_dir, kRealNgxDllName));
    }

    wchar_t cwd[MAX_PATH] = {};
    const DWORD cwd_len = GetCurrentDirectoryW(MAX_PATH, cwd);
    if (cwd_len > 0 && cwd_len < MAX_PATH) {
        candidates.push_back(JoinPath(cwd, kBackupNgxDllName));
        candidates.push_back(JoinPath(cwd, kRealNgxDllName));
    }

    wchar_t system_dir[MAX_PATH] = {};
    const UINT system_len = GetSystemDirectoryW(system_dir, MAX_PATH);
    if (system_len > 0 && system_len < MAX_PATH) {
        candidates.push_back(JoinPath(system_dir, kRealNgxDllName));
    }

    return candidates;
}

HMODULE TryLoadCandidate(const std::wstring& candidate, const std::wstring& self_path) {
    if (candidate.empty()) return nullptr;
    if (!self_path.empty() && SamePath(candidate, self_path)) {
        OSSG_LOG_WARN("ngx", "skipping self while resolving real nvngx_dlss.dll");
        return nullptr;
    }

    const DWORD attrs = GetFileAttributesW(candidate.c_str());
    if (attrs == INVALID_FILE_ATTRIBUTES || (attrs & FILE_ATTRIBUTE_DIRECTORY)) {
        return nullptr;
    }

    HMODULE module = LoadLibraryExW(candidate.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) {
        OSSG_LOG_WARN("ngx", "failed to load real nvngx_dlss.dll candidate (err=%lu)",
                      GetLastError());
        return nullptr;
    }

    const std::wstring loaded_path = ModulePath(module);
    if (!self_path.empty() && !loaded_path.empty() && SamePath(loaded_path, self_path)) {
        OSSG_LOG_WARN("ngx", "resolved nvngx_dlss.dll to self; ignoring candidate");
        FreeLibrary(module);
        return nullptr;
    }

    return module;
}

void LoadRealNgxOnce() {
    const std::wstring self_path = SelfModulePath();
    const std::vector<std::wstring> candidates = BuildCandidatePaths();

    for (const std::wstring& candidate : candidates) {
        if (HMODULE module = TryLoadCandidate(candidate, self_path)) {
            g_real_ngx = module;
            OSSG_LOG_INFO("ngx", "loaded real nvngx_dlss.dll");
            return;
        }
    }

    OSSG_LOG_ERROR("ngx",
                   "real nvngx_dlss.dll not found in OSS_NGX_REAL_DLL, game backup, cwd, exe dir, or system32");
}

HMODULE RealNgxModule() {
    std::call_once(g_load_once, LoadRealNgxOnce);
    return g_real_ngx;
}

} // namespace

void* ResolveExport(const char* name) {
    if (!name || !*name) return nullptr;

    HMODULE module = RealNgxModule();
    if (!module) return nullptr;

    FARPROC proc = GetProcAddress(module, name);
    if (!proc) {
        OSSG_LOG_ERROR("ngx", "real nvngx_dlss.dll missing export: %s", name);
        return nullptr;
    }
    return reinterpret_cast<void*>(proc);
}

} // namespace oss_gaussian::ngx
