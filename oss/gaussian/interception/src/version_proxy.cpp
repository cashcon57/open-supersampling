// =============================================================================
//  version_proxy.cpp
//
//  Forwarders for the small VERSION.dll export surface. This lets the capture
//  DLL use `version.dll` as a game-local loader profile while preserving normal
//  VERSION API behavior by forwarding every call to System32\version.dll.
// =============================================================================
#include "version_proxy.h"

#include "log.h"

#include <Windows.h>

#include <mutex>

namespace oss_gaussian {

namespace {

HMODULE g_systemVersion = nullptr;
std::mutex g_systemVersionMu;

bool EnsureSystemVersionLoadedLocked() {
    if (g_systemVersion) return true;
    g_systemVersion = LoadLibraryW(L"C:\\Windows\\System32\\version.dll");
    if (!g_systemVersion) {
        OSSG_LOG_ERROR("version_proxy",
                       "LoadLibraryW(System32\\version.dll) failed, GetLastError=%lu",
                       GetLastError());
        return false;
    }
    OSSG_LOG_INFO("version_proxy", "system32 version.dll loaded at %p",
                  reinterpret_cast<void*>(g_systemVersion));
    return true;
}

template <typename Fn>
Fn ResolveVersionOnce(const char* name, Fn& cache) {
    if (cache) return cache;

    std::lock_guard<std::mutex> lk(g_systemVersionMu);
    if (cache) return cache;
    if (!EnsureSystemVersionLoadedLocked()) return nullptr;
    cache = reinterpret_cast<Fn>(GetProcAddress(g_systemVersion, name));
    if (!cache) {
        OSSG_LOG_WARN("version_proxy",
                      "%s: forward miss; system32 version.dll did not export this symbol",
                      name);
    }
    return cache;
}

} // namespace

bool OssGaussianVersionProxyAttach() {
    std::lock_guard<std::mutex> lk(g_systemVersionMu);
    return EnsureSystemVersionLoadedLocked();
}

void OssGaussianVersionProxyDetach() {
    std::lock_guard<std::mutex> lk(g_systemVersionMu);
    if (g_systemVersion) {
        FreeLibrary(g_systemVersion);
        g_systemVersion = nullptr;
    }
}

} // namespace oss_gaussian

extern "C" {

BOOL WINAPI OssgGetFileVersionInfoA(LPCSTR file, DWORD handle, DWORD len, LPVOID data) {
    using Fn = BOOL(WINAPI*)(LPCSTR, DWORD, DWORD, LPVOID);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("GetFileVersionInfoA", fn);
    return real ? real(file, handle, len, data) : FALSE;
}

BOOL WINAPI OssgGetFileVersionInfoW(LPCWSTR file, DWORD handle, DWORD len, LPVOID data) {
    using Fn = BOOL(WINAPI*)(LPCWSTR, DWORD, DWORD, LPVOID);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("GetFileVersionInfoW", fn);
    return real ? real(file, handle, len, data) : FALSE;
}

BOOL WINAPI OssgGetFileVersionInfoExA(DWORD flags, LPCSTR file, DWORD handle, DWORD len, LPVOID data) {
    using Fn = BOOL(WINAPI*)(DWORD, LPCSTR, DWORD, DWORD, LPVOID);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("GetFileVersionInfoExA", fn);
    return real ? real(flags, file, handle, len, data) : FALSE;
}

BOOL WINAPI OssgGetFileVersionInfoExW(DWORD flags, LPCWSTR file, DWORD handle, DWORD len, LPVOID data) {
    using Fn = BOOL(WINAPI*)(DWORD, LPCWSTR, DWORD, DWORD, LPVOID);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("GetFileVersionInfoExW", fn);
    return real ? real(flags, file, handle, len, data) : FALSE;
}

DWORD WINAPI OssgGetFileVersionInfoByHandle(DWORD handle, DWORD zero, DWORD len, LPVOID data) {
    using Fn = DWORD(WINAPI*)(DWORD, DWORD, DWORD, LPVOID);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("GetFileVersionInfoByHandle", fn);
    return real ? real(handle, zero, len, data) : 0;
}

DWORD WINAPI OssgGetFileVersionInfoSizeA(LPCSTR file, LPDWORD handle) {
    using Fn = DWORD(WINAPI*)(LPCSTR, LPDWORD);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("GetFileVersionInfoSizeA", fn);
    return real ? real(file, handle) : 0;
}

DWORD WINAPI OssgGetFileVersionInfoSizeW(LPCWSTR file, LPDWORD handle) {
    using Fn = DWORD(WINAPI*)(LPCWSTR, LPDWORD);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("GetFileVersionInfoSizeW", fn);
    return real ? real(file, handle) : 0;
}

DWORD WINAPI OssgGetFileVersionInfoSizeExA(DWORD flags, LPCSTR file, LPDWORD handle) {
    using Fn = DWORD(WINAPI*)(DWORD, LPCSTR, LPDWORD);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("GetFileVersionInfoSizeExA", fn);
    return real ? real(flags, file, handle) : 0;
}

DWORD WINAPI OssgGetFileVersionInfoSizeExW(DWORD flags, LPCWSTR file, LPDWORD handle) {
    using Fn = DWORD(WINAPI*)(DWORD, LPCWSTR, LPDWORD);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("GetFileVersionInfoSizeExW", fn);
    return real ? real(flags, file, handle) : 0;
}

DWORD WINAPI OssgVerFindFileA(DWORD flags, LPSTR file, LPSTR win_dir, LPSTR app_dir,
                              LPSTR cur_dir, PUINT cur_dir_len, LPSTR dest_dir,
                              PUINT dest_dir_len) {
    using Fn = DWORD(WINAPI*)(DWORD, LPSTR, LPSTR, LPSTR, LPSTR, PUINT, LPSTR, PUINT);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("VerFindFileA", fn);
    return real ? real(flags, file, win_dir, app_dir, cur_dir, cur_dir_len, dest_dir, dest_dir_len) : 0;
}

DWORD WINAPI OssgVerFindFileW(DWORD flags, LPWSTR file, LPWSTR win_dir, LPWSTR app_dir,
                              LPWSTR cur_dir, PUINT cur_dir_len, LPWSTR dest_dir,
                              PUINT dest_dir_len) {
    using Fn = DWORD(WINAPI*)(DWORD, LPWSTR, LPWSTR, LPWSTR, LPWSTR, PUINT, LPWSTR, PUINT);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("VerFindFileW", fn);
    return real ? real(flags, file, win_dir, app_dir, cur_dir, cur_dir_len, dest_dir, dest_dir_len) : 0;
}

DWORD WINAPI OssgVerInstallFileA(DWORD flags, LPSTR src_file, LPSTR dest_file,
                                 LPSTR src_dir, LPSTR dest_dir, LPSTR cur_dir,
                                 LPSTR tmp_file, PUINT tmp_file_len) {
    using Fn = DWORD(WINAPI*)(DWORD, LPSTR, LPSTR, LPSTR, LPSTR, LPSTR, LPSTR, PUINT);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("VerInstallFileA", fn);
    return real ? real(flags, src_file, dest_file, src_dir, dest_dir, cur_dir, tmp_file, tmp_file_len) : 0;
}

DWORD WINAPI OssgVerInstallFileW(DWORD flags, LPWSTR src_file, LPWSTR dest_file,
                                 LPWSTR src_dir, LPWSTR dest_dir, LPWSTR cur_dir,
                                 LPWSTR tmp_file, PUINT tmp_file_len) {
    using Fn = DWORD(WINAPI*)(DWORD, LPWSTR, LPWSTR, LPWSTR, LPWSTR, LPWSTR, LPWSTR, PUINT);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("VerInstallFileW", fn);
    return real ? real(flags, src_file, dest_file, src_dir, dest_dir, cur_dir, tmp_file, tmp_file_len) : 0;
}

DWORD WINAPI OssgVerLanguageNameA(DWORD lang, LPSTR lang_name, DWORD lang_name_len) {
    using Fn = DWORD(WINAPI*)(DWORD, LPSTR, DWORD);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("VerLanguageNameA", fn);
    return real ? real(lang, lang_name, lang_name_len) : 0;
}

DWORD WINAPI OssgVerLanguageNameW(DWORD lang, LPWSTR lang_name, DWORD lang_name_len) {
    using Fn = DWORD(WINAPI*)(DWORD, LPWSTR, DWORD);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("VerLanguageNameW", fn);
    return real ? real(lang, lang_name, lang_name_len) : 0;
}

BOOL WINAPI OssgVerQueryValueA(LPCVOID block, LPCSTR sub_block, LPVOID* buffer, PUINT len) {
    using Fn = BOOL(WINAPI*)(LPCVOID, LPCSTR, LPVOID*, PUINT);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("VerQueryValueA", fn);
    return real ? real(block, sub_block, buffer, len) : FALSE;
}

BOOL WINAPI OssgVerQueryValueW(LPCVOID block, LPCWSTR sub_block, LPVOID* buffer, PUINT len) {
    using Fn = BOOL(WINAPI*)(LPCVOID, LPCWSTR, LPVOID*, PUINT);
    static Fn fn = nullptr;
    auto real = oss_gaussian::ResolveVersionOnce<Fn>("VerQueryValueW", fn);
    return real ? real(block, sub_block, buffer, len) : FALSE;
}

} // extern "C"
