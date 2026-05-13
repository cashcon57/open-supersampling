// =============================================================================
//  test_version_proxy.cpp
//
//  Smoke test for the optional VERSION.dll loader profile.
// =============================================================================
#include <Windows.h>

#include <chrono>
#include <cstdio>
#include <thread>

namespace {

int Fail(const char* msg, DWORD err = 0) {
    std::fprintf(stderr, "[test_version_proxy] FAIL: %s (GetLastError=%lu)\n", msg, err);
    return 1;
}

constexpr const char* kRequiredExports[] = {
    "GetFileVersionInfoA",
    "GetFileVersionInfoByHandle",
    "GetFileVersionInfoExA",
    "GetFileVersionInfoExW",
    "GetFileVersionInfoSizeA",
    "GetFileVersionInfoSizeExA",
    "GetFileVersionInfoSizeExW",
    "GetFileVersionInfoSizeW",
    "GetFileVersionInfoW",
    "VerFindFileA",
    "VerFindFileW",
    "VerInstallFileA",
    "VerInstallFileW",
    "VerLanguageNameA",
    "VerLanguageNameW",
    "VerQueryValueA",
    "VerQueryValueW",
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_version_proxy <path-to-version.dll>\n");
        return 2;
    }

    HMODULE proxy = LoadLibraryA(argv[1]);
    if (!proxy) return Fail("LoadLibraryA(proxy) failed", GetLastError());

    using InitFn = unsigned int(WINAPI*)();
    auto pIsInitialized =
        reinterpret_cast<InitFn>(GetProcAddress(proxy, "oss_gaussian_is_initialized"));
    if (!pIsInitialized) {
        FreeLibrary(proxy);
        return Fail("oss_gaussian_is_initialized resolve");
    }
    bool initialized = false;
    for (int i = 0; i < 100; ++i) {
        if (pIsInitialized() != 0) {
            initialized = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    if (!initialized) {
        FreeLibrary(proxy);
        return Fail("async DLL initializer did not complete");
    }

    int missing = 0;
    for (const char* name : kRequiredExports) {
        if (!GetProcAddress(proxy, name)) {
            std::fprintf(stderr, "[test_version_proxy] missing export: %s\n", name);
            ++missing;
        }
    }
    if (missing) {
        FreeLibrary(proxy);
        return Fail("one or more required exports missing");
    }

    using VerLanguageNameAFn = DWORD(WINAPI*)(DWORD, LPSTR, DWORD);
    auto pLang = reinterpret_cast<VerLanguageNameAFn>(GetProcAddress(proxy, "VerLanguageNameA"));
    char buf[128] = {};
    DWORD n = pLang ? pLang(0x0409, buf, static_cast<DWORD>(sizeof(buf))) : 0;
    if (n == 0 || buf[0] == '\0') {
        FreeLibrary(proxy);
        return Fail("VerLanguageNameA forward returned no language");
    }

    FreeLibrary(proxy);
    std::printf("[test_version_proxy] OK: VERSION exports resolved, language=%s\n", buf);
    return 0;
}
