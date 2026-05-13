// =============================================================================
//  test_ffx_api_proxy.cpp
//
//  Export-surface smoke for the generic FidelityFX API proxy target.
// =============================================================================
#include <Windows.h>

#include <cstdio>

namespace {

constexpr const char* kRequiredExports[] = {
    "ffxConfigure",
    "ffxCreateContext",
    "ffxDestroyContext",
    "ffxDispatch",
    "ffxQuery",
};

int Fail(const char* msg, DWORD err = 0) {
    std::fprintf(stderr, "[test_ffx_api_proxy] FAIL: %s (GetLastError=%lu)\n", msg, err);
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_ffx_api_proxy <path-to-dll>\n");
        return 2;
    }

    HMODULE proxy = LoadLibraryA(argv[1]);
    if (!proxy) return Fail("LoadLibraryA(proxy) failed", GetLastError());

    int missing = 0;
    for (const char* name : kRequiredExports) {
        if (!GetProcAddress(proxy, name)) {
            std::fprintf(stderr, "[test_ffx_api_proxy] missing export: %s\n", name);
            ++missing;
        }
    }
    if (missing) {
        FreeLibrary(proxy);
        return Fail("one or more required exports missing");
    }

    using QueryFn = unsigned int (*)(void*, void*);
    auto query = reinterpret_cast<QueryFn>(GetProcAddress(proxy, "ffxQuery"));
    if (!query) {
        FreeLibrary(proxy);
        return Fail("ffxQuery resolve");
    }
    if (query(nullptr, nullptr) != 6u) {
        FreeLibrary(proxy);
        return Fail("ffxQuery should fail closed without real FidelityFX DLL");
    }

    FreeLibrary(proxy);
    std::printf("[test_ffx_api_proxy] OK: %zu exports resolved\n",
                sizeof(kRequiredExports) / sizeof(kRequiredExports[0]));
    return 0;
}
