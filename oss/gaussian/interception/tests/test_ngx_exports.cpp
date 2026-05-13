// =============================================================================
//  test_ngx_exports.cpp
//
//  Resolution-only smoke test for the NGX export surface. This intentionally
//  does not call into the functions: on developer machines without NVIDIA NGX,
//  calls would correctly fail, and on machines with NGX loaded, null D3D12
//  arguments would be unsafe for the real runtime.
// =============================================================================
#include <Windows.h>

#include <cstdio>

namespace {

constexpr const char* kRequiredNgxExports[] = {
    "NVSDK_NGX_D3D12_Init",
    "NVSDK_NGX_D3D12_Init_Ext",
    "NVSDK_NGX_D3D12_Shutdown1",
    "NVSDK_NGX_D3D12_GetCapabilityParameters",
    "NVSDK_NGX_D3D12_AllocateParameters",
    "NVSDK_NGX_D3D12_DestroyParameters",
    "NVSDK_NGX_D3D12_CreateFeature",
    "NVSDK_NGX_D3D12_EvaluateFeature",
    "NVSDK_NGX_D3D12_ReleaseFeature",
    "NVSDK_NGX_D3D12_GetScratchBufferSize",
};

int Fail(const char* msg, DWORD err = 0) {
    std::fprintf(stderr, "[test_ngx_exports] FAIL: %s (GetLastError=%lu)\n", msg, err);
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_ngx_exports <path-to-dxgi.dll>\n");
        return 2;
    }

    HMODULE proxy = LoadLibraryA(argv[1]);
    if (!proxy) return Fail("LoadLibraryA(proxy) failed", GetLastError());

    int missing = 0;
    for (const char* name : kRequiredNgxExports) {
        if (!GetProcAddress(proxy, name)) {
            std::fprintf(stderr, "[test_ngx_exports] missing export: %s\n", name);
            ++missing;
        }
    }

    FreeLibrary(proxy);
    if (missing) return Fail("one or more NGX exports missing");

    std::printf("[test_ngx_exports] OK: %zu NGX exports resolved\n",
                sizeof(kRequiredNgxExports) / sizeof(kRequiredNgxExports[0]));
    return 0;
}
