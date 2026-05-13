// =============================================================================
//  test_ffx_backend_dx12_proxy.cpp
//
//  Export-surface smoke for the FidelityFX DX12 backend proxy target.
// =============================================================================
#include <Windows.h>

#include <cstdio>

namespace {

constexpr const char* kRequiredExports[] = {
    "?ffxFrameInterpolationUiComposition@@YAHPEBUFfxPresentCallbackDescription@@@Z",
    "GetFfxResourceDescriptionDX12",
    "ffxAssertReport",
    "ffxAssertSetPrintingCallback",
    "ffxCreateFrameinterpolationSwapchainDX12",
    "ffxCreateFrameinterpolationSwapchainForHwndDX12",
    "ffxGetCommandListDX12",
    "ffxGetCommandQueueDX12",
    "ffxGetDX12SwapchainPtr",
    "ffxGetDeviceDX12",
    "ffxGetFrameinterpolationCommandlistDX12",
    "ffxGetFrameinterpolationTextureDX12",
    "ffxGetInterfaceDX12",
    "ffxGetResourceDX12",
    "ffxGetScratchMemorySizeDX12",
    "ffxGetSurfaceFormatDX12",
    "ffxGetSwapchainDX12",
    "ffxRegisterFrameinterpolationUiResourceDX12",
    "ffxReplaceSwapchainForFrameinterpolationDX12",
    "ffxSetFrameGenerationConfigToSwapchainDX12",
    "ffxWaitForPresents",
};

int Fail(const char* msg, DWORD err = 0) {
    std::fprintf(stderr, "[test_ffx_backend_dx12_proxy] FAIL: %s (GetLastError=%lu)\n",
                 msg, err);
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_ffx_backend_dx12_proxy <path-to-dll>\n");
        return 2;
    }

    HMODULE proxy = LoadLibraryA(argv[1]);
    if (!proxy) return Fail("LoadLibraryA(proxy) failed", GetLastError());

    int missing = 0;
    for (const char* name : kRequiredExports) {
        if (!GetProcAddress(proxy, name)) {
            std::fprintf(stderr, "[test_ffx_backend_dx12_proxy] missing export: %s\n", name);
            ++missing;
        }
    }
    if (missing) {
        FreeLibrary(proxy);
        return Fail("one or more required exports missing");
    }

    using ScratchFn = size_t (*)(size_t);
    auto scratch = reinterpret_cast<ScratchFn>(
        GetProcAddress(proxy, "ffxGetScratchMemorySizeDX12"));
    if (!scratch) {
        FreeLibrary(proxy);
        return Fail("ffxGetScratchMemorySizeDX12 resolve");
    }
    if (scratch(1) != 0) {
        FreeLibrary(proxy);
        return Fail("ffxGetScratchMemorySizeDX12 should return 0 without real backend");
    }

    FreeLibrary(proxy);
    std::printf("[test_ffx_backend_dx12_proxy] OK: %zu exports resolved\n",
                sizeof(kRequiredExports) / sizeof(kRequiredExports[0]));
    return 0;
}
