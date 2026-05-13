// =============================================================================
//  test_ffx_fsr3_proxy.cpp
//
//  Export-surface smoke for the FSR3 proxy target. This does not require AMD
//  FSR binaries; known wrapper calls should fail closed when the real DLL is
//  absent.
// =============================================================================
#include <Windows.h>

#include <cstdint>
#include <cstdio>

namespace {

constexpr const char* kRequiredExports[] = {
    "ffxAssertReport",
    "ffxAssertSetPrintingCallback",
    "ffxFsr3ConfigureFrameGeneration",
    "ffxFsr3ContextCreate",
    "ffxFsr3ContextDestroy",
    "ffxFsr3ContextDispatchUpscale",
    "ffxFsr3ContextGenerateReactiveMask",
    "ffxFsr3DispatchFrameGeneration",
    "ffxFsr3GetJitterOffset",
    "ffxFsr3GetJitterPhaseCount",
    "ffxFsr3GetRenderResolutionFromQualityMode",
    "ffxFsr3GetUpscaleRatioFromQualityMode",
    "ffxFsr3ResourceIsNull",
    "ffxFsr3SkipPresent",
    "ffxSafeReleaseCopyResource",
    "ffxSafeReleasePipeline",
    "ffxSafeReleaseResource",
};

int Fail(const char* msg, DWORD err = 0) {
    std::fprintf(stderr, "[test_ffx_fsr3_proxy] FAIL: %s (GetLastError=%lu)\n", msg, err);
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_ffx_fsr3_proxy <path-to-ffx_fsr3_x64.dll>\n");
        return 2;
    }

    HMODULE proxy = LoadLibraryA(argv[1]);
    if (!proxy) return Fail("LoadLibraryA(proxy) failed", GetLastError());

    int missing = 0;
    for (const char* name : kRequiredExports) {
        if (!GetProcAddress(proxy, name)) {
            std::fprintf(stderr, "[test_ffx_fsr3_proxy] missing export: %s\n", name);
            ++missing;
        }
    }
    if (missing) {
        FreeLibrary(proxy);
        return Fail("one or more required exports missing");
    }

    using DispatchFn = int32_t (*)(void*, const void*);
    auto dispatch = reinterpret_cast<DispatchFn>(
        GetProcAddress(proxy, "ffxFsr3ContextDispatchUpscale"));
    if (!dispatch) {
        FreeLibrary(proxy);
        return Fail("ffxFsr3ContextDispatchUpscale resolve");
    }
    const int32_t dispatch_result = dispatch(nullptr, nullptr);
    if (dispatch_result == 0) {
        FreeLibrary(proxy);
        return Fail("ffxFsr3ContextDispatchUpscale unexpectedly succeeded without real FSR3");
    }

    using RatioFn = float (*)(int32_t);
    auto ratio = reinterpret_cast<RatioFn>(
        GetProcAddress(proxy, "ffxFsr3GetUpscaleRatioFromQualityMode"));
    if (!ratio) {
        FreeLibrary(proxy);
        return Fail("ffxFsr3GetUpscaleRatioFromQualityMode resolve");
    }
    if (ratio(0) != 0.0f) {
        FreeLibrary(proxy);
        return Fail("ffxFsr3GetUpscaleRatioFromQualityMode should return 0 without real FSR3");
    }

    FreeLibrary(proxy);
    std::printf("[test_ffx_fsr3_proxy] OK: %zu exports resolved; missing-real calls failed closed\n",
                sizeof(kRequiredExports) / sizeof(kRequiredExports[0]));
    return 0;
}
