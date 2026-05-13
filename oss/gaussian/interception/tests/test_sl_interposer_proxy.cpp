// =============================================================================
//  test_sl_interposer_proxy.cpp
//
//  Export-surface smoke for the Streamline proxy target. This does not require
//  NVIDIA Streamline binaries; wrapper calls should fail closed when the real
//  `oss_sl_real.dll` is absent.
// =============================================================================
#include <Windows.h>

#include <cstdint>
#include <cstdio>

namespace {

constexpr const char* kRequiredExports[] = {
    "CreateDXGIFactory",
    "CreateDXGIFactory1",
    "CreateDXGIFactory2",
    "D3D11CreateDevice",
    "D3D11CreateDeviceAndSwapChain",
    "D3D12CreateDevice",
    "D3D12CreateRootSignatureDeserializer",
    "D3D12CreateVersionedRootSignatureDeserializer",
    "D3D12EnableExperimentalFeatures",
    "D3D12GetDebugInterface",
    "D3D12GetInterface",
    "D3D12SerializeRootSignature",
    "D3D12SerializeVersionedRootSignature",
    "DXGIGetDebugInterface1",
    "slAllocateResources",
    "slEvaluateFeature",
    "slFreeResources",
    "slGetFeatureFunction",
    "slGetFeatureRequirements",
    "slGetFeatureVersion",
    "slGetNativeInterface",
    "slGetNewFrameToken",
    "slInit",
    "slIsFeatureLoaded",
    "slIsFeatureSupported",
    "slSetConstants",
    "slSetD3DDevice",
    "slSetFeatureLoaded",
    "slSetTag",
    "slSetTagForFrame",
    "slSetVulkanInfo",
    "slShutdown",
    "slUpgradeInterface",
};

int Fail(const char* msg, DWORD err = 0) {
    std::fprintf(stderr, "[test_sl_interposer_proxy] FAIL: %s (GetLastError=%lu)\n", msg, err);
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_sl_interposer_proxy <path-to-sl.interposer.dll>\n");
        return 2;
    }

    HMODULE proxy = LoadLibraryA(argv[1]);
    if (!proxy) return Fail("LoadLibraryA(proxy) failed", GetLastError());

    int missing = 0;
    for (const char* name : kRequiredExports) {
        if (!GetProcAddress(proxy, name)) {
            std::fprintf(stderr, "[test_sl_interposer_proxy] missing export: %s\n", name);
            ++missing;
        }
    }
    if (missing) {
        FreeLibrary(proxy);
        return Fail("one or more required exports missing");
    }

    using EvaluateFn = int32_t (*)(uint32_t, const void*, const void**, uint32_t, void*);
    auto evaluate = reinterpret_cast<EvaluateFn>(GetProcAddress(proxy, "slEvaluateFeature"));
    if (!evaluate) {
        FreeLibrary(proxy);
        return Fail("slEvaluateFeature resolve");
    }
    const int32_t result = evaluate(0, nullptr, nullptr, 0, nullptr);
    if (result == 0) {
        FreeLibrary(proxy);
        return Fail("slEvaluateFeature unexpectedly succeeded without real Streamline");
    }

    FreeLibrary(proxy);
    std::printf("[test_sl_interposer_proxy] OK: %zu exports resolved; missing-real call failed closed\n",
                sizeof(kRequiredExports) / sizeof(kRequiredExports[0]));
    return 0;
}
