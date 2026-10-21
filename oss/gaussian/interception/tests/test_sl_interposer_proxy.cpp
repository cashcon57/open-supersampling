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

struct ExportSpec {
    const char* name;
    WORD ordinal;
};

constexpr ExportSpec kRequiredExports[] = {
    {"CreateDXGIFactory", 1},
    {"CreateDXGIFactory1", 2},
    {"CreateDXGIFactory2", 3},
    {"D3D11CreateDevice", 4},
    {"D3D11CreateDeviceAndSwapChain", 5},
    {"D3D12CreateDevice", 6},
    {"D3D12CreateRootSignatureDeserializer", 7},
    {"D3D12CreateVersionedRootSignatureDeserializer", 8},
    {"D3D12EnableExperimentalFeatures", 9},
    {"D3D12GetDebugInterface", 10},
    {"D3D12GetInterface", 11},
    {"D3D12SerializeRootSignature", 12},
    {"D3D12SerializeVersionedRootSignature", 13},
    {"DXGIGetDebugInterface1", 14},
    {"slAllocateResources", 15},
    {"slEvaluateFeature", 16},
    {"slFreeResources", 17},
    {"slGetFeatureFunction", 18},
    {"slGetFeatureRequirements", 19},
    {"slGetFeatureVersion", 20},
    {"slGetNativeInterface", 21},
    {"slGetNewFrameToken", 22},
    {"slInit", 23},
    {"slIsFeatureLoaded", 24},
    {"slIsFeatureSupported", 25},
    {"slSetConstants", 26},
    {"slSetD3DDevice", 27},
    {"slSetFeatureLoaded", 28},
    {"slSetTag", 29},
    {"slSetVulkanInfo", 30},
    {"slShutdown", 31},
    {"slUpgradeInterface", 32},
};

constexpr const char* kForbiddenExports[] = {
    "OssgSlEvaluateFeature",
    "OssgSlGetFeatureFunction",
    "OssgSlInit",
    "OssgSlSetD3DDevice",
    "OssgSlSetTag",
    "OssgSlSetTagForFrame",
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
    for (const ExportSpec& spec : kRequiredExports) {
        FARPROC by_name = GetProcAddress(proxy, spec.name);
        FARPROC by_ordinal = GetProcAddress(proxy, MAKEINTRESOURCEA(spec.ordinal));
        if (!by_name || !by_ordinal) {
            std::fprintf(stderr, "[test_sl_interposer_proxy] missing export: %s @%u\n",
                         spec.name, spec.ordinal);
            ++missing;
        } else if (by_name != by_ordinal) {
            std::fprintf(stderr,
                         "[test_sl_interposer_proxy] name/ordinal mismatch: %s @%u\n",
                         spec.name, spec.ordinal);
            ++missing;
        }
    }
    for (const char* name : kForbiddenExports) {
        if (GetProcAddress(proxy, name)) {
            std::fprintf(stderr, "[test_sl_interposer_proxy] unexpected export: %s\n", name);
            ++missing;
        }
    }
    if (!GetProcAddress(proxy, "slSetTagForFrame")) {
        std::fprintf(stderr, "[test_sl_interposer_proxy] missing compatibility export: slSetTagForFrame\n");
        ++missing;
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
    std::printf("[test_sl_interposer_proxy] OK: %zu ordinal exports resolved; missing-real call failed closed\n",
                sizeof(kRequiredExports) / sizeof(kRequiredExports[0]));
    return 0;
}
