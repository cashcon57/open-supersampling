// =============================================================================
//  test_ffx_api_proxy.cpp
//
//  Export-surface smoke for the generic FidelityFX API proxy target.
// =============================================================================
#include <Windows.h>

#include <cstdint>
#include <cstdio>

namespace {

constexpr const char* kRequiredExports[] = {
    "ffxConfigure",
    "ffxCreateContext",
    "ffxDestroyContext",
    "ffxDispatch",
    "ffxQuery",
};

constexpr uint64_t kFfxApiDispatchDescTypeUpscale = 0x00010001u;

struct FfxApiHeader {
    uint64_t type;
    FfxApiHeader* p_next;
};

struct FfxApiDimensions2D {
    uint32_t width;
    uint32_t height;
};

struct FfxApiFloatCoords2D {
    float x;
    float y;
};

struct FfxApiResourceDescription {
    uint32_t type;
    uint32_t format;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t mip_count;
    uint32_t flags;
    uint32_t usage;
};

struct FfxApiResource {
    void* resource;
    FfxApiResourceDescription description;
    uint32_t state;
};

struct FfxDispatchDescUpscale {
    FfxApiHeader header;
    void* command_list;
    FfxApiResource color;
    FfxApiResource depth;
    FfxApiResource motion_vectors;
    FfxApiResource exposure;
    FfxApiResource reactive;
    FfxApiResource transparency_and_composition;
    FfxApiResource output;
    FfxApiFloatCoords2D jitter_offset;
    FfxApiFloatCoords2D motion_vector_scale;
    FfxApiDimensions2D render_size;
    FfxApiDimensions2D upscale_size;
    bool enable_sharpening;
    float sharpness;
    float frame_time_delta;
    float pre_exposure;
    bool reset;
    float camera_near;
    float camera_far;
    float camera_fov_angle_vertical;
    float view_space_to_meters_factor;
    uint32_t flags;
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

    using DispatchFn = unsigned int (*)(void*, const FfxApiHeader*);
    auto dispatch = reinterpret_cast<DispatchFn>(GetProcAddress(proxy, "ffxDispatch"));
    if (!dispatch) {
        FreeLibrary(proxy);
        return Fail("ffxDispatch resolve");
    }
    FfxDispatchDescUpscale upscale{};
    upscale.header.type = kFfxApiDispatchDescTypeUpscale;
    upscale.color.resource = reinterpret_cast<void*>(0x1111);
    upscale.color.description.width = 1280;
    upscale.color.description.height = 720;
    upscale.depth.resource = reinterpret_cast<void*>(0x2222);
    upscale.motion_vectors.resource = reinterpret_cast<void*>(0x3333);
    upscale.output.resource = reinterpret_cast<void*>(0x4444);
    upscale.render_size = {1280, 720};
    upscale.upscale_size = {1920, 1080};
    upscale.motion_vector_scale = {1280.0f, 720.0f};
    upscale.pre_exposure = 1.0f;
    if (dispatch(nullptr, &upscale.header) != 6u) {
        FreeLibrary(proxy);
        return Fail("ffxDispatch should fail closed without real FidelityFX DLL");
    }

    FreeLibrary(proxy);
    std::printf("[test_ffx_api_proxy] OK: %zu exports resolved\n",
                sizeof(kRequiredExports) / sizeof(kRequiredExports[0]));
    return 0;
}
