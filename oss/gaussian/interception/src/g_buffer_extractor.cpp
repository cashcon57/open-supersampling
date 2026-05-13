// =============================================================================
//  g_buffer_extractor.cpp
//
//  Modeled on OptiScaler's `Inputs/DLSSFeatureDx12.cpp::Evaluate(...)` (the
//  parameter-unpacking section), with the parameter-key constants matching
//  NVIDIA's `nvsdk_ngx_defs.h`. See:
//    - https://github.com/optiscaler/OptiScaler/blob/master/OptiScaler/Inputs/DLSSFeatureDx12.cpp
//    - https://github.com/NVIDIA/DLSS/blob/main/include/nvsdk_ngx_defs.h
//
//  This file intentionally mirrors only the public NVSDK_NGX_Parameter virtual
//  surface needed for reads. We do not vendor NVIDIA headers into this repo.
//
//  Copyright 2026 OSS-Gaussian contributors
//  Licensed under the Apache License, Version 2.0 (see ../LICENSE).
// =============================================================================
#include "g_buffer_extractor.h"

#include "log.h"

#include <d3d12.h>
#include <d3d11.h>

namespace oss_gaussian {

namespace {

constexpr uint64_t kFramesToLogAtInfo = 5;

constexpr int kNgxSuccess = 0x1;

// Parameter key strings. Names taken from `nvsdk_ngx_defs.h`. Keep in sync
// when the real NGX header is vendored in T2.4.
constexpr const char* kKeyColor             = "Color";
constexpr const char* kKeyOutput            = "Output";
constexpr const char* kKeyDepth             = "Depth";
constexpr const char* kKeyMotionVectors     = "MotionVectors";
constexpr const char* kKeyExposureTexture   = "ExposureTexture";
constexpr const char* kKeyJitterX           = "Jitter.Offset.X";
constexpr const char* kKeyJitterY           = "Jitter.Offset.Y";
constexpr const char* kKeyMVScaleX          = "MV.Scale.X";
constexpr const char* kKeyMVScaleY          = "MV.Scale.Y";
constexpr const char* kKeyExposureScale     = "DLSS.Exposure.Scale";
constexpr const char* kKeyReset             = "Reset";
constexpr const char* kKeyFeatureFlags      = "DLSS.Feature.Create.Flags";
constexpr const char* kKeySubrectBaseX      = "DLSS.Input.Color.Subrect.Base.X";
constexpr const char* kKeySubrectBaseY      = "DLSS.Input.Color.Subrect.Base.Y";
constexpr const char* kKeySubrectRenderW    = "DLSS.Render.Subrect.Dimensions.Width";
constexpr const char* kKeySubrectRenderH    = "DLSS.Render.Subrect.Dimensions.Height";
constexpr const char* kKeyOutputWidth       = "OutWidth";
constexpr const char* kKeyOutputHeight      = "OutHeight";

// Silence unused-warning until T2.6 wires these into real Get* calls.
[[maybe_unused]] constexpr const char* kAllKeys[] = {
    kKeyColor, kKeyOutput, kKeyDepth, kKeyMotionVectors, kKeyExposureTexture,
    kKeyJitterX, kKeyJitterY, kKeyMVScaleX, kKeyMVScaleY, kKeyExposureScale,
    kKeyReset, kKeyFeatureFlags,
    kKeySubrectBaseX, kKeySubrectBaseY, kKeySubrectRenderW, kKeySubrectRenderH,
    kKeyOutputWidth, kKeyOutputHeight,
};

// ABI mirror of NVIDIA's public NVSDK_NGX_Parameter C++ interface. Keep this
// in the exact documented virtual order; adding data members would break calls.
struct NgxParameterAbi {
    virtual void Set(const char*, unsigned long long) = 0;
    virtual void Set(const char*, float) = 0;
    virtual void Set(const char*, double) = 0;
    virtual void Set(const char*, unsigned int) = 0;
    virtual void Set(const char*, int) = 0;
    virtual void Set(const char*, ID3D11Resource*) = 0;
    virtual void Set(const char*, ID3D12Resource*) = 0;
    virtual void Set(const char*, void*) = 0;
    virtual int Get(const char*, unsigned long long*) const = 0;
    virtual int Get(const char*, float*) const = 0;
    virtual int Get(const char*, double*) const = 0;
    virtual int Get(const char*, unsigned int*) const = 0;
    virtual int Get(const char*, int*) const = 0;
    virtual int Get(const char*, ID3D11Resource**) const = 0;
    virtual int Get(const char*, ID3D12Resource**) const = 0;
    virtual int Get(const char*, void**) const = 0;
    virtual void Reset() = 0;
};

template <typename T>
bool TryGet(const NgxParameterAbi& params, const char* key, T* value) {
    return params.Get(key, value) == kNgxSuccess;
}

bool TryGetResource(const NgxParameterAbi& params, const char* key, void** value) {
    ID3D12Resource* resource = nullptr;
    if (params.Get(key, &resource) == kNgxSuccess) {
        *value = resource;
        return true;
    }
    return false;
}

void FillMissingDimensionsFromResources(OssGaussianFrame& frame) {
    if (frame.color &&
        (frame.subrect_render_width == 0 || frame.subrect_render_height == 0)) {
        auto* color = static_cast<ID3D12Resource*>(frame.color);
        const D3D12_RESOURCE_DESC desc = color->GetDesc();
        if (frame.subrect_render_width == 0) {
            frame.subrect_render_width = static_cast<uint32_t>(desc.Width);
        }
        if (frame.subrect_render_height == 0) {
            frame.subrect_render_height = desc.Height;
        }
    }

    if (frame.output && (frame.output_width == 0 || frame.output_height == 0)) {
        auto* output = static_cast<ID3D12Resource*>(frame.output);
        const D3D12_RESOURCE_DESC desc = output->GetDesc();
        if (frame.output_width == 0) {
            frame.output_width = static_cast<uint32_t>(desc.Width);
        }
        if (frame.output_height == 0) {
            frame.output_height = desc.Height;
        }
    }
}

} // namespace

OssGaussianStatus ReadFromNgxParameters(const NVSDK_NGX_Parameter* params,
                                        uint64_t                   frame_index,
                                        GBufferFrame*              out) {
    if (!params || !out) return OSS_GAUSSIAN_ERR_INVALID_ARG;

    // Zero-init the public struct so missing optional keys read as 0.
    out->pub = OssGaussianFrame{};
    out->pub.frame_index = frame_index;
    out->pub.exposure_scale = 1.0f;
    out->pub.mv_scale_x = 1.0f;
    out->pub.mv_scale_y = 1.0f;

    const auto& p = *reinterpret_cast<const NgxParameterAbi*>(params);
    TryGetResource(p, kKeyColor, &out->pub.color);
    TryGetResource(p, kKeyOutput, &out->pub.output);
    TryGetResource(p, kKeyDepth, &out->pub.depth);
    TryGetResource(p, kKeyMotionVectors, &out->pub.motion_vectors);
    TryGetResource(p, kKeyExposureTexture, &out->pub.exposure_texture);

    TryGet(p, kKeyJitterX, &out->pub.jitter_offset_x);
    TryGet(p, kKeyJitterY, &out->pub.jitter_offset_y);
    TryGet(p, kKeyMVScaleX, &out->pub.mv_scale_x);
    TryGet(p, kKeyMVScaleY, &out->pub.mv_scale_y);
    TryGet(p, kKeyExposureScale, &out->pub.exposure_scale);

    TryGet(p, kKeyReset, &out->pub.reset);
    TryGet(p, kKeyFeatureFlags, &out->pub.feature_create_flags);
    TryGet(p, kKeySubrectBaseX, &out->pub.subrect_base_x);
    TryGet(p, kKeySubrectBaseY, &out->pub.subrect_base_y);
    TryGet(p, kKeySubrectRenderW, &out->pub.subrect_render_width);
    TryGet(p, kKeySubrectRenderH, &out->pub.subrect_render_height);
    TryGet(p, kKeyOutputWidth, &out->pub.output_width);
    TryGet(p, kKeyOutputHeight, &out->pub.output_height);

    FillMissingDimensionsFromResources(out->pub);

    return OSS_GAUSSIAN_OK;
}

void LogFrameSummary(const GBufferFrame& frame) {
    const auto& f = frame.pub;
    LogLevel    lvl =
        (f.frame_index < kFramesToLogAtInfo) ? LogLevel::Info : LogLevel::Trace;

    LogFmt(lvl, "gbuf",
        "frame=%llu color=%p depth=%p mv=%p output=%p "
        "jitter=(%.4f,%.4f) mv_scale=(%.4f,%.4f) "
        "reset=%u flags=0x%x render=%ux%u out=%ux%u",
        static_cast<unsigned long long>(f.frame_index),
        f.color, f.depth, f.motion_vectors, f.output,
        f.jitter_offset_x, f.jitter_offset_y,
        f.mv_scale_x,      f.mv_scale_y,
        f.reset, f.feature_create_flags,
        f.subrect_render_width, f.subrect_render_height,
        f.output_width,         f.output_height);
}

} // namespace oss_gaussian
