// =============================================================================
//  g_buffer_extractor.cpp
//
//  Modeled on OptiScaler's `Inputs/DLSSFeatureDx12.cpp::Evaluate(...)` (the
//  parameter-unpacking section), with the parameter-key constants matching
//  NVIDIA's `nvsdk_ngx_defs.h`. See:
//    - https://github.com/optiscaler/OptiScaler/blob/master/OptiScaler/Inputs/DLSSFeatureDx12.cpp
//    - https://github.com/NVIDIA/DLSS/blob/main/include/nvsdk_ngx_defs.h
//
//  Sprint 2 scaffolding ONLY: stubs out the per-key reads. T2.6 wires them to
//  real NVSDK_NGX_Parameter_GetVoidPointer / GetFloat / GetUInt calls once
//  the NGX headers are vendored in `third_party/DLSS/`.
//
//  Copyright 2026 OSS-Gaussian contributors
//  Licensed under the Apache License, Version 2.0 (see ../LICENSE).
// =============================================================================
#include "g_buffer_extractor.h"

#include "log.h"

namespace oss_gaussian {

namespace {

constexpr uint64_t kFramesToLogAtInfo = 5;

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
constexpr const char* kKeyExposureScale     = "Exposure.Scale";
constexpr const char* kKeyReset             = "Reset";
constexpr const char* kKeyFeatureFlags      = "DLSS.Feature.Create.Flags";
constexpr const char* kKeySubrectBaseX      = "Subrect.Base.X";
constexpr const char* kKeySubrectBaseY      = "Subrect.Base.Y";
constexpr const char* kKeySubrectRenderW    = "Subrect.Rendering.Width";
constexpr const char* kKeySubrectRenderH    = "Subrect.Rendering.Height";
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

} // namespace

OssGaussianStatus ReadFromNgxParameters(const NVSDK_NGX_Parameter* params,
                                        uint64_t                   frame_index,
                                        GBufferFrame*              out) {
    if (!params || !out) return OSS_GAUSSIAN_ERR_INVALID_ARG;

    // Zero-init the public struct so missing optional keys read as 0.
    out->pub = OssGaussianFrame{};
    out->pub.frame_index = frame_index;

    // -------------------------------------------------------------------------
    // T2.6 will replace this body with real NGX parameter reads:
    //
    //   params->Get(kKeyColor,         &out->pub.color);
    //   params->Get(kKeyOutput,        &out->pub.output);
    //   params->Get(kKeyDepth,         &out->pub.depth);
    //   params->Get(kKeyMotionVectors, &out->pub.motion_vectors);
    //   params->Get(kKeyJitterX,       &out->pub.jitter_offset_x);
    //   ... etc for every key in kAllKeys[] ...
    //
    // For Sprint 2 scaffolding, we just acknowledge the call and return OK.
    // The DLL is not wired into the game yet, so this path is unreachable.
    // -------------------------------------------------------------------------

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
