// =============================================================================
//  g_buffer_extractor.h
//
//  Reads G-buffer-relevant resources + scalars from an NVSDK_NGX_Parameter
//  dictionary and presents them to the rest of the DLL as a typed struct.
//
//  Modeled on OptiScaler's `Inputs/DLSSFeatureDx12.cpp::Evaluate(...)` parameter
//  unpacking pattern (https://github.com/optiscaler/OptiScaler). Sprint 2 only
//  reads the dict; we do not yet copy resources to readback heaps — that lands
//  in T2.7 (EXR dump pipeline).
//
//  Copyright 2026 OSS-Gaussian contributors
//  Licensed under the Apache License, Version 2.0 (see ../LICENSE).
// =============================================================================
#ifndef OSS_GAUSSIAN_GBUFFER_EXTRACTOR_H
#define OSS_GAUSSIAN_GBUFFER_EXTRACTOR_H

#include <cstdint>

#include "../include/oss_gaussian_interception.h"

// Forward-declare the NGX parameter type to avoid pulling nvsdk_ngx.h into the
// scaffold. The real header is added when T2.4 vendors it under third_party/.
struct NVSDK_NGX_Parameter;

namespace oss_gaussian {

/// Snapshot of one EvaluateFeature call. Mirrors `OssGaussianFrame` but lives
/// in C++ space; the public C struct is what we hand to callbacks.
struct GBufferFrame {
    OssGaussianFrame pub{}; // populated; passed verbatim to client callbacks.

    // Internal-only diagnostics (resource descs, format ids) live here once
    // T2.6 lands. Keep struct trivially-copyable for now.
};

/// Extract every G-buffer key from the NGX parameter dict into `out`.
/// Returns OSS_GAUSSIAN_OK on success, OSS_GAUSSIAN_ERR_INVALID_ARG on null
/// inputs. Missing optional keys are zeroed silently.
///
/// `frame_index` is set by the caller (a per-DLL counter incremented in
/// EvaluateFeature).
OssGaussianStatus ReadFromNgxParameters(
    const NVSDK_NGX_Parameter* params,
    uint64_t                   frame_index,
    GBufferFrame*              out);

/// Log a one-line summary of the frame at INFO level for the first N frames,
/// TRACE thereafter. Cheap; safe to call every EvaluateFeature.
void LogFrameSummary(const GBufferFrame& frame);

} // namespace oss_gaussian

#endif // OSS_GAUSSIAN_GBUFFER_EXTRACTOR_H
