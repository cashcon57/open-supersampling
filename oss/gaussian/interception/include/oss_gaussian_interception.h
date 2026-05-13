// =============================================================================
//  oss_gaussian_interception.h
//
//  Public C API for the OSS-Gaussian D3D12 interception DLL. The eventual
//  Gaussian canvas + renderer (Sprint 4 / Sprint 5) consume this surface.
//
//  Copyright 2026 OSS-Gaussian contributors
//  Licensed under the Apache License, Version 2.0 (see ../LICENSE).
//
//  Modeled on the public surface of OptiScaler's `Inputs/IFeature_Dx12.h`
//  (https://github.com/optiscaler/OptiScaler) — pared down to what Sprint 2
//  actually exercises. Everything beyond Sprint 2 (live shared-memory IPC,
//  callback-based per-frame consumption) is declared here as an opaque hook
//  so callers do not need to recompile when Sprint 5 lights it up.
// =============================================================================
#ifndef OSS_GAUSSIAN_INTERCEPTION_H
#define OSS_GAUSSIAN_INTERCEPTION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(OSS_GAUSSIAN_BUILDING_DLL)
#    define OSS_GAUSSIAN_API __declspec(dllexport)
#  else
#    define OSS_GAUSSIAN_API __declspec(dllimport)
#  endif
#else
#  define OSS_GAUSSIAN_API
#endif

// -----------------------------------------------------------------------------
//  Status codes. HRESULT-shaped so the C++ side can interop with the rest of
//  the D3D12 surface without translation.
// -----------------------------------------------------------------------------
typedef int32_t OssGaussianStatus;
#define OSS_GAUSSIAN_OK                  ((OssGaussianStatus) 0)
#define OSS_GAUSSIAN_ERR_NOT_INITIALIZED ((OssGaussianStatus)-1)
#define OSS_GAUSSIAN_ERR_INVALID_ARG     ((OssGaussianStatus)-2)
#define OSS_GAUSSIAN_ERR_NOT_IMPLEMENTED ((OssGaussianStatus)-3)
#define OSS_GAUSSIAN_ERR_INTERNAL        ((OssGaussianStatus)-4)

// -----------------------------------------------------------------------------
//  Per-frame G-buffer descriptor handed to client callbacks.
//
//  Pointer-typed fields are `void*` to keep this header free of D3D12 includes;
//  callers cast to `ID3D12Resource*` on the C++ side.
//
//  Field names match NVSDK_NGX_Parameter keys
//  (`nvsdk_ngx_defs.h::NVSDK_NGX_Parameter_*`).
// -----------------------------------------------------------------------------
typedef struct OssGaussianFrame {
    uint64_t frame_index;          // Monotonic, set by the DLL.

    void*    color;                // ID3D12Resource* — input low-res HDR color.
    void*    output;               // ID3D12Resource* — UAV we are expected to fill.
    void*    depth;                // ID3D12Resource* — depth buffer.
    void*    motion_vectors;       // ID3D12Resource* — RG, scaled by mv_scale_x/y.
    void*    exposure_texture;     // ID3D12Resource* — optional autoexposure (may be null).

    float    jitter_offset_x;      // Subpixel jitter in pixels.
    float    jitter_offset_y;
    float    mv_scale_x;           // Multiplier on motion-vector channels.
    float    mv_scale_y;
    float    exposure_scale;       // Optional scalar exposure (1.0 if unused).

    uint32_t reset;                // Camera-cut signal. Non-zero => prior history invalid.
    uint32_t feature_create_flags; // DLSS feature flags from creation (MVJittered etc.).

    uint32_t subrect_base_x;
    uint32_t subrect_base_y;
    uint32_t subrect_render_width;
    uint32_t subrect_render_height;
    uint32_t output_width;
    uint32_t output_height;

    uint32_t resource_states_valid; // Bitmask: 1=color, 2=output, 4=depth, 8=motion, 16=exposure.
    uint32_t color_state;           // D3D12_RESOURCE_STATES when known.
    uint32_t output_state;
    uint32_t depth_state;
    uint32_t motion_vectors_state;
    uint32_t exposure_texture_state;
} OssGaussianFrame;

// -----------------------------------------------------------------------------
//  Render mode toggle (T2.9). Mode A pass-through, Mode B writes Output ourselves.
// -----------------------------------------------------------------------------
typedef enum OssGaussianRenderMode {
    OSS_GAUSSIAN_MODE_PASSTHROUGH = 0,
    OSS_GAUSSIAN_MODE_OSS_RENDER  = 1,
} OssGaussianRenderMode;

// -----------------------------------------------------------------------------
//  Callback signature. Invoked from the game thread inside our spoofed
//  `NVSDK_NGX_D3D12_EvaluateFeature`, after we have read the parameter dict
//  but before forwarding to real DLSS. Callbacks MUST NOT block; the budget
//  is single-digit milliseconds.
//
//  Returning a non-zero status logs but does not abort the frame.
// -----------------------------------------------------------------------------
typedef OssGaussianStatus (*OssGaussianFrameCallback)(
    const OssGaussianFrame* frame,
    void*                   user_data);

// -----------------------------------------------------------------------------
//  Public API.
// -----------------------------------------------------------------------------

/// Set (or clear, with NULL) the per-frame callback. Thread-safe.
OSS_GAUSSIAN_API OssGaussianStatus
oss_gaussian_set_callback(OssGaussianFrameCallback callback, void* user_data);

/// Switch active render mode. Effective starting on the next EvaluateFeature.
OSS_GAUSSIAN_API OssGaussianStatus
oss_gaussian_set_render_mode(OssGaussianRenderMode mode);

/// Read current render mode.
OSS_GAUSSIAN_API OssGaussianRenderMode
oss_gaussian_get_render_mode(void);

/// Non-zero after the async process-attach initializer has completed.
OSS_GAUSSIAN_API uint32_t
oss_gaussian_is_initialized(void);

/// Build version string (e.g. "0.1.0+sprint2"). Pointer is to static storage.
OSS_GAUSSIAN_API const char*
oss_gaussian_version(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // OSS_GAUSSIAN_INTERCEPTION_H
