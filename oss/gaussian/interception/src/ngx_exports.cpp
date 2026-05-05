// =============================================================================
//  ngx_exports.cpp
//
//  Stub implementations of the 10 NVSDK_NGX_D3D12_* exports. Each stub:
//    1. Logs the call via OutputDebugStringA + the file logger.
//    2. Returns NVSDK_NGX_Result_Success (or a benign code) so the game
//       continues. The pass-through plumbing to real `nvngx_dlss.dll` lands
//       in T2.5 (`ngx_passthrough.cpp`).
//
//  Modeled on:
//    - OptiScaler `OptiScaler/NVNGX_DLSS.cpp` (the export bag).
//      https://github.com/optiscaler/OptiScaler/blob/master/OptiScaler/NVNGX_DLSS.cpp
//    - PotatoOfDoom/CyberFSR2 (smaller analog for Cyberpunk specifically).
//      https://github.com/PotatoOfDoom/CyberFSR2
//    - NVIDIA/DLSS public headers (signatures).
//      https://github.com/NVIDIA/DLSS/blob/main/include/nvsdk_ngx.h
//
//  Signatures here intentionally use local typedefs / forward decls so the
//  scaffold compiles before NVIDIA/DLSS headers are vendored under
//  third_party/DLSS/. Once vendored, replace these typedefs with the real
//  `#include <nvsdk_ngx.h>` and remove the forwards.
//
//  Copyright 2026 OSS-Gaussian contributors
//  Licensed under the Apache License, Version 2.0 (see ../LICENSE).
// =============================================================================
#include "log.h"

#include "../oss_capture.h"

#include <Windows.h>

// -----------------------------------------------------------------------------
//  Local mirrors of the public NGX C surface. Sourced from
//  NVIDIA/DLSS `nvsdk_ngx.h`. ABI-stable across SDK versions for these symbols.
// -----------------------------------------------------------------------------

// NVIDIA's headers define this as __cdecl on Windows. Mirror it locally so
// the scaffold compiles before nvsdk_ngx.h is vendored.
#ifndef NVSDK_CONV
#  define NVSDK_CONV __cdecl
#endif

extern "C" {

typedef int          NVSDK_NGX_Result;
typedef int          NVSDK_NGX_Feature;
typedef int          NVSDK_NGX_Version;
struct  NVSDK_NGX_Parameter;
struct  NVSDK_NGX_Handle;
struct  NVSDK_NGX_FeatureCommonInfo;
struct  ID3D12Device;
struct  ID3D12GraphicsCommandList;

typedef void (*PFN_NVSDK_NGX_ProgressCallback)(float progress, bool* should_cancel);

// Per nvsdk_ngx_defs.h — the values we hand back. 1 == Success.
constexpr NVSDK_NGX_Result NVSDK_NGX_Result_Success = 0x1;

} // extern "C"

namespace {

inline void StubLog(const char* fn) {
    char buf[160];
    _snprintf_s(buf, _countof(buf), _TRUNCATE, "[oss-gaussian] called: %s\n", fn);
    OutputDebugStringA(buf);
    OSSG_LOG_INFO("ngx", "stub: %s", fn);
}

} // namespace

// -----------------------------------------------------------------------------
//  Exports. Order + names match nvngx_dlss.dll on disk so a `dumpbin /exports`
//  matches the real DLL one-for-one.
//
//  Mark all 10 with __declspec(dllexport) for now. T2.3 adds a .def file that
//  also re-exports `dxgi.dll` symbols by forwarding strings.
// -----------------------------------------------------------------------------

extern "C" {

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_Init(unsigned long long /*InApplicationId*/,
                     const wchar_t*     /*InApplicationDataPath*/,
                     ID3D12Device*      /*InDevice*/,
                     NVSDK_NGX_Version  /*InSDKVersion*/) {
    StubLog("NVSDK_NGX_D3D12_Init");
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_Init_Ext(unsigned long long                 /*InApplicationId*/,
                         const wchar_t*                     /*InApplicationDataPath*/,
                         ID3D12Device*                      /*InDevice*/,
                         const NVSDK_NGX_FeatureCommonInfo* /*InFeatureInfo*/,
                         NVSDK_NGX_Version                  /*InSDKVersion*/) {
    StubLog("NVSDK_NGX_D3D12_Init_Ext");
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_Shutdown1(ID3D12Device* /*InDevice*/) {
    StubLog("NVSDK_NGX_D3D12_Shutdown1");
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_GetCapabilityParameters(NVSDK_NGX_Parameter** /*OutParameters*/) {
    StubLog("NVSDK_NGX_D3D12_GetCapabilityParameters");
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_AllocateParameters(NVSDK_NGX_Parameter** /*OutParameters*/) {
    StubLog("NVSDK_NGX_D3D12_AllocateParameters");
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_DestroyParameters(NVSDK_NGX_Parameter* /*InParameters*/) {
    StubLog("NVSDK_NGX_D3D12_DestroyParameters");
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_CreateFeature(ID3D12GraphicsCommandList* /*InCmdList*/,
                              NVSDK_NGX_Feature          /*InFeatureID*/,
                              NVSDK_NGX_Parameter*       /*InParameters*/,
                              NVSDK_NGX_Handle**         /*OutHandle*/) {
    StubLog("NVSDK_NGX_D3D12_CreateFeature");
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_EvaluateFeature(ID3D12GraphicsCommandList*     InCmdList,
                                const NVSDK_NGX_Handle*        InFeatureHandle,
                                const NVSDK_NGX_Parameter*     InParameters,
                                PFN_NVSDK_NGX_ProgressCallback /*InCallback*/) {
    StubLog("NVSDK_NGX_D3D12_EvaluateFeature");
    oss_capture_on_ngx_evaluate_feature(InCmdList, InFeatureHandle, InParameters);
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_ReleaseFeature(NVSDK_NGX_Handle* /*InHandle*/) {
    StubLog("NVSDK_NGX_D3D12_ReleaseFeature");
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_GetScratchBufferSize(ID3D12Device*        /*InDevice*/,
                                     NVSDK_NGX_Feature    /*InFeatureID*/,
                                     const NVSDK_NGX_Parameter* /*InParameters*/,
                                     size_t*              OutSizeInBytes) {
    StubLog("NVSDK_NGX_D3D12_GetScratchBufferSize");
    if (OutSizeInBytes) *OutSizeInBytes = 0;
    return NVSDK_NGX_Result_Success;
}

} // extern "C"
