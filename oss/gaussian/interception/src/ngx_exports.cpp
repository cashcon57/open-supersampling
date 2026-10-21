// =============================================================================
//  ngx_exports.cpp
//
//  Pass-through implementations of the 10 NVSDK_NGX_D3D12_* exports. Each
//  wrapper lazily resolves the same symbol from the real nvngx_dlss.dll and
//  forwards the original arguments. EvaluateFeature additionally notifies the
//  OSS capture layer before forwarding to DLSS.
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
//  third_party/DLSS/.
//
//  Copyright 2026 OSS-Gaussian contributors
//  Licensed under the Apache License, Version 2.0 (see ../LICENSE).
// =============================================================================
#include "log.h"
#include "ngx_passthrough.h"

#include "../oss_capture.h"

#include <Windows.h>

namespace {

inline void ForwardLog(const char* fn) {
    char buf[160];
    _snprintf_s(buf, _countof(buf), _TRUNCATE, "[oss-gaussian] called: %s\n", fn);
    OutputDebugStringA(buf);
    OSSG_LOG_TRACE("ngx", "forward: %s", fn);
}

inline NVSDK_NGX_Result MissingRealNgx(const char* fn) {
    OSSG_LOG_ERROR("ngx", "%s: real nvngx_dlss.dll or export unavailable", fn);
    return NVSDK_NGX_Result_FAIL;
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
NVSDK_NGX_D3D12_Init(unsigned long long InApplicationId,
                     const wchar_t*     InApplicationDataPath,
                     ID3D12Device*      InDevice,
                     NVSDK_NGX_Version  InSDKVersion) {
    ForwardLog("NVSDK_NGX_D3D12_Init");
    auto real = oss_gaussian::ngx::ResolveTyped<oss_gaussian::ngx::D3D12InitFn>(
        "NVSDK_NGX_D3D12_Init");
    if (!real) return MissingRealNgx("NVSDK_NGX_D3D12_Init");
    return real(InApplicationId, InApplicationDataPath, InDevice, InSDKVersion);
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_Init_Ext(unsigned long long                 InApplicationId,
                         const wchar_t*                     InApplicationDataPath,
                         ID3D12Device*                      InDevice,
                         NVSDK_NGX_Version                  InSDKVersion,
                         const NVSDK_NGX_Parameter*         InParameters) {
    ForwardLog("NVSDK_NGX_D3D12_Init_Ext");
    auto real = oss_gaussian::ngx::ResolveTyped<oss_gaussian::ngx::D3D12InitExtFn>(
        "NVSDK_NGX_D3D12_Init_Ext");
    if (!real) return MissingRealNgx("NVSDK_NGX_D3D12_Init_Ext");
    return real(InApplicationId, InApplicationDataPath, InDevice, InSDKVersion, InParameters);
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_Shutdown1(ID3D12Device* InDevice) {
    ForwardLog("NVSDK_NGX_D3D12_Shutdown1");
    auto real = oss_gaussian::ngx::ResolveTyped<oss_gaussian::ngx::D3D12Shutdown1Fn>(
        "NVSDK_NGX_D3D12_Shutdown1");
    if (!real) return MissingRealNgx("NVSDK_NGX_D3D12_Shutdown1");
    return real(InDevice);
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_GetCapabilityParameters(NVSDK_NGX_Parameter** OutParameters) {
    ForwardLog("NVSDK_NGX_D3D12_GetCapabilityParameters");
    auto real =
        oss_gaussian::ngx::ResolveTyped<oss_gaussian::ngx::D3D12GetCapabilityParametersFn>(
            "NVSDK_NGX_D3D12_GetCapabilityParameters");
    if (!real) {
        if (OutParameters) *OutParameters = nullptr;
        return MissingRealNgx("NVSDK_NGX_D3D12_GetCapabilityParameters");
    }
    return real(OutParameters);
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_AllocateParameters(NVSDK_NGX_Parameter** OutParameters) {
    ForwardLog("NVSDK_NGX_D3D12_AllocateParameters");
    auto real =
        oss_gaussian::ngx::ResolveTyped<oss_gaussian::ngx::D3D12AllocateParametersFn>(
            "NVSDK_NGX_D3D12_AllocateParameters");
    if (!real) {
        if (OutParameters) *OutParameters = nullptr;
        return MissingRealNgx("NVSDK_NGX_D3D12_AllocateParameters");
    }
    return real(OutParameters);
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_DestroyParameters(NVSDK_NGX_Parameter* InParameters) {
    ForwardLog("NVSDK_NGX_D3D12_DestroyParameters");
    auto real =
        oss_gaussian::ngx::ResolveTyped<oss_gaussian::ngx::D3D12DestroyParametersFn>(
            "NVSDK_NGX_D3D12_DestroyParameters");
    if (!real) return MissingRealNgx("NVSDK_NGX_D3D12_DestroyParameters");
    return real(InParameters);
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_CreateFeature(ID3D12GraphicsCommandList* InCmdList,
                              NVSDK_NGX_Feature          InFeatureID,
                              NVSDK_NGX_Parameter*       InParameters,
                              NVSDK_NGX_Handle**         OutHandle) {
    ForwardLog("NVSDK_NGX_D3D12_CreateFeature");
    auto real = oss_gaussian::ngx::ResolveTyped<oss_gaussian::ngx::D3D12CreateFeatureFn>(
        "NVSDK_NGX_D3D12_CreateFeature");
    if (!real) {
        if (OutHandle) *OutHandle = nullptr;
        return MissingRealNgx("NVSDK_NGX_D3D12_CreateFeature");
    }
    return real(InCmdList, InFeatureID, InParameters, OutHandle);
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_EvaluateFeature(ID3D12GraphicsCommandList*     InCmdList,
                                const NVSDK_NGX_Handle*        InFeatureHandle,
                                const NVSDK_NGX_Parameter*     InParameters,
                                PFN_NVSDK_NGX_ProgressCallback InCallback) {
    ForwardLog("NVSDK_NGX_D3D12_EvaluateFeature");
    auto real =
        oss_gaussian::ngx::ResolveTyped<oss_gaussian::ngx::D3D12EvaluateFeatureFn>(
            "NVSDK_NGX_D3D12_EvaluateFeature");
    if (!real) return MissingRealNgx("NVSDK_NGX_D3D12_EvaluateFeature");
    void* ticket = oss_capture_on_ngx_evaluate_feature_begin(
        InCmdList,
        InFeatureHandle,
        InParameters);
    const NVSDK_NGX_Result result =
        real(InCmdList, InFeatureHandle, InParameters, InCallback);
    oss_capture_on_ngx_evaluate_feature_end(
        ticket,
        InCmdList,
        InFeatureHandle,
        InParameters,
        result);
    return result;
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_ReleaseFeature(NVSDK_NGX_Handle* InHandle) {
    ForwardLog("NVSDK_NGX_D3D12_ReleaseFeature");
    auto real = oss_gaussian::ngx::ResolveTyped<oss_gaussian::ngx::D3D12ReleaseFeatureFn>(
        "NVSDK_NGX_D3D12_ReleaseFeature");
    if (!real) return MissingRealNgx("NVSDK_NGX_D3D12_ReleaseFeature");
    return real(InHandle);
}

__declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NVSDK_NGX_D3D12_GetScratchBufferSize(NVSDK_NGX_Feature    InFeatureID,
                                     const NVSDK_NGX_Parameter* InParameters,
                                     size_t*              OutSizeInBytes) {
    ForwardLog("NVSDK_NGX_D3D12_GetScratchBufferSize");
    auto real =
        oss_gaussian::ngx::ResolveTyped<oss_gaussian::ngx::D3D12GetScratchBufferSizeFn>(
            "NVSDK_NGX_D3D12_GetScratchBufferSize");
    if (!real) {
        if (OutSizeInBytes) *OutSizeInBytes = 0;
        return MissingRealNgx("NVSDK_NGX_D3D12_GetScratchBufferSize");
    }
    return real(InFeatureID, InParameters, OutSizeInBytes);
}

} // extern "C"
