// =============================================================================
//  ngx_passthrough.h
//
//  Lazy loader/resolver for the real NVIDIA DLSS NGX DLL.
// =============================================================================
#ifndef OSS_GAUSSIAN_NGX_PASSTHROUGH_H
#define OSS_GAUSSIAN_NGX_PASSTHROUGH_H

#include "ngx_types.h"

namespace oss_gaussian::ngx {

void* ResolveExport(const char* name);

template <typename Fn>
Fn ResolveTyped(const char* name) {
    return reinterpret_cast<Fn>(ResolveExport(name));
}

using D3D12InitFn = NVSDK_NGX_Result(NVSDK_CONV*)(
    unsigned long long,
    const wchar_t*,
    ID3D12Device*,
    NVSDK_NGX_Version);

using D3D12InitExtFn = NVSDK_NGX_Result(NVSDK_CONV*)(
    unsigned long long,
    const wchar_t*,
    ID3D12Device*,
    NVSDK_NGX_Version,
    const NVSDK_NGX_Parameter*);

using D3D12Shutdown1Fn = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12Device*);

using D3D12GetCapabilityParametersFn =
    NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter**);

using D3D12AllocateParametersFn =
    NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter**);

using D3D12DestroyParametersFn =
    NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter*);

using D3D12CreateFeatureFn = NVSDK_NGX_Result(NVSDK_CONV*)(
    ID3D12GraphicsCommandList*,
    NVSDK_NGX_Feature,
    NVSDK_NGX_Parameter*,
    NVSDK_NGX_Handle**);

using D3D12EvaluateFeatureFn = NVSDK_NGX_Result(NVSDK_CONV*)(
    ID3D12GraphicsCommandList*,
    const NVSDK_NGX_Handle*,
    const NVSDK_NGX_Parameter*,
    PFN_NVSDK_NGX_ProgressCallback);

using D3D12ReleaseFeatureFn =
    NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);

using D3D12GetScratchBufferSizeFn = NVSDK_NGX_Result(NVSDK_CONV*)(
    NVSDK_NGX_Feature,
    const NVSDK_NGX_Parameter*,
    size_t*);

} // namespace oss_gaussian::ngx

#endif // OSS_GAUSSIAN_NGX_PASSTHROUGH_H
