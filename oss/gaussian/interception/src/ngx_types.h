// =============================================================================
//  ngx_types.h
//
//  Minimal local mirrors of the public NGX D3D12 C surface used by our export
//  shim. Keep this header NVIDIA-SDK-free so the capture DLL builds before the
//  proprietary/public DLSS headers are vendored.
// =============================================================================
#ifndef OSS_GAUSSIAN_NGX_TYPES_H
#define OSS_GAUSSIAN_NGX_TYPES_H

#include <stddef.h>

#ifndef NVSDK_CONV
#  define NVSDK_CONV __cdecl
#endif

extern "C" {

typedef int NVSDK_NGX_Result;
typedef int NVSDK_NGX_Feature;
typedef int NVSDK_NGX_Version;

struct NVSDK_NGX_Parameter;
struct NVSDK_NGX_Handle;
struct NVSDK_NGX_FeatureCommonInfo;
struct ID3D12Device;
struct ID3D12GraphicsCommandList;

typedef void (*PFN_NVSDK_NGX_ProgressCallback)(float progress, bool* should_cancel);

} // extern "C"

constexpr NVSDK_NGX_Result NVSDK_NGX_Result_Success = 0x1;
constexpr NVSDK_NGX_Result NVSDK_NGX_Result_FAIL = static_cast<NVSDK_NGX_Result>(0xBAD00000);

#endif // OSS_GAUSSIAN_NGX_TYPES_H
