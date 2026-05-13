// =============================================================================
//  ffx_backend_dx12_proxy.h
//
//  FSR/FidelityFX DX12 backend proxy support for games that import
//  ffx_backend_dx12_x64.dll directly.
// =============================================================================
#ifndef OSS_GAUSSIAN_FFX_BACKEND_DX12_PROXY_H
#define OSS_GAUSSIAN_FFX_BACKEND_DX12_PROXY_H

namespace oss_gaussian::ffx_backend_dx12 {

void* ResolveExport(const char* name);

template <typename Fn>
Fn ResolveTyped(const char* name) {
    return reinterpret_cast<Fn>(ResolveExport(name));
}

} // namespace oss_gaussian::ffx_backend_dx12

#endif // OSS_GAUSSIAN_FFX_BACKEND_DX12_PROXY_H
