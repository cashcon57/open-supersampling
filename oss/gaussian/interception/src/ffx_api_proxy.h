// =============================================================================
//  ffx_api_proxy.h
//
//  Generic FidelityFX API proxy support for games that import
//  amd_fidelityfx_dx12.dll directly.
// =============================================================================
#ifndef OSS_GAUSSIAN_FFX_API_PROXY_H
#define OSS_GAUSSIAN_FFX_API_PROXY_H

namespace oss_gaussian::ffx_api {

void* ResolveExport(const char* name);

template <typename Fn>
Fn ResolveTyped(const char* name) {
    return reinterpret_cast<Fn>(ResolveExport(name));
}

} // namespace oss_gaussian::ffx_api

#endif // OSS_GAUSSIAN_FFX_API_PROXY_H
