// =============================================================================
//  ffx_fsr3_proxy.h
//
//  FSR3 proxy support for titles that import ffx_fsr3_x64.dll directly.
// =============================================================================
#ifndef OSS_GAUSSIAN_FFX_FSR3_PROXY_H
#define OSS_GAUSSIAN_FFX_FSR3_PROXY_H

namespace oss_gaussian::ffx_fsr3 {

void* ResolveExport(const char* name);

template <typename Fn>
Fn ResolveTyped(const char* name) {
    return reinterpret_cast<Fn>(ResolveExport(name));
}

} // namespace oss_gaussian::ffx_fsr3

#endif // OSS_GAUSSIAN_FFX_FSR3_PROXY_H
