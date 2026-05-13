// =============================================================================
//  sl_interposer_proxy.h
//
//  Streamline interposer proxy support. The target is installed as
//  `sl.interposer.dll` and forwards to the game's original copy renamed to
//  `oss_sl_real.dll`.
// =============================================================================
#ifndef OSS_GAUSSIAN_SL_INTERPOSER_PROXY_H
#define OSS_GAUSSIAN_SL_INTERPOSER_PROXY_H

namespace oss_gaussian::sl_proxy {

void* ResolveExport(const char* name);

template <typename Fn>
Fn ResolveTyped(const char* name) {
    return reinterpret_cast<Fn>(ResolveExport(name));
}

} // namespace oss_gaussian::sl_proxy

#endif // OSS_GAUSSIAN_SL_INTERPOSER_PROXY_H
