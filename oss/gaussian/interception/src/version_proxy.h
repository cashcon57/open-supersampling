// =============================================================================
//  version_proxy.h
//
//  Optional VERSION.dll forwarders for games where dxgi.dll is not a reliable
//  executable loader path. The same capture module can be copied as
//  version.dll and still initialize the D3D12/NGX hooks.
// =============================================================================
#ifndef OSS_GAUSSIAN_VERSION_PROXY_H
#define OSS_GAUSSIAN_VERSION_PROXY_H

namespace oss_gaussian {

bool OssGaussianVersionProxyAttach();
void OssGaussianVersionProxyDetach();

} // namespace oss_gaussian

#endif // OSS_GAUSSIAN_VERSION_PROXY_H
