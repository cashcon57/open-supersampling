// =============================================================================
//  dxgi_proxy.h
//
//  DXGI export forwarding for the OSS-Gaussian interception DLL.
//
//  When our DLL is dropped into a game's bin\x64\ folder under the name
//  `dxgi.dll`, Windows' game-local DLL search resolves it before the system
//  copy at C:\Windows\System32\dxgi.dll. Any DXGI export the host process
//  imports must therefore be forwarded to the real system DLL, otherwise the
//  loader will fail to resolve the import and the game won't start.
//
//  Implementation strategy: runtime LoadLibrary + GetProcAddress trampoline.
//  We deliberately do NOT statically link against system32's dxgi.lib — that
//  triggers known load-order traps (Windows can resolve the import to our
//  own DLL recursively) and bakes ordinal/symbol assumptions into the binary
//  that don't survive Windows version drift. The runtime trampoline pattern
//  is what OptiScaler, ReShade, and Special K all use; we follow suit.
//
//  Game-agnostic: nothing in this module is specific to Cyberpunk 2077.
//  Any DLSS-shipping DX12 title (Hogwarts Legacy, Portal RTX, Alan Wake 2,
//  etc.) that loads dxgi.dll from its own bin folder will work with the
//  same forwarders.
//
//  Fallback: a `.def` file with `EXPORTS Foo=C:/Windows/System32/dxgi.Foo`
//  forwarding strings is a viable alternative (see MSDN "Module-Definition
//  (.def) Files"). It's documented in the README but not implemented here
//  because the runtime trampoline is more flexible — we can log misses,
//  reload on failure, and add new symbols without re-running the linker.
//
//  Copyright 2026 OSS-Gaussian contributors
//  Licensed under the Apache License, Version 2.0 (see ../LICENSE).
// =============================================================================
#ifndef OSS_GAUSSIAN_DXGI_PROXY_H
#define OSS_GAUSSIAN_DXGI_PROXY_H

#include <Windows.h>

namespace oss_gaussian {

/// LoadLibrary the real C:\Windows\System32\dxgi.dll and cache its HMODULE so
/// the per-export trampolines can resolve their target on first call.
///
/// Called from DllMain DLL_PROCESS_ATTACH. Returns true on success. On
/// failure the DLL still loads — individual forwarders log a miss and return
/// E_NOTIMPL. We never abort process load from DllMain.
bool OssGaussianDxgiProxyAttach();

/// Free the cached system dxgi.dll HMODULE. Called from DllMain
/// DLL_PROCESS_DETACH. Idempotent.
void OssGaussianDxgiProxyDetach();

} // namespace oss_gaussian

#endif // OSS_GAUSSIAN_DXGI_PROXY_H
