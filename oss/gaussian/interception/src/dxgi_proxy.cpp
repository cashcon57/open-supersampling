// =============================================================================
//  dxgi_proxy.cpp
//
//  See dxgi_proxy.h for design rationale. This TU defines:
//    1. The cached HMODULE of C:\Windows\System32\dxgi.dll (`g_systemDxgi`).
//    2. OssGaussianDxgiProxyAttach / Detach lifecycle helpers.
//    3. A thin `__declspec(dllexport)` trampoline per DXGI symbol that resolves
//       the target via GetProcAddress on first call (cached in a function-local
//       static) and forwards arguments unchanged.
//
//  Game-agnostic. No Cyberpunk-specific logic; the union of exports across
//  recent Win10/11 system dxgi.dll versions is covered.
//
//  Copyright 2026 OSS-Gaussian contributors
//  Licensed under the Apache License, Version 2.0 (see ../LICENSE).
// =============================================================================
#include "dxgi_proxy.h"

#include "log.h"

#include <Windows.h>
#include <dxgi.h>
#include <dxgi1_3.h>

namespace oss_gaussian {

namespace {

// HMODULE for C:\Windows\System32\dxgi.dll. Set in Attach, cleared in Detach.
// Loader-locked access only on Attach/Detach; per-call reads are racy-tolerant
// (the value is only ever set once, then cleared at process exit).
HMODULE g_systemDxgi = nullptr;

// Resolve a symbol from g_systemDxgi, log a miss if absent. Templated so each
// trampoline gets its own function-local static cache.
template <typename Fn>
Fn ResolveOnce(const char* name, Fn& cache) {
    if (cache) return cache;
    if (!g_systemDxgi) {
        OSSG_LOG_WARN("dxgi_proxy",
                      "%s: forward miss; system32 dxgi.dll not loaded", name);
        return nullptr;
    }
    cache = reinterpret_cast<Fn>(GetProcAddress(g_systemDxgi, name));
    if (!cache) {
        OSSG_LOG_WARN("dxgi_proxy",
                      "%s: forward miss; system32 dxgi.dll did not export this symbol",
                      name);
    }
    return cache;
}

} // namespace

bool OssGaussianDxgiProxyAttach() {
    if (g_systemDxgi) return true;

    // Absolute path. Never rely on PATH/SafeDllSearchMode — that risks the
    // loader resolving "dxgi.dll" to our own DLL and infinite-recursing.
    g_systemDxgi = LoadLibraryW(L"C:\\Windows\\System32\\dxgi.dll");
    if (!g_systemDxgi) {
        OSSG_LOG_ERROR("dxgi_proxy",
                       "LoadLibraryW(System32\\dxgi.dll) failed, GetLastError=%lu",
                       GetLastError());
        return false;
    }
    OSSG_LOG_INFO("dxgi_proxy", "system32 dxgi.dll loaded at %p",
                  reinterpret_cast<void*>(g_systemDxgi));
    return true;
}

void OssGaussianDxgiProxyDetach() {
    if (g_systemDxgi) {
        FreeLibrary(g_systemDxgi);
        g_systemDxgi = nullptr;
    }
}

} // namespace oss_gaussian

// -----------------------------------------------------------------------------
//  Forwarder macro. Each export resolves its target lazily and forwards the
//  call. Misses log via OSSG_LOG_WARN and return a benign failure code.
// -----------------------------------------------------------------------------

// Trampoline for HRESULT-returning exports.
#define OSSG_DXGI_FWD_HRESULT(NAME, SIG, ARGS)                                \
    extern "C" __declspec(dllexport) HRESULT WINAPI NAME SIG {                \
        using Fn = HRESULT(WINAPI*) SIG;                                      \
        static Fn fn = nullptr;                                               \
        Fn resolved = ::oss_gaussian::ResolveOnce<Fn>(#NAME, fn);             \
        if (!resolved) return E_NOTIMPL;                                      \
        return resolved ARGS;                                                 \
    }

// Trampoline for void-returning exports.
#define OSSG_DXGI_FWD_VOID(NAME, SIG, ARGS)                                   \
    extern "C" __declspec(dllexport) void WINAPI NAME SIG {                   \
        using Fn = void(WINAPI*) SIG;                                         \
        static Fn fn = nullptr;                                               \
        Fn resolved = ::oss_gaussian::ResolveOnce<Fn>(#NAME, fn);             \
        if (!resolved) return;                                                \
        resolved ARGS;                                                        \
    }

// Trampoline for int-returning exports (PIX capture state, etc.).
#define OSSG_DXGI_FWD_INT(NAME, SIG, ARGS)                                    \
    extern "C" __declspec(dllexport) int WINAPI NAME SIG {                    \
        using Fn = int(WINAPI*) SIG;                                          \
        static Fn fn = nullptr;                                               \
        Fn resolved = ::oss_gaussian::ResolveOnce<Fn>(#NAME, fn);             \
        if (!resolved) return 0;                                              \
        return resolved ARGS;                                                 \
    }

// -----------------------------------------------------------------------------
//  Documented public DXGI exports.
// -----------------------------------------------------------------------------

OSSG_DXGI_FWD_HRESULT(CreateDXGIFactory,  (REFIID riid, void** ppFactory), (riid, ppFactory))
OSSG_DXGI_FWD_HRESULT(CreateDXGIFactory1, (REFIID riid, void** ppFactory), (riid, ppFactory))
OSSG_DXGI_FWD_HRESULT(CreateDXGIFactory2, (UINT Flags, REFIID riid, void** ppFactory), (Flags, riid, ppFactory))
OSSG_DXGI_FWD_HRESULT(DXGIDeclareAdapterRemovalSupport, (void), ())
OSSG_DXGI_FWD_HRESULT(DXGIGetDebugInterface1, (UINT Flags, REFIID riid, void** pDebug), (Flags, riid, pDebug))

// Undocumented but ABI-stable exports observed across Win10/11 system32
// dxgi.dll. Argument types are deliberately `void*` — every caller already
// knows the real signature; our forwarders just shuffle register/stack
// bytes through. HRESULT is conventional for the DXGI* family; PIX hooks
// return int/void. ResolveOnce logs misses on version drift.
OSSG_DXGI_FWD_HRESULT(DXGIDisableVBlankVirtualization, (void), ())
OSSG_DXGI_FWD_HRESULT(DXGIDumpJournal, (void* pCallback, void* pContext), (pCallback, pContext))
OSSG_DXGI_FWD_HRESULT(DXGIReportAdapterConfiguration, (void* pConfig), (pConfig))
OSSG_DXGI_FWD_HRESULT(ApplyCompatResolutionQuirking, (void), ())
OSSG_DXGI_FWD_HRESULT(CompatString, (void* a, void* b, void* c, void* d), (a, b, c, d))
OSSG_DXGI_FWD_HRESULT(CompatValue, (const char* name, void* value), (name, value))
OSSG_DXGI_FWD_HRESULT(DXGID3D10CreateDevice, (HMODULE hModule, void* pFactory, void* pAdapter, UINT Flags, void* pUnknown, void** ppDevice), (hModule, pFactory, pAdapter, Flags, pUnknown, ppDevice))
OSSG_DXGI_FWD_HRESULT(DXGID3D10CreateLayeredDevice, (void* p1, void* p2, void* p3, void* p4, void* p5), (p1, p2, p3, p4, p5))
OSSG_DXGI_FWD_HRESULT(DXGID3D10GetLayeredDeviceSize, (const void* pLayers, UINT NumLayers), (pLayers, NumLayers))
OSSG_DXGI_FWD_HRESULT(DXGID3D10RegisterLayers, (const void* pLayers, UINT NumLayers), (pLayers, NumLayers))
OSSG_DXGI_FWD_HRESULT(SetAppCompatStringPointer, (UINT_PTR dwId, const char* pString), (dwId, pString))
OSSG_DXGI_FWD_HRESULT(PIXBeginCapture, (DWORD flags, const void* pParams), (flags, pParams))
OSSG_DXGI_FWD_HRESULT(PIXEndCapture, (BOOL discard), (discard))
OSSG_DXGI_FWD_INT(PIXGetCaptureState, (void), ())
