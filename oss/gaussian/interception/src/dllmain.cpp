// =============================================================================
//  dllmain.cpp
//
//  DLL entry point. Sprint 2 scaffolding: initializes the file logger on
//  attach, tears it down on detach, and exposes the public C API declared
//  in include/oss_gaussian_interception.h.
//
//  Detours hook installation lives in this file once T2.2 vendors Detours;
//  the call sites below are stubbed with `// TODO(T2.x):` markers so the
//  diff at sprint time is small.
//
//  Modeled on:
//    - OptiScaler `OptiScaler/Hooks/HooksDx.cpp` and `dllmain.cpp` for the
//      Detours-based DXGI/D3D12 hook list.
//      https://github.com/optiscaler/OptiScaler/blob/master/OptiScaler/Hooks/HooksDx.cpp
//    - Microsoft Detours sample DLL (`samples/simple/simple.cpp`).
//      https://github.com/microsoft/Detours/blob/main/samples/simple/simple.cpp
//
//  Copyright 2026 OSS-Gaussian contributors
//  Licensed under the Apache License, Version 2.0 (see ../LICENSE).
// =============================================================================
#define OSS_GAUSSIAN_BUILDING_DLL 1

#include "../include/oss_gaussian_interception.h"
#include "log.h"

#include <Windows.h>

#include <atomic>
#include <mutex>

namespace oss_gaussian {

namespace {

std::atomic<OssGaussianRenderMode> g_render_mode{OSS_GAUSSIAN_MODE_PASSTHROUGH};

std::mutex                         g_callback_mu;
OssGaussianFrameCallback           g_callback     = nullptr;
void*                              g_callback_ud  = nullptr;

// Set by DllMain ATTACH; cleared on DETACH. Used as a sanity check elsewhere.
std::atomic<bool>                  g_dll_attached{false};

void OnAttach(HMODULE self) {
    (void)self;
    LogInit();
    OSSG_LOG_INFO("dll", "OSS-Gaussian interception DLL attached (build %s)",
                  oss_gaussian_version());

    // TODO(T2.2): DetourTransactionBegin() / DetourUpdateThread(GetCurrentThread()).
    // TODO(T2.3): Forward DXGI exports to system32 dxgi.dll (load HMODULE here).
    // TODO(T2.5): LoadLibrary detour for nvngx_dlss.dll → our HMODULE.
    // TODO(T2.8): Hook IDXGIFactory::CreateSwapChain* → vtable-patch Present.
    // TODO(T2.9): SetWindowsHookExW(WH_KEYBOARD_LL, ...) for F11 toggle.
    // TODO(T2.x): DetourTransactionCommit().

    g_dll_attached.store(true, std::memory_order_release);
}

void OnDetach() {
    g_dll_attached.store(false, std::memory_order_release);

    // TODO(T2.x): Detach all Detours (DetourTransactionBegin/Commit with
    // DetourDetach for every fn-ptr installed in OnAttach).

    OSSG_LOG_INFO("dll", "OSS-Gaussian interception DLL detaching");
    LogShutdown();
}

} // namespace

// -----------------------------------------------------------------------------
//  Public C API implementations.
// -----------------------------------------------------------------------------

// Internal helper: retrieve the active callback under lock. Used by EvaluateFeature
// once T2.6 wires the param-dict reader. Exposed in this TU only.
void InvokeFrameCallback(const OssGaussianFrame& frame) {
    OssGaussianFrameCallback cb = nullptr;
    void*                    ud = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_callback_mu);
        cb = g_callback;
        ud = g_callback_ud;
    }
    if (!cb) return;

    OssGaussianStatus s = cb(&frame, ud);
    if (s != OSS_GAUSSIAN_OK) {
        OSSG_LOG_WARN("cb", "frame callback returned status=%d (frame=%llu)",
                      static_cast<int>(s),
                      static_cast<unsigned long long>(frame.frame_index));
    }
}

} // namespace oss_gaussian

// -----------------------------------------------------------------------------
extern "C" {

OSS_GAUSSIAN_API OssGaussianStatus
oss_gaussian_set_callback(OssGaussianFrameCallback callback, void* user_data) {
    std::lock_guard<std::mutex> lk(oss_gaussian::g_callback_mu);
    oss_gaussian::g_callback    = callback;
    oss_gaussian::g_callback_ud = user_data;
    OSSG_LOG_INFO("api", "set_callback: %s",
                  callback ? "installed" : "cleared");
    return OSS_GAUSSIAN_OK;
}

OSS_GAUSSIAN_API OssGaussianStatus
oss_gaussian_set_render_mode(OssGaussianRenderMode mode) {
    if (mode != OSS_GAUSSIAN_MODE_PASSTHROUGH &&
        mode != OSS_GAUSSIAN_MODE_OSS_RENDER) {
        return OSS_GAUSSIAN_ERR_INVALID_ARG;
    }
    oss_gaussian::g_render_mode.store(mode, std::memory_order_release);
    OSSG_LOG_INFO("api", "set_render_mode: %d", static_cast<int>(mode));
    return OSS_GAUSSIAN_OK;
}

OSS_GAUSSIAN_API OssGaussianRenderMode
oss_gaussian_get_render_mode(void) {
    return oss_gaussian::g_render_mode.load(std::memory_order_acquire);
}

OSS_GAUSSIAN_API const char*
oss_gaussian_version(void) {
    return "0.1.0+sprint2-scaffold";
}

} // extern "C"

// -----------------------------------------------------------------------------
//  DllMain.
// -----------------------------------------------------------------------------
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID /*reserved*/) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        // Disable per-thread callbacks; we don't need them and they cost cycles
        // in a game process with many short-lived threads.
        DisableThreadLibraryCalls(hModule);
        oss_gaussian::OnAttach(hModule);
        break;

    case DLL_PROCESS_DETACH:
        oss_gaussian::OnDetach();
        break;

    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    default:
        break;
    }
    return TRUE;
}
