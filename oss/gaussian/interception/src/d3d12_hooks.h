// =============================================================================
//  d3d12_hooks.h
//
//  Detours-based D3D12 + DXGI hooks for the OSS-Gaussian capture path.
//
//  Surface (called from dllmain.cpp OnAttach/OnDetach):
//      InstallD3D12Hooks()    — vtable-patch IDXGISwapChain::Present and
//                                ID3D12CommandQueue::ExecuteCommandLists
//                                via Detours-installed dummy device probe.
//      UninstallD3D12Hooks()  — symmetric teardown.
//
//  Why probe via dummy device: IDXGISwapChain and ID3D12CommandQueue are
//  COM interfaces — they don't have stable trampoline addresses. We create
//  a throwaway swapchain + queue, read their vtable slots, then patch those
//  slots in process-wide. Same trick OptiScaler uses (see HooksDx.cpp:
//  HooksDx::HookCommandList()).
//
//  Copyright 2026 OSS-Gaussian contributors. Apache 2.0.
// =============================================================================
#ifndef OSS_GAUSSIAN_D3D12_HOOKS_H
#define OSS_GAUSSIAN_D3D12_HOOKS_H

#include <Windows.h>

namespace oss_gaussian {

// Install vtable hooks on Present + ExecuteCommandLists. Idempotent: returns
// true if the hook is already (or now) installed. Logs into LogModule("hooks").
bool InstallD3D12Hooks();

// Set the desired CaptureSampler configuration before the first frame. Must
// be called BEFORE InstallD3D12Hooks (or before the first present after
// install) to take effect. Pass capture_mode strings: "trickle" | "lite" |
// "regular" | "INSANE". Empty string preserves the default.
void ConfigureCaptureSampler(const char* capture_mode);

// Symmetric teardown. Idempotent.
void UninstallD3D12Hooks();

// Return whether hooks are currently active (atomic snapshot).
bool AreD3D12HooksActive();

// Per-frame counter, monotonic across the DLL lifetime. Incremented inside
// the Present hook before invoking the callback. Useful for debug logging
// downstream.
unsigned long long CurrentFrameIndex();

}  // namespace oss_gaussian

#endif  // OSS_GAUSSIAN_D3D12_HOOKS_H
