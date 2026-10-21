// =============================================================================
//  d3d12_hooks.h
//
//  Detours-based D3D12 + DXGI hooks for the OSS-Gaussian capture path.
//
//  Surface (called from dllmain.cpp's async init worker / explicit teardown):
//      InstallD3D12Hooks()    - vtable-patch IDXGISwapChain::Present /
//                                Present1, IDXGIFactory swap-chain creation,
//                                and ID3D12CommandQueue::ExecuteCommandLists
//                                via a throwaway runtime probe.
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

#include <cstdint>

namespace oss_gaussian {

enum class D3D12HookInstallState : uint32_t {
    NotInstalled = 0,
    Installed = 1,
    Degraded = 2,
    Failed = 3,
};

struct D3D12HookStatus {
    D3D12HookInstallState state = D3D12HookInstallState::NotInstalled;
    bool installed = false;
    bool degraded = false;
    bool present_hooked = false;
    bool present1_hooked = false;
    bool execute_command_lists_hooked = false;
    bool create_device_hooked = false;
    bool create_command_queue_hooked = false;
    bool swap_chain_creation_hooked = false;
    bool readback_degraded = false;
    long last_detour_error = 0;
    long last_probe_hresult = 0;
    uint64_t frame_count = 0;
    uint64_t present_count = 0;
    uint64_t present1_count = 0;
    uint64_t create_device_count = 0;
    uint64_t create_command_queue_count = 0;
    uint64_t execute_command_lists_count = 0;
    uint64_t swap_chain_creation_count = 0;
    uint64_t capture_keep_count = 0;
    uint64_t degraded_capture_count = 0;
};

struct D3D12HookVTableProbe {
    void* present = nullptr;
    void* present1 = nullptr;
    void* execute_command_lists = nullptr;
    void* create_swap_chain = nullptr;
    void* create_swap_chain_for_hwnd = nullptr;
    void* create_swap_chain_for_core_window = nullptr;
    void* create_swap_chain_for_composition = nullptr;
};

// Install vtable hooks on Present + ExecuteCommandLists. Idempotent: returns
// true if the hook is already (or now) installed. Logs into LogModule("hooks").
bool InstallD3D12Hooks();

// Fast path for dxgi.dll proxy forwarders: once the game creates an
// IDXGIFactory through our local DXGI proxy, attach swap-chain creation hooks
// directly from that real factory vtable. This avoids doing a full D3D12 probe
// under loader-sensitive startup paths and lets us late-attach Present from the
// actual game swapchain.
void NotifyDxgiFactoryCreatedForCapture(void* factory);

// Set the desired CaptureSampler configuration before the first frame. Must
// be called BEFORE InstallD3D12Hooks (or before the first present after
// install) to take effect. Pass capture_mode strings: "trickle" | "lite" |
// "regular" | "INSANE". Empty string preserves the default.
void ConfigureCaptureSampler(const char* capture_mode);

// Symmetric teardown. Idempotent.
void UninstallD3D12Hooks();

// Return whether hooks are currently active (atomic snapshot).
bool AreD3D12HooksActive();

// Structured status for logs, smoke tests, and external diagnostics. The hook
// layer can be active but degraded when optional DXGI 1.2 creation/Present1
// detours could not be attached.
D3D12HookStatus GetD3D12HookStatus();

const char* D3D12HookInstallStateName(D3D12HookInstallState state);

// Per-frame counter, monotonic across the DLL lifetime. Incremented inside
// the Present hook before invoking the callback. Useful for debug logging
// downstream.
unsigned long long CurrentFrameIndex();

// Test seam: read known vtable slots from already-created COM objects without
// attaching Detours. Arguments are void* so tests can provide mocks or real
// COM instances without this header pulling in D3D12/DXGI.
bool InspectD3D12HookVTablesForTesting(
    void* swap_chain,
    void* command_queue,
    void* factory,
    D3D12HookVTableProbe* out);

}  // namespace oss_gaussian

#endif  // OSS_GAUSSIAN_D3D12_HOOKS_H
