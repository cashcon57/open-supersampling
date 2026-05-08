// =============================================================================
//  d3d12_hooks.cpp
//
//  Real D3D12 + DXGI hook implementation. This is what turns the DLL from
//  "logging stub" into "frames flow through the capture path."
//
//  Architecture:
//
//    1. InstallD3D12Hooks() creates a throwaway swapchain + command queue
//       to read the COM vtable addresses of:
//         - IDXGISwapChain::Present                (slot 8 in IDXGISwapChain)
//         - ID3D12CommandQueue::ExecuteCommandLists (slot 10)
//
//    2. We Detours-attach to those vtable slots. Detours installs a 5-byte
//       trampoline jmp and stores the original prologue so we can chain to
//       the real implementation.
//
//    3. The Present hook:
//         a. Increments g_frame_counter
//         b. Captures backbuffer texture pointer + DXGI_FORMAT
//         c. Builds an OssCaptureCandidate
//         d. Asks CaptureSampler::Consider() (already implemented, mature)
//         e. If KEEP, queues a staging-copy + EXR-write task
//         f. Forwards to the real Present
//
//    4. The ExecuteCommandLists hook retains the latest queue pointer so the
//       Present hook has a queue to schedule the staging copy on.
//
//    5. Worker thread drains the EXR-write queue off the render thread.
//
//  Reference (closely studied):
//    - OptiScaler `OptiScaler/Hooks/HooksDx.cpp` HookCommandList /
//      HookSwapChainPresent / HookQueueExecuteCommandLists. Apache 2.0
//      compatible reference for the vtable-probe pattern.
//      https://github.com/optiscaler/OptiScaler/blob/master/OptiScaler/Hooks/HooksDx.cpp
//    - Microsoft Detours samples/simple/simple.cpp.
//      https://github.com/microsoft/Detours
//
//  Status: SCAFFOLDED. Compiles on Windows; runtime testing on 3080 Ti
//  desktop (Tailnet alias `3080ti-windows`) is the next step. The Present
//  hook is end-to-end wired but the staging copy is stubbed (logs only)
//  pending a real D3D12 readback heap implementation in staging_copy.cpp.
//
//  Copyright 2026 OSS-Gaussian contributors. Apache 2.0.
// =============================================================================
#define OSS_GAUSSIAN_BUILDING_DLL 1

#include "d3d12_hooks.h"
#include "log.h"
#include "../oss_capture.h"
#include "../include/oss_gaussian_interception.h"

#include <Windows.h>
#include <atomic>
#include <mutex>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "../third_party/Detours/include/detours.h"

using Microsoft::WRL::ComPtr;

namespace oss_gaussian {

namespace {

// ----------------------------------------------------------------------
//  Shared state.
// ----------------------------------------------------------------------

std::atomic<bool>                  g_hooks_installed{false};
std::atomic<unsigned long long>    g_frame_counter{0};

// Latest command queue seen by the ExecuteCommandLists hook. Used by the
// Present hook to schedule readback on the correct queue.
std::mutex                         g_queue_mu;
ComPtr<ID3D12CommandQueue>         g_last_queue;

// Function-pointer typedefs matching D3D12 / DXGI vtables.
typedef HRESULT (STDMETHODCALLTYPE *PFN_Present)(IDXGISwapChain*, UINT, UINT);
typedef void    (STDMETHODCALLTYPE *PFN_ExecuteCommandLists)(
    ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);

// Originals (filled in InstallD3D12Hooks; used by the trampolines).
PFN_Present              g_orig_Present              = nullptr;
PFN_ExecuteCommandLists  g_orig_ExecuteCommandLists  = nullptr;

// CaptureSampler instance. Lazily constructed. Configuration comes from
// ConfigureCaptureSampler() (called by dllmain after LoadDllConfig); if no
// configure call was made, defaults to oss_capture_default_config().
std::mutex                                g_sampler_mu;
std::unique_ptr<capture::CaptureSampler>  g_sampler;
std::string                               g_pending_capture_mode;

capture::CaptureSampler* GetOrCreateSampler() {
    std::lock_guard<std::mutex> lk(g_sampler_mu);
    if (!g_sampler) {
        OssCaptureConfig cfg = oss_capture_default_config();
        if (!g_pending_capture_mode.empty()) {
            // Map capture_mode string → OssCaptureMode enum and apply preset.
            OssCaptureMode mode = OSS_CAPTURE_MODE_LITE;  // safe default
            if (g_pending_capture_mode == "trickle") mode = OSS_CAPTURE_MODE_TRICKLE;
            else if (g_pending_capture_mode == "regular") mode = OSS_CAPTURE_MODE_REGULAR;
            else if (g_pending_capture_mode == "INSANE")  mode = OSS_CAPTURE_MODE_INSANE;
            else if (g_pending_capture_mode == "lite")    mode = OSS_CAPTURE_MODE_LITE;
            oss_capture_apply_mode_preset(&cfg, mode);
        }
        g_sampler = std::make_unique<capture::CaptureSampler>(cfg);
        OSSG_LOG_INFO("hooks", "CaptureSampler constructed (mode=%s)",
                      g_pending_capture_mode.empty() ? "default" : g_pending_capture_mode.c_str());
    }
    return g_sampler.get();
}

// ----------------------------------------------------------------------
//  Helper: extract backbuffer descriptor from IDXGISwapChain.
// ----------------------------------------------------------------------

bool GetBackbufferDesc(
    IDXGISwapChain* swap_chain,
    UINT*           width_out,
    UINT*           height_out,
    DXGI_FORMAT*    format_out
) {
    DXGI_SWAP_CHAIN_DESC sc_desc{};
    if (FAILED(swap_chain->GetDesc(&sc_desc))) {
        return false;
    }
    *width_out  = sc_desc.BufferDesc.Width;
    *height_out = sc_desc.BufferDesc.Height;
    *format_out = sc_desc.BufferDesc.Format;
    return true;
}

// ----------------------------------------------------------------------
//  The hooks themselves.
// ----------------------------------------------------------------------

// Wallclock seconds since DLL attach. For OssCaptureCandidate.timestamp_seconds.
double SecondsSinceStart() {
    static const LARGE_INTEGER s_start = []() {
        LARGE_INTEGER li{};
        QueryPerformanceCounter(&li);
        return li;
    }();
    static const double s_freq = []() {
        LARGE_INTEGER fr{};
        QueryPerformanceFrequency(&fr);
        return static_cast<double>(fr.QuadPart);
    }();
    LARGE_INTEGER now{};
    QueryPerformanceCounter(&now);
    return static_cast<double>(now.QuadPart - s_start.QuadPart) / s_freq;
}

HRESULT STDMETHODCALLTYPE Hooked_Present(
    IDXGISwapChain* This,
    UINT            SyncInterval,
    UINT            Flags
) {
    __try {
        const unsigned long long frame_idx =
            g_frame_counter.fetch_add(1, std::memory_order_relaxed) + 1;

        UINT w = 0, h = 0;
        DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
        if (GetBackbufferDesc(This, &w, &h, &fmt)) {
            // Build the candidate. For Sprint 2.x we don't have G-buffer
            // pointers yet (those land in T2.6 once we add NGX EvaluateFeature
            // unpack); for now we record present-tick timing + format-validity
            // so the sampler decides on cadence and obvious-failure modes.
            //
            // Motion magnitude, perceptual hash, and full-frame validity will
            // populate after T2.6 (NGX hook) and an early-frame readback for
            // the phash. Without those, the sampler runs in a degenerate mode
            // that mostly logs presents — useful for shakedown.
            OssCaptureCandidate cand{};
            cand.frame_index                       = frame_idx;
            cand.timestamp_seconds                 = SecondsSinceStart();
            cand.seconds_since_last_candidate      = 0.0;
            cand.seconds_since_previous_candidate  = 0.0;
            cand.motion_mean_magnitude_px          = 0.0f;
            cand.motion_below_threshold_seconds    = 0.0;
            cand.perceptual_hash_64                = 0;
            cand.depth_degenerate                  = 0;
            cand.motion_vectors_nan                = 0;
            // Format validity bit: certain HDR formats are unsupported pending
            // a colorimetric path. Flag and let sampler down-vote them.
            cand.unsupported_rt_format = (fmt == DXGI_FORMAT_R10G10B10A2_UNORM ||
                                          fmt == DXGI_FORMAT_R11G11B10_FLOAT ||
                                          fmt == DXGI_FORMAT_R16G16B16A16_FLOAT) ? 0 : 1;

            capture::CaptureSampler* sampler = GetOrCreateSampler();
            OssCaptureDecision decision = sampler->Consider(cand);

            if (decision.capture) {
                OSSG_LOG_INFO("hooks",
                              "frame %llu: KEEP (rule=%d, mode=%s, w=%u, h=%u, fmt=%d)",
                              frame_idx, static_cast<int>(decision.rule),
                              decision.capture_mode_name, w, h, static_cast<int>(fmt));
                // TODO(staging-copy.cpp): kick off async readback + EXR write.
                // The staging path is implemented in staging_copy.{h,cpp} but
                // not yet wired here pending NGX EvaluateFeature integration —
                // we want to capture LR + G-buffers, not just the post-DLSS
                // backbuffer (which leaks DLSS hallucination into our training
                // corpus). For now: log only.
            }
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        // Hook callbacks must NEVER unwind into game code; if anything in the
        // capture path faults we swallow it and pass through cleanly.
        OSSG_LOG_ERROR("hooks", "Hooked_Present caught SEH; falling through");
    }

    return g_orig_Present(This, SyncInterval, Flags);
}

void STDMETHODCALLTYPE Hooked_ExecuteCommandLists(
    ID3D12CommandQueue*       This,
    UINT                      NumCommandLists,
    ID3D12CommandList* const* ppCommandLists
) {
    __try {
        // Retain the latest queue. Most engines submit on a single graphics
        // queue per frame; we keep the most recent pointer for staging copy.
        if (This) {
            std::lock_guard<std::mutex> lk(g_queue_mu);
            if (g_last_queue.Get() != This) {
                g_last_queue = This;
                OSSG_LOG_INFO("hooks", "ExecuteCommandLists: retained new queue %p", (void*)This);
            }
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        OSSG_LOG_ERROR("hooks", "Hooked_ExecuteCommandLists caught SEH");
    }

    g_orig_ExecuteCommandLists(This, NumCommandLists, ppCommandLists);
}

// ----------------------------------------------------------------------
//  vtable-probe: spin up a throwaway swapchain + queue to read function
//  pointers from the runtime, then Detours-attach to those slots.
//
//  Failure modes:
//    - D3D12CreateDevice fails (no DX12 hardware) → log + bail
//    - Swapchain create fails (no window) → use a hidden temp window
//    - Detours attach fails → unwind partial state, log, bail
//
//  We use a hidden HWND so we don't pop a visible window in the game's UI
//  thread.
// ----------------------------------------------------------------------

bool ProbeAndHook() {
    HRESULT hr;

    ComPtr<ID3D12Device> device;
    hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(hr)) {
        OSSG_LOG_ERROR("hooks", "D3D12CreateDevice failed (0x%08lx); cannot probe", hr);
        return false;
    }

    D3D12_COMMAND_QUEUE_DESC qdesc{};
    qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qdesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    ComPtr<ID3D12CommandQueue> queue;
    hr = device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue));
    if (FAILED(hr)) {
        OSSG_LOG_ERROR("hooks", "CreateCommandQueue failed (0x%08lx)", hr);
        return false;
    }

    // Hidden window for the dummy swapchain.
    WNDCLASSEXW wc{};
    wc.cbSize        = sizeof(wc);
    wc.lpfnWndProc   = DefWindowProcW;
    wc.hInstance     = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"OSS_GAUSSIAN_PROBE";
    RegisterClassExW(&wc);
    HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"oss-probe",
                                WS_OVERLAPPEDWINDOW, 0, 0, 16, 16,
                                nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd) {
        OSSG_LOG_ERROR("hooks", "CreateWindowExW failed (le=%lu)", GetLastError());
        return false;
    }

    ComPtr<IDXGIFactory4> factory;
    hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        OSSG_LOG_ERROR("hooks", "CreateDXGIFactory1 failed (0x%08lx)", hr);
        DestroyWindow(hwnd);
        return false;
    }

    DXGI_SWAP_CHAIN_DESC1 sc{};
    sc.Width              = 16;
    sc.Height             = 16;
    sc.Format             = DXGI_FORMAT_R8G8B8A8_UNORM;
    sc.BufferCount        = 2;
    sc.BufferUsage        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sc.SwapEffect         = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sc.SampleDesc.Count   = 1;

    ComPtr<IDXGISwapChain1> swap_chain1;
    hr = factory->CreateSwapChainForHwnd(queue.Get(), hwnd, &sc, nullptr, nullptr, &swap_chain1);
    if (FAILED(hr)) {
        OSSG_LOG_ERROR("hooks", "CreateSwapChainForHwnd failed (0x%08lx)", hr);
        DestroyWindow(hwnd);
        return false;
    }

    // Read vtable slots. IDXGISwapChain::Present is slot 8 (after IUnknown:
    // QueryInterface, AddRef, Release, then IDXGIObject: SetPrivateData,
    // SetPrivateDataInterface, GetPrivateData, GetParent — that's 7 — then
    // IDXGIDeviceSubObject: GetDevice — total 8 entries before Present).
    void** swap_vt  = *reinterpret_cast<void***>(swap_chain1.Get());
    void** queue_vt = *reinterpret_cast<void***>(queue.Get());
    g_orig_Present              = reinterpret_cast<PFN_Present>(swap_vt[8]);
    g_orig_ExecuteCommandLists  = reinterpret_cast<PFN_ExecuteCommandLists>(queue_vt[10]);

    // Attach via Detours.
    LONG err = NO_ERROR;
    err = DetourTransactionBegin();
    err |= DetourUpdateThread(GetCurrentThread());
    err |= DetourAttach(reinterpret_cast<PVOID*>(&g_orig_Present),             reinterpret_cast<PVOID>(Hooked_Present));
    err |= DetourAttach(reinterpret_cast<PVOID*>(&g_orig_ExecuteCommandLists), reinterpret_cast<PVOID>(Hooked_ExecuteCommandLists));
    if (err != NO_ERROR) {
        OSSG_LOG_ERROR("hooks", "Detour attach failed (err=%ld)", err);
        DetourTransactionAbort();
        DestroyWindow(hwnd);
        return false;
    }
    err = DetourTransactionCommit();
    if (err != NO_ERROR) {
        OSSG_LOG_ERROR("hooks", "Detour commit failed (err=%ld)", err);
        DestroyWindow(hwnd);
        return false;
    }

    DestroyWindow(hwnd);

    OSSG_LOG_INFO("hooks", "D3D12 hooks installed (Present=%p, ExecuteCommandLists=%p)",
                  reinterpret_cast<void*>(g_orig_Present),
                  reinterpret_cast<void*>(g_orig_ExecuteCommandLists));
    return true;
}

}  // namespace

// ----------------------------------------------------------------------
//  Public surface.
// ----------------------------------------------------------------------

bool InstallD3D12Hooks() {
    if (g_hooks_installed.load(std::memory_order_acquire)) {
        return true;
    }
    bool ok = ProbeAndHook();
    if (ok) {
        g_hooks_installed.store(true, std::memory_order_release);
    }
    return ok;
}

void UninstallD3D12Hooks() {
    if (!g_hooks_installed.load(std::memory_order_acquire)) return;

    LONG err = NO_ERROR;
    err = DetourTransactionBegin();
    err |= DetourUpdateThread(GetCurrentThread());
    if (g_orig_Present) {
        err |= DetourDetach(reinterpret_cast<PVOID*>(&g_orig_Present),             reinterpret_cast<PVOID>(Hooked_Present));
    }
    if (g_orig_ExecuteCommandLists) {
        err |= DetourDetach(reinterpret_cast<PVOID*>(&g_orig_ExecuteCommandLists), reinterpret_cast<PVOID>(Hooked_ExecuteCommandLists));
    }
    if (err != NO_ERROR) {
        OSSG_LOG_ERROR("hooks", "Detour detach failed (err=%ld); aborting transaction", err);
        DetourTransactionAbort();
    } else {
        DetourTransactionCommit();
    }

    {
        std::lock_guard<std::mutex> lk(g_queue_mu);
        g_last_queue.Reset();
    }
    {
        std::lock_guard<std::mutex> lk(g_sampler_mu);
        g_sampler.reset();
    }

    g_hooks_installed.store(false, std::memory_order_release);
    OSSG_LOG_INFO("hooks", "D3D12 hooks uninstalled");
}

bool AreD3D12HooksActive() {
    return g_hooks_installed.load(std::memory_order_acquire);
}

unsigned long long CurrentFrameIndex() {
    return g_frame_counter.load(std::memory_order_relaxed);
}

void ConfigureCaptureSampler(const char* capture_mode) {
    if (!capture_mode || capture_mode[0] == '\0') return;
    std::lock_guard<std::mutex> lk(g_sampler_mu);
    g_pending_capture_mode = capture_mode;
    // If sampler already exists, drop it so the next Consider() rebuilds with
    // the new mode. Race-free under the same lock.
    g_sampler.reset();
    OSSG_LOG_INFO("hooks", "ConfigureCaptureSampler: mode set to %s", capture_mode);
}

}  // namespace oss_gaussian
