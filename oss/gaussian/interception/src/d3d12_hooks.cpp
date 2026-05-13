// =============================================================================
//  d3d12_hooks.cpp
//
//  Native D3D12 + DXGI hook plumbing for the OSS-Gaussian capture path.
//
//  Current production behavior:
//    - Probe real runtime COM vtables with a throwaway D3D12 device, command
//      queue, factory, hidden window, and swap chain.
//    - Detour IDXGISwapChain::Present and ID3D12CommandQueue::
//      ExecuteCommandLists as the core path.
//    - Best-effort detour IDXGISwapChain1::Present1 plus IDXGIFactory/
//      IDXGIFactory2 swap-chain creation methods so creation-time queues and
//      swap chains are logged/retained when available.
//    - Feed Present metadata into CaptureSampler and the oss_capture hook seam.
//
//  Fallback behavior:
//    Provider adapters (NGX/Streamline/FSR) are preferred because they expose
//    LR color, output, depth, and motion vectors. When a game reaches Present
//    but has not exercised a provider adapter yet, the hook can still capture
//    the swapchain backbuffer as a raw local frame so field smoke tests prove
//    the end-to-end capture path without uploading anything.
//
//  Copyright 2026 OSS-Gaussian contributors. Apache 2.0.
// =============================================================================
#define OSS_GAUSSIAN_BUILDING_DLL 1

#include "d3d12_hooks.h"

#if !defined(_WIN32)

namespace oss_gaussian {

bool InstallD3D12Hooks() { return false; }
void ConfigureCaptureSampler(const char*) {}
void UninstallD3D12Hooks() {}
bool AreD3D12HooksActive() { return false; }
unsigned long long CurrentFrameIndex() { return 0; }

D3D12HookStatus GetD3D12HookStatus() {
    D3D12HookStatus status{};
    status.state = D3D12HookInstallState::Failed;
    return status;
}

const char* D3D12HookInstallStateName(D3D12HookInstallState state) {
    switch (state) {
        case D3D12HookInstallState::NotInstalled: return "not-installed";
        case D3D12HookInstallState::Installed: return "installed";
        case D3D12HookInstallState::Degraded: return "degraded";
        case D3D12HookInstallState::Failed: return "failed";
    }
    return "unknown";
}

bool InspectD3D12HookVTablesForTesting(
    void*,
    void*,
    void*,
    D3D12HookVTableProbe*) {
    return false;
}

}  // namespace oss_gaussian

#else

#include "log.h"
#include "staging_copy.h"
#include "../oss_capture.h"
#include "../include/oss_gaussian_interception.h"

#include <Windows.h>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "../third_party/Detours/src/detours.h"

using Microsoft::WRL::ComPtr;

namespace oss_gaussian {

namespace {

constexpr UINT kPresentSlot = 8;
constexpr UINT kPresent1Slot = 22;
constexpr UINT kExecuteCommandListsSlot = 10;
constexpr UINT kCreateSwapChainSlot = 10;
constexpr UINT kCreateSwapChainForHwndSlot = 15;
constexpr UINT kCreateSwapChainForCoreWindowSlot = 16;
constexpr UINT kCreateSwapChainForCompositionSlot = 24;
constexpr bool kReadbackDegraded = false;

std::mutex g_install_mu;
std::mutex g_present_attach_mu;
std::atomic<bool> g_hooks_installed{false};
std::atomic<uint32_t> g_install_state{
    static_cast<uint32_t>(D3D12HookInstallState::NotInstalled)};

std::atomic<bool> g_present_hooked{false};
std::atomic<bool> g_present1_hooked{false};
std::atomic<bool> g_execute_hooked{false};
std::atomic<bool> g_swapchain_creation_hooked{false};

std::atomic<long> g_last_detour_error{NO_ERROR};
std::atomic<long> g_last_probe_hresult{S_OK};

std::atomic<unsigned long long> g_frame_counter{0};
std::atomic<unsigned long long> g_present_count{0};
std::atomic<unsigned long long> g_present1_count{0};
std::atomic<unsigned long long> g_execute_count{0};
std::atomic<unsigned long long> g_swapchain_create_count{0};
std::atomic<unsigned long long> g_capture_keep_count{0};
std::atomic<unsigned long long> g_degraded_capture_count{0};

std::mutex g_queue_mu;
ComPtr<ID3D12CommandQueue> g_last_queue;

std::mutex g_sampler_mu;
std::unique_ptr<capture::CaptureSampler> g_sampler;
std::string g_pending_capture_mode;

std::mutex g_timing_mu;
double g_previous_present_seconds = 0.0;
double g_static_motion_seconds = 0.0;

std::once_flag g_present_session_once;
std::string g_present_session_uuid;

struct PresentBurstState {
    bool active = false;
    uint32_t next_index = 0;
    uint32_t burst_n = 0;
    OssCaptureBurstTier tier = OSS_CAPTURE_TIER_NONE;
    char tier_name[8]{};
    OssCaptureMode capture_mode = OSS_CAPTURE_MODE_LITE;
    char capture_mode_name[8]{};
    char burst_uuid[37]{};
};

std::mutex g_present_burst_mu;
PresentBurstState g_present_burst;

typedef HRESULT (STDMETHODCALLTYPE *PFN_Present)(IDXGISwapChain*, UINT, UINT);
typedef HRESULT (STDMETHODCALLTYPE *PFN_Present1)(
    IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);
typedef void (STDMETHODCALLTYPE *PFN_ExecuteCommandLists)(
    ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);
typedef HRESULT (STDMETHODCALLTYPE *PFN_CreateSwapChain)(
    IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**);
typedef HRESULT (STDMETHODCALLTYPE *PFN_CreateSwapChainForHwnd)(
    IDXGIFactory2*,
    IUnknown*,
    HWND,
    const DXGI_SWAP_CHAIN_DESC1*,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*,
    IDXGIOutput*,
    IDXGISwapChain1**);
typedef HRESULT (STDMETHODCALLTYPE *PFN_CreateSwapChainForCoreWindow)(
    IDXGIFactory2*,
    IUnknown*,
    IUnknown*,
    const DXGI_SWAP_CHAIN_DESC1*,
    IDXGIOutput*,
    IDXGISwapChain1**);
typedef HRESULT (STDMETHODCALLTYPE *PFN_CreateSwapChainForComposition)(
    IDXGIFactory2*,
    IUnknown*,
    const DXGI_SWAP_CHAIN_DESC1*,
    IDXGIOutput*,
    IDXGISwapChain1**);

PFN_Present g_orig_Present = nullptr;
PFN_Present1 g_orig_Present1 = nullptr;
PFN_ExecuteCommandLists g_orig_ExecuteCommandLists = nullptr;
PFN_CreateSwapChain g_orig_CreateSwapChain = nullptr;
PFN_CreateSwapChainForHwnd g_orig_CreateSwapChainForHwnd = nullptr;
PFN_CreateSwapChainForCoreWindow g_orig_CreateSwapChainForCoreWindow = nullptr;
PFN_CreateSwapChainForComposition g_orig_CreateSwapChainForComposition = nullptr;

HRESULT STDMETHODCALLTYPE Hooked_Present(IDXGISwapChain* This, UINT SyncInterval, UINT Flags);
HRESULT STDMETHODCALLTYPE Hooked_Present1(
    IDXGISwapChain1* This,
    UINT SyncInterval,
    UINT PresentFlags,
    const DXGI_PRESENT_PARAMETERS* pPresentParameters);

D3D12HookInstallState LoadInstallState() {
    return static_cast<D3D12HookInstallState>(
        g_install_state.load(std::memory_order_acquire));
}

void StoreInstallState(D3D12HookInstallState state) {
    g_install_state.store(static_cast<uint32_t>(state), std::memory_order_release);
}

OssCaptureMode ParseCaptureMode(const std::string& mode_name) {
    if (mode_name == "trickle") return OSS_CAPTURE_MODE_TRICKLE;
    if (mode_name == "regular") return OSS_CAPTURE_MODE_REGULAR;
    if (mode_name == "INSANE") return OSS_CAPTURE_MODE_INSANE;
    return OSS_CAPTURE_MODE_LITE;
}

capture::CaptureSampler* GetOrCreateSampler() {
    std::lock_guard<std::mutex> lk(g_sampler_mu);
    if (!g_sampler) {
        OssCaptureConfig cfg = oss_capture_default_config();
        if (!g_pending_capture_mode.empty()) {
            oss_capture_apply_mode_preset(&cfg, ParseCaptureMode(g_pending_capture_mode));
        }
        g_sampler = std::make_unique<capture::CaptureSampler>(cfg);
        OSSG_LOG_INFO(
            "hooks",
            "CaptureSampler constructed (mode=%s)",
            g_pending_capture_mode.empty() ? "default" : g_pending_capture_mode.c_str());
    }
    return g_sampler.get();
}

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

bool IsCaptureCandidateFormat(DXGI_FORMAT format) {
    switch (format) {
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        case DXGI_FORMAT_R10G10B10A2_UNORM:
        case DXGI_FORMAT_R11G11B10_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            return true;
        default:
            return false;
    }
}

uint64_t MetadataHash64(
    unsigned long long frame_index,
    UINT width,
    UINT height,
    DXGI_FORMAT format) {
    uint64_t h = 1469598103934665603ull;
    const uint64_t values[] = {
        static_cast<uint64_t>(frame_index),
        static_cast<uint64_t>(width),
        static_cast<uint64_t>(height),
        static_cast<uint64_t>(format),
    };
    for (uint64_t value : values) {
        h ^= value;
        h *= 1099511628211ull;
    }
    return h;
}

bool GetBackbufferDesc(
    IDXGISwapChain* swap_chain,
    UINT* width_out,
    UINT* height_out,
    DXGI_FORMAT* format_out) {
    if (!swap_chain || !width_out || !height_out || !format_out) {
        return false;
    }

    DXGI_SWAP_CHAIN_DESC desc{};
    if (SUCCEEDED(swap_chain->GetDesc(&desc))) {
        *width_out = desc.BufferDesc.Width;
        *height_out = desc.BufferDesc.Height;
        *format_out = desc.BufferDesc.Format;
        if (*width_out != 0 && *height_out != 0 && *format_out != DXGI_FORMAT_UNKNOWN) {
            return true;
        }
    }

    ComPtr<IDXGISwapChain1> swap_chain1;
    if (SUCCEEDED(swap_chain->QueryInterface(IID_PPV_ARGS(&swap_chain1)))) {
        DXGI_SWAP_CHAIN_DESC1 desc1{};
        if (SUCCEEDED(swap_chain1->GetDesc1(&desc1))) {
            *width_out = desc1.Width;
            *height_out = desc1.Height;
            *format_out = desc1.Format;
            return true;
        }
    }

    return false;
}

void RetainQueue(ID3D12CommandQueue* queue, const char* source) {
    if (!queue) return;

    std::lock_guard<std::mutex> lk(g_queue_mu);
    if (g_last_queue.Get() != queue) {
        g_last_queue = queue;
        OSSG_LOG_INFO("hooks", "%s: retained command queue %p", source, queue);
    }
}

bool AttachPresentHookFromRealSwapChain(IDXGISwapChain* swap_chain, const char* source) {
    if (!swap_chain || g_present_hooked.load(std::memory_order_acquire)) {
        return g_present_hooked.load(std::memory_order_acquire);
    }

    std::lock_guard<std::mutex> lk(g_present_attach_mu);
    if (g_present_hooked.load(std::memory_order_acquire)) {
        return true;
    }

    auto swap_vt = *reinterpret_cast<void***>(swap_chain);
    if (!swap_vt || !swap_vt[kPresentSlot]) {
        OSSG_LOG_WARN("hooks", "%s: cannot late-attach Present; vtable slot missing", source);
        return false;
    }

    g_orig_Present = reinterpret_cast<PFN_Present>(swap_vt[kPresentSlot]);

    LONG err = DetourTransactionBegin();
    if (err == NO_ERROR) err = DetourUpdateThread(GetCurrentThread());
    if (err == NO_ERROR) {
        err = DetourAttach(
            reinterpret_cast<PVOID*>(&g_orig_Present),
            reinterpret_cast<PVOID>(Hooked_Present));
    }
    if (err != NO_ERROR) {
        g_last_detour_error.store(err, std::memory_order_release);
        DetourTransactionAbort();
        OSSG_LOG_WARN("hooks", "%s: late Present DetourAttach failed err=%ld", source, err);
        return false;
    }
    err = DetourTransactionCommit();
    g_last_detour_error.store(err, std::memory_order_release);
    if (err != NO_ERROR) {
        OSSG_LOG_WARN("hooks", "%s: late Present DetourCommit failed err=%ld", source, err);
        return false;
    }
    g_present_hooked.store(true, std::memory_order_release);
    OSSG_LOG_INFO("hooks", "%s: late-attached IDXGISwapChain::Present from real swapchain", source);

    if (swap_vt[kPresent1Slot] && !g_present1_hooked.load(std::memory_order_acquire)) {
        auto present1 = reinterpret_cast<PFN_Present1>(swap_vt[kPresent1Slot]);
        err = DetourTransactionBegin();
        if (err == NO_ERROR) err = DetourUpdateThread(GetCurrentThread());
        if (err == NO_ERROR) {
            g_orig_Present1 = present1;
            err = DetourAttach(
                reinterpret_cast<PVOID*>(&g_orig_Present1),
                reinterpret_cast<PVOID>(Hooked_Present1));
        }
        if (err == NO_ERROR) {
            err = DetourTransactionCommit();
        } else {
            DetourTransactionAbort();
        }
        g_last_detour_error.store(err, std::memory_order_release);
        if (err == NO_ERROR) {
            g_present1_hooked.store(true, std::memory_order_release);
            OSSG_LOG_INFO("hooks", "%s: late-attached IDXGISwapChain1::Present1", source);
        } else {
            OSSG_LOG_WARN("hooks", "%s: late Present1 attach skipped err=%ld", source, err);
        }
    }

    return true;
}

ComPtr<ID3D12CommandQueue> CurrentQueue() {
    std::lock_guard<std::mutex> lk(g_queue_mu);
    return g_last_queue;
}

std::wstring Utf8ToWide(const std::string& value) {
    if (value.empty()) return std::wstring();
    int needed = MultiByteToWideChar(
        CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0);
    std::wstring out(static_cast<size_t>(needed), L'\0');
    MultiByteToWideChar(
        CP_UTF8, 0, value.data(), static_cast<int>(value.size()), out.data(), needed);
    return out;
}

std::string WideToUtf8(const std::wstring& value) {
    if (value.empty()) return std::string();
    int needed = WideCharToMultiByte(
        CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(needed), '\0');
    WideCharToMultiByte(
        CP_UTF8, 0, value.data(), static_cast<int>(value.size()), out.data(), needed, nullptr, nullptr);
    return out;
}

std::string JsonEscape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char ch : value) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(ch); break;
        }
    }
    return out;
}

std::string PresentSessionUuid() {
    std::call_once(g_present_session_once, []() {
        SYSTEMTIME st{};
        GetSystemTime(&st);
        DWORD pid = GetCurrentProcessId();
        char buf[96]{};
        snprintf(
            buf,
            sizeof(buf),
            "present-%04u%02u%02u-%02u%02u%02u-%lu",
            st.wYear,
            st.wMonth,
            st.wDay,
            st.wHour,
            st.wMinute,
            st.wSecond,
            static_cast<unsigned long>(pid));
        g_present_session_uuid = buf;
    });
    return g_present_session_uuid;
}

double UnixNowSeconds() {
    FILETIME ft{};
    GetSystemTimeAsFileTime(&ft);
    ULARGE_INTEGER uli{};
    uli.LowPart = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;
    constexpr unsigned long long kUnixEpoch100ns = 116444736000000000ull;
    return static_cast<double>(uli.QuadPart - kUnixEpoch100ns) / 10000000.0;
}

std::wstring LocalPendingRoot() {
    std::wstring configured = capture::CurrentPendingRoot();
    if (!configured.empty()) return configured;

    wchar_t buf[MAX_PATH]{};
    DWORD len = GetEnvironmentVariableW(L"LOCALAPPDATA", buf, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        return std::wstring(buf) + L"\\oss-capture\\pending";
    }
    return L".\\oss-capture\\pending";
}

std::wstring BuildPresentExrPath(const OssCaptureConfig& cfg, unsigned long long frame_index) {
    std::string game_id = cfg.game_id[0] ? cfg.game_id : "unknown-game";
    std::string session_uuid = PresentSessionUuid();

    std::ostringstream frame_name;
    frame_name << "present-" << std::setw(10) << std::setfill('0') << frame_index << ".exr";
    return LocalPendingRoot() + L"\\" + Utf8ToWide(game_id) + L"\\" +
           Utf8ToWide(session_uuid) + L"\\" + Utf8ToWide(frame_name.str());
}

void WritePresentMetadata(
    const std::wstring& exr_path,
    const OssCaptureConfig& cfg,
    const std::string& session_uuid,
    const std::string& frame_uuid,
    unsigned long long frame_index,
    UINT width,
    UINT height,
    DXGI_FORMAT format,
    const OssCaptureDecision& decision,
    uint32_t burst_index) {
    try {
        std::filesystem::path json_path(exr_path);
        json_path += L".json";
        std::ofstream meta(json_path, std::ios::binary | std::ios::trunc);
        if (!meta.is_open()) {
            OSSG_LOG_ERROR("hooks", "present metadata open failed path=%s", WideToUtf8(json_path.wstring()).c_str());
            return;
        }
        const char* game_id = cfg.game_id[0] ? cfg.game_id : "unknown-game";
        const char* storage_mode =
            cfg.capture_storage_mode[0] ? cfg.capture_storage_mode : "local";
        meta << "{\n";
        meta << "  \"schema_version\": 1,\n";
        meta << "  \"capture_kind\": \"present_raw_backbuffer\",\n";
        meta << "  \"game_id\": \"" << JsonEscape(game_id) << "\",\n";
        meta << "  \"capture_storage_mode\": \"" << JsonEscape(storage_mode) << "\",\n";
        meta << "  \"session_uuid\": \"" << JsonEscape(session_uuid) << "\",\n";
        meta << "  \"frame_uuid\": \"" << JsonEscape(frame_uuid) << "\",\n";
        meta << "  \"frame_index\": " << frame_index << ",\n";
        meta << "  \"captured_at_unix\": " << UnixNowSeconds() << ",\n";
        meta << "  \"sequence_index\": " << frame_index << ",\n";
        meta << "  \"sequence_reset\": 0,\n";
        meta << "  \"width\": " << width << ",\n";
        meta << "  \"height\": " << height << ",\n";
        meta << "  \"lr_resolution\": null,\n";
        meta << "  \"hr_resolution\": [" << width << ", " << height << "],\n";
        meta << "  \"dxgi_format\": " << static_cast<int>(format) << ",\n";
        meta << "  \"channels\": {\n";
        meta << "    \"present_color_raw\": true,\n";
        meta << "    \"lr_color\": false,\n";
        meta << "    \"hr_output\": false,\n";
        meta << "    \"depth\": false,\n";
        meta << "    \"motion_vectors\": false\n";
        meta << "  },\n";
        meta << "  \"capture_rule\": " << static_cast<int>(decision.rule) << ",\n";
        meta << "  \"capture_mode\": \"" << JsonEscape(decision.capture_mode_name) << "\",\n";
        meta << "  \"burst_uuid\": \"" << JsonEscape(decision.burst_uuid) << "\",\n";
        meta << "  \"burst_index\": " << burst_index << ",\n";
        meta << "  \"burst_n\": " << decision.burst_n << ",\n";
        meta << "  \"burst_tier\": \"" << JsonEscape(decision.burst_tier_name) << "\",\n";
        meta << "  \"raw_path\": \"" << JsonEscape(WideToUtf8(exr_path) + ".raw") << "\"\n";
        meta << "}\n";
    } catch (...) {
        OSSG_LOG_ERROR("hooks", "present metadata write threw");
    }
}

OssCaptureDecision ContinuePresentBurst(uint32_t* burst_index_out) {
    OssCaptureDecision decision{};
    std::lock_guard<std::mutex> lk(g_present_burst_mu);
    if (!g_present_burst.active || g_present_burst.next_index >= g_present_burst.burst_n) {
        return decision;
    }
    decision.capture = 1u;
    decision.rule = OSS_CAPTURE_RULE_TEMPORAL_STRIDE;
    decision.burst_n = g_present_burst.burst_n;
    decision.burst_tier = g_present_burst.tier;
    decision.capture_mode = g_present_burst.capture_mode;
    strncpy_s(decision.burst_uuid, g_present_burst.burst_uuid, _TRUNCATE);
    strncpy_s(decision.burst_tier_name, g_present_burst.tier_name, _TRUNCATE);
    strncpy_s(decision.capture_mode_name, g_present_burst.capture_mode_name, _TRUNCATE);
    if (burst_index_out) *burst_index_out = g_present_burst.next_index;
    ++g_present_burst.next_index;
    if (g_present_burst.next_index >= g_present_burst.burst_n) {
        g_present_burst = PresentBurstState{};
    }
    return decision;
}

uint32_t StartPresentBurst(const OssCaptureDecision& decision) {
    if (!decision.capture || decision.burst_n <= 1u ||
        decision.burst_tier == OSS_CAPTURE_TIER_NONE) {
        return 0u;
    }
    std::lock_guard<std::mutex> lk(g_present_burst_mu);
    g_present_burst = PresentBurstState{};
    g_present_burst.active = true;
    g_present_burst.next_index = 1u;
    g_present_burst.burst_n = decision.burst_n;
    g_present_burst.tier = decision.burst_tier;
    g_present_burst.capture_mode = decision.capture_mode;
    strncpy_s(g_present_burst.burst_uuid, decision.burst_uuid, _TRUNCATE);
    strncpy_s(g_present_burst.tier_name, decision.burst_tier_name, _TRUNCATE);
    strncpy_s(g_present_burst.capture_mode_name, decision.capture_mode_name, _TRUNCATE);
    return 0u;
}

void TryRetainQueueFromUnknown(IUnknown* maybe_queue, const char* source) {
    if (!maybe_queue) return;

    ComPtr<ID3D12CommandQueue> queue;
    if (SUCCEEDED(maybe_queue->QueryInterface(IID_PPV_ARGS(&queue)))) {
        RetainQueue(queue.Get(), source);
    }
}

void LogSwapChainCreated(
    const char* api,
    IUnknown* device_or_queue,
    IDXGISwapChain* swap_chain) {
    g_swapchain_create_count.fetch_add(1, std::memory_order_relaxed);
    TryRetainQueueFromUnknown(device_or_queue, api);
    AttachPresentHookFromRealSwapChain(swap_chain, api);

    UINT width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    if (GetBackbufferDesc(swap_chain, &width, &height, &format)) {
        OSSG_LOG_INFO(
            "hooks",
            "%s: swap_chain=%p size=%ux%u fmt=%d",
            api,
            swap_chain,
            width,
            height,
            static_cast<int>(format));
    } else {
        OSSG_LOG_INFO("hooks", "%s: swap_chain=%p desc unavailable", api, swap_chain);
    }
}

OssCaptureCandidate BuildCandidate(
    unsigned long long frame_index,
    double now_seconds,
    UINT width,
    UINT height,
    DXGI_FORMAT format) {
    double previous_delta = 0.0;
    double static_motion_seconds = 0.0;
    {
        std::lock_guard<std::mutex> lk(g_timing_mu);
        if (g_previous_present_seconds > 0.0) {
            previous_delta = now_seconds - g_previous_present_seconds;
            if (previous_delta < 0.0) previous_delta = 0.0;
        }
        g_previous_present_seconds = now_seconds;
        g_static_motion_seconds += previous_delta;
        static_motion_seconds = g_static_motion_seconds;
    }

    OssCaptureCandidate candidate{};
    candidate.frame_index = frame_index;
    candidate.timestamp_seconds = now_seconds;
    candidate.seconds_since_last_candidate = previous_delta;
    candidate.seconds_since_previous_candidate = previous_delta;
    candidate.motion_mean_magnitude_px = 0.0f;
    candidate.motion_below_threshold_seconds = static_motion_seconds;
    candidate.perceptual_hash_64 = MetadataHash64(frame_index, width, height, format);
    candidate.depth_degenerate = 0;
    candidate.motion_vectors_nan = 0;
    candidate.unsupported_rt_format = IsCaptureCandidateFormat(format) ? 0u : 1u;
    return candidate;
}

void ProcessPresent(IDXGISwapChain* swap_chain, const char* api_name) {
    oss_capture_on_present(swap_chain);

    const unsigned long long frame_index =
        g_frame_counter.fetch_add(1, std::memory_order_relaxed) + 1ull;

    UINT width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    if (!GetBackbufferDesc(swap_chain, &width, &height, &format)) {
        OSSG_LOG_WARN("hooks", "%s: frame %llu desc unavailable", api_name, frame_index);
        return;
    }

    const double now = SecondsSinceStart();
    OssCaptureCandidate candidate = BuildCandidate(frame_index, now, width, height, format);
    uint32_t burst_index = 0u;
    OssCaptureDecision decision = ContinuePresentBurst(&burst_index);
    if (!decision.capture) {
        decision = GetOrCreateSampler()->Consider(candidate);
        burst_index = StartPresentBurst(decision);
    }

    if (!decision.capture) {
        return;
    }

    g_capture_keep_count.fetch_add(1, std::memory_order_relaxed);
    ComPtr<ID3D12Resource> backbuffer;
    UINT buffer_index = 0;
    ComPtr<IDXGISwapChain3> swap_chain3;
    if (SUCCEEDED(swap_chain->QueryInterface(IID_PPV_ARGS(&swap_chain3)))) {
        buffer_index = swap_chain3->GetCurrentBackBufferIndex();
    }
    HRESULT buffer_hr = swap_chain->GetBuffer(buffer_index, IID_PPV_ARGS(&backbuffer));
    if (FAILED(buffer_hr) || !backbuffer) {
        g_degraded_capture_count.fetch_add(1, std::memory_order_relaxed);
        OSSG_LOG_INFO(
            "hooks",
            "%s: frame %llu KEEP rule=%d mode=%s size=%ux%u fmt=%d readback=degraded(GetBuffer failed hr=0x%08lx)",
            api_name,
            frame_index,
            static_cast<int>(decision.rule),
            decision.capture_mode_name,
            width,
            height,
            static_cast<int>(format),
            buffer_hr);
        return;
    }

    ComPtr<ID3D12CommandQueue> queue = CurrentQueue();
    if (!queue) {
        g_degraded_capture_count.fetch_add(1, std::memory_order_relaxed);
        OSSG_LOG_INFO(
            "hooks",
            "%s: frame %llu KEEP rule=%d mode=%s size=%ux%u fmt=%d backbuffer=%p readback=degraded(no command queue)",
            api_name,
            frame_index,
            static_cast<int>(decision.rule),
            decision.capture_mode_name,
            width,
            height,
            static_cast<int>(format),
            backbuffer.Get());
        return;
    }

    ComPtr<ID3D12Device> device;
    HRESULT device_hr = backbuffer->GetDevice(IID_PPV_ARGS(&device));
    if (FAILED(device_hr) || !InitStagingCopy(device.Get())) {
        g_degraded_capture_count.fetch_add(1, std::memory_order_relaxed);
        OSSG_LOG_INFO(
            "hooks",
            "%s: frame %llu KEEP rule=%d mode=%s size=%ux%u fmt=%d backbuffer=%p readback=degraded(device/staging unavailable hr=0x%08lx)",
            api_name,
            frame_index,
            static_cast<int>(decision.rule),
            decision.capture_mode_name,
            width,
            height,
            static_cast<int>(format),
            backbuffer.Get(),
            device_hr);
        return;
    }

    OssCaptureConfig cfg = capture::CurrentCaptureConfig();
    const std::string session_uuid = PresentSessionUuid();
    std::wstring exr_path = BuildPresentExrPath(cfg, frame_index);
    const std::string frame_uuid =
        WideToUtf8(std::filesystem::path(exr_path).stem().wstring());
    std::filesystem::create_directories(std::filesystem::path(exr_path).parent_path());
    const std::string exr_utf8 = WideToUtf8(exr_path);
    bool scheduled = ScheduleReadback(
        queue.Get(),
        backbuffer.Get(),
        width,
        height,
        static_cast<int32_t>(format),
        exr_utf8.c_str(),
        frame_index);
    if (scheduled) {
        WritePresentMetadata(
            exr_path,
            cfg,
            session_uuid,
            frame_uuid,
            frame_index,
            width,
            height,
            format,
            decision,
            burst_index);
    } else {
        g_degraded_capture_count.fetch_add(1, std::memory_order_relaxed);
    }

    OSSG_LOG_INFO(
        "hooks",
        "%s: frame %llu KEEP rule=%d mode=%s burst=%u/%u size=%ux%u fmt=%d backbuffer=%p "
        "readback=%s path=%s.raw",
        api_name,
        frame_index,
        static_cast<int>(decision.rule),
        decision.capture_mode_name,
        burst_index,
        decision.burst_n,
        width,
        height,
        static_cast<int>(format),
        backbuffer.Get(),
        scheduled ? "scheduled" : "degraded(ScheduleReadback failed)",
        exr_utf8.c_str());
}

void SafeProcessPresent(IDXGISwapChain* swap_chain, const char* api_name) {
#if defined(_MSC_VER)
    __try {
        ProcessPresent(swap_chain, api_name);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        OSSG_LOG_ERROR("hooks", "%s hook caught SEH; falling through", api_name);
    }
#else
    try {
        ProcessPresent(swap_chain, api_name);
    } catch (...) {
        OSSG_LOG_ERROR("hooks", "%s hook caught exception; falling through", api_name);
    }
#endif
}

void SafeTrackExecuteCommandLists(ID3D12CommandQueue* queue) {
#if defined(_MSC_VER)
    __try {
        RetainQueue(queue, "ExecuteCommandLists");
        oss_capture_on_execute_command_lists(queue);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        OSSG_LOG_ERROR("hooks", "ExecuteCommandLists hook caught SEH");
    }
#else
    try {
        RetainQueue(queue, "ExecuteCommandLists");
        oss_capture_on_execute_command_lists(queue);
    } catch (...) {
        OSSG_LOG_ERROR("hooks", "ExecuteCommandLists hook caught exception");
    }
#endif
}

void SafeLogSwapChainCreated(
    const char* api,
    IUnknown* device_or_queue,
    IDXGISwapChain* swap_chain) {
#if defined(_MSC_VER)
    __try {
        LogSwapChainCreated(api, device_or_queue, swap_chain);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        OSSG_LOG_ERROR("hooks", "%s creation hook caught SEH", api);
    }
#else
    try {
        LogSwapChainCreated(api, device_or_queue, swap_chain);
    } catch (...) {
        OSSG_LOG_ERROR("hooks", "%s creation hook caught exception", api);
    }
#endif
}

HRESULT STDMETHODCALLTYPE Hooked_Present(
    IDXGISwapChain* This,
    UINT SyncInterval,
    UINT Flags) {
    g_present_count.fetch_add(1, std::memory_order_relaxed);
    SafeProcessPresent(This, "Present");
    return g_orig_Present(This, SyncInterval, Flags);
}

HRESULT STDMETHODCALLTYPE Hooked_Present1(
    IDXGISwapChain1* This,
    UINT SyncInterval,
    UINT PresentFlags,
    const DXGI_PRESENT_PARAMETERS* pPresentParameters) {
    g_present1_count.fetch_add(1, std::memory_order_relaxed);
    SafeProcessPresent(This, "Present1");
    return g_orig_Present1(This, SyncInterval, PresentFlags, pPresentParameters);
}

void STDMETHODCALLTYPE Hooked_ExecuteCommandLists(
    ID3D12CommandQueue* This,
    UINT NumCommandLists,
    ID3D12CommandList* const* ppCommandLists) {
    g_execute_count.fetch_add(1, std::memory_order_relaxed);
    SafeTrackExecuteCommandLists(This);
    g_orig_ExecuteCommandLists(This, NumCommandLists, ppCommandLists);
    oss_capture_on_execute_command_lists_executed(
        This,
        NumCommandLists,
        reinterpret_cast<void* const*>(ppCommandLists));
}

HRESULT STDMETHODCALLTYPE Hooked_CreateSwapChain(
    IDXGIFactory* This,
    IUnknown* pDevice,
    DXGI_SWAP_CHAIN_DESC* pDesc,
    IDXGISwapChain** ppSwapChain) {
    HRESULT hr = g_orig_CreateSwapChain(This, pDevice, pDesc, ppSwapChain);
    if (SUCCEEDED(hr) && ppSwapChain && *ppSwapChain) {
        SafeLogSwapChainCreated("CreateSwapChain", pDevice, *ppSwapChain);
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE Hooked_CreateSwapChainForHwnd(
    IDXGIFactory2* This,
    IUnknown* pDevice,
    HWND hWnd,
    const DXGI_SWAP_CHAIN_DESC1* pDesc,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* pFullscreenDesc,
    IDXGIOutput* pRestrictToOutput,
    IDXGISwapChain1** ppSwapChain) {
    HRESULT hr = g_orig_CreateSwapChainForHwnd(
        This, pDevice, hWnd, pDesc, pFullscreenDesc, pRestrictToOutput, ppSwapChain);
    if (SUCCEEDED(hr) && ppSwapChain && *ppSwapChain) {
        SafeLogSwapChainCreated("CreateSwapChainForHwnd", pDevice, *ppSwapChain);
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE Hooked_CreateSwapChainForCoreWindow(
    IDXGIFactory2* This,
    IUnknown* pDevice,
    IUnknown* pWindow,
    const DXGI_SWAP_CHAIN_DESC1* pDesc,
    IDXGIOutput* pRestrictToOutput,
    IDXGISwapChain1** ppSwapChain) {
    HRESULT hr = g_orig_CreateSwapChainForCoreWindow(
        This, pDevice, pWindow, pDesc, pRestrictToOutput, ppSwapChain);
    if (SUCCEEDED(hr) && ppSwapChain && *ppSwapChain) {
        SafeLogSwapChainCreated("CreateSwapChainForCoreWindow", pDevice, *ppSwapChain);
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE Hooked_CreateSwapChainForComposition(
    IDXGIFactory2* This,
    IUnknown* pDevice,
    const DXGI_SWAP_CHAIN_DESC1* pDesc,
    IDXGIOutput* pRestrictToOutput,
    IDXGISwapChain1** ppSwapChain) {
    HRESULT hr = g_orig_CreateSwapChainForComposition(
        This, pDevice, pDesc, pRestrictToOutput, ppSwapChain);
    if (SUCCEEDED(hr) && ppSwapChain && *ppSwapChain) {
        SafeLogSwapChainCreated("CreateSwapChainForComposition", pDevice, *ppSwapChain);
    }
    return hr;
}

bool CreateProbeWindow(HWND* hwnd_out, ATOM* atom_out) {
    if (!hwnd_out || !atom_out) return false;

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"OSS_GAUSSIAN_D3D12_HOOK_PROBE";

    ATOM atom = RegisterClassExW(&wc);
    if (!atom && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
        OSSG_LOG_ERROR("hooks", "RegisterClassExW failed (le=%lu)", GetLastError());
        return false;
    }

    HWND hwnd = CreateWindowExW(
        0,
        wc.lpszClassName,
        L"oss-d3d12-probe",
        WS_OVERLAPPEDWINDOW,
        0,
        0,
        16,
        16,
        nullptr,
        nullptr,
        wc.hInstance,
        nullptr);
    if (!hwnd) {
        OSSG_LOG_ERROR("hooks", "CreateWindowExW failed (le=%lu)", GetLastError());
        return false;
    }

    *hwnd_out = hwnd;
    *atom_out = atom;
    return true;
}

void DestroyProbeWindow(HWND hwnd, ATOM atom) {
    if (hwnd) {
        DestroyWindow(hwnd);
    }
    if (atom) {
        UnregisterClassW(L"OSS_GAUSSIAN_D3D12_HOOK_PROBE", GetModuleHandleW(nullptr));
    }
}

bool AttachRequiredHooks() {
    LONG err = DetourTransactionBegin();
    if (err == NO_ERROR) err = DetourUpdateThread(GetCurrentThread());
    if (err == NO_ERROR && g_orig_Present) {
        err = DetourAttach(
            reinterpret_cast<PVOID*>(&g_orig_Present),
            reinterpret_cast<PVOID>(Hooked_Present));
    }
    if (err == NO_ERROR && g_orig_ExecuteCommandLists) {
        err = DetourAttach(
            reinterpret_cast<PVOID*>(&g_orig_ExecuteCommandLists),
            reinterpret_cast<PVOID>(Hooked_ExecuteCommandLists));
    }

    if (err != NO_ERROR) {
        g_last_detour_error.store(err, std::memory_order_release);
        OSSG_LOG_ERROR("hooks", "required DetourAttach failed (err=%ld)", err);
        DetourTransactionAbort();
        return false;
    }

    err = DetourTransactionCommit();
    g_last_detour_error.store(err, std::memory_order_release);
    if (err != NO_ERROR) {
        OSSG_LOG_ERROR("hooks", "required DetourTransactionCommit failed (err=%ld)", err);
        return false;
    }

    g_present_hooked.store(g_orig_Present != nullptr, std::memory_order_release);
    g_execute_hooked.store(g_orig_ExecuteCommandLists != nullptr, std::memory_order_release);
    return true;
}

bool AttachOptionalHooks() {
    LONG err = DetourTransactionBegin();
    if (err == NO_ERROR) err = DetourUpdateThread(GetCurrentThread());
    if (err == NO_ERROR && g_orig_Present1) {
        err = DetourAttach(
            reinterpret_cast<PVOID*>(&g_orig_Present1),
            reinterpret_cast<PVOID>(Hooked_Present1));
    }
    if (err == NO_ERROR && g_orig_CreateSwapChain) {
        err = DetourAttach(
            reinterpret_cast<PVOID*>(&g_orig_CreateSwapChain),
            reinterpret_cast<PVOID>(Hooked_CreateSwapChain));
    }
    if (err == NO_ERROR && g_orig_CreateSwapChainForHwnd) {
        err = DetourAttach(
            reinterpret_cast<PVOID*>(&g_orig_CreateSwapChainForHwnd),
            reinterpret_cast<PVOID>(Hooked_CreateSwapChainForHwnd));
    }
    if (err == NO_ERROR && g_orig_CreateSwapChainForCoreWindow) {
        err = DetourAttach(
            reinterpret_cast<PVOID*>(&g_orig_CreateSwapChainForCoreWindow),
            reinterpret_cast<PVOID>(Hooked_CreateSwapChainForCoreWindow));
    }
    if (err == NO_ERROR && g_orig_CreateSwapChainForComposition) {
        err = DetourAttach(
            reinterpret_cast<PVOID*>(&g_orig_CreateSwapChainForComposition),
            reinterpret_cast<PVOID>(Hooked_CreateSwapChainForComposition));
    }

    if (err != NO_ERROR) {
        g_last_detour_error.store(err, std::memory_order_release);
        OSSG_LOG_WARN("hooks", "optional DetourAttach failed (err=%ld); keeping core hooks", err);
        DetourTransactionAbort();
        return false;
    }

    err = DetourTransactionCommit();
    g_last_detour_error.store(err, std::memory_order_release);
    if (err != NO_ERROR) {
        OSSG_LOG_WARN(
            "hooks",
            "optional DetourTransactionCommit failed (err=%ld); keeping core hooks",
            err);
        return false;
    }

    g_present1_hooked.store(g_orig_Present1 != nullptr, std::memory_order_release);
    g_swapchain_creation_hooked.store(
        g_orig_CreateSwapChain != nullptr ||
            g_orig_CreateSwapChainForHwnd != nullptr ||
            g_orig_CreateSwapChainForCoreWindow != nullptr ||
            g_orig_CreateSwapChainForComposition != nullptr,
        std::memory_order_release);
    return true;
}

bool ProbeAndHook() {
    HRESULT hr = S_OK;

    HMODULE system_d3d12 = LoadLibraryW(L"C:\\Windows\\System32\\d3d12.dll");
    if (!system_d3d12) {
        OSSG_LOG_ERROR(
            "hooks",
            "LoadLibraryW(System32\\d3d12.dll) failed (err=%lu)",
            GetLastError());
        return false;
    }
    using D3D12CreateDeviceFn = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
    auto create_d3d12_device =
        reinterpret_cast<D3D12CreateDeviceFn>(GetProcAddress(system_d3d12, "D3D12CreateDevice"));
    if (!create_d3d12_device) {
        OSSG_LOG_ERROR("hooks", "GetProcAddress(D3D12CreateDevice) failed (err=%lu)",
                       GetLastError());
        FreeLibrary(system_d3d12);
        return false;
    }

    ComPtr<ID3D12Device> device;
    hr = create_d3d12_device(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    g_last_probe_hresult.store(hr, std::memory_order_release);
    if (FAILED(hr)) {
        OSSG_LOG_ERROR("hooks", "D3D12CreateDevice failed (0x%08lx); hooks inactive", hr);
        return false;
    }

    D3D12_COMMAND_QUEUE_DESC qdesc{};
    qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qdesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    ComPtr<ID3D12CommandQueue> queue;
    hr = device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue));
    g_last_probe_hresult.store(hr, std::memory_order_release);
    if (FAILED(hr)) {
        OSSG_LOG_ERROR("hooks", "CreateCommandQueue failed (0x%08lx)", hr);
        return false;
    }

    HWND hwnd = nullptr;
    ATOM atom = 0;
    if (!CreateProbeWindow(&hwnd, &atom)) {
        return false;
    }

    HMODULE system_dxgi = LoadLibraryW(L"C:\\Windows\\System32\\dxgi.dll");
    if (!system_dxgi) {
        OSSG_LOG_ERROR(
            "hooks",
            "LoadLibraryW(System32\\dxgi.dll) failed (err=%lu)",
            GetLastError());
        DestroyProbeWindow(hwnd, atom);
        return false;
    }
    using CreateDXGIFactory1Fn = HRESULT(WINAPI*)(REFIID, void**);
    auto create_dxgi_factory1 =
        reinterpret_cast<CreateDXGIFactory1Fn>(GetProcAddress(system_dxgi, "CreateDXGIFactory1"));
    if (!create_dxgi_factory1) {
        OSSG_LOG_ERROR("hooks", "GetProcAddress(CreateDXGIFactory1) failed (err=%lu)",
                       GetLastError());
        FreeLibrary(system_dxgi);
        DestroyProbeWindow(hwnd, atom);
        return false;
    }

    ComPtr<IDXGIFactory4> factory;
    hr = create_dxgi_factory1(IID_PPV_ARGS(&factory));
    g_last_probe_hresult.store(hr, std::memory_order_release);
    if (FAILED(hr)) {
        OSSG_LOG_ERROR("hooks", "CreateDXGIFactory1 failed (0x%08lx)", hr);
        DestroyProbeWindow(hwnd, atom);
        return false;
    }

    DXGI_SWAP_CHAIN_DESC1 sc{};
    sc.Width = 16;
    sc.Height = 16;
    sc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sc.BufferCount = 2;
    sc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sc.SampleDesc.Count = 1;

    ComPtr<IDXGISwapChain1> swap_chain1;
    hr = factory->CreateSwapChainForHwnd(queue.Get(), hwnd, &sc, nullptr, nullptr, &swap_chain1);
    g_last_probe_hresult.store(hr, std::memory_order_release);
    if (FAILED(hr)) {
        OSSG_LOG_WARN("hooks",
                      "CreateSwapChainForHwnd probe failed (0x%08lx); installing "
                      "ExecuteCommandLists/factory hooks without Present",
                      hr);
        auto queue_vt = *reinterpret_cast<void***>(queue.Get());
        auto factory_vt = *reinterpret_cast<void***>(factory.Get());
        if (queue_vt) {
            g_orig_ExecuteCommandLists =
                reinterpret_cast<PFN_ExecuteCommandLists>(queue_vt[kExecuteCommandListsSlot]);
        }
        if (factory_vt) {
            g_orig_CreateSwapChain =
                reinterpret_cast<PFN_CreateSwapChain>(factory_vt[kCreateSwapChainSlot]);
            g_orig_CreateSwapChainForHwnd =
                reinterpret_cast<PFN_CreateSwapChainForHwnd>(
                    factory_vt[kCreateSwapChainForHwndSlot]);
            g_orig_CreateSwapChainForCoreWindow =
                reinterpret_cast<PFN_CreateSwapChainForCoreWindow>(
                    factory_vt[kCreateSwapChainForCoreWindowSlot]);
            g_orig_CreateSwapChainForComposition =
                reinterpret_cast<PFN_CreateSwapChainForComposition>(
                    factory_vt[kCreateSwapChainForCompositionSlot]);
        }
        DestroyProbeWindow(hwnd, atom);
        if (!g_orig_ExecuteCommandLists) {
            OSSG_LOG_ERROR("hooks", "ExecuteCommandLists vtable slot missing");
            return false;
        }
    } else {
        D3D12HookVTableProbe probe{};
        if (!InspectD3D12HookVTablesForTesting(
                swap_chain1.Get(),
                queue.Get(),
                factory.Get(),
                &probe)) {
            OSSG_LOG_ERROR("hooks", "vtable inspection failed");
            DestroyProbeWindow(hwnd, atom);
            return false;
        }

        g_orig_Present = reinterpret_cast<PFN_Present>(probe.present);
        g_orig_Present1 = reinterpret_cast<PFN_Present1>(probe.present1);
        g_orig_ExecuteCommandLists =
            reinterpret_cast<PFN_ExecuteCommandLists>(probe.execute_command_lists);
        g_orig_CreateSwapChain = reinterpret_cast<PFN_CreateSwapChain>(probe.create_swap_chain);
        g_orig_CreateSwapChainForHwnd =
            reinterpret_cast<PFN_CreateSwapChainForHwnd>(probe.create_swap_chain_for_hwnd);
        g_orig_CreateSwapChainForCoreWindow =
            reinterpret_cast<PFN_CreateSwapChainForCoreWindow>(
                probe.create_swap_chain_for_core_window);
        g_orig_CreateSwapChainForComposition =
            reinterpret_cast<PFN_CreateSwapChainForComposition>(
                probe.create_swap_chain_for_composition);

        DestroyProbeWindow(hwnd, atom);
    }

    if (!g_orig_ExecuteCommandLists) {
        OSSG_LOG_ERROR(
            "hooks",
            "required vtable slot missing (ExecuteCommandLists=%p)",
            reinterpret_cast<void*>(g_orig_ExecuteCommandLists));
        return false;
    }

    DetourRestoreAfterWith();

    if (!AttachRequiredHooks()) {
        return false;
    }

    const bool optional_ok = AttachOptionalHooks();
    StoreInstallState(optional_ok && !kReadbackDegraded
                          ? D3D12HookInstallState::Installed
                          : D3D12HookInstallState::Degraded);

    OSSG_LOG_INFO(
        "hooks",
        "D3D12 hooks %s (Present=%d Present1=%d ExecuteCommandLists=%d "
        "SwapChainCreate=%d)",
        optional_ok
            ? "installed with degraded readback"
            : "installed with degraded optional DXGI coverage and readback",
        g_present_hooked.load(std::memory_order_acquire) ? 1 : 0,
        g_present1_hooked.load(std::memory_order_acquire) ? 1 : 0,
        g_execute_hooked.load(std::memory_order_acquire) ? 1 : 0,
        g_swapchain_creation_hooked.load(std::memory_order_acquire) ? 1 : 0);
    return true;
}

void ResetRuntimeStateAfterDetach() {
    {
        std::lock_guard<std::mutex> lk(g_queue_mu);
        g_last_queue.Reset();
    }
    {
        std::lock_guard<std::mutex> lk(g_sampler_mu);
        g_sampler.reset();
    }
    {
        std::lock_guard<std::mutex> lk(g_timing_mu);
        g_previous_present_seconds = 0.0;
        g_static_motion_seconds = 0.0;
    }
}

}  // namespace

bool InstallD3D12Hooks() {
    std::lock_guard<std::mutex> lk(g_install_mu);
    if (g_hooks_installed.load(std::memory_order_acquire)) {
        return true;
    }

    const bool ok = ProbeAndHook();
    g_hooks_installed.store(ok, std::memory_order_release);
    if (!ok) {
        StoreInstallState(D3D12HookInstallState::Failed);
    }
    return ok;
}

void UninstallD3D12Hooks() {
    std::lock_guard<std::mutex> lk(g_install_mu);
    if (!g_hooks_installed.load(std::memory_order_acquire)) {
        return;
    }

    LONG err = DetourTransactionBegin();
    if (err == NO_ERROR) err = DetourUpdateThread(GetCurrentThread());
    if (err == NO_ERROR && g_present_hooked.load(std::memory_order_acquire)) {
        err = DetourDetach(
            reinterpret_cast<PVOID*>(&g_orig_Present),
            reinterpret_cast<PVOID>(Hooked_Present));
    }
    if (err == NO_ERROR && g_present1_hooked.load(std::memory_order_acquire)) {
        err = DetourDetach(
            reinterpret_cast<PVOID*>(&g_orig_Present1),
            reinterpret_cast<PVOID>(Hooked_Present1));
    }
    if (err == NO_ERROR && g_execute_hooked.load(std::memory_order_acquire)) {
        err = DetourDetach(
            reinterpret_cast<PVOID*>(&g_orig_ExecuteCommandLists),
            reinterpret_cast<PVOID>(Hooked_ExecuteCommandLists));
    }
    if (err == NO_ERROR && g_swapchain_creation_hooked.load(std::memory_order_acquire)) {
        if (g_orig_CreateSwapChain) {
            err = DetourDetach(
                reinterpret_cast<PVOID*>(&g_orig_CreateSwapChain),
                reinterpret_cast<PVOID>(Hooked_CreateSwapChain));
        }
        if (err == NO_ERROR && g_orig_CreateSwapChainForHwnd) {
            err = DetourDetach(
                reinterpret_cast<PVOID*>(&g_orig_CreateSwapChainForHwnd),
                reinterpret_cast<PVOID>(Hooked_CreateSwapChainForHwnd));
        }
        if (err == NO_ERROR && g_orig_CreateSwapChainForCoreWindow) {
            err = DetourDetach(
                reinterpret_cast<PVOID*>(&g_orig_CreateSwapChainForCoreWindow),
                reinterpret_cast<PVOID>(Hooked_CreateSwapChainForCoreWindow));
        }
        if (err == NO_ERROR && g_orig_CreateSwapChainForComposition) {
            err = DetourDetach(
                reinterpret_cast<PVOID*>(&g_orig_CreateSwapChainForComposition),
                reinterpret_cast<PVOID>(Hooked_CreateSwapChainForComposition));
        }
    }

    if (err != NO_ERROR) {
        g_last_detour_error.store(err, std::memory_order_release);
        OSSG_LOG_ERROR("hooks", "Detour detach failed (err=%ld); aborting transaction", err);
        DetourTransactionAbort();
    } else {
        err = DetourTransactionCommit();
        g_last_detour_error.store(err, std::memory_order_release);
        if (err != NO_ERROR) {
            OSSG_LOG_ERROR("hooks", "Detour detach commit failed (err=%ld)", err);
        }
    }

    ShutdownStagingCopy();
    ResetRuntimeStateAfterDetach();

    g_present_hooked.store(false, std::memory_order_release);
    g_present1_hooked.store(false, std::memory_order_release);
    g_execute_hooked.store(false, std::memory_order_release);
    g_swapchain_creation_hooked.store(false, std::memory_order_release);
    g_hooks_installed.store(false, std::memory_order_release);
    StoreInstallState(D3D12HookInstallState::NotInstalled);

    OSSG_LOG_INFO("hooks", "D3D12 hooks uninstalled");
}

bool AreD3D12HooksActive() {
    return g_hooks_installed.load(std::memory_order_acquire);
}

D3D12HookStatus GetD3D12HookStatus() {
    D3D12HookStatus status{};
    status.state = LoadInstallState();
    status.installed = g_hooks_installed.load(std::memory_order_acquire);
    status.degraded = status.state == D3D12HookInstallState::Degraded;
    status.present_hooked = g_present_hooked.load(std::memory_order_acquire);
    status.present1_hooked = g_present1_hooked.load(std::memory_order_acquire);
    status.execute_command_lists_hooked = g_execute_hooked.load(std::memory_order_acquire);
    status.swap_chain_creation_hooked =
        g_swapchain_creation_hooked.load(std::memory_order_acquire);
    status.readback_degraded = kReadbackDegraded;
    status.last_detour_error = g_last_detour_error.load(std::memory_order_acquire);
    status.last_probe_hresult = g_last_probe_hresult.load(std::memory_order_acquire);
    status.frame_count = g_frame_counter.load(std::memory_order_relaxed);
    status.present_count = g_present_count.load(std::memory_order_relaxed);
    status.present1_count = g_present1_count.load(std::memory_order_relaxed);
    status.execute_command_lists_count = g_execute_count.load(std::memory_order_relaxed);
    status.swap_chain_creation_count = g_swapchain_create_count.load(std::memory_order_relaxed);
    status.capture_keep_count = g_capture_keep_count.load(std::memory_order_relaxed);
    status.degraded_capture_count = g_degraded_capture_count.load(std::memory_order_relaxed);
    return status;
}

const char* D3D12HookInstallStateName(D3D12HookInstallState state) {
    switch (state) {
        case D3D12HookInstallState::NotInstalled: return "not-installed";
        case D3D12HookInstallState::Installed: return "installed";
        case D3D12HookInstallState::Degraded: return "degraded";
        case D3D12HookInstallState::Failed: return "failed";
    }
    return "unknown";
}

unsigned long long CurrentFrameIndex() {
    return g_frame_counter.load(std::memory_order_relaxed);
}

void ConfigureCaptureSampler(const char* capture_mode) {
    std::lock_guard<std::mutex> lk(g_sampler_mu);
    if (!capture_mode || capture_mode[0] == '\0') {
        g_pending_capture_mode.clear();
    } else {
        g_pending_capture_mode = capture_mode;
    }
    g_sampler.reset();
    OSSG_LOG_INFO(
        "hooks",
        "ConfigureCaptureSampler: mode set to %s",
        g_pending_capture_mode.empty() ? "(default)" : g_pending_capture_mode.c_str());
}

bool InspectD3D12HookVTablesForTesting(
    void* swap_chain,
    void* command_queue,
    void* factory,
    D3D12HookVTableProbe* out) {
    if (!swap_chain || !command_queue || !factory || !out) {
        return false;
    }

    auto swap_vt = *reinterpret_cast<void***>(swap_chain);
    auto queue_vt = *reinterpret_cast<void***>(command_queue);
    auto factory_vt = *reinterpret_cast<void***>(factory);
    if (!swap_vt || !queue_vt || !factory_vt) {
        return false;
    }

    D3D12HookVTableProbe probe{};
    probe.present = swap_vt[kPresentSlot];
    probe.present1 = swap_vt[kPresent1Slot];
    probe.execute_command_lists = queue_vt[kExecuteCommandListsSlot];
    probe.create_swap_chain = factory_vt[kCreateSwapChainSlot];
    probe.create_swap_chain_for_hwnd = factory_vt[kCreateSwapChainForHwndSlot];
    probe.create_swap_chain_for_core_window = factory_vt[kCreateSwapChainForCoreWindowSlot];
    probe.create_swap_chain_for_composition = factory_vt[kCreateSwapChainForCompositionSlot];

    if (!probe.present || !probe.execute_command_lists) {
        return false;
    }

    *out = probe;
    return true;
}

}  // namespace oss_gaussian

#endif  // defined(_WIN32)
