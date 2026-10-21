// =============================================================================
//  dllmain.cpp
//
//  DLL entry point. DllMain itself only performs loader-lock-safe scheduling;
//  heavyweight logger/config/proxy/hook setup runs on a worker thread after
//  process attach returns.
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

#include "../oss_capture.h"
#include "../include/oss_gaussian_interception.h"
#include "d3d12_hooks.h"
#include "dll_config.h"
#include "dxgi_proxy.h"
#include "log.h"
#include "version_proxy.h"

#include <Windows.h>

#include <atomic>
#include <cctype>
#include <mutex>
#include <string>

namespace oss_gaussian {

namespace {

std::atomic<OssGaussianRenderMode> g_render_mode{OSS_GAUSSIAN_MODE_PASSTHROUGH};

std::mutex                         g_callback_mu;
OssGaussianFrameCallback           g_callback     = nullptr;
void*                              g_callback_ud  = nullptr;

// Set by DllMain ATTACH; cleared on DETACH. Used as a sanity check elsewhere.
std::atomic<bool>                  g_dll_attached{false};
std::atomic<bool>                  g_init_started{false};
std::atomic<bool>                  g_native_init_entered{false};
std::atomic<bool>                  g_init_completed{false};
std::atomic<bool>                  g_shutdown_requested{false};
std::atomic<bool>                  g_proxy_inline_init_started{false};
HMODULE                            g_self_module{nullptr};

DWORD WINAPI InitThreadProc(LPVOID param);

void WriteInitMarker(const char* phase) {
    wchar_t enabled[8] = {};
    if (GetEnvironmentVariableW(L"OSS_GAUSSIAN_INIT_MARKER", enabled, 8) == 0) return;
    if (enabled[0] != L'1' && enabled[0] != L't' && enabled[0] != L'T') return;

    wchar_t base[MAX_PATH] = {};
    if (GetEnvironmentVariableW(L"LOCALAPPDATA", base, MAX_PATH) == 0) return;

    wchar_t dir[MAX_PATH] = {};
    if (swprintf_s(dir, L"%s\\oss-gaussian", base) < 0) return;
    CreateDirectoryW(dir, nullptr);

    wchar_t path[MAX_PATH] = {};
    if (swprintf_s(path, L"%s\\init-marker.log", dir) < 0) return;

    HANDLE h = CreateFileW(
        path,
        FILE_APPEND_DATA,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr,
        OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (h == INVALID_HANDLE_VALUE) return;

    char line[160] = {};
    DWORD pid = GetCurrentProcessId();
    DWORD tid = GetCurrentThreadId();
    SYSTEMTIME st{};
    GetLocalTime(&st);
    int n = _snprintf_s(line, _countof(line), _TRUNCATE,
        "%04u-%02u-%02uT%02u:%02u:%02u.%03u pid=%lu tid=%lu phase=%s\n",
        st.wYear, st.wMonth, st.wDay,
        st.wHour, st.wMinute, st.wSecond, st.wMilliseconds,
        static_cast<unsigned long>(pid),
        static_cast<unsigned long>(tid),
        phase ? phase : "(null)");
    if (n > 0) {
        DWORD written = 0;
        WriteFile(h, line, static_cast<DWORD>(n), &written, nullptr);
    }
    CloseHandle(h);
}

OssCaptureMode ParseCaptureMode(const std::string& mode) {
    if (mode == "trickle") return OSS_CAPTURE_MODE_TRICKLE;
    if (mode == "regular") return OSS_CAPTURE_MODE_REGULAR;
    if (mode == "insane" || mode == "INSANE") return OSS_CAPTURE_MODE_INSANE;
    return OSS_CAPTURE_MODE_LITE;
}

bool EnvFlagDisabled(const wchar_t* name) {
    wchar_t value[16] = {};
    const DWORD n = GetEnvironmentVariableW(name, value, static_cast<DWORD>(_countof(value)));
    if (n == 0 || n >= _countof(value)) return false;
    return value[0] == L'0' || value[0] == L'f' || value[0] == L'F' ||
           value[0] == L'n' || value[0] == L'N';
}

std::wstring Utf8ToWideLocal(const std::string& value) {
    if (value.empty()) return std::wstring();
    const int needed = MultiByteToWideChar(
        CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0);
    if (needed <= 0) return std::wstring();
    std::wstring out(static_cast<size_t>(needed), L'\0');
    MultiByteToWideChar(
        CP_UTF8, 0, value.data(), static_cast<int>(value.size()), out.data(), needed);
    return out;
}

void ConfigureNativeCapture(const DllConfig& dll_cfg) {
    OssCaptureConfig cfg = oss_capture_default_config();
    oss_capture_apply_mode_preset(&cfg, ParseCaptureMode(dll_cfg.capture_mode));
    cfg.capture_normals = 0;
    cfg.capture_albedo = 0;
    cfg.capture_roughness = 0;
    cfg.capture_metallic = 0;
    cfg.capture_emissive = 0;
    cfg.two_tier_enabled = 0;
    const std::string game_id = dll_cfg.game_id.empty() ? "unknown-game" : dll_cfg.game_id;
    strncpy_s(cfg.game_id, game_id.c_str(), _TRUNCATE);
    strncpy_s(cfg.game_version, "unknown", _TRUNCATE);
    // Capture metadata gets the per-install consent_id, never install_token:
    // install_token is the upload bearer credential and capture files can be
    // shared or left on disk.
    const std::string consent_id =
        dll_cfg.consent_id.empty() ? "unspecified" : dll_cfg.consent_id;
    strncpy_s(cfg.user_consent_token, consent_id.c_str(), _TRUNCATE);
    const std::string storage_mode =
        dll_cfg.capture_storage_mode.empty() ? "local" : dll_cfg.capture_storage_mode;
    strncpy_s(cfg.capture_storage_mode, storage_mode.c_str(), _TRUNCATE);
    if (!dll_cfg.pending_dir.empty()) {
        const std::wstring pending = Utf8ToWideLocal(dll_cfg.pending_dir);
        if (!pending.empty()) {
            wcsncpy_s(cfg.pending_root, pending.c_str(), _TRUNCATE);
        }
    }
    if (!oss_capture_configure(&cfg)) {
        OSSG_LOG_WARN("dll", "oss_capture_configure failed; NGX training capture inactive");
    }
}

bool PinThisModuleForProcessLifetime() {
    HMODULE pinned_module = nullptr;
    if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            reinterpret_cast<LPCWSTR>(&g_render_mode),
            &pinned_module)) {
        return true;
    }

    OutputDebugStringA(
        "[oss-gaussian] DllMain: failed to pin module; async hook init disabled\n");
    return false;
}

bool ScheduleAttach(HMODULE self) {
    WriteInitMarker("schedule-start");
    bool expected = false;
    if (!g_init_started.compare_exchange_strong(
            expected,
            true,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        WriteInitMarker("schedule-already-started");
        return true;
    }

    WriteInitMarker("create-thread-start");
    HANDLE thread = CreateThread(
        nullptr,
        0,
        InitThreadProc,
        self,
        0,
        nullptr);
    if (!thread) {
        g_init_started.store(false, std::memory_order_release);
        WriteInitMarker("create-thread-failed");
        OutputDebugStringA(
            "[oss-gaussian] DllMain: CreateThread failed; hook init disabled\n");
        return false;
    }

    CloseHandle(thread);
    WriteInitMarker("create-thread-done");
    return true;
}

bool InitInDllMainRequested() {
    wchar_t enabled[8] = {};
    if (GetEnvironmentVariableW(L"OSS_GAUSSIAN_INIT_IN_DLLMAIN", enabled, 8) == 0) return false;
    return enabled[0] == L'1' || enabled[0] == L't' || enabled[0] == L'T';
}

void InitializeNativeHooks(HMODULE self) {
    (void)self;
    if (g_shutdown_requested.load(std::memory_order_acquire)) {
        return;
    }

    LogInit();
    OSSG_LOG_INFO("dll", "OSS-Gaussian interception DLL attached (build %s)",
                  oss_gaussian_version());

#ifndef OSS_GAUSSIAN_NO_SYSTEM_PROXY_ATTACH
    // T2.3: Forward DXGI exports to system32 dxgi.dll. The forwarders
    // themselves live in dxgi_proxy.cpp; this just primes the cached HMODULE.
    // Failure is non-fatal: forwarders log a miss and return E_NOTIMPL.
    if (OssGaussianDxgiProxyAttach()) {
        OSSG_LOG_INFO("dll", "DXGI proxy attached (system32 dxgi.dll cached)");
    } else {
        OSSG_LOG_ERROR("dll",
                       "DXGI proxy attach failed; forwarded exports will return E_NOTIMPL");
    }
    if (OssGaussianVersionProxyAttach()) {
        OSSG_LOG_INFO("dll", "VERSION proxy attached (system32 version.dll cached)");
    } else {
        OSSG_LOG_ERROR("dll",
                       "VERSION proxy attach failed; forwarded exports will fail closed");
    }
#else
    OSSG_LOG_INFO("dll", "system proxy attach skipped for non-system proxy target");
#endif

    // T2.7: Load per-game config.json (capture mode, runtime knobs). Best-guess
    // the game_id from the host EXE basename — installer puts config under
    // %LOCALAPPDATA%\oss-capture\<game_id>\config.json so the basename usually
    // matches.
    std::string game_id_hint;
    {
        char exe_path[MAX_PATH];
        DWORD len = GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
        if (len > 0 && len < MAX_PATH) {
            std::string ep(exe_path);
            size_t slash = ep.find_last_of("\\/");
            std::string basename = (slash == std::string::npos) ? ep : ep.substr(slash + 1);
            // Strip extension
            size_t dot = basename.find_last_of('.');
            if (dot != std::string::npos) basename = basename.substr(0, dot);
            // Lowercase
            for (auto& c : basename) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
            game_id_hint = basename;
        }
    }
    DllConfig dll_cfg = LoadDllConfig(game_id_hint);
    OSSG_LOG_INFO("dll", "DllConfig: loaded=%d, source=%s, mode=%s",
                  dll_cfg.is_loaded ? 1 : 0,
                  dll_cfg.loaded_from.empty() ? "(defaults)" : dll_cfg.loaded_from.c_str(),
                  dll_cfg.capture_mode.empty() ? "(default)" : dll_cfg.capture_mode.c_str());

    // Push capture_mode to the hook layer so the sampler is constructed with
    // the right preset on first Present.
    if (!dll_cfg.capture_mode.empty()) {
        ConfigureCaptureSampler(dll_cfg.capture_mode.c_str());
    }
    ConfigureNativeCapture(dll_cfg);

    const bool d3d12_hooks_enabled =
        dll_cfg.d3d12_hooks_enabled && !EnvFlagDisabled(L"OSS_GAUSSIAN_D3D12_HOOKS");
    if (!d3d12_hooks_enabled) {
        OSSG_LOG_INFO(
            "dll",
            "D3D12 hooks disabled by config/env; provider capture remains available");
    } else if (InstallD3D12Hooks()) {
        OSSG_LOG_INFO("dll", "D3D12 hooks installed (Present + ExecuteCommandLists)");
    } else {
        OSSG_LOG_ERROR("dll", "D3D12 hook install failed; capture path inactive");
    }

    // TODO(T2.5): LoadLibrary detour for nvngx_dlss.dll → our HMODULE.
    // TODO(T2.6): Hook NVSDK_NGX_D3D12_EvaluateFeature param-dict unpack.
    // TODO(T2.9): SetWindowsHookExW(WH_KEYBOARD_LL, ...) for F11 toggle.

    g_dll_attached.store(true, std::memory_order_release);
    g_init_completed.store(true, std::memory_order_release);
}

void EnsureInitializedFromProxyExportImpl() {
#ifdef OSS_GAUSSIAN_PROXY_EXPORT_LIGHTWEIGHT_INIT
    LogInit();
    return;
#else
    if (g_init_completed.load(std::memory_order_acquire) ||
        g_shutdown_requested.load(std::memory_order_acquire)) {
        return;
    }

    WriteInitMarker("proxy-ensure-start");
    bool expected = false;
    if (!g_proxy_inline_init_started.compare_exchange_strong(
            expected,
            true,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        WriteInitMarker("proxy-ensure-already-started");
        return;
    }

    WriteInitMarker("proxy-inline-begin");
    bool native_expected = false;
    if (g_native_init_entered.compare_exchange_strong(
            native_expected,
            true,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        PinThisModuleForProcessLifetime();
        InitializeNativeHooks(g_self_module);
    } else {
        WriteInitMarker("proxy-inline-native-init-already-entered");
    }
    WriteInitMarker("proxy-inline-end");
#endif
}

DWORD WINAPI InitThreadProc(LPVOID param) {
    WriteInitMarker("begin");
    bool native_expected = false;
    if (g_native_init_entered.compare_exchange_strong(
            native_expected,
            true,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        PinThisModuleForProcessLifetime();
        InitializeNativeHooks(static_cast<HMODULE>(param));
    } else {
        WriteInitMarker("thread-native-init-already-entered");
    }
    WriteInitMarker("end");
    return 0;
}

void OnDetach(LPVOID reserved) {
    g_shutdown_requested.store(true, std::memory_order_release);
    g_dll_attached.store(false, std::memory_order_release);
    g_init_completed.store(false, std::memory_order_release);

    if (!reserved) {
        OutputDebugStringA(
            "[oss-gaussian] DllMain: dynamic detach observed; "
            "heavy cleanup skipped under loader lock\n");
    }
}

} // namespace

// -----------------------------------------------------------------------------
//  Public C API implementations.
// -----------------------------------------------------------------------------

void OssGaussianEnsureInitializedFromProxyExport() {
    EnsureInitializedFromProxyExportImpl();
}

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

OSS_GAUSSIAN_API uint32_t
oss_gaussian_is_initialized(void) {
    return oss_gaussian::g_init_completed.load(std::memory_order_acquire) ? 1u : 0u;
}

OSS_GAUSSIAN_API const char*
oss_gaussian_version(void) {
    return "0.1.0+sprint2-scaffold";
}

} // extern "C"

// -----------------------------------------------------------------------------
//  DllMain.
// -----------------------------------------------------------------------------
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID reserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        oss_gaussian::g_self_module = hModule;
        oss_gaussian::WriteInitMarker("dllmain-attach");
        // Disable per-thread callbacks; we don't need them and they cost cycles
        // in a game process with many short-lived threads.
        DisableThreadLibraryCalls(hModule);
        oss_gaussian::g_dll_attached.store(true, std::memory_order_release);
#ifndef OSS_GAUSSIAN_DEFER_NATIVE_ATTACH
        oss_gaussian::ScheduleAttach(hModule);
#else
        oss_gaussian::WriteInitMarker("native-attach-deferred");
#endif
        if (oss_gaussian::InitInDllMainRequested()) {
            oss_gaussian::WriteInitMarker("dllmain-direct-init-begin");
            bool native_expected = false;
            if (oss_gaussian::g_native_init_entered.compare_exchange_strong(
                    native_expected,
                    true,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                oss_gaussian::PinThisModuleForProcessLifetime();
                oss_gaussian::InitializeNativeHooks(hModule);
            } else {
                oss_gaussian::WriteInitMarker("dllmain-direct-native-init-already-entered");
            }
            oss_gaussian::WriteInitMarker("dllmain-direct-init-end");
        }
        oss_gaussian::WriteInitMarker("dllmain-attach-done");
        break;

    case DLL_PROCESS_DETACH:
        oss_gaussian::OnDetach(reserved);
        break;

    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    default:
        break;
    }
    return TRUE;
}
