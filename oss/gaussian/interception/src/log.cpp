// =============================================================================
//  log.cpp
//
//  Modeled on OptiScaler's `Logger.cpp` (https://github.com/optiscaler/OptiScaler).
//  Stripped to a single mutex-guarded FILE* and OutputDebugStringA mirror.
//
//  Copyright 2026 OSS-Gaussian contributors
//  Licensed under the Apache License, Version 2.0 (see ../LICENSE).
// =============================================================================
#include "log.h"

#include <Windows.h>
#include <Shlwapi.h>
#include <ShlObj.h>

#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <share.h>

#pragma comment(lib, "Shlwapi.lib")
#pragma comment(lib, "Shell32.lib")

namespace oss_gaussian {

namespace {

std::mutex g_log_mu;
FILE*      g_log_fp     = nullptr;
LogLevel   g_min_level  = LogLevel::Info;

const char* LevelTag(LogLevel l) {
    switch (l) {
        case LogLevel::Trace: return "TRACE";
        case LogLevel::Info:  return "INFO";
        case LogLevel::Warn:  return "WARN";
        case LogLevel::Error: return "ERROR";
    }
    return "?";
}

bool TryBuildLogPathFromBase(const wchar_t* base_dir, wchar_t* out_path, size_t out_cch) {
    if (!base_dir || !base_dir[0]) return false;

    wchar_t dir[MAX_PATH] = {};
    if (swprintf_s(dir, L"%s\\oss-gaussian", base_dir) < 0) return false;
    if (!CreateDirectoryW(dir, nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) {
        return false;
    }
    return swprintf_s(out_path, out_cch, L"%s\\interception.log", dir) >= 0;
}

/// Resolve a per-user log path and ensure the directory exists.
bool ResolveLogPath(wchar_t* out_path, size_t out_cch) {
    wchar_t explicit_dir[MAX_PATH] = {};
    if (GetEnvironmentVariableW(L"OSS_GAUSSIAN_LOG_DIR", explicit_dir, MAX_PATH) > 0 &&
        TryBuildLogPathFromBase(explicit_dir, out_path, out_cch)) {
        return true;
    }

    wchar_t local_appdata[MAX_PATH] = {};
    if (GetEnvironmentVariableW(L"LOCALAPPDATA", local_appdata, MAX_PATH) > 0 &&
        TryBuildLogPathFromBase(local_appdata, out_path, out_cch)) {
        return true;
    }

    PWSTR known_local_appdata = nullptr;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &known_local_appdata))) {
        bool ok = TryBuildLogPathFromBase(known_local_appdata, out_path, out_cch);
        CoTaskMemFree(known_local_appdata);
        if (ok) return true;
    }

    wchar_t temp_dir[MAX_PATH] = {};
    if (GetTempPathW(MAX_PATH, temp_dir) > 0 &&
        TryBuildLogPathFromBase(temp_dir, out_path, out_cch)) {
        return true;
    }

    return false;
}

} // namespace

bool LogInit() {
    std::lock_guard<std::mutex> lk(g_log_mu);
    if (g_log_fp) return true;

    wchar_t path[MAX_PATH] = {};
    if (!ResolveLogPath(path, MAX_PATH)) {
        OutputDebugStringA("[oss-gaussian] LogInit: ResolveLogPath failed\n");
        return false;
    }

    g_log_fp = _wfsopen(path, L"a", _SH_DENYNO);
    if (!g_log_fp) {
        OutputDebugStringA("[oss-gaussian] LogInit: fopen failed\n");
        g_log_fp = nullptr;
        return false;
    }

    // Header line on each session start.
    SYSTEMTIME st{};
    GetLocalTime(&st);
    fprintf(g_log_fp,
            "\n=== oss-gaussian session %04u-%02u-%02u %02u:%02u:%02u ===\n",
            st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    fflush(g_log_fp);
    return true;
}

void LogShutdown() {
    std::lock_guard<std::mutex> lk(g_log_mu);
    if (g_log_fp) {
        fflush(g_log_fp);
        fclose(g_log_fp);
        g_log_fp = nullptr;
    }
}

void LogSetLevel(LogLevel level) {
    std::lock_guard<std::mutex> lk(g_log_mu);
    g_min_level = level;
}

void LogWrite(LogLevel level, const char* tag, const std::string& msg) {
    if (static_cast<int>(level) < static_cast<int>(g_min_level)) return;

    SYSTEMTIME st{};
    GetLocalTime(&st);

    char line[2048];
    int n = _snprintf_s(line, _countof(line), _TRUNCATE,
        "[%02u:%02u:%02u.%03u][%-5s][%s] %s\n",
        st.wHour, st.wMinute, st.wSecond, st.wMilliseconds,
        LevelTag(level), tag ? tag : "-", msg.c_str());
    if (n < 0) return;

    OutputDebugStringA(line);

    std::lock_guard<std::mutex> lk(g_log_mu);
    if (g_log_fp) {
        fputs(line, g_log_fp);
        fflush(g_log_fp); // line-buffered; cheap and safe vs game crashes.
    }
}

void LogFmt(LogLevel level, const char* tag, const char* fmt, ...) {
    if (static_cast<int>(level) < static_cast<int>(g_min_level)) return;

    char buf[1536];
    va_list ap;
    va_start(ap, fmt);
    int n = _vsnprintf_s(buf, _countof(buf), _TRUNCATE, fmt, ap);
    va_end(ap);
    if (n < 0) {
        // Truncated. Buf is still null-terminated by _vsnprintf_s.
    }
    LogWrite(level, tag, std::string(buf));
}

} // namespace oss_gaussian
