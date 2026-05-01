// =============================================================================
//  log.h
//
//  Tiny line-buffered file logger writing to
//  %LOCALAPPDATA%\oss-gaussian\interception.log.
//
//  Modeled on OptiScaler's `Logger.h` (https://github.com/optiscaler/OptiScaler)
//  but cut down to the few primitives Sprint 2 actually needs. No spdlog
//  dependency — keeps the DLL footprint small and the link line short.
//
//  Copyright 2026 OSS-Gaussian contributors
//  Licensed under the Apache License, Version 2.0 (see ../LICENSE).
// =============================================================================
#ifndef OSS_GAUSSIAN_LOG_H
#define OSS_GAUSSIAN_LOG_H

#include <string>

namespace oss_gaussian {

enum class LogLevel : int {
    Trace = 0,
    Info  = 1,
    Warn  = 2,
    Error = 3,
};

/// Open the log file under %LOCALAPPDATA%\oss-gaussian. Idempotent.
/// Returns true if the file is writable. On failure, logging silently drops
/// (we never want logging to crash the game).
bool LogInit();

/// Flush + close. Called from DllMain DETACH.
void LogShutdown();

/// Set runtime minimum level. Default is Info.
void LogSetLevel(LogLevel level);

/// Write one line. Always also calls OutputDebugStringA so that a debugger
/// attached to Cyberpunk2077.exe sees the same stream.
void LogWrite(LogLevel level, const char* tag, const std::string& msg);

/// Convenience: printf-style formatting.
void LogFmt(LogLevel level, const char* tag, const char* fmt, ...);

} // namespace oss_gaussian

// Compact macros — keep call-sites short, no streams.
#define OSSG_LOG_TRACE(tag, ...) ::oss_gaussian::LogFmt(::oss_gaussian::LogLevel::Trace, (tag), __VA_ARGS__)
#define OSSG_LOG_INFO(tag,  ...) ::oss_gaussian::LogFmt(::oss_gaussian::LogLevel::Info,  (tag), __VA_ARGS__)
#define OSSG_LOG_WARN(tag,  ...) ::oss_gaussian::LogFmt(::oss_gaussian::LogLevel::Warn,  (tag), __VA_ARGS__)
#define OSSG_LOG_ERROR(tag, ...) ::oss_gaussian::LogFmt(::oss_gaussian::LogLevel::Error, (tag), __VA_ARGS__)

#endif // OSS_GAUSSIAN_LOG_H
