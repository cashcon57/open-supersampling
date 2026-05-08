// =============================================================================
//  dll_config.h
//
//  DLL-side reader for the per-game config.json produced by
//  scripts/build_capture_installer.py and dropped at one of:
//      %LOCALAPPDATA%\oss-capture\<game_id>\config.json
//      %PROGRAMDATA%\oss-capture\<game_id>\config.json
//      <game-folder>\oss-capture-config.json   (game-relative override)
//
//  Reads only the subset of the schema the DLL actually needs at runtime.
//  Falls back to safe defaults if the file is missing, malformed, or has
//  unrecognized keys (forward-compat with newer installer schemas).
//
//  The reader does NOT take a runtime dependency on nlohmann/json — it
//  parses a hand-rolled subset of JSON (top-level object, string/int/bool
//  values, single-level nested object). That keeps the DLL footprint
//  small and the build dep graph lean. The full schema validation happens
//  on the Python side at install time.
//
//  Status: SCAFFOLDED. Compiles on Windows. Runtime tested only via the
//  unit test in tests/dll_config/ once it lands.
//
//  Copyright 2026 OSS-Gaussian contributors. Apache 2.0.
// =============================================================================
#ifndef OSS_GAUSSIAN_DLL_CONFIG_H
#define OSS_GAUSSIAN_DLL_CONFIG_H

#include <string>
#include <stdint.h>

namespace oss_gaussian {

struct DllConfig {
    // Identity
    std::string game_id;            // e.g. "cyberpunk2077"
    std::string capture_mode;       // "trickle" | "lite" | "regular" | "INSANE"
    std::string install_token;
    std::string capture_api_base;   // https://capture-ingest.opensupersampling.com
    int         schema_version = 0;

    // Runtime knobs
    double      suggested_capture_rate_per_min = 3.0;
    uint64_t    pending_dir_cap_bytes          = 2ULL * 1024 * 1024 * 1024;
    uint64_t    max_frame_bytes                = 16ULL * 1024 * 1024;
    int         uploader_retry_attempts        = 5;
    int         uploader_retry_max_seconds     = 30 * 60;

    // Source of the loaded config (for logging). Empty if defaults.
    std::string loaded_from;

    // True if config.json was successfully parsed (any key recognized).
    bool        is_loaded = false;
};

// Try in order:
//   1. %LOCALAPPDATA%\oss-capture\<best-guess-game-id>\config.json
//   2. %PROGRAMDATA%\oss-capture\<best-guess-game-id>\config.json
//   3. <game-folder>\oss-capture-config.json
// `game_id_hint` is used to disambiguate — typically the basename of the
// host process EXE without extension, lowercased. Pass empty string to skip
// the per-game lookup and only try the game-folder override.
//
// Always returns a populated DllConfig; check `is_loaded` to know whether
// values came from disk or are the safe defaults.
DllConfig LoadDllConfig(const std::string& game_id_hint);

// Locate the config file at one of the well-known paths above. Returns
// empty string if no config found. Used by LoadDllConfig + tests.
std::string FindConfigPath(const std::string& game_id_hint);

// Parse a JSON document into a DllConfig. Public for unit tests; production
// callers should use LoadDllConfig which handles file I/O.
DllConfig ParseDllConfig(const std::string& json_text, const std::string& source_path = "");

}  // namespace oss_gaussian

#endif  // OSS_GAUSSIAN_DLL_CONFIG_H
