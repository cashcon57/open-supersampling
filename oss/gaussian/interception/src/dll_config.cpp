// =============================================================================
//  dll_config.cpp
//
//  Hand-rolled JSON-subset parser for the OSS-capture config.json schema.
//  See dll_config.h for design notes.
//
//  Supported JSON subset (matches what scripts/build_capture_installer.py
//  produces):
//    - Top-level object only ({...})
//    - Keys: double-quoted string
//    - Values: double-quoted string, integer, fractional number, true/false,
//              single-level nested object (e.g. "endpoints":{...})
//    - Whitespace, commas, basic escape sequences (\", \\, \n, \t)
//
//  Unsupported (intentionally):
//    - Arrays (no need for current schema)
//    - Multi-level nested objects beyond depth 2
//    - Unicode escapes (\uXXXX)
//
//  Failure mode: any parse error → return DllConfig with is_loaded=false
//  (defaults) and log a warning. The DLL never refuses to load a game just
//  because config.json is malformed.
//
//  Copyright 2026 OSS-Gaussian contributors. Apache 2.0.
// =============================================================================
#define OSS_GAUSSIAN_BUILDING_DLL 1

#include "dll_config.h"
#include "log.h"

#include <Windows.h>
#include <ShlObj.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace oss_gaussian {

namespace {

// ----------------------------------------------------------------------
//  Path helpers.
// ----------------------------------------------------------------------

std::string WideToUtf8(const std::wstring& w) {
    if (w.empty()) return {};
    int needed = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
                                     nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(needed), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
                        out.data(), needed, nullptr, nullptr);
    return out;
}

std::string ShellPath(REFKNOWNFOLDERID folder_id) {
    PWSTR buf = nullptr;
    HRESULT hr = SHGetKnownFolderPath(folder_id, 0, nullptr, &buf);
    std::string out;
    if (SUCCEEDED(hr) && buf) out = WideToUtf8(buf);
    if (buf) CoTaskMemFree(buf);
    return out;
}

bool FileExists(const std::string& path) {
    DWORD attrs = GetFileAttributesA(path.c_str());
    return (attrs != INVALID_FILE_ATTRIBUTES) && !(attrs & FILE_ATTRIBUTE_DIRECTORY);
}

std::string ReadAllText(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// ----------------------------------------------------------------------
//  Lexer + parser. Walks a string offset; never throws.
// ----------------------------------------------------------------------

struct Cursor {
    const char* p;
    const char* end;

    bool  has(size_t n = 1) const { return p + n <= end; }
    char  peek(size_t off = 0) const { return has(off + 1) ? p[off] : '\0'; }
    char  advance() { return has() ? *p++ : '\0'; }
    void  skip_ws() {
        while (has() && (std::isspace(static_cast<unsigned char>(*p)) || *p == ',')) ++p;
    }
};

bool ParseString(Cursor& c, std::string& out) {
    c.skip_ws();
    if (c.advance() != '"') return false;
    out.clear();
    while (c.has()) {
        char ch = c.advance();
        if (ch == '"') return true;
        if (ch == '\\' && c.has()) {
            char esc = c.advance();
            switch (esc) {
                case '"':  out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/'); break;
                case 'n':  out.push_back('\n'); break;
                case 't':  out.push_back('\t'); break;
                case 'r':  out.push_back('\r'); break;
                default:   out.push_back(esc); break;  // permissive
            }
        } else {
            out.push_back(ch);
        }
    }
    return false;
}

bool ParseNumber(Cursor& c, double& out_dbl, int64_t& out_int, bool& is_float) {
    c.skip_ws();
    const char* start = c.p;
    is_float = false;
    if (c.peek() == '-' || c.peek() == '+') c.advance();
    while (c.has() && (std::isdigit(static_cast<unsigned char>(c.peek())))) c.advance();
    if (c.peek() == '.') {
        is_float = true;
        c.advance();
        while (c.has() && std::isdigit(static_cast<unsigned char>(c.peek()))) c.advance();
    }
    if (c.peek() == 'e' || c.peek() == 'E') {
        is_float = true;
        c.advance();
        if (c.peek() == '-' || c.peek() == '+') c.advance();
        while (c.has() && std::isdigit(static_cast<unsigned char>(c.peek()))) c.advance();
    }
    if (c.p == start) return false;
    std::string num(start, c.p);
    if (is_float) {
        out_dbl = std::strtod(num.c_str(), nullptr);
        out_int = static_cast<int64_t>(out_dbl);
    } else {
        out_int = static_cast<int64_t>(std::strtoll(num.c_str(), nullptr, 10));
        out_dbl = static_cast<double>(out_int);
    }
    return true;
}

bool ParseBool(Cursor& c, bool& out) {
    c.skip_ws();
    if (c.has(4) && std::strncmp(c.p, "true", 4) == 0) { c.p += 4; out = true;  return true; }
    if (c.has(5) && std::strncmp(c.p, "false", 5) == 0) { c.p += 5; out = false; return true; }
    return false;
}

// Skip a JSON value of any supported subset. Used to walk past nested objects
// we don't care about deeply (e.g. consent.standard_disclosure body text).
bool SkipValue(Cursor& c) {
    c.skip_ws();
    char ch = c.peek();
    if (ch == '"') {
        std::string s;
        return ParseString(c, s);
    }
    if (ch == '{' || ch == '[') {
        char open = c.advance();
        char close = (open == '{') ? '}' : ']';
        int depth = 1;
        while (c.has() && depth > 0) {
            char x = c.advance();
            if (x == '"') {
                std::string s;
                c.p--;
                if (!ParseString(c, s)) return false;
            } else if (x == open) {
                ++depth;
            } else if (x == close) {
                --depth;
            }
        }
        return depth == 0;
    }
    if (ch == 't' || ch == 'f') {
        bool b;
        return ParseBool(c, b);
    }
    if (ch == 'n' && c.has(4) && std::strncmp(c.p, "null", 4) == 0) {
        c.p += 4;
        return true;
    }
    if (ch == '-' || ch == '+' || std::isdigit(static_cast<unsigned char>(ch))) {
        double dv; int64_t iv; bool isf;
        return ParseNumber(c, dv, iv, isf);
    }
    return false;
}

bool AssignField(DllConfig& cfg, const std::string& key, Cursor& c) {
    c.skip_ws();
    char ch = c.peek();

    if (ch == '"') {
        std::string s;
        if (!ParseString(c, s)) return false;
        if      (key == "game_id")          cfg.game_id          = s;
        else if (key == "capture_mode")     cfg.capture_mode     = s;
        else if (key == "install_token")    cfg.install_token    = s;
        else if (key == "capture_api_base") cfg.capture_api_base = s;
        return true;
    }
    if (ch == 't' || ch == 'f') {
        bool b;
        if (!ParseBool(c, b)) return false;
        // No bool fields in our subset yet, but accept gracefully.
        return true;
    }
    if (ch == '-' || ch == '+' || std::isdigit(static_cast<unsigned char>(ch))) {
        double dv; int64_t iv; bool isf;
        if (!ParseNumber(c, dv, iv, isf)) return false;
        if      (key == "schema_version")                 cfg.schema_version                 = static_cast<int>(iv);
        else if (key == "suggested_capture_rate_per_min") cfg.suggested_capture_rate_per_min = dv;
        else if (key == "pending_dir_cap_bytes")          cfg.pending_dir_cap_bytes          = static_cast<uint64_t>(iv);
        else if (key == "max_frame_bytes")                cfg.max_frame_bytes                = static_cast<uint64_t>(iv);
        else if (key == "uploader_retry_attempts")        cfg.uploader_retry_attempts        = static_cast<int>(iv);
        else if (key == "uploader_retry_max_seconds")     cfg.uploader_retry_max_seconds     = static_cast<int>(iv);
        return true;
    }
    if (ch == '{' || ch == '[') {
        // Skip nested objects/arrays — we don't read endpoints/consent at the
        // DLL layer (the uploader handles those).
        return SkipValue(c);
    }
    if (ch == 'n' && c.has(4) && std::strncmp(c.p, "null", 4) == 0) {
        c.p += 4;
        return true;
    }
    return false;
}

}  // namespace

// ----------------------------------------------------------------------
//  Public surface.
// ----------------------------------------------------------------------

DllConfig ParseDllConfig(const std::string& json_text, const std::string& source_path) {
    DllConfig cfg;
    cfg.loaded_from = source_path;

    Cursor c{json_text.data(), json_text.data() + json_text.size()};
    c.skip_ws();
    if (c.advance() != '{') {
        OSSG_LOG_ERROR("config", "ParseDllConfig: top-level JSON must be an object (source=%s)",
                       source_path.c_str());
        return cfg;
    }

    int recognized = 0;
    while (true) {
        c.skip_ws();
        if (c.peek() == '}') { c.advance(); break; }
        if (!c.has()) break;

        std::string key;
        if (!ParseString(c, key)) {
            OSSG_LOG_ERROR("config", "ParseDllConfig: expected string key (source=%s)",
                           source_path.c_str());
            return cfg;
        }
        c.skip_ws();
        if (c.advance() != ':') {
            OSSG_LOG_ERROR("config", "ParseDllConfig: expected ':' after key %s",
                           key.c_str());
            return cfg;
        }
        if (!AssignField(cfg, key, c)) {
            OSSG_LOG_WARN("config", "ParseDllConfig: failed to read value for key %s",
                          key.c_str());
            // Continue trying remaining keys; one bad value should not nuke the load.
        } else {
            ++recognized;
        }
    }

    cfg.is_loaded = (recognized > 0);
    OSSG_LOG_INFO("config",
                  "ParseDllConfig: %d keys recognized; mode=%s, max_frame=%llu",
                  recognized,
                  cfg.capture_mode.empty() ? "(default)" : cfg.capture_mode.c_str(),
                  static_cast<unsigned long long>(cfg.max_frame_bytes));
    return cfg;
}

std::string FindConfigPath(const std::string& game_id_hint) {
    auto try_paths = [&](const std::string& base) -> std::string {
        if (base.empty()) return {};
        if (!game_id_hint.empty()) {
            std::string p = base + "\\oss-capture\\" + game_id_hint + "\\config.json";
            if (FileExists(p)) return p;
        }
        // Generic fallback under the same root (no per-game subfolder).
        std::string p2 = base + "\\oss-capture\\config.json";
        if (FileExists(p2)) return p2;
        return {};
    };

    std::string path = try_paths(ShellPath(FOLDERID_LocalAppData));
    if (!path.empty()) return path;
    path = try_paths(ShellPath(FOLDERID_ProgramData));
    if (!path.empty()) return path;

    // Game-folder override: same dir as the host EXE.
    char exe_path[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        std::string ep(exe_path);
        size_t slash = ep.find_last_of("\\/");
        if (slash != std::string::npos) {
            std::string p = ep.substr(0, slash) + "\\oss-capture-config.json";
            if (FileExists(p)) return p;
        }
    }
    return {};
}

DllConfig LoadDllConfig(const std::string& game_id_hint) {
    std::string path = FindConfigPath(game_id_hint);
    if (path.empty()) {
        OSSG_LOG_INFO("config", "no config.json found; using defaults (game_id_hint=%s)",
                      game_id_hint.c_str());
        return DllConfig{};
    }
    std::string text = ReadAllText(path);
    if (text.empty()) {
        OSSG_LOG_WARN("config", "config file empty/unreadable: %s", path.c_str());
        return DllConfig{};
    }
    DllConfig cfg = ParseDllConfig(text, path);
    return cfg;
}

}  // namespace oss_gaussian
