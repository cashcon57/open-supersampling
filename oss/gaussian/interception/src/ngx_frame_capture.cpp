// =============================================================================
//  ngx_frame_capture.cpp
//
//  Records GPU->CPU copies for DLSS SR resources on the game's open NGX command
//  list, then writes a same-canvas multi-channel EXR after the matching
//  ExecuteCommandLists call signals a fence.
// =============================================================================
#define OSS_GAUSSIAN_BUILDING_DLL 1

#include "ngx_frame_capture.h"

#include "log.h"
#include "../oss_capture.h"

#include <Windows.h>
#include <d3d12.h>
#include <dxgiformat.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace oss_gaussian {
namespace {

enum class CaptureRole {
    LrColor,
    HrOutput,
    Depth,
    Motion,
};

struct TextureReadback {
    CaptureRole role = CaptureRole::LrColor;
    ComPtr<ID3D12Resource> source;
    ComPtr<ID3D12Resource> readback;
    bool source_state_known = false;
    D3D12_RESOURCE_STATES source_state = D3D12_RESOURCE_STATE_COMMON;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT num_rows = 0;
    UINT64 row_size_bytes = 0;
    UINT64 total_bytes = 0;
    D3D12_RESOURCE_DESC desc{};
};

struct PendingBatch {
    ID3D12CommandList* command_list = nullptr;
    UINT64 fence_value = 0;
    std::vector<TextureReadback> textures;
    std::wstring exr_path;
    std::string json_path;
    std::string game_id;
    std::string game_version;
    std::string frame_uuid;
    std::string session_uuid;
    std::string consent_token;
    std::string capture_storage_mode;
    std::string provider;
    OssCaptureDecision decision{};
    OssGaussianFrame frame{};
    bool drop_after_fence = false;
    double captured_at_unix = 0.0;
    double captured_at_monotonic_seconds = 0.0;
    float motion_mean_px = 0.0f;
    uint64_t phash = 0;
};

ComPtr<ID3D12Device> g_device;
ComPtr<ID3D12Fence> g_fence;
HANDLE g_fence_event = nullptr;
std::atomic<UINT64> g_next_fence_value{1};

std::mutex g_pending_mu;
std::vector<std::shared_ptr<PendingBatch>> g_pending;
std::vector<std::shared_ptr<PendingBatch>> g_unfenced_quarantine;

std::mutex g_worker_mu;
std::condition_variable g_worker_cv;
std::deque<std::shared_ptr<PendingBatch>> g_worker_q;
std::atomic<bool> g_worker_started{false};
std::thread g_worker;

std::once_flag g_session_once;
std::string g_session_uuid;
std::chrono::steady_clock::time_point g_start_time = std::chrono::steady_clock::now();
std::mutex g_candidate_mu;
double g_last_candidate_time = 0.0;

float HalfToFloat(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

double UnixNow() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}

double SecondsSinceStart() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - g_start_time).count();
}

std::string GuidString() {
    GUID guid{};
    if (FAILED(CoCreateGuid(&guid))) {
        const uint64_t ticks = static_cast<uint64_t>(UnixNow() * 1000000.0);
        char fallback[37]{};
        std::snprintf(
            fallback,
            sizeof(fallback),
            "00000000-0000-4000-8000-%012llx",
            static_cast<unsigned long long>(ticks & 0xffffffffffffull));
        return fallback;
    }
    char buf[37]{};
    std::snprintf(
        buf,
        sizeof(buf),
        "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        static_cast<unsigned long>(guid.Data1),
        guid.Data2,
        guid.Data3,
        guid.Data4[0],
        guid.Data4[1],
        guid.Data4[2],
        guid.Data4[3],
        guid.Data4[4],
        guid.Data4[5],
        guid.Data4[6],
        guid.Data4[7]);
    return buf;
}

const std::string& SessionUuid() {
    std::call_once(g_session_once, [] { g_session_uuid = GuidString(); });
    return g_session_uuid;
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

std::wstring LocalAppDataPendingRoot() {
    std::wstring configured = capture::CurrentPendingRoot();
    if (!configured.empty()) {
        return configured;
    }
    wchar_t buf[MAX_PATH]{};
    DWORD len = GetEnvironmentVariableW(L"LOCALAPPDATA", buf, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        return std::wstring(buf) + L"\\oss-capture\\pending";
    }
    return L".\\oss-capture\\pending";
}

std::wstring BuildExrPath(
    const std::string& game_id,
    const std::string& session_uuid,
    const std::string& frame_uuid) {
    return LocalAppDataPendingRoot() + L"\\" + Utf8ToWide(game_id) + L"\\" +
           Utf8ToWide(session_uuid) + L"\\" + Utf8ToWide(frame_uuid) + L".exr";
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

const char* ModeName(OssCaptureMode mode) {
    switch (mode) {
        case OSS_CAPTURE_MODE_TRICKLE: return "trickle";
        case OSS_CAPTURE_MODE_LITE: return "lite";
        case OSS_CAPTURE_MODE_REGULAR: return "regular";
        case OSS_CAPTURE_MODE_INSANE: return "INSANE";
    }
    return "lite";
}

const char* RoleName(CaptureRole role) {
    switch (role) {
        case CaptureRole::LrColor: return "lr_color";
        case CaptureRole::HrOutput: return "hr_output";
        case CaptureRole::Depth: return "depth";
        case CaptureRole::Motion: return "motion_vectors";
    }
    return "unknown";
}

const char* DxgiFormatName(DXGI_FORMAT format) {
    switch (format) {
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return "R16G16B16A16_FLOAT";
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return "R32G32B32A32_FLOAT";
        case DXGI_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return "R8G8B8A8_UNORM_SRGB";
        case DXGI_FORMAT_B8G8R8A8_UNORM: return "B8G8R8A8_UNORM";
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return "B8G8R8A8_UNORM_SRGB";
        case DXGI_FORMAT_R10G10B10A2_UNORM: return "R10G10B10A2_UNORM";
        case DXGI_FORMAT_D32_FLOAT: return "D32_FLOAT";
        case DXGI_FORMAT_R32_FLOAT: return "R32_FLOAT";
        case DXGI_FORMAT_R32_TYPELESS: return "R32_TYPELESS";
        case DXGI_FORMAT_R16_FLOAT: return "R16_FLOAT";
        case DXGI_FORMAT_R16G16_FLOAT: return "R16G16_FLOAT";
        case DXGI_FORMAT_R32G32_FLOAT: return "R32G32_FLOAT";
        default: return "UNKNOWN";
    }
}

void WriteTextureMeta(std::ofstream& meta, const TextureReadback* texture, const char* indent) {
    if (!texture) {
        meta << "null";
        return;
    }
    meta << "{\n";
    meta << indent << "  \"role\": \"" << RoleName(texture->role) << "\",\n";
    meta << indent << "  \"width\": " << texture->desc.Width << ",\n";
    meta << indent << "  \"height\": " << texture->desc.Height << ",\n";
    meta << indent << "  \"dxgi_format\": " << static_cast<int>(texture->desc.Format) << ",\n";
    meta << indent << "  \"dxgi_format_name\": \"" << DxgiFormatName(texture->desc.Format) << "\",\n";
    meta << indent << "  \"source_state_known\": "
         << (texture->source_state_known ? "true" : "false") << ",\n";
    meta << indent << "  \"source_state\": " << static_cast<uint32_t>(texture->source_state) << ",\n";
    meta << indent << "  \"row_pitch_bytes\": " << texture->footprint.Footprint.RowPitch << ",\n";
    meta << indent << "  \"total_bytes\": " << texture->total_bytes << "\n";
    meta << indent << "}";
}

bool EnsureDevice(ID3D12Resource* resource) {
    if (g_device) return true;
    if (!resource) return false;
    ComPtr<ID3D12Device> device;
    if (FAILED(resource->GetDevice(IID_PPV_ARGS(&device)))) return false;
    g_device = device;
    if (FAILED(g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence)))) {
        g_device.Reset();
        return false;
    }
    g_fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    return g_fence_event != nullptr;
}

bool SupportedResource(const D3D12_RESOURCE_DESC& desc) {
    return desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
           desc.DepthOrArraySize == 1 &&
           desc.MipLevels >= 1 &&
           desc.SampleDesc.Count == 1 &&
           desc.Width > 0 &&
           desc.Height > 0;
}

bool CreateReadback(TextureReadback& texture) {
    if (!g_device || !texture.source) return false;
    texture.desc = texture.source->GetDesc();
    if (!SupportedResource(texture.desc)) return false;
    g_device->GetCopyableFootprints(
        &texture.desc,
        0,
        1,
        0,
        &texture.footprint,
        &texture.num_rows,
        &texture.row_size_bytes,
        &texture.total_bytes);
    if (texture.total_bytes == 0) return false;

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = texture.total_bytes;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return SUCCEEDED(g_device->CreateCommittedResource(
        &heap,
        D3D12_HEAP_FLAG_NONE,
        &buffer,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&texture.readback)));
}

D3D12_RESOURCE_STATES AssumedState(CaptureRole role) {
    if (role == CaptureRole::HrOutput) return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    return static_cast<D3D12_RESOURCE_STATES>(
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
}

uint32_t StateBit(CaptureRole role) {
    switch (role) {
        case CaptureRole::LrColor: return 1u << 0u;
        case CaptureRole::HrOutput: return 1u << 1u;
        case CaptureRole::Depth: return 1u << 2u;
        case CaptureRole::Motion: return 1u << 3u;
    }
    return 0u;
}

D3D12_RESOURCE_STATES FrameState(const OssGaussianFrame& frame, CaptureRole role) {
    switch (role) {
        case CaptureRole::LrColor:
            return static_cast<D3D12_RESOURCE_STATES>(frame.color_state);
        case CaptureRole::HrOutput:
            return static_cast<D3D12_RESOURCE_STATES>(frame.output_state);
        case CaptureRole::Depth:
            return static_cast<D3D12_RESOURCE_STATES>(frame.depth_state);
        case CaptureRole::Motion:
            return static_cast<D3D12_RESOURCE_STATES>(frame.motion_vectors_state);
    }
    return D3D12_RESOURCE_STATE_COMMON;
}

TextureReadback MakeReadback(CaptureRole role, ID3D12Resource* source, const OssGaussianFrame& frame) {
    TextureReadback texture{};
    texture.role = role;
    texture.source = source;
    texture.source_state_known = (frame.resource_states_valid & StateBit(role)) != 0u;
    texture.source_state = FrameState(frame, role);
    return texture;
}

void RecordReadbackCopy(ID3D12GraphicsCommandList* command_list, TextureReadback& texture) {
    const D3D12_RESOURCE_STATES state_before =
        texture.source_state_known ? texture.source_state : AssumedState(texture.role);
    D3D12_RESOURCE_BARRIER before{};
    before.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    before.Transition.pResource = texture.source.Get();
    before.Transition.Subresource = 0;
    before.Transition.StateBefore = state_before;
    before.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;

    D3D12_RESOURCE_BARRIER after = before;
    after.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    after.Transition.StateAfter = state_before;

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = texture.readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = texture.footprint;

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = texture.source.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;

    command_list->ResourceBarrier(1, &before);
    command_list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    command_list->ResourceBarrier(1, &after);
}

const TextureReadback* FindTexture(const PendingBatch& batch, CaptureRole role) {
    for (const TextureReadback& texture : batch.textures) {
        if (texture.role == role) return &texture;
    }
    return nullptr;
}

const uint8_t* PixelPtr(const TextureReadback& texture, const void* mapped, uint32_t x, uint32_t y) {
    const auto* base = static_cast<const uint8_t*>(mapped);
    return base + texture.footprint.Offset +
           static_cast<size_t>(y) * texture.footprint.Footprint.RowPitch +
           static_cast<size_t>(x) * texture.row_size_bytes / std::max<UINT>(texture.footprint.Footprint.Width, 1);
}

bool DecodeRgb(const TextureReadback& texture, const void* mapped, uint32_t x, uint32_t y, float* rgb) {
    const uint8_t* p = PixelPtr(texture, mapped, x, y);
    switch (texture.desc.Format) {
        case DXGI_FORMAT_R16G16B16A16_FLOAT: {
            const auto* v = reinterpret_cast<const uint16_t*>(p);
            rgb[0] = HalfToFloat(v[0]);
            rgb[1] = HalfToFloat(v[1]);
            rgb[2] = HalfToFloat(v[2]);
            return true;
        }
        case DXGI_FORMAT_R32G32B32A32_FLOAT: {
            const auto* v = reinterpret_cast<const float*>(p);
            rgb[0] = v[0];
            rgb[1] = v[1];
            rgb[2] = v[2];
            return true;
        }
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: {
            rgb[0] = p[0] / 255.0f;
            rgb[1] = p[1] / 255.0f;
            rgb[2] = p[2] / 255.0f;
            return true;
        }
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: {
            rgb[0] = p[2] / 255.0f;
            rgb[1] = p[1] / 255.0f;
            rgb[2] = p[0] / 255.0f;
            return true;
        }
        case DXGI_FORMAT_R10G10B10A2_UNORM: {
            uint32_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            rgb[0] = static_cast<float>(v & 0x3ffu) / 1023.0f;
            rgb[1] = static_cast<float>((v >> 10) & 0x3ffu) / 1023.0f;
            rgb[2] = static_cast<float>((v >> 20) & 0x3ffu) / 1023.0f;
            return true;
        }
        default:
            return false;
    }
}

bool DecodeDepth(const TextureReadback& texture, const void* mapped, uint32_t x, uint32_t y, float* depth) {
    const uint8_t* p = PixelPtr(texture, mapped, x, y);
    switch (texture.desc.Format) {
        case DXGI_FORMAT_D32_FLOAT:
        case DXGI_FORMAT_R32_FLOAT:
        case DXGI_FORMAT_R32_TYPELESS:
            std::memcpy(depth, p, sizeof(float));
            return std::isfinite(*depth);
        case DXGI_FORMAT_R16_FLOAT: {
            uint16_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            *depth = HalfToFloat(v);
            return std::isfinite(*depth);
        }
        default:
            return false;
    }
}

bool DecodeMotion(const TextureReadback& texture, const void* mapped, uint32_t x, uint32_t y, float* motion) {
    const uint8_t* p = PixelPtr(texture, mapped, x, y);
    switch (texture.desc.Format) {
        case DXGI_FORMAT_R16G16_FLOAT: {
            const auto* v = reinterpret_cast<const uint16_t*>(p);
            motion[0] = HalfToFloat(v[0]);
            motion[1] = HalfToFloat(v[1]);
            return std::isfinite(motion[0]) && std::isfinite(motion[1]);
        }
        case DXGI_FORMAT_R16G16B16A16_FLOAT: {
            const auto* v = reinterpret_cast<const uint16_t*>(p);
            motion[0] = HalfToFloat(v[0]);
            motion[1] = HalfToFloat(v[1]);
            return std::isfinite(motion[0]) && std::isfinite(motion[1]);
        }
        case DXGI_FORMAT_R32G32_FLOAT: {
            const auto* v = reinterpret_cast<const float*>(p);
            motion[0] = v[0];
            motion[1] = v[1];
            return std::isfinite(motion[0]) && std::isfinite(motion[1]);
        }
        case DXGI_FORMAT_R32G32B32A32_FLOAT: {
            const auto* v = reinterpret_cast<const float*>(p);
            motion[0] = v[0];
            motion[1] = v[1];
            return std::isfinite(motion[0]) && std::isfinite(motion[1]);
        }
        default:
            return false;
    }
}

uint32_t ScaleCoord(uint32_t value, uint32_t from_extent, uint32_t to_extent) {
    if (to_extent <= 1 || from_extent <= 1) return 0;
    return std::min<uint32_t>(
        from_extent - 1,
        static_cast<uint32_t>((static_cast<uint64_t>(value) * from_extent) / to_extent));
}

uint32_t ClampExtent(uint32_t requested, uint32_t fallback, uint32_t maximum) {
    const uint32_t value = requested ? requested : fallback;
    return std::max<uint32_t>(1u, std::min(value, maximum));
}

uint32_t ClampBase(uint32_t requested, uint32_t maximum) {
    return std::min(requested, maximum > 0 ? maximum - 1u : 0u);
}

bool MapTexture(const TextureReadback& texture, void** mapped) {
    D3D12_RANGE range{0, static_cast<SIZE_T>(texture.total_bytes)};
    return SUCCEEDED(texture.readback->Map(0, &range, mapped)) && *mapped;
}

void UnmapTexture(const TextureReadback& texture) {
    D3D12_RANGE no_write{0, 0};
    texture.readback->Unmap(0, &no_write);
}

bool BuildPayloadAndWrite(const PendingBatch& batch) {
    const TextureReadback* lr = FindTexture(batch, CaptureRole::LrColor);
    const TextureReadback* hr = FindTexture(batch, CaptureRole::HrOutput);
    const TextureReadback* depth = FindTexture(batch, CaptureRole::Depth);
    const TextureReadback* motion = FindTexture(batch, CaptureRole::Motion);
    if (!lr || !hr || !depth || !motion) return false;

    void* lr_map = nullptr;
    void* hr_map = nullptr;
    void* depth_map = nullptr;
    void* motion_map = nullptr;
    if (!MapTexture(*lr, &lr_map) ||
        !MapTexture(*hr, &hr_map) ||
        !MapTexture(*depth, &depth_map) ||
        !MapTexture(*motion, &motion_map)) {
        if (lr_map) UnmapTexture(*lr);
        if (hr_map) UnmapTexture(*hr);
        if (depth_map) UnmapTexture(*depth);
        if (motion_map) UnmapTexture(*motion);
        return false;
    }

    const uint32_t lr_desc_w = static_cast<uint32_t>(lr->desc.Width);
    const uint32_t lr_desc_h = lr->desc.Height;
    const uint32_t hr_desc_w = static_cast<uint32_t>(hr->desc.Width);
    const uint32_t hr_desc_h = hr->desc.Height;
    const uint32_t render_base_x = ClampBase(batch.frame.subrect_base_x, lr_desc_w);
    const uint32_t render_base_y = ClampBase(batch.frame.subrect_base_y, lr_desc_h);
    const uint32_t render_w = ClampExtent(
        batch.frame.subrect_render_width,
        lr_desc_w - render_base_x,
        lr_desc_w - render_base_x);
    const uint32_t render_h = ClampExtent(
        batch.frame.subrect_render_height,
        lr_desc_h - render_base_y,
        lr_desc_h - render_base_y);
    const uint32_t output_w = ClampExtent(batch.frame.output_width, hr_desc_w, hr_desc_w);
    const uint32_t output_h = ClampExtent(batch.frame.output_height, hr_desc_h, hr_desc_h);
    std::vector<float> lr_rgb(static_cast<size_t>(output_w) * output_h * 3u);
    std::vector<float> hr_rgb(static_cast<size_t>(output_w) * output_h * 3u);
    std::vector<float> depth_z(static_cast<size_t>(output_w) * output_h);
    std::vector<float> motion_xy(static_cast<size_t>(output_w) * output_h * 2u);
    bool ok = true;
    double motion_sum = 0.0;
    for (uint32_t y = 0; y < output_h && ok; ++y) {
        for (uint32_t x = 0; x < output_w; ++x) {
            const uint32_t sub_x = ScaleCoord(x, render_w, output_w);
            const uint32_t sub_y = ScaleCoord(y, render_h, output_h);
            const uint32_t lr_x = std::min<uint32_t>(lr_desc_w - 1u, render_base_x + sub_x);
            const uint32_t lr_y = std::min<uint32_t>(lr_desc_h - 1u, render_base_y + sub_y);
            const uint32_t d_x = ScaleCoord(x, static_cast<uint32_t>(depth->desc.Width), output_w);
            const uint32_t d_y = ScaleCoord(y, depth->desc.Height, output_h);
            const uint32_t mv_x = ScaleCoord(x, static_cast<uint32_t>(motion->desc.Width), output_w);
            const uint32_t mv_y = ScaleCoord(y, motion->desc.Height, output_h);
            const size_t i = static_cast<size_t>(y) * output_w + x;

            ok = DecodeRgb(*lr, lr_map, lr_x, lr_y, &lr_rgb[i * 3u]) &&
                 DecodeRgb(*hr, hr_map, x, y, &hr_rgb[i * 3u]) &&
                 DecodeDepth(*depth, depth_map, d_x, d_y, &depth_z[i]) &&
                 DecodeMotion(*motion, motion_map, mv_x, mv_y, &motion_xy[i * 2u]);
            motion_xy[i * 2u + 0u] *= batch.frame.mv_scale_x;
            motion_xy[i * 2u + 1u] *= batch.frame.mv_scale_y;
            motion_sum += std::sqrt(
                motion_xy[i * 2u + 0u] * motion_xy[i * 2u + 0u] +
                motion_xy[i * 2u + 1u] * motion_xy[i * 2u + 1u]);
        }
    }

    UnmapTexture(*lr);
    UnmapTexture(*hr);
    UnmapTexture(*depth);
    UnmapTexture(*motion);
    if (!ok) return false;

    OssCaptureFramePayload payload{};
    payload.lr_rgb = {lr_rgb.data(), output_w, output_h, 3};
    payload.hr_rgb = {hr_rgb.data(), output_w, output_h, 3};
    payload.depth_z = {depth_z.data(), output_w, output_h, 1};
    payload.motion_xy = {motion_xy.data(), output_w, output_h, 2};
    payload.burst_tier = batch.decision.burst_tier;
    payload.capture_mode = batch.decision.capture_mode;
    payload.capture_lr = 1;
    payload.capture_hr = batch.decision.capture_hr ? 1u : 0u;
    payload.capture_depth = 1;
    payload.capture_motion = 1;
    payload.capture_normals = 0;
    if (!oss_capture_write_exr(batch.exr_path.c_str(), &payload)) return false;

    const double mean_motion =
        motion_sum / static_cast<double>(std::max<size_t>(1, static_cast<size_t>(output_w) * output_h));
    std::ofstream meta(batch.json_path, std::ios::binary | std::ios::trunc);
    if (!meta) return false;
    meta << "{\n";
    meta << "  \"schema_version\": 1,\n";
    meta << "  \"capture_kind\": \"sr\",\n";
    meta << "  \"provider\": \"" << JsonEscape(batch.provider.empty() ? "unknown" : batch.provider) << "\",\n";
    meta << "  \"game_id\": \"" << JsonEscape(batch.game_id) << "\",\n";
    meta << "  \"game_version\": \"" << JsonEscape(batch.game_version) << "\",\n";
    meta << "  \"session_uuid\": \"" << batch.session_uuid << "\",\n";
    meta << "  \"frame_uuid\": \"" << batch.frame_uuid << "\",\n";
    meta << "  \"frame_index\": " << batch.frame.frame_index << ",\n";
    meta << "  \"captured_at_unix\": " << batch.captured_at_unix << ",\n";
    meta << "  \"captured_at_monotonic_seconds\": " << batch.captured_at_monotonic_seconds << ",\n";
    meta << "  \"sequence_index\": " << batch.frame.frame_index << ",\n";
    meta << "  \"sequence_reset\": " << batch.frame.reset << ",\n";
    meta << "  \"lr_resolution\": [" << render_w << ", " << render_h << "],\n";
    meta << "  \"hr_resolution\": [" << output_w << ", " << output_h << "],\n";
    meta << "  \"render_resolution\": [" << render_w << ", " << render_h << "],\n";
    meta << "  \"output_resolution\": [" << output_w << ", " << output_h << "],\n";
    meta << "  \"dxgi_formats\": {\n";
    meta << "    \"color\": \"" << DxgiFormatName(lr->desc.Format) << "\",\n";
    meta << "    \"output\": \"" << DxgiFormatName(hr->desc.Format) << "\",\n";
    meta << "    \"depth\": \"" << DxgiFormatName(depth->desc.Format) << "\",\n";
    meta << "    \"motion_vectors\": \"" << DxgiFormatName(motion->desc.Format) << "\"\n";
    meta << "  },\n";
    meta << "  \"render_subrect\": {\n";
    meta << "    \"base_x\": " << batch.frame.subrect_base_x << ",\n";
    meta << "    \"base_y\": " << batch.frame.subrect_base_y << ",\n";
    meta << "    \"width\": " << render_w << ",\n";
    meta << "    \"height\": " << render_h << "\n";
    meta << "  },\n";
    meta << "  \"hr_source\": \"" << (batch.decision.capture_hr ? JsonEscape(batch.provider) : "none") << "\",\n";
    meta << "  \"jitter_offset_uv\": [" << batch.frame.jitter_offset_x << ", " << batch.frame.jitter_offset_y << "],\n";
    meta << "  \"jitter_offset_px\": [" << batch.frame.jitter_offset_x << ", " << batch.frame.jitter_offset_y << "],\n";
    meta << "  \"motion_vector_scale\": [" << batch.frame.mv_scale_x << ", " << batch.frame.mv_scale_y << "],\n";
    meta << "  \"exposure_scale\": " << batch.frame.exposure_scale << ",\n";
    meta << "  \"feature_create_flags\": " << batch.frame.feature_create_flags << ",\n";
    meta << "  \"motion_mean_magnitude_px\": " << mean_motion << ",\n";
    meta << "  \"channel_presence\": {\n";
    meta << "    \"lr\": true,\n";
    meta << "    \"hr\": " << (batch.decision.capture_hr ? "true" : "false") << ",\n";
    meta << "    \"depth\": true,\n";
    meta << "    \"motion\": true,\n";
    meta << "    \"normals\": false,\n";
    meta << "    \"albedo\": false,\n";
    meta << "    \"roughness\": false,\n";
    meta << "    \"metallic\": false,\n";
    meta << "    \"emissive\": false,\n";
    meta << "    \"exposure_texture\": " << (batch.frame.exposure_texture ? "true" : "false") << "\n";
    meta << "  },\n";
    meta << "  \"resources\": {\n";
    meta << "    \"lr_color\": ";
    WriteTextureMeta(meta, lr, "    ");
    meta << ",\n";
    meta << "    \"hr_output\": ";
    WriteTextureMeta(meta, hr, "    ");
    meta << ",\n";
    meta << "    \"depth\": ";
    WriteTextureMeta(meta, depth, "    ");
    meta << ",\n";
    meta << "    \"motion_vectors\": ";
    WriteTextureMeta(meta, motion, "    ");
    meta << "\n";
    meta << "  },\n";
    char hash_buf[32]{};
    std::snprintf(hash_buf, sizeof(hash_buf), "0x%016llx", static_cast<unsigned long long>(batch.phash));
    meta << "  \"perceptual_hash_64\": \"" << hash_buf << "\",\n";
    meta << "  \"capture_mode\": \"" << ModeName(batch.decision.capture_mode) << "\",\n";
    if (batch.decision.burst_tier != OSS_CAPTURE_TIER_NONE &&
        batch.decision.burst_uuid[0] != '\0') {
        meta << "  \"burst_uuid\": \"" << batch.decision.burst_uuid << "\",\n";
        meta << "  \"burst_index\": 0,\n";
        meta << "  \"burst_n\": " << batch.decision.burst_n << ",\n";
        meta << "  \"burst_tier\": \"" << batch.decision.burst_tier_name << "\",\n";
    }
    meta << "  \"capture_storage_mode\": \"" << JsonEscape(batch.capture_storage_mode) << "\",\n";
    meta << "  \"user_consent_token\": \"" << JsonEscape(batch.consent_token) << "\",\n";
    meta << "  \"uploader_version\": \"native-0.1.0\"\n";
    meta << "}\n";
    return meta.good();
}

void WorkerLoop() {
    for (;;) {
        std::shared_ptr<PendingBatch> batch;
        {
            std::unique_lock<std::mutex> lk(g_worker_mu);
            g_worker_cv.wait(lk, [] { return !g_worker_q.empty(); });
            batch = std::move(g_worker_q.front());
            g_worker_q.pop_front();
        }
        if (!batch) break;
        if (g_fence->GetCompletedValue() < batch->fence_value) {
            if (FAILED(g_fence->SetEventOnCompletion(batch->fence_value, g_fence_event))) {
                OSSG_LOG_ERROR("ngx_capture", "SetEventOnCompletion failed");
                continue;
            }
            WaitForSingleObject(g_fence_event, INFINITE);
        }
        if (batch->drop_after_fence) {
            OSSG_LOG_WARN(
                "ngx_capture",
                "retired dropped capture frame=%llu after fence",
                static_cast<unsigned long long>(batch->frame.frame_index));
            continue;
        }
        if (BuildPayloadAndWrite(*batch)) {
            OSSG_LOG_INFO(
                "ngx_capture",
                "wrote capture exr=%s",
                WideToUtf8(batch->exr_path).c_str());
        } else {
            OSSG_LOG_ERROR("ngx_capture", "failed to write capture frame");
        }
    }
}

void EnsureWorker() {
    bool expected = false;
    if (g_worker_started.compare_exchange_strong(expected, true)) {
        g_worker = std::thread(WorkerLoop);
        g_worker.detach();
    }
}

uint64_t CandidateHash(const OssGaussianFrame& frame) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix(frame.frame_index);
    mix(reinterpret_cast<uintptr_t>(frame.color));
    mix(reinterpret_cast<uintptr_t>(frame.output));
    return h;
}

OssCaptureDecision ConsiderFrameCandidate(const OssGaussianFrame& frame, float motion_hint) {
    const double now = SecondsSinceStart();
    double delta = 0.0;
    {
        std::lock_guard<std::mutex> lk(g_candidate_mu);
        if (g_last_candidate_time > 0.0) delta = std::max(0.0, now - g_last_candidate_time);
        g_last_candidate_time = now;
    }
    OssCaptureCandidate candidate{};
    candidate.frame_index = frame.frame_index;
    candidate.timestamp_seconds = now;
    candidate.seconds_since_last_candidate = delta;
    candidate.seconds_since_previous_candidate = delta;
    candidate.motion_mean_magnitude_px = motion_hint;
    candidate.motion_below_threshold_seconds = motion_hint < 0.5f ? delta : 0.0;
    candidate.perceptual_hash_64 = CandidateHash(frame);
    candidate.depth_degenerate = frame.depth ? 0u : 1u;
    candidate.motion_vectors_nan = frame.motion_vectors ? 0u : 1u;
    candidate.unsupported_rt_format = 0u;
    return oss_capture_consider_candidate(&candidate);
}

std::shared_ptr<PendingBatch> TrackPendingForFence(std::unique_ptr<PendingBatch> owned) {
    std::shared_ptr<PendingBatch> batch(owned.release());
    {
        std::lock_guard<std::mutex> lk(g_pending_mu);
        g_pending.push_back(batch);
    }
    EnsureWorker();
    return batch;
}

} // namespace

void* BeginUpscalerFrameCapture(
    const char* provider,
    ID3D12GraphicsCommandList* command_list,
    const OssGaussianFrame& frame) {
    if (!command_list || !frame.color || !frame.output || !frame.depth || !frame.motion_vectors) {
        return nullptr;
    }
    if (!EnsureDevice(static_cast<ID3D12Resource*>(frame.output))) {
        OSSG_LOG_WARN("ngx_capture", "device/fence init failed");
        return nullptr;
    }

    OssCaptureDecision decision = ConsiderFrameCandidate(frame, 1.0f);
    if (!decision.capture) return nullptr;

    auto batch = std::make_unique<PendingBatch>();
    batch->command_list = command_list;
    batch->decision = decision;
    batch->frame = frame;
    batch->captured_at_unix = UnixNow();
    batch->captured_at_monotonic_seconds = SecondsSinceStart();
    batch->phash = CandidateHash(frame);
    OssCaptureConfig cfg = capture::CurrentCaptureConfig();
    batch->game_id = cfg.game_id[0] ? cfg.game_id : "cyberpunk-2077";
    batch->game_version = cfg.game_version[0] ? cfg.game_version : "unknown";
    batch->consent_token = cfg.user_consent_token;
    batch->capture_storage_mode = cfg.capture_storage_mode[0] ? cfg.capture_storage_mode : "local";
    batch->provider = provider && provider[0] ? provider : "unknown";
    batch->session_uuid = SessionUuid();
    batch->frame_uuid = GuidString();
    batch->exr_path = BuildExrPath(batch->game_id, batch->session_uuid, batch->frame_uuid);
    batch->json_path = WideToUtf8(batch->exr_path.substr(0, batch->exr_path.size() - 4) + L".json");

    batch->textures = {
        MakeReadback(CaptureRole::LrColor, static_cast<ID3D12Resource*>(frame.color), frame),
        MakeReadback(CaptureRole::Depth, static_cast<ID3D12Resource*>(frame.depth), frame),
        MakeReadback(CaptureRole::Motion, static_cast<ID3D12Resource*>(frame.motion_vectors), frame),
    };
    for (TextureReadback& texture : batch->textures) {
        if (!CreateReadback(texture)) {
            OSSG_LOG_WARN("ngx_capture", "readback allocation/format unsupported role=%d", static_cast<int>(texture.role));
            return nullptr;
        }
    }

    std::filesystem::create_directories(std::filesystem::path(batch->exr_path).parent_path());
    for (TextureReadback& texture : batch->textures) {
        RecordReadbackCopy(command_list, texture);
    }
    return batch.release();
}

void EndUpscalerFrameCapture(void* ticket, bool succeeded, int provider_result) {
    std::unique_ptr<PendingBatch> owned(static_cast<PendingBatch*>(ticket));
    if (!owned) return;
    if (!succeeded) {
        OSSG_LOG_WARN(
            "ngx_capture",
            "dropping capture because %s dispatch failed result=0x%08x",
            owned->provider.empty() ? "upscaler" : owned->provider.c_str(),
            provider_result);
        owned->drop_after_fence = true;
        TrackPendingForFence(std::move(owned));
        return;
    }

    ID3D12GraphicsCommandList* command_list =
        static_cast<ID3D12GraphicsCommandList*>(owned->command_list);
    TextureReadback output = MakeReadback(
        CaptureRole::HrOutput,
        static_cast<ID3D12Resource*>(owned->frame.output),
        owned->frame);
    if (!CreateReadback(output)) {
        OSSG_LOG_WARN("ngx_capture", "output readback allocation/format unsupported");
        owned->drop_after_fence = true;
        TrackPendingForFence(std::move(owned));
        return;
    }
    RecordReadbackCopy(command_list, output);
    owned->textures.push_back(std::move(output));

    std::shared_ptr<PendingBatch> batch = TrackPendingForFence(std::move(owned));
    OSSG_LOG_INFO(
        "ngx_capture",
        "scheduled frame=%llu exr=%s",
        static_cast<unsigned long long>(batch->frame.frame_index),
        WideToUtf8(batch->exr_path).c_str());
}

void* BeginNgxFrameCapture(
    ID3D12GraphicsCommandList* command_list,
    const OssGaussianFrame& frame) {
    return BeginUpscalerFrameCapture("dlss", command_list, frame);
}

void EndNgxFrameCapture(void* ticket, int ngx_result) {
    EndUpscalerFrameCapture(ticket, ngx_result == 0x1, ngx_result);
}

void NotifyNgxCaptureCommandListsExecuted(
    ID3D12CommandQueue* queue,
    unsigned int command_list_count,
    ID3D12CommandList* const* command_lists) {
    if (!queue || !command_lists || !g_fence) return;
    std::vector<std::shared_ptr<PendingBatch>> matched;
    {
        std::lock_guard<std::mutex> lk(g_pending_mu);
        auto it = g_pending.begin();
        while (it != g_pending.end()) {
            bool found = false;
            for (unsigned int i = 0; i < command_list_count; ++i) {
                if ((*it)->command_list == command_lists[i]) {
                    found = true;
                    break;
                }
            }
            if (found) {
                matched.push_back(*it);
                it = g_pending.erase(it);
            } else {
                ++it;
            }
        }
    }
    if (matched.empty()) return;

    const UINT64 fence_value = g_next_fence_value.fetch_add(1, std::memory_order_relaxed);
    if (FAILED(queue->Signal(g_fence.Get(), fence_value))) {
        OSSG_LOG_ERROR("ngx_capture", "queue Signal failed for capture batch");
        std::lock_guard<std::mutex> lk(g_pending_mu);
        for (auto& batch : matched) {
            g_unfenced_quarantine.push_back(batch);
        }
        if (g_unfenced_quarantine.size() > 64) {
            const size_t excess = g_unfenced_quarantine.size() - 64;
            g_unfenced_quarantine.erase(
                g_unfenced_quarantine.begin(),
                g_unfenced_quarantine.begin() + excess);
        }
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_worker_mu);
        for (auto& batch : matched) {
            batch->fence_value = fence_value;
            g_worker_q.push_back(batch);
        }
    }
    g_worker_cv.notify_all();
}

} // namespace oss_gaussian
