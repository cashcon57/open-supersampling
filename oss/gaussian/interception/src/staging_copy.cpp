// =============================================================================
//  staging_copy.cpp
//
//  Async D3D12 readback worker. See staging_copy.h header for design notes.
//
//  Implementation overview:
//
//    Init: capture device, configure the bounded readback pool, create fence +
//          event, spin up worker thread.
//
//    ScheduleReadback (render thread):
//      - Pull a free slot from the pool (drop frame if exhausted).
//      - Build a one-shot ID3D12CommandList: CopyTextureRegion src → readback.
//      - ExecuteCommandLists on the provided queue, Signal fence with a unique
//        completion value.
//      - Push (slot, completion_value, exr_path, frame_index) onto a queue.
//
//    Worker thread (loop):
//      - Pop next request.
//      - Wait on fence event for completion_value.
//      - Map readback resource, hand the bytes + metadata to the EXR writer
//        (oss_capture_write_exr).
//      - Unmap, release back to pool.
//
//  Failure modes (logged, never fatal):
//    - Device removed mid-flight → Worker logs, releases slot, continues.
//    - Pool/backpressure exhausted → ScheduleReadback returns false; caller
//      drops frame.
//    - EXR write fails → telemetry++, log, continue.
//
//  Status: SCAFFOLDED. The full D3D12 readback path with command list building,
//  fence signaling, and worker drain is implemented in skeleton form below.
//  TestableReadiness work to do on 3080 Ti before unleashing on Cyberpunk:
//    a) Validate readback descriptor packing for DXGI_FORMAT_R10G10B10A2_UNORM
//       (Cyberpunk's HDR backbuffer format on most setups).
//    b) Verify worker thread shutdown drains correctly under fast game-quit.
//    c) Pool-size auto-tuning based on present cadence.
//
//  Copyright 2026 OSS-Gaussian contributors. Apache 2.0.
// =============================================================================
#define OSS_GAUSSIAN_BUILDING_DLL 1

#include "staging_copy.h"
#include "log.h"
#include "../oss_capture.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <d3d12.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

namespace oss_gaussian {

namespace {

// ----------------------------------------------------------------------
//  Pool of recycled readback heaps. Defaults target burst capture without
//  hiding runaway IO pressure; env overrides are clamped to keep game-process
//  memory bounded.
// ----------------------------------------------------------------------
constexpr size_t kDefaultPoolSize = 32;
constexpr size_t kMinPoolSize = 1;
constexpr size_t kMaxPoolSize = 128;

struct StagingRuntimeConfig {
    size_t pool_size = kDefaultPoolSize;
    size_t max_in_flight = kDefaultPoolSize;
};

struct StagingSlot {
    ComPtr<ID3D12Resource> readback_buffer;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cmd_list;
    UINT64 size_bytes;
    bool   in_use;

    StagingSlot() : size_bytes(0), in_use(false) {}
};

ComPtr<ID3D12Device>            g_device;
ComPtr<ID3D12Fence>             g_fence;
HANDLE                          g_fence_event = nullptr;
std::atomic<UINT64>             g_next_fence_value{1};

std::mutex                      g_pool_mu;
std::vector<StagingSlot>        g_pool;

// Worker queue.
struct Request {
    size_t              slot_idx;
    UINT64              fence_value;
    UINT                width;
    UINT                height;
    int32_t             dxgi_format;
    std::string         out_exr_path;
    unsigned long long  frame_index;
};

std::mutex                      g_worker_mu;
std::condition_variable         g_worker_cv;
std::deque<Request>             g_worker_q;
std::atomic<bool>               g_worker_run{false};
std::thread                     g_worker_thread;

// Telemetry.
std::atomic<unsigned long long> g_t_requested{0};
std::atomic<unsigned long long> g_t_scheduled{0};
std::atomic<unsigned long long> g_t_completed{0};
std::atomic<unsigned long long> g_t_dropped_pool{0};
std::atomic<unsigned long long> g_t_dropped_backpressure{0};
std::atomic<unsigned long long> g_t_failed_copy{0};
std::atomic<unsigned long long> g_t_failed_write{0};
std::atomic<unsigned long long> g_t_in_flight{0};
std::atomic<unsigned long long> g_t_queued{0};
std::atomic<unsigned long long> g_t_high_water_in_flight{0};
std::atomic<unsigned long long> g_t_high_water_queued{0};

std::mutex                      g_config_mu;
StagingRuntimeConfig            g_config;

// ----------------------------------------------------------------------
//  Helpers.
// ----------------------------------------------------------------------

unsigned long long ClampU64(unsigned long long value, unsigned long long lo, unsigned long long hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

unsigned long long ReadEnvU64(const wchar_t* name, unsigned long long fallback) {
    wchar_t buf[32]{};
    DWORD len = GetEnvironmentVariableW(name, buf, static_cast<DWORD>(_countof(buf)));
    if (len == 0 || len >= _countof(buf)) return fallback;

    wchar_t* end = nullptr;
    unsigned long long parsed = wcstoull(buf, &end, 10);
    if (end == buf) return fallback;
    return parsed;
}

StagingRuntimeConfig LoadRuntimeConfig() {
    StagingRuntimeConfig cfg{};
    cfg.pool_size = static_cast<size_t>(ClampU64(
        ReadEnvU64(L"OSS_GAUSSIAN_STAGING_POOL_SIZE", kDefaultPoolSize),
        kMinPoolSize,
        kMaxPoolSize));
    cfg.max_in_flight = static_cast<size_t>(ClampU64(
        ReadEnvU64(L"OSS_GAUSSIAN_STAGING_MAX_IN_FLIGHT", cfg.pool_size),
        1,
        cfg.pool_size));
    return cfg;
}

void UpdateHighWater(std::atomic<unsigned long long>& target, unsigned long long value) {
    unsigned long long observed = target.load(std::memory_order_relaxed);
    while (value > observed &&
           !target.compare_exchange_weak(
               observed,
               value,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

void NoteQueued() {
    const unsigned long long queued = g_t_queued.fetch_add(1, std::memory_order_relaxed) + 1ull;
    UpdateHighWater(g_t_high_water_queued, queued);
}

void NoteDequeued() {
    g_t_queued.fetch_sub(1, std::memory_order_relaxed);
}

void NoteInFlightStarted() {
    const unsigned long long in_flight =
        g_t_in_flight.fetch_add(1, std::memory_order_relaxed) + 1ull;
    UpdateHighWater(g_t_high_water_in_flight, in_flight);
}

void NoteInFlightFinished() {
    g_t_in_flight.fetch_sub(1, std::memory_order_relaxed);
}

bool ShouldLogCounter(unsigned long long value) {
    return value == 1ull || value == 2ull || value == 4ull ||
           (value != 0ull && (value % 16ull) == 0ull);
}

UINT64 BytesPerPixel(DXGI_FORMAT fmt) {
    switch (fmt) {
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_R10G10B10A2_UNORM:
        case DXGI_FORMAT_R11G11B10_FLOAT:
            return 4;
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
            return 8;
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            return 16;
        default:
            return 4;
    }
}

UINT64 AlignUp(UINT64 value, UINT64 alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

bool ComputeReadbackLayout(
    UINT width,
    UINT height,
    DXGI_FORMAT fmt,
    UINT64* row_pitch_out,
    UINT64* total_bytes_out) {
    if (!width || !height || !row_pitch_out || !total_bytes_out) return false;

    const UINT64 bpp = BytesPerPixel(fmt);
    if (width > std::numeric_limits<UINT64>::max() / bpp) return false;

    const UINT64 raw_row_bytes = static_cast<UINT64>(width) * bpp;
    const UINT64 row_pitch = AlignUp(raw_row_bytes, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    if (row_pitch > std::numeric_limits<UINT>::max()) return false;
    if (height > std::numeric_limits<UINT64>::max() / row_pitch) return false;

    *row_pitch_out = row_pitch;
    *total_bytes_out = static_cast<UINT64>(height) * row_pitch;
    return true;
}

bool CreateSlot(StagingSlot& slot, UINT64 size_bytes) {
    HRESULT hr;
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width              = size_bytes;
    rd.Height             = 1;
    rd.DepthOrArraySize   = 1;
    rd.MipLevels          = 1;
    rd.SampleDesc.Count   = 1;
    rd.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    hr = g_device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&slot.readback_buffer));
    if (FAILED(hr)) {
        OSSG_LOG_ERROR("staging", "readback CreateCommittedResource failed (0x%08lx)", hr);
        return false;
    }

    hr = g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&slot.alloc));
    if (FAILED(hr)) {
        OSSG_LOG_ERROR("staging", "CreateCommandAllocator failed (0x%08lx)", hr);
        return false;
    }

    hr = g_device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, slot.alloc.Get(), nullptr,
        IID_PPV_ARGS(&slot.cmd_list));
    if (FAILED(hr)) {
        OSSG_LOG_ERROR("staging", "CreateCommandList failed (0x%08lx)", hr);
        return false;
    }
    slot.cmd_list->Close();
    slot.size_bytes = size_bytes;
    return true;
}

int AcquireSlot(UINT64 needed_bytes) {
    std::lock_guard<std::mutex> lk(g_pool_mu);
    for (size_t i = 0; i < g_pool.size(); ++i) {
        if (!g_pool[i].in_use && g_pool[i].size_bytes >= needed_bytes) {
            g_pool[i].in_use = true;
            return static_cast<int>(i);
        }
    }
    // Try resizing a free slot.
    for (size_t i = 0; i < g_pool.size(); ++i) {
        if (!g_pool[i].in_use) {
            g_pool[i] = StagingSlot{};
            if (CreateSlot(g_pool[i], needed_bytes)) {
                g_pool[i].in_use = true;
                return static_cast<int>(i);
            }
            return -1;
        }
    }
    return -1;
}

void ReleaseSlot(size_t idx) {
    std::lock_guard<std::mutex> lk(g_pool_mu);
    if (idx < g_pool.size()) {
        g_pool[idx].in_use = false;
    }
}

// ----------------------------------------------------------------------
//  Worker thread — drains requests, waits on fence, writes EXR.
// ----------------------------------------------------------------------

void WorkerLoop() {
    OSSG_LOG_INFO("staging", "worker thread up");
    for (;;) {
        Request req;
        {
            std::unique_lock<std::mutex> lk(g_worker_mu);
            g_worker_cv.wait(lk, [] {
                return !g_worker_run.load(std::memory_order_acquire) || !g_worker_q.empty();
            });
            if (!g_worker_run.load(std::memory_order_acquire) && g_worker_q.empty()) break;
            req = std::move(g_worker_q.front());
            g_worker_q.pop_front();
            NoteDequeued();
        }

        // Wait for the fence to indicate the readback finished.
        if (g_fence->GetCompletedValue() < req.fence_value) {
            HRESULT event_hr = g_fence->SetEventOnCompletion(req.fence_value, g_fence_event);
            if (FAILED(event_hr)) {
                g_t_failed_copy.fetch_add(1, std::memory_order_relaxed);
                OSSG_LOG_ERROR(
                    "staging",
                    "SetEventOnCompletion failed (hr=0x%08lx, fence=%llu)",
                    event_hr,
                    static_cast<unsigned long long>(req.fence_value));
                ReleaseSlot(req.slot_idx);
                NoteInFlightFinished();
                continue;
            }
            DWORD wait_result = WaitForSingleObject(g_fence_event, INFINITE);
            if (wait_result != WAIT_OBJECT_0) {
                g_t_failed_copy.fetch_add(1, std::memory_order_relaxed);
                OSSG_LOG_ERROR(
                    "staging",
                    "readback fence wait failed (result=%lu, fence=%llu)",
                    wait_result,
                    static_cast<unsigned long long>(req.fence_value));
                ReleaseSlot(req.slot_idx);
                NoteInFlightFinished();
                continue;
            }
        }

        // Map, write, unmap.
        //
        // NOTE: oss_capture_write_exr expects OssCaptureImageView with
        // `const float* pixels` (linear-light float channels). Our readback
        // here is raw bytes in the source dxgi_format (e.g., R10G10B10A2,
        // R11G11B10F, R16G16B16A16F). Converting to linear float requires:
        //   - Format-specific unpack (10-10-10-2 → fp32, 11-11-10F → fp32, etc.)
        //   - Optional sRGB→linear if the format is *_UNORM_SRGB
        //   - 4-channel alignment for the EXR writer's R/G/B/A schema
        //
        // For Sprint 2.x we WRITE A RAW .bin DUMP next to the requested EXR
        // path so field testing can validate the readback path end-to-end.
        // The format-correct EXR write lands once NGX EvaluateFeature gives
        // us the LR + G-buffer float pointers directly (T2.6); at that point
        // the post-Present backbuffer is no longer the desired capture (it's
        // post-DLSS-hallucination).
        StagingSlot& slot = g_pool[req.slot_idx];
        void* mapped_ptr = nullptr;
        D3D12_RANGE read_range{0, slot.size_bytes};
        HRESULT hr = slot.readback_buffer->Map(0, &read_range, &mapped_ptr);
        if (SUCCEEDED(hr) && mapped_ptr) {
            // Write raw bytes alongside the EXR path with .raw extension.
            std::string raw_path = req.out_exr_path + ".raw";
            FILE* fp = nullptr;
            fopen_s(&fp, raw_path.c_str(), "wb");
            if (fp) {
                fwrite(mapped_ptr, 1, slot.size_bytes, fp);
                fclose(fp);
                g_t_completed.fetch_add(1, std::memory_order_relaxed);
            } else {
                g_t_failed_write.fetch_add(1, std::memory_order_relaxed);
                OSSG_LOG_ERROR("staging", "fopen_s failed for %s", raw_path.c_str());
            }

            D3D12_RANGE no_write{0, 0};
            slot.readback_buffer->Unmap(0, &no_write);
        } else {
            g_t_failed_copy.fetch_add(1, std::memory_order_relaxed);
            OSSG_LOG_ERROR("staging", "readback Map failed (hr=0x%08lx)", hr);
        }

        ReleaseSlot(req.slot_idx);
        NoteInFlightFinished();
    }
    OSSG_LOG_INFO("staging", "worker thread down");
}

}  // namespace

// ----------------------------------------------------------------------
//  Public surface.
// ----------------------------------------------------------------------

bool InitStagingCopy(ID3D12Device* device) {
    if (!device) return false;
    if (g_worker_run.load(std::memory_order_acquire)) return true;

    {
        std::lock_guard<std::mutex> cfg_lk(g_config_mu);
        g_config = LoadRuntimeConfig();
    }
    {
        std::lock_guard<std::mutex> pool_lk(g_pool_mu);
        g_pool.clear();
        g_pool.resize(g_config.pool_size);
    }

    g_device = device;
    HRESULT hr = g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence));
    if (FAILED(hr)) {
        OSSG_LOG_ERROR("staging", "CreateFence failed (0x%08lx)", hr);
        return false;
    }
    g_fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!g_fence_event) {
        OSSG_LOG_ERROR("staging", "CreateEvent failed (le=%lu)", GetLastError());
        g_fence.Reset();
        g_device.Reset();
        return false;
    }

    g_worker_run.store(true, std::memory_order_release);
    g_worker_thread = std::thread(WorkerLoop);

    OSSG_LOG_INFO(
        "staging",
        "InitStagingCopy: pool=%zu max_in_flight=%zu fence=%p",
        g_config.pool_size,
        g_config.max_in_flight,
        (void*)g_fence.Get());
    return true;
}

bool ScheduleReadback(
    ID3D12CommandQueue*  queue,
    ID3D12Resource*      src,
    UINT                 width,
    UINT                 height,
    int32_t              dxgi_format,
    const char*          out_exr_path,
    unsigned long long   frame_index
) {
    if (!queue || !src || !out_exr_path) return false;
    if (!g_worker_run.load(std::memory_order_acquire)) return false;

    g_t_requested.fetch_add(1, std::memory_order_relaxed);

    size_t pool_size = kDefaultPoolSize;
    size_t max_in_flight = kDefaultPoolSize;
    {
        std::lock_guard<std::mutex> cfg_lk(g_config_mu);
        pool_size = g_config.pool_size;
        max_in_flight = g_config.max_in_flight;
    }

    const unsigned long long in_flight_now =
        g_t_in_flight.load(std::memory_order_relaxed);
    if (in_flight_now >= max_in_flight) {
        const unsigned long long dropped =
            g_t_dropped_backpressure.fetch_add(1, std::memory_order_relaxed) + 1ull;
        if (ShouldLogCounter(dropped)) {
            OSSG_LOG_WARN(
                "staging",
                "dropping readback for backpressure frame=%llu in_flight=%llu max_in_flight=%zu "
                "queued=%llu dropped_backpressure=%llu",
                frame_index,
                in_flight_now,
                max_in_flight,
                g_t_queued.load(std::memory_order_relaxed),
                dropped);
        }
        return false;
    }

    UINT64 row_pitch = 0;
    UINT64 needed = 0;
    if (!ComputeReadbackLayout(
            width,
            height,
            static_cast<DXGI_FORMAT>(dxgi_format),
            &row_pitch,
            &needed)) {
        g_t_failed_copy.fetch_add(1, std::memory_order_relaxed);
        OSSG_LOG_ERROR(
            "staging",
            "invalid readback layout width=%u height=%u format=%d",
            width,
            height,
            dxgi_format);
        return false;
    }
    int idx = AcquireSlot(needed);
    if (idx < 0) {
        const unsigned long long dropped =
            g_t_dropped_pool.fetch_add(1, std::memory_order_relaxed) + 1ull;
        if (ShouldLogCounter(dropped)) {
            OSSG_LOG_WARN(
                "staging",
                "dropping readback because pool is full frame=%llu pool=%zu in_flight=%llu "
                "queued=%llu dropped_pool_full=%llu",
                frame_index,
                pool_size,
                g_t_in_flight.load(std::memory_order_relaxed),
                g_t_queued.load(std::memory_order_relaxed),
                dropped);
        }
        return false;
    }

    StagingSlot& slot = g_pool[idx];
    HRESULT hr = slot.alloc->Reset();
    if (FAILED(hr)) {
        g_t_failed_copy.fetch_add(1, std::memory_order_relaxed);
        ReleaseSlot(static_cast<size_t>(idx));
        OSSG_LOG_ERROR("staging", "CommandAllocator::Reset failed (0x%08lx)", hr);
        return false;
    }
    hr = slot.cmd_list->Reset(slot.alloc.Get(), nullptr);
    if (FAILED(hr)) {
        g_t_failed_copy.fetch_add(1, std::memory_order_relaxed);
        ReleaseSlot(static_cast<size_t>(idx));
        OSSG_LOG_ERROR("staging", "CommandList::Reset failed (0x%08lx)", hr);
        return false;
    }

    D3D12_RESOURCE_BARRIER to_copy{};
    to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_copy.Transition.pResource = src;
    to_copy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    to_copy.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    to_copy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    slot.cmd_list->ResourceBarrier(1, &to_copy);

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = slot.readback_buffer.Get();
    dst.Type      = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint.Format   = static_cast<DXGI_FORMAT>(dxgi_format);
    dst.PlacedFootprint.Footprint.Width    = width;
    dst.PlacedFootprint.Footprint.Height   = height;
    dst.PlacedFootprint.Footprint.Depth    = 1;
    dst.PlacedFootprint.Footprint.RowPitch = static_cast<UINT>(row_pitch);

    D3D12_TEXTURE_COPY_LOCATION src_loc{};
    src_loc.pResource        = src;
    src_loc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src_loc.SubresourceIndex = 0;

    slot.cmd_list->CopyTextureRegion(&dst, 0, 0, 0, &src_loc, nullptr);

    D3D12_RESOURCE_BARRIER to_present = to_copy;
    to_present.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    to_present.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    slot.cmd_list->ResourceBarrier(1, &to_present);

    hr = slot.cmd_list->Close();
    if (FAILED(hr)) {
        g_t_failed_copy.fetch_add(1, std::memory_order_relaxed);
        ReleaseSlot(static_cast<size_t>(idx));
        OSSG_LOG_ERROR("staging", "CommandList::Close failed (0x%08lx)", hr);
        return false;
    }

    ID3D12CommandList* lists[] = {slot.cmd_list.Get()};
    queue->ExecuteCommandLists(1, lists);

    UINT64 fence_val = g_next_fence_value.fetch_add(1, std::memory_order_relaxed);
    hr = queue->Signal(g_fence.Get(), fence_val);
    if (FAILED(hr)) {
        g_t_failed_copy.fetch_add(1, std::memory_order_relaxed);
        OSSG_LOG_ERROR(
            "staging",
            "CommandQueue::Signal failed (0x%08lx); retaining slot to avoid reuse while copy may be in-flight",
            hr);
        return false;
    }

    NoteInFlightStarted();
    {
        std::lock_guard<std::mutex> lk(g_worker_mu);
        Request req;
        req.slot_idx     = static_cast<size_t>(idx);
        req.fence_value  = fence_val;
        req.width        = width;
        req.height       = height;
        req.dxgi_format  = dxgi_format;
        req.out_exr_path = out_exr_path;
        req.frame_index  = frame_index;
        g_worker_q.push_back(std::move(req));
        NoteQueued();
    }
    g_t_scheduled.fetch_add(1, std::memory_order_relaxed);
    g_worker_cv.notify_one();
    OSSG_LOG_INFO(
        "staging",
        "scheduled present readback frame=%llu size=%ux%u fmt=%d row_pitch=%llu bytes=%llu "
        "in_flight=%llu queued=%llu path=%s.raw",
        frame_index,
        width,
        height,
        dxgi_format,
        static_cast<unsigned long long>(row_pitch),
        static_cast<unsigned long long>(needed),
        g_t_in_flight.load(std::memory_order_relaxed),
        g_t_queued.load(std::memory_order_relaxed),
        out_exr_path);
    return true;
}

void ShutdownStagingCopy() {
    if (!g_worker_run.load(std::memory_order_acquire)) return;

    g_worker_run.store(false, std::memory_order_release);
    g_worker_cv.notify_all();
    if (g_worker_thread.joinable()) g_worker_thread.join();

    if (g_fence_event) {
        CloseHandle(g_fence_event);
        g_fence_event = nullptr;
    }
    g_fence.Reset();
    g_device.Reset();

    {
        std::lock_guard<std::mutex> lk(g_pool_mu);
        for (size_t i = 0; i < g_pool.size(); ++i) {
            g_pool[i] = StagingSlot{};
        }
        g_pool.clear();
    }

    StagingCopyStats stats = GetStagingCopyStats();
    OSSG_LOG_INFO(
        "staging",
        "ShutdownStagingCopy complete requested=%llu scheduled=%llu completed=%llu "
        "dropped_pool=%llu dropped_backpressure=%llu failed_copy=%llu failed_write=%llu "
        "high_water_in_flight=%llu high_water_queued=%llu",
        stats.requested,
        stats.scheduled,
        stats.completed,
        stats.dropped_pool_full,
        stats.dropped_backpressure,
        stats.failed_copy,
        stats.failed_write,
        stats.high_water_in_flight,
        stats.high_water_queued);
}

StagingCopyStats GetStagingCopyStats() {
    StagingCopyStats s{};
    s.requested         = g_t_requested.load(std::memory_order_relaxed);
    s.scheduled         = g_t_scheduled.load(std::memory_order_relaxed);
    s.completed         = g_t_completed.load(std::memory_order_relaxed);
    s.dropped_pool_full = g_t_dropped_pool.load(std::memory_order_relaxed);
    s.dropped_backpressure = g_t_dropped_backpressure.load(std::memory_order_relaxed);
    s.failed_copy       = g_t_failed_copy.load(std::memory_order_relaxed);
    s.failed_write      = g_t_failed_write.load(std::memory_order_relaxed);
    s.in_flight         = g_t_in_flight.load(std::memory_order_relaxed);
    s.queued            = g_t_queued.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> cfg_lk(g_config_mu);
        s.pool_size     = g_config.pool_size;
        s.max_in_flight = g_config.max_in_flight;
    }
    s.high_water_in_flight = g_t_high_water_in_flight.load(std::memory_order_relaxed);
    s.high_water_queued = g_t_high_water_queued.load(std::memory_order_relaxed);
    return s;
}

}  // namespace oss_gaussian
