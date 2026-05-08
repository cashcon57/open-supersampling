// =============================================================================
//  staging_copy.cpp
//
//  Async D3D12 readback worker. See staging_copy.h header for design notes.
//
//  Implementation overview:
//
//    Init: capture device, allocate a 1-frame readback heap pool (recycled),
//          create fence + event, spin up worker thread.
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
//    - Pool exhausted → ScheduleReadback returns false; caller drops frame.
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
#include <deque>
#include <mutex>
#include <string>
#include <thread>

#include <d3d12.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

namespace oss_gaussian {

namespace {

// ----------------------------------------------------------------------
//  Pool of recycled readback heaps. Sized for ~4 in-flight frames.
// ----------------------------------------------------------------------
constexpr size_t kPoolSize = 4;

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
StagingSlot                     g_pool[kPoolSize];

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
std::atomic<unsigned long long> g_t_completed{0};
std::atomic<unsigned long long> g_t_dropped_pool{0};
std::atomic<unsigned long long> g_t_failed_copy{0};
std::atomic<unsigned long long> g_t_failed_write{0};

// ----------------------------------------------------------------------
//  Helpers.
// ----------------------------------------------------------------------

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
    for (size_t i = 0; i < kPoolSize; ++i) {
        if (!g_pool[i].in_use && g_pool[i].size_bytes >= needed_bytes) {
            g_pool[i].in_use = true;
            return static_cast<int>(i);
        }
    }
    // Try resizing a free slot.
    for (size_t i = 0; i < kPoolSize; ++i) {
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
    if (idx < kPoolSize) {
        g_pool[idx].in_use = false;
    }
}

// ----------------------------------------------------------------------
//  Worker thread — drains requests, waits on fence, writes EXR.
// ----------------------------------------------------------------------

void WorkerLoop() {
    OSSG_LOG_INFO("staging", "worker thread up");
    while (g_worker_run.load(std::memory_order_acquire)) {
        Request req;
        {
            std::unique_lock<std::mutex> lk(g_worker_mu);
            g_worker_cv.wait(lk, [] {
                return !g_worker_run.load(std::memory_order_acquire) || !g_worker_q.empty();
            });
            if (!g_worker_run.load(std::memory_order_acquire) && g_worker_q.empty()) break;
            req = std::move(g_worker_q.front());
            g_worker_q.pop_front();
        }

        // Wait for the fence to indicate the readback finished.
        if (g_fence->GetCompletedValue() < req.fence_value) {
            g_fence->SetEventOnCompletion(req.fence_value, g_fence_event);
            WaitForSingleObject(g_fence_event, INFINITE);
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

    g_device = device;
    HRESULT hr = g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence));
    if (FAILED(hr)) {
        OSSG_LOG_ERROR("staging", "CreateFence failed (0x%08lx)", hr);
        return false;
    }
    g_fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!g_fence_event) {
        OSSG_LOG_ERROR("staging", "CreateEvent failed (le=%lu)", GetLastError());
        return false;
    }

    g_worker_run.store(true, std::memory_order_release);
    g_worker_thread = std::thread(WorkerLoop);

    OSSG_LOG_INFO("staging", "InitStagingCopy: pool=%zu, fence=%p", kPoolSize, (void*)g_fence.Get());
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

    UINT64 needed = static_cast<UINT64>(width) * height *
                    BytesPerPixel(static_cast<DXGI_FORMAT>(dxgi_format));
    int idx = AcquireSlot(needed);
    if (idx < 0) {
        g_t_dropped_pool.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    StagingSlot& slot = g_pool[idx];
    slot.alloc->Reset();
    slot.cmd_list->Reset(slot.alloc.Get(), nullptr);

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = slot.readback_buffer.Get();
    dst.Type      = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint.Format   = static_cast<DXGI_FORMAT>(dxgi_format);
    dst.PlacedFootprint.Footprint.Width    = width;
    dst.PlacedFootprint.Footprint.Height   = height;
    dst.PlacedFootprint.Footprint.Depth    = 1;
    dst.PlacedFootprint.Footprint.RowPitch =
        static_cast<UINT>(width * BytesPerPixel(static_cast<DXGI_FORMAT>(dxgi_format)));

    D3D12_TEXTURE_COPY_LOCATION src_loc{};
    src_loc.pResource        = src;
    src_loc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src_loc.SubresourceIndex = 0;

    slot.cmd_list->CopyTextureRegion(&dst, 0, 0, 0, &src_loc, nullptr);
    slot.cmd_list->Close();

    ID3D12CommandList* lists[] = {slot.cmd_list.Get()};
    queue->ExecuteCommandLists(1, lists);

    UINT64 fence_val = g_next_fence_value.fetch_add(1, std::memory_order_relaxed);
    queue->Signal(g_fence.Get(), fence_val);

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
    }
    g_worker_cv.notify_one();
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
        for (size_t i = 0; i < kPoolSize; ++i) {
            g_pool[i] = StagingSlot{};
        }
    }

    OSSG_LOG_INFO("staging", "ShutdownStagingCopy complete");
}

StagingCopyStats GetStagingCopyStats() {
    StagingCopyStats s;
    s.requested         = g_t_requested.load(std::memory_order_relaxed);
    s.completed         = g_t_completed.load(std::memory_order_relaxed);
    s.dropped_pool_full = g_t_dropped_pool.load(std::memory_order_relaxed);
    s.failed_copy       = g_t_failed_copy.load(std::memory_order_relaxed);
    s.failed_write      = g_t_failed_write.load(std::memory_order_relaxed);
    return s;
}

}  // namespace oss_gaussian
