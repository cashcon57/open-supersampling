// =============================================================================
//  staging_copy.h
//
//  GPU→CPU readback for the OSS-Gaussian capture path. Allocates a D3D12
//  readback heap, schedules a copy from a GPU resource, signals a fence,
//  and waits asynchronously on a worker thread before invoking the EXR
//  writer.
//
//  Design constraints:
//    - Must NOT block the render thread (the Present hook calls Submit()
//      and returns immediately; the worker thread does the wait + write).
//    - Must NOT keep a fence alive across game shutdown; UninstallStagingCopy
//      flushes pending writes and tears down the worker.
//    - Must handle the readback-heap allocation pool: we recycle a small
//      pool of staging buffers (default 4) to avoid hot-path malloc.
//
//  Status: SCAFFOLDED. Compiles on Windows; the actual D3D12 copy commands
//  + fence path are stubbed pending real-game testing on 3080 Ti.
//
//  Copyright 2026 OSS-Gaussian contributors. Apache 2.0.
// =============================================================================
#ifndef OSS_GAUSSIAN_STAGING_COPY_H
#define OSS_GAUSSIAN_STAGING_COPY_H

#include <Windows.h>
#include <stdint.h>

struct ID3D12Resource;
struct ID3D12Device;
struct ID3D12CommandQueue;

namespace oss_gaussian {

// Initialize the staging-copy subsystem. Captures a device pointer and spins
// up the worker thread. Idempotent; second call is a no-op. Returns false
// if device init fails (game shutting down, OOM, etc.).
bool InitStagingCopy(ID3D12Device* device);

// Schedule an async GPU→CPU readback of `src` on `queue`. The provided EXR
// path will be written by the worker thread once the copy completes. Returns
// true if the request was queued; false if the staging pool is exhausted
// (in which case the caller drops this frame).
//
// `src` MUST be in a state that can be transitioned to D3D12_RESOURCE_STATE_
// COPY_SOURCE. The hook layer is responsible for the prior-state knowledge.
bool ScheduleReadback(
    ID3D12CommandQueue*  queue,
    ID3D12Resource*      src,
    UINT                 width,
    UINT                 height,
    int32_t              dxgi_format,
    const char*          out_exr_path,
    unsigned long long   frame_index
);

// Drain pending readbacks and shut down the worker thread. Called from
// UninstallD3D12Hooks() and from DLL_PROCESS_DETACH. Idempotent.
void ShutdownStagingCopy();

// Telemetry for the dashboard / log inspector.
struct StagingCopyStats {
    unsigned long long requested;
    unsigned long long completed;
    unsigned long long dropped_pool_full;
    unsigned long long failed_copy;
    unsigned long long failed_write;
};
StagingCopyStats GetStagingCopyStats();

}  // namespace oss_gaussian

#endif  // OSS_GAUSSIAN_STAGING_COPY_H
