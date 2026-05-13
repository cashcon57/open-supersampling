// =============================================================================
//  ngx_frame_capture.h
//
//  Cyberpunk-first NGX/DLSS resource capture path. The NGX export wrapper calls
//  this after real DLSS evaluation, while the game's command list is still open.
// =============================================================================
#ifndef OSS_GAUSSIAN_NGX_FRAME_CAPTURE_H
#define OSS_GAUSSIAN_NGX_FRAME_CAPTURE_H

#include "../include/oss_gaussian_interception.h"

struct ID3D12CommandList;
struct ID3D12CommandQueue;
struct ID3D12GraphicsCommandList;

namespace oss_gaussian {

void* BeginNgxFrameCapture(
    ID3D12GraphicsCommandList* command_list,
    const OssGaussianFrame& frame);

void EndNgxFrameCapture(void* ticket, int ngx_result);

void* BeginUpscalerFrameCapture(
    const char* provider,
    ID3D12GraphicsCommandList* command_list,
    const OssGaussianFrame& frame);

void EndUpscalerFrameCapture(void* ticket, bool succeeded, int provider_result);

void NotifyNgxCaptureCommandListsExecuted(
    ID3D12CommandQueue* queue,
    unsigned int command_list_count,
    ID3D12CommandList* const* command_lists);

} // namespace oss_gaussian

#endif // OSS_GAUSSIAN_NGX_FRAME_CAPTURE_H
