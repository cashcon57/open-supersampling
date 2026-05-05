// =============================================================================
//  oss_capture.h
//
//  Capture-mode client API for the OSS-Gaussian DXGI/D3D12 proxy DLL.
// =============================================================================
#ifndef OSS_CAPTURE_H
#define OSS_CAPTURE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#include <cstddef>
#endif

#if defined(_WIN32)
#  if defined(OSS_GAUSSIAN_BUILDING_DLL)
#    define OSS_CAPTURE_API __declspec(dllexport)
#  else
#    define OSS_CAPTURE_API __declspec(dllimport)
#  endif
#else
#  define OSS_CAPTURE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum OssCaptureDecisionRule {
    OSS_CAPTURE_RULE_ACCEPT = 0,
    OSS_CAPTURE_RULE_TEMPORAL_STRIDE = 1,
    OSS_CAPTURE_RULE_MOTION_BUCKET = 2,
    OSS_CAPTURE_RULE_PERCEPTUAL_DEDUP = 3,
    OSS_CAPTURE_RULE_GBUFFER_SANITY = 4,
    OSS_CAPTURE_RULE_POST_LOADING_GUARD = 5,
} OssCaptureDecisionRule;

typedef enum OssCaptureBurstTier {
    OSS_CAPTURE_TIER_NONE = 0,
    OSS_CAPTURE_TIER_SHORT = 1,
    OSS_CAPTURE_TIER_LONG = 2,
} OssCaptureBurstTier;

typedef struct OssCaptureConfig {
    char     game_id[64];
    char     game_version[64];
    char     user_consent_token[160];
    wchar_t  pending_root[260];
    double   capture_stride_seconds; // Compatibility alias; prefer stride_seconds.
    uint32_t burst_n;
    double   stride_seconds;
    int      short_burst_n;
    double   short_stride_seconds;
    int      long_burst_n;
    double   long_stride_seconds;
    int      long_capture_hr;
    int      two_tier_enabled;
    double   dedup_window_seconds;
    double   loading_gap_seconds;
    uint32_t max_motion_bucket_samples;
    uint32_t dedup_hamming_threshold;
} OssCaptureConfig;

typedef struct OssCaptureCandidate {
    uint64_t frame_index;
    double   timestamp_seconds;
    double   seconds_since_last_candidate;
    double   seconds_since_previous_candidate;
    float    motion_mean_magnitude_px;
    uint64_t perceptual_hash_64;
    uint32_t depth_degenerate;
    uint32_t motion_vectors_nan;
    uint32_t unsupported_rt_format;
} OssCaptureCandidate;

typedef struct OssCaptureDecision {
    uint32_t               capture;
    OssCaptureDecisionRule rule;
    uint32_t               burst_n;
    char                   burst_uuid[37];
    OssCaptureBurstTier    burst_tier;
    char                   burst_tier_name[8];
    uint32_t               capture_hr;
} OssCaptureDecision;

typedef struct OssCaptureBurstFrame {
    uint32_t active;
    uint32_t burst_index;
    uint32_t burst_n;
    char     burst_uuid[37];
    OssCaptureBurstTier burst_tier;
    char     burst_tier_name[8];
    uint32_t capture_hr;
} OssCaptureBurstFrame;

typedef struct OssCaptureImageView {
    const float* pixels;
    uint32_t     width;
    uint32_t     height;
    uint32_t     channels;
} OssCaptureImageView;

typedef struct OssCaptureFramePayload {
    OssCaptureImageView lr_rgb;
    OssCaptureImageView hr_rgb;
    OssCaptureImageView depth_z;
    OssCaptureImageView motion_xy;
    OssCaptureImageView normals_xyz;
    OssCaptureBurstTier burst_tier;
    uint32_t capture_hr;
} OssCaptureFramePayload;

OSS_CAPTURE_API OssCaptureConfig oss_capture_default_config(void);
OSS_CAPTURE_API int oss_capture_configure(const OssCaptureConfig* config);
OSS_CAPTURE_API OssCaptureDecision oss_capture_consider_candidate(const OssCaptureCandidate* candidate);
OSS_CAPTURE_API uint64_t oss_capture_phash64_rgb8(const uint8_t* rgb, uint32_t width, uint32_t height, uint32_t stride_bytes);
OSS_CAPTURE_API uint32_t oss_capture_hamming_distance64(uint64_t a, uint64_t b);
OSS_CAPTURE_API int oss_capture_write_exr(const wchar_t* path, const OssCaptureFramePayload* payload);
OSS_CAPTURE_API uint32_t oss_capture_consume_present_burst(OssCaptureBurstFrame* out);

// Hook entry points used by the D3D12/DXGI detour layer. They intentionally use
// void* to keep this header free of Windows/D3D12 includes.
OSS_CAPTURE_API void oss_capture_on_present(void* swap_chain);
OSS_CAPTURE_API void oss_capture_on_execute_command_lists(void* command_queue);
OSS_CAPTURE_API void oss_capture_on_ngx_evaluate_feature(void* command_list, const void* ngx_handle, const void* ngx_params);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus
namespace oss_gaussian::capture {

class CaptureSampler {
public:
    explicit CaptureSampler(const OssCaptureConfig& config);

    OssCaptureDecision Consider(const OssCaptureCandidate& candidate);
    void Reset();

private:
    OssCaptureConfig config_{};
    double           last_short_event_time_ = -1.0e30;
    double           last_long_event_time_ = -1.0e30;
    uint32_t         motion_buckets_[8]{};

    struct RecentHash {
        uint64_t hash = 0;
        double   timestamp = 0.0;
    };
    RecentHash recent_hashes_[64]{};
    size_t     recent_count_ = 0;
    size_t     recent_next_ = 0;
};

} // namespace oss_gaussian::capture
#endif

#endif // OSS_CAPTURE_H
