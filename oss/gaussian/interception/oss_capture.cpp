#define OSS_GAUSSIAN_BUILDING_DLL 1

#include "oss_capture.h"

#include "src/log.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <iterator>
#include <mutex>

#if defined(_WIN32)
#  include <Windows.h>
#endif

namespace oss_gaussian::capture {
namespace {

std::mutex g_sampler_mu;
CaptureSampler g_sampler(oss_capture_default_config());
std::atomic<bool> g_enabled{false};

uint32_t motion_bucket(float magnitude_px) {
    if (magnitude_px < 0.5f) return 0;
    if (magnitude_px < 1.0f) return 1;
    if (magnitude_px < 2.0f) return 2;
    if (magnitude_px < 4.0f) return 3;
    if (magnitude_px < 8.0f) return 4;
    if (magnitude_px < 16.0f) return 5;
    if (magnitude_px < 32.0f) return 6;
    return 7;
}

OssCaptureDecision reject(OssCaptureDecisionRule rule) {
    return OssCaptureDecision{0u, rule};
}

OssCaptureDecision accept() {
    return OssCaptureDecision{1u, OSS_CAPTURE_RULE_ACCEPT};
}

void safe_log_exception(const char* hook) {
    OSSG_LOG_ERROR("capture", "capture hook swallowed exception in %s", hook);
}

void on_present_impl(void* swap_chain) {
    (void)swap_chain;
    // Integration point: capture-mode Present hook records the HR backbuffer
    // resource descriptor and lets the original Present proceed immediately.
    OSSG_LOG_TRACE("capture", "Present observed for capture-mode backbuffer");
}

void on_execute_command_lists_impl(void* command_queue) {
    (void)command_queue;
    // Integration point: retain the queue used for async texture copy/readback.
    OSSG_LOG_TRACE("capture", "ExecuteCommandLists observed for capture-mode queue");
}

void on_ngx_evaluate_feature_impl(void* command_list, const void* ngx_handle, const void* ngx_params) {
    (void)command_list;
    (void)ngx_handle;
    (void)ngx_params;
    // Integration point: unpack DLSS parameters into LR/depth/motion/normals,
    // compute candidate stats, run CaptureSampler, and only enqueue a readback
    // + EXR write after an ACCEPT decision. Rejects never touch disk.
    OSSG_LOG_TRACE("capture", "NGX EvaluateFeature observed for capture-mode candidate");
}

} // namespace

CaptureSampler::CaptureSampler(const OssCaptureConfig& config) : config_(config) {}

void CaptureSampler::Reset() {
    last_accept_time_ = -1.0e30;
    std::fill(std::begin(motion_buckets_), std::end(motion_buckets_), 0u);
    std::fill(std::begin(recent_hashes_), std::end(recent_hashes_), RecentHash{});
    recent_count_ = 0;
    recent_next_ = 0;
}

OssCaptureDecision CaptureSampler::Consider(const OssCaptureCandidate& candidate) {
    // 1. Temporal stride: cap candidates before any expensive work.
    if (candidate.seconds_since_last_candidate < config_.capture_stride_seconds) {
        return reject(OSS_CAPTURE_RULE_TEMPORAL_STRIDE);
    }

    // 2. Motion bucket: keep session motion distribution from collapsing to one
    // repeated bucket such as static menus or idle camera holds.
    const uint32_t bucket = motion_bucket(candidate.motion_mean_magnitude_px);
    if (motion_buckets_[bucket] >= config_.max_motion_bucket_samples) {
        return reject(OSS_CAPTURE_RULE_MOTION_BUCKET);
    }

    // 3. Perceptual dedup: compare against recent accepted hashes only.
    for (size_t i = 0; i < recent_count_; ++i) {
        const RecentHash& recent = recent_hashes_[i];
        if (candidate.timestamp_seconds - recent.timestamp > config_.dedup_window_seconds) {
            continue;
        }
        if (oss_capture_hamming_distance64(candidate.perceptual_hash_64, recent.hash) <
            config_.dedup_hamming_threshold) {
            return reject(OSS_CAPTURE_RULE_PERCEPTUAL_DEDUP);
        }
    }

    // 4. G-buffer sanity: bad depth/MV/format candidates are freed in memory and
    // never reach the writer.
    if (candidate.depth_degenerate || candidate.motion_vectors_nan || candidate.unsupported_rt_format) {
        return reject(OSS_CAPTURE_RULE_GBUFFER_SANITY);
    }

    // 5. First-frame-after-loading-screen guard: skip a candidate after a long
    // silence, which usually means a loading transition or menu.
    if (candidate.seconds_since_previous_candidate > config_.loading_gap_seconds) {
        return reject(OSS_CAPTURE_RULE_POST_LOADING_GUARD);
    }

    ++motion_buckets_[bucket];
    last_accept_time_ = candidate.timestamp_seconds;
    recent_hashes_[recent_next_] = RecentHash{candidate.perceptual_hash_64, candidate.timestamp_seconds};
    recent_next_ = (recent_next_ + 1u) % std::size(recent_hashes_);
    recent_count_ = std::min<size_t>(recent_count_ + 1u, std::size(recent_hashes_));
    return accept();
}

} // namespace oss_gaussian::capture

extern "C" {

OssCaptureConfig oss_capture_default_config(void) {
    OssCaptureConfig cfg{};
    std::strncpy(cfg.game_id, "unknown-game", sizeof(cfg.game_id) - 1u);
    std::strncpy(cfg.game_version, "unknown", sizeof(cfg.game_version) - 1u);
    cfg.capture_stride_seconds = 20.0;
    cfg.dedup_window_seconds = 300.0;
    cfg.loading_gap_seconds = 30.0;
    cfg.max_motion_bucket_samples = 24u;
    cfg.dedup_hamming_threshold = 5u;
    return cfg;
}

int oss_capture_configure(const OssCaptureConfig* config) {
    if (!config || config->capture_stride_seconds <= 0.0 || config->dedup_hamming_threshold == 0u) {
        return 0;
    }
    std::lock_guard<std::mutex> lk(oss_gaussian::capture::g_sampler_mu);
    oss_gaussian::capture::g_sampler = oss_gaussian::capture::CaptureSampler(*config);
    oss_gaussian::capture::g_enabled.store(true, std::memory_order_release);
    OSSG_LOG_INFO("capture", "capture-mode configured game_id=%s stride=%.2fs", config->game_id, config->capture_stride_seconds);
    return 1;
}

OssCaptureDecision oss_capture_consider_candidate(const OssCaptureCandidate* candidate) {
    if (!candidate) {
        return OssCaptureDecision{0u, OSS_CAPTURE_RULE_GBUFFER_SANITY};
    }
    std::lock_guard<std::mutex> lk(oss_gaussian::capture::g_sampler_mu);
    return oss_gaussian::capture::g_sampler.Consider(*candidate);
}

void oss_capture_on_present(void* swap_chain) {
#if defined(_MSC_VER)
    __try {
        oss_gaussian::capture::on_present_impl(swap_chain);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        oss_gaussian::capture::safe_log_exception("Present");
    }
#else
    try {
        oss_gaussian::capture::on_present_impl(swap_chain);
    } catch (...) {
        oss_gaussian::capture::safe_log_exception("Present");
    }
#endif
}

void oss_capture_on_execute_command_lists(void* command_queue) {
#if defined(_MSC_VER)
    __try {
        oss_gaussian::capture::on_execute_command_lists_impl(command_queue);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        oss_gaussian::capture::safe_log_exception("ExecuteCommandLists");
    }
#else
    try {
        oss_gaussian::capture::on_execute_command_lists_impl(command_queue);
    } catch (...) {
        oss_gaussian::capture::safe_log_exception("ExecuteCommandLists");
    }
#endif
}

void oss_capture_on_ngx_evaluate_feature(void* command_list, const void* ngx_handle, const void* ngx_params) {
#if defined(_MSC_VER)
    __try {
        oss_gaussian::capture::on_ngx_evaluate_feature_impl(command_list, ngx_handle, ngx_params);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        oss_gaussian::capture::safe_log_exception("NVSDK_NGX_D3D12_EvaluateFeature");
    }
#else
    try {
        oss_gaussian::capture::on_ngx_evaluate_feature_impl(command_list, ngx_handle, ngx_params);
    } catch (...) {
        oss_gaussian::capture::safe_log_exception("NVSDK_NGX_D3D12_EvaluateFeature");
    }
#endif
}

} // extern "C"
