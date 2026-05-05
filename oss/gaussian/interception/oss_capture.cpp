#define OSS_GAUSSIAN_BUILDING_DLL 1

#include "oss_capture.h"

#include "src/log.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <iterator>
#include <mutex>
#include <cstdio>

#if defined(_WIN32)
#  include <Windows.h>
#endif

namespace oss_gaussian::capture {
namespace {

std::mutex g_sampler_mu;
CaptureSampler g_sampler(oss_capture_default_config());
std::atomic<bool> g_enabled{false};

std::mutex g_burst_mu;
OssCaptureBurstFrame g_active_burst{};
std::atomic<uint64_t> g_burst_counter{1};

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

double stride_seconds(const OssCaptureConfig& config) {
    if (config.stride_seconds > 0.0) {
        return config.stride_seconds;
    }
    return config.capture_stride_seconds;
}

double short_stride_seconds(const OssCaptureConfig& config) {
    if (config.stride_seconds > 0.0 &&
        config.stride_seconds != 80.0 &&
        config.short_stride_seconds == 80.0) {
        return config.stride_seconds;
    }
    if (config.short_stride_seconds > 0.0) {
        return config.short_stride_seconds;
    }
    return stride_seconds(config);
}

double long_stride_seconds(const OssCaptureConfig& config) {
    return config.long_stride_seconds > 0.0 ? config.long_stride_seconds : 1800.0;
}

uint32_t short_burst_n(const OssCaptureConfig& config) {
    if (config.burst_n > 0u &&
        config.burst_n != 2u &&
        config.short_burst_n == 2) {
        return config.burst_n;
    }
    if (config.short_burst_n > 0) {
        return static_cast<uint32_t>(config.short_burst_n);
    }
    return std::max<uint32_t>(config.burst_n, 1u);
}

uint32_t long_burst_n(const OssCaptureConfig& config) {
    return config.long_burst_n > 0 ? static_cast<uint32_t>(config.long_burst_n) : 60u;
}

const char* tier_name(OssCaptureBurstTier tier) {
    switch (tier) {
    case OSS_CAPTURE_TIER_SHORT:
        return "short";
    case OSS_CAPTURE_TIER_LONG:
        return "long";
    case OSS_CAPTURE_TIER_NONE:
    default:
        return "";
    }
}

void make_burst_uuid(char out[37]) {
    const uint64_t id = g_burst_counter.fetch_add(1, std::memory_order_relaxed);
    std::snprintf(
        out,
        37,
        "%08llx-%04llx-%04llx-%04llx-%012llx",
        static_cast<unsigned long long>((id >> 32) & 0xffffffffull),
        static_cast<unsigned long long>((id >> 16) & 0xffffull),
        static_cast<unsigned long long>(id & 0xffffull),
        static_cast<unsigned long long>((id ^ 0x4f535343ull) & 0xffffull),
        static_cast<unsigned long long>(id & 0xffffffffffffull));
}

OssCaptureDecision reject(OssCaptureDecisionRule rule) {
    OssCaptureDecision decision{};
    decision.capture = 0u;
    decision.rule = rule;
    return decision;
}

void set_tier_name(char out[8], OssCaptureBurstTier tier) {
    std::strncpy(out, tier_name(tier), 7u);
    out[7] = '\0';
}

OssCaptureDecision accept(const OssCaptureConfig& config, OssCaptureBurstTier tier) {
    OssCaptureDecision decision{};
    decision.capture = 1u;
    decision.rule = OSS_CAPTURE_RULE_ACCEPT;
    decision.burst_tier = tier;
    set_tier_name(decision.burst_tier_name, tier);
    decision.burst_n =
        (tier == OSS_CAPTURE_TIER_LONG) ? long_burst_n(config) : short_burst_n(config);
    decision.capture_hr =
        (tier == OSS_CAPTURE_TIER_LONG) ? static_cast<uint32_t>(config.long_capture_hr != 0) : 1u;
    make_burst_uuid(decision.burst_uuid);
    return decision;
}

void safe_log_exception(const char* hook) {
    OSSG_LOG_ERROR("capture", "capture hook swallowed exception in %s", hook);
}

void on_present_impl(void* swap_chain) {
    (void)swap_chain;
    OssCaptureBurstFrame burst_frame{};
    if (oss_capture_consume_present_burst(&burst_frame)) {
        // Integration point: capture-mode Present hook records this HR
        // backbuffer as burst_index within burst_uuid and enqueues async
        // CPU readback/write. The original Present still proceeds immediately.
        OSSG_LOG_TRACE(
            "capture",
            "enqueue %s burst frame uuid=%s index=%u/%u capture_hr=%u",
            burst_frame.burst_tier_name,
            burst_frame.burst_uuid,
            burst_frame.burst_index,
            burst_frame.burst_n,
            burst_frame.capture_hr);
        return;
    }
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
    // compute candidate stats, run CaptureSampler, and arm a burst. Present
    // consumes N consecutive swap-chain frames after ACCEPT. Rejects never touch
    // disk.
    OSSG_LOG_TRACE("capture", "NGX EvaluateFeature observed for capture-mode candidate");
}

} // namespace

CaptureSampler::CaptureSampler(const OssCaptureConfig& config) : config_(config) {}

void CaptureSampler::Reset() {
    last_short_event_time_ = -1.0e30;
    last_long_event_time_ = -1.0e30;
    std::fill(std::begin(motion_buckets_), std::end(motion_buckets_), 0u);
    std::fill(std::begin(recent_hashes_), std::end(recent_hashes_), RecentHash{});
    recent_count_ = 0;
    recent_next_ = 0;
}

OssCaptureDecision CaptureSampler::Consider(const OssCaptureCandidate& candidate) {
    // 1. Tier stride gates. Long takes priority when both windows are open.
    OssCaptureBurstTier tier = OSS_CAPTURE_TIER_NONE;
    if (config_.two_tier_enabled &&
        candidate.timestamp_seconds - last_long_event_time_ >= long_stride_seconds(config_)) {
        tier = OSS_CAPTURE_TIER_LONG;
    } else if (candidate.timestamp_seconds - last_short_event_time_ >= short_stride_seconds(config_)) {
        tier = OSS_CAPTURE_TIER_SHORT;
    }
    if (tier == OSS_CAPTURE_TIER_NONE) {
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
    if (tier == OSS_CAPTURE_TIER_LONG) {
        last_long_event_time_ = candidate.timestamp_seconds;
    } else {
        last_short_event_time_ = candidate.timestamp_seconds;
    }
    recent_hashes_[recent_next_] = RecentHash{candidate.perceptual_hash_64, candidate.timestamp_seconds};
    recent_next_ = (recent_next_ + 1u) % std::size(recent_hashes_);
    recent_count_ = std::min<size_t>(recent_count_ + 1u, std::size(recent_hashes_));
    return accept(config_, tier);
}

} // namespace oss_gaussian::capture

extern "C" {

OssCaptureConfig oss_capture_default_config(void) {
    OssCaptureConfig cfg{};
    std::strncpy(cfg.game_id, "unknown-game", sizeof(cfg.game_id) - 1u);
    std::strncpy(cfg.game_version, "unknown", sizeof(cfg.game_version) - 1u);
    cfg.capture_stride_seconds = 80.0;
    cfg.burst_n = 2u;
    cfg.stride_seconds = 80.0;
    cfg.short_burst_n = 2;
    cfg.short_stride_seconds = 80.0;
    cfg.long_burst_n = 60;
    cfg.long_stride_seconds = 1800.0;
    cfg.long_capture_hr = 0;
    cfg.two_tier_enabled = 0;
    cfg.dedup_window_seconds = 300.0;
    cfg.loading_gap_seconds = 30.0;
    cfg.max_motion_bucket_samples = 24u;
    cfg.dedup_hamming_threshold = 5u;
    return cfg;
}

int oss_capture_configure(const OssCaptureConfig* config) {
    const double config_stride =
        config ? oss_gaussian::capture::short_stride_seconds(*config) : 0.0;
    if (!config ||
        config_stride <= 0.0 ||
        oss_gaussian::capture::short_burst_n(*config) == 0u ||
        oss_gaussian::capture::long_burst_n(*config) == 0u ||
        oss_gaussian::capture::long_stride_seconds(*config) <= 0.0 ||
        config->dedup_hamming_threshold == 0u) {
        return 0;
    }
    std::lock_guard<std::mutex> lk(oss_gaussian::capture::g_sampler_mu);
    oss_gaussian::capture::g_sampler = oss_gaussian::capture::CaptureSampler(*config);
    oss_gaussian::capture::g_enabled.store(true, std::memory_order_release);
    OSSG_LOG_INFO(
        "capture",
        "capture-mode configured game_id=%s short=(n=%u stride=%.2fs) long=(enabled=%d n=%u stride=%.2fs capture_hr=%d)",
        config->game_id,
        oss_gaussian::capture::short_burst_n(*config),
        oss_gaussian::capture::short_stride_seconds(*config),
        config->two_tier_enabled,
        oss_gaussian::capture::long_burst_n(*config),
        oss_gaussian::capture::long_stride_seconds(*config),
        config->long_capture_hr);
    return 1;
}

OssCaptureDecision oss_capture_consider_candidate(const OssCaptureCandidate* candidate) {
    if (!candidate) {
        return OssCaptureDecision{0u, OSS_CAPTURE_RULE_GBUFFER_SANITY};
    }
    std::lock_guard<std::mutex> lk(oss_gaussian::capture::g_sampler_mu);
    OssCaptureDecision decision = oss_gaussian::capture::g_sampler.Consider(*candidate);
    if (decision.capture) {
        std::lock_guard<std::mutex> burst_lk(oss_gaussian::capture::g_burst_mu);
        oss_gaussian::capture::g_active_burst = OssCaptureBurstFrame{};
        oss_gaussian::capture::g_active_burst.active = 1u;
        oss_gaussian::capture::g_active_burst.burst_index = 0u;
        oss_gaussian::capture::g_active_burst.burst_n = decision.burst_n;
        oss_gaussian::capture::g_active_burst.burst_tier = decision.burst_tier;
        oss_gaussian::capture::g_active_burst.capture_hr = decision.capture_hr;
        std::strncpy(
            oss_gaussian::capture::g_active_burst.burst_tier_name,
            decision.burst_tier_name,
            sizeof(oss_gaussian::capture::g_active_burst.burst_tier_name) - 1u);
        std::strncpy(
            oss_gaussian::capture::g_active_burst.burst_uuid,
            decision.burst_uuid,
            sizeof(oss_gaussian::capture::g_active_burst.burst_uuid) - 1u);
    }
    return decision;
}

uint32_t oss_capture_consume_present_burst(OssCaptureBurstFrame* out) {
    std::lock_guard<std::mutex> lk(oss_gaussian::capture::g_burst_mu);
    if (!oss_gaussian::capture::g_active_burst.active ||
        oss_gaussian::capture::g_active_burst.burst_index >= oss_gaussian::capture::g_active_burst.burst_n) {
        if (out) {
            *out = OssCaptureBurstFrame{};
        }
        return 0u;
    }

    if (out) {
        *out = oss_gaussian::capture::g_active_burst;
    }
    ++oss_gaussian::capture::g_active_burst.burst_index;
    if (oss_gaussian::capture::g_active_burst.burst_index >= oss_gaussian::capture::g_active_burst.burst_n) {
        oss_gaussian::capture::g_active_burst.active = 0u;
    }
    return 1u;
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
