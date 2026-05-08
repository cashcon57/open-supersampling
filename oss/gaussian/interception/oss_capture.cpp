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

const char* mode_name(OssCaptureMode mode) {
    switch (mode) {
    case OSS_CAPTURE_MODE_TRICKLE:
        return "trickle";
    case OSS_CAPTURE_MODE_LITE:
        return "lite";
    case OSS_CAPTURE_MODE_REGULAR:
        return "regular";
    case OSS_CAPTURE_MODE_INSANE:
        return "INSANE";
    default:
        return "lite";
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

void set_mode_name(char out[8], OssCaptureMode mode) {
    std::strncpy(out, mode_name(mode), 7u);
    out[7] = '\0';
}

void populate_mode(OssCaptureDecision& decision, const OssCaptureConfig& config) {
    decision.capture_mode = config.mode;
    set_mode_name(decision.capture_mode_name, config.mode);
    decision.supersample_gt =
        (config.mode == OSS_CAPTURE_MODE_INSANE && config.enable_supersample_gt != 0) ? 1u : 0u;
}

OssCaptureDecision accept_static_single(const OssCaptureConfig& config) {
    OssCaptureDecision decision{};
    decision.capture = 1u;
    decision.rule = OSS_CAPTURE_RULE_ACCEPT;
    decision.burst_tier = OSS_CAPTURE_TIER_NONE;
    set_tier_name(decision.burst_tier_name, OSS_CAPTURE_TIER_NONE);
    decision.burst_n = 1u;
    decision.capture_hr = static_cast<uint32_t>(config.capture_hr_on_t0 != 0);
    decision.capture_hr_on_t0 = decision.capture_hr;
    decision.capture_hr_on_tplus = decision.capture_hr;
    populate_mode(decision, config);
    return decision;
}

OssCaptureDecision accept_burst(const OssCaptureConfig& config, OssCaptureBurstTier tier) {
    OssCaptureDecision decision{};
    decision.capture = 1u;
    decision.rule = OSS_CAPTURE_RULE_ACCEPT;
    decision.burst_tier = tier;
    set_tier_name(decision.burst_tier_name, tier);
    decision.burst_n =
        (tier == OSS_CAPTURE_TIER_LONG) ? long_burst_n(config) : short_burst_n(config);
    if (tier == OSS_CAPTURE_TIER_LONG) {
        decision.capture_hr_on_t0 = static_cast<uint32_t>(config.long_capture_hr != 0);
        decision.capture_hr_on_tplus = static_cast<uint32_t>(config.long_capture_hr != 0);
    } else {
        decision.capture_hr_on_t0 = static_cast<uint32_t>(config.capture_hr_on_t0 != 0);
        decision.capture_hr_on_tplus = static_cast<uint32_t>(config.capture_hr_on_tplus != 0);
    }
    decision.capture_hr = decision.capture_hr_on_t0;
    populate_mode(decision, config);
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
    // T2.6: unpack DLSS parameters into LR/depth/motion/normals, compute
    // candidate stats, run CaptureSampler, arm a burst. Present consumes
    // N consecutive swap-chain frames after ACCEPT. Rejects never touch disk.
    //
    // Implementation outline (pending NVIDIA/DLSS SDK headers vendored under
    // third_party/DLSS/ — license review required before redistribution):
    //
    //   // Cast to the typed parameter dictionary
    //   const NVSDK_NGX_Parameter* p =
    //       reinterpret_cast<const NVSDK_NGX_Parameter*>(ngx_params);
    //
    //   // Resource pointers (ID3D12Resource*) and dimensions
    //   ID3D12Resource* color   = nullptr;  p->Get(NVSDK_NGX_Parameter_Color,         (void**)&color);
    //   ID3D12Resource* output  = nullptr;  p->Get(NVSDK_NGX_Parameter_Output,        (void**)&output);
    //   ID3D12Resource* depth   = nullptr;  p->Get(NVSDK_NGX_Parameter_Depth,         (void**)&depth);
    //   ID3D12Resource* motion  = nullptr;  p->Get(NVSDK_NGX_Parameter_MotionVectors, (void**)&motion);
    //   uint32_t lr_w = 0, lr_h = 0;
    //   p->Get(NVSDK_NGX_Parameter_Width,  &lr_w);
    //   p->Get(NVSDK_NGX_Parameter_Height, &lr_h);
    //   float jitter_x = 0.0f, jitter_y = 0.0f;
    //   p->Get(NVSDK_NGX_Parameter_Jitter_Offset_X, &jitter_x);
    //   p->Get(NVSDK_NGX_Parameter_Jitter_Offset_Y, &jitter_y);
    //
    //   // Build the candidate
    //   OssCaptureCandidate cand{};
    //   cand.frame_index = oss_gaussian::CurrentFrameIndex();
    //   cand.timestamp_seconds = SecondsSinceStart();
    //   // motion_mean_magnitude_px requires reading 'motion' contents — use
    //   // the staging-copy path with a small downscaled snapshot
    //   // perceptual_hash_64 = oss_capture_phash64_rgb8(...) — same approach
    //   // depth_degenerate = check first-pixel depth == 1.0 (cleared)
    //
    //   OssCaptureDecision decision = oss_capture_consider_candidate(&cand);
    //   if (decision.capture) {
    //       // Schedule readback of color + depth + motion via staging_copy.cpp,
    //       // then invoke oss_capture_write_exr() with the OssCaptureFramePayload
    //   }
    //
    // Why not implement directly: the NVSDK_NGX_Parameter virtual interface
    // is defined in nvsdk_ngx.h which is licensed redistribution. The DLSS
    // SDK is at github.com/NVIDIA/DLSS but redistribution requires license
    // attestation. Vendoring + license review = separate sprint task.
    //
    // For now, log the call so we can verify the hook fires when DLSS-enabled
    // games run with our DLL injected.
    OSSG_LOG_INFO("capture",
                  "NGX EvaluateFeature observed (cmd_list=%p, handle=%p, params=%p) — "
                  "param-dict unpack pending DLSS SDK header vendoring (T2.6)",
                  command_list, ngx_handle, ngx_params);
}

} // namespace

CaptureSampler::CaptureSampler(const OssCaptureConfig& config) : config_(config) {}

void CaptureSampler::Reset() {
    last_short_event_time_ = -1.0e30;
    last_long_event_time_ = -1.0e30;
    last_static_single_time_ = -1.0e30;
    last_static_candidate_time_ = -1.0e30;
    last_opportunistic_pair_time_ = -1.0e30;
    std::fill(std::begin(motion_buckets_), std::end(motion_buckets_), 0u);
    std::fill(std::begin(recent_hashes_), std::end(recent_hashes_), RecentHash{});
    recent_count_ = 0;
    recent_next_ = 0;
}

OssCaptureDecision CaptureSampler::Consider(const OssCaptureCandidate& candidate) {
    const bool is_static =
        config_.enable_static_frame_trigger != 0 &&
        candidate.motion_mean_magnitude_px < static_cast<float>(config_.static_motion_threshold_px) &&
        candidate.motion_below_threshold_seconds >= config_.static_dwell_seconds;
    const bool moving_after_static =
        config_.enable_opportunistic_pair != 0 &&
        candidate.motion_mean_magnitude_px >= static_cast<float>(config_.static_motion_threshold_px) &&
        candidate.timestamp_seconds - last_static_candidate_time_ <= config_.opportunistic_pair_motion_window_s &&
        candidate.timestamp_seconds - last_opportunistic_pair_time_ >=
            static_cast<double>(config_.opportunistic_pair_min_period_s);

    OssCaptureBurstTier tier = OSS_CAPTURE_TIER_NONE;
    bool static_single = false;

    // 1. Mode-specific stride gates. Long takes priority when both burst
    // windows are open. Static singles are independent non-burst captures.
    if (moving_after_static) {
        tier = OSS_CAPTURE_TIER_SHORT;
    } else if (config_.mode != OSS_CAPTURE_MODE_TRICKLE &&
               config_.two_tier_enabled &&
               candidate.timestamp_seconds - last_long_event_time_ >= long_stride_seconds(config_)) {
        tier = OSS_CAPTURE_TIER_LONG;
    } else if (config_.mode != OSS_CAPTURE_MODE_TRICKLE &&
               candidate.timestamp_seconds - last_short_event_time_ >= short_stride_seconds(config_)) {
        tier = OSS_CAPTURE_TIER_SHORT;
    } else if (is_static &&
               candidate.timestamp_seconds - last_static_single_time_ >=
                   static_cast<double>(config_.static_min_period_seconds)) {
        static_single = true;
    } else {
        if (is_static) {
            last_static_candidate_time_ = candidate.timestamp_seconds;
        }
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
    if (static_single) {
        last_static_single_time_ = candidate.timestamp_seconds;
        last_static_candidate_time_ = candidate.timestamp_seconds;
    } else if (moving_after_static) {
        last_short_event_time_ = candidate.timestamp_seconds;
        last_opportunistic_pair_time_ = candidate.timestamp_seconds;
    } else if (tier == OSS_CAPTURE_TIER_LONG) {
        last_long_event_time_ = candidate.timestamp_seconds;
    } else {
        last_short_event_time_ = candidate.timestamp_seconds;
    }
    recent_hashes_[recent_next_] = RecentHash{candidate.perceptual_hash_64, candidate.timestamp_seconds};
    recent_next_ = (recent_next_ + 1u) % std::size(recent_hashes_);
    recent_count_ = std::min<size_t>(recent_count_ + 1u, std::size(recent_hashes_));
    return static_single ? accept_static_single(config_) : accept_burst(config_, tier);
}

} // namespace oss_gaussian::capture

extern "C" {

OssCaptureConfig oss_capture_default_config(void) {
    OssCaptureConfig cfg{};
    std::strncpy(cfg.game_id, "unknown-game", sizeof(cfg.game_id) - 1u);
    std::strncpy(cfg.game_version, "unknown", sizeof(cfg.game_version) - 1u);
    oss_capture_apply_mode_preset(&cfg, OSS_CAPTURE_MODE_LITE);
    cfg.dedup_window_seconds = 300.0;
    cfg.loading_gap_seconds = 30.0;
    cfg.max_motion_bucket_samples = 24u;
    cfg.dedup_hamming_threshold = 5u;
    return cfg;
}

int oss_capture_apply_mode_preset(OssCaptureConfig* config, OssCaptureMode mode) {
    if (!config) {
        return 0;
    }
    config->mode = mode;
    config->capture_lr = 1;
    config->capture_depth = 1;
    config->capture_motion = 1;
    config->capture_normals = 1;
    config->capture_albedo = 0;
    config->capture_roughness = 0;
    config->capture_metallic = 0;
    config->capture_emissive = 0;
    config->fp32_depth_motion = 0;
    config->enable_supersample_gt = 0;
    config->enable_dlaa_capture = 0;
    config->enable_multi_dlss_mode = 0;
    config->enable_scene_cut_burst = 0;
    config->scene_cut_burst_n = 0;
    config->static_motion_threshold_px = 0.5;
    config->static_dwell_seconds = 1.5;
    config->enable_opportunistic_pair = 0;
    config->opportunistic_pair_motion_window_s = 5.0;
    config->opportunistic_pair_min_period_s = 1200;
    config->long_capture_hr = 0;

    switch (mode) {
    case OSS_CAPTURE_MODE_TRICKLE:
        config->capture_stride_seconds = 300.0;
        config->burst_n = 2u;
        config->stride_seconds = 300.0;
        config->short_burst_n = 2;
        config->short_stride_seconds = 300.0;
        config->long_burst_n = 0;
        config->long_stride_seconds = 1800.0;
        config->two_tier_enabled = 0;
        config->capture_hr_on_t0 = 1;
        config->capture_hr_on_tplus = 0;
        config->enable_static_frame_trigger = 1;
        config->static_min_period_seconds = 300;
        config->enable_opportunistic_pair = 1;
        config->dedup_window_seconds = 1800.0;
        config->dedup_hamming_threshold = 10u;
        break;
    case OSS_CAPTURE_MODE_REGULAR:
        config->capture_stride_seconds = 40.0;
        config->burst_n = 4u;
        config->stride_seconds = 40.0;
        config->short_burst_n = 4;
        config->short_stride_seconds = 40.0;
        config->long_burst_n = 60;
        config->long_stride_seconds = 600.0;
        config->two_tier_enabled = 1;
        config->capture_hr_on_t0 = 1;
        config->capture_hr_on_tplus = 1;
        config->capture_albedo = 1;
        config->capture_roughness = 1;
        config->enable_static_frame_trigger = 1;
        config->static_min_period_seconds = 600;
        break;
    case OSS_CAPTURE_MODE_INSANE:
        config->capture_stride_seconds = 40.0;
        config->burst_n = 4u;
        config->stride_seconds = 40.0;
        config->short_burst_n = 4;
        config->short_stride_seconds = 40.0;
        config->long_burst_n = 240;
        config->long_stride_seconds = 300.0;
        config->two_tier_enabled = 1;
        config->capture_hr_on_t0 = 1;
        config->capture_hr_on_tplus = 1;
        config->capture_albedo = 1;
        config->capture_roughness = 1;
        config->capture_metallic = 1;
        config->capture_emissive = 1;
        config->fp32_depth_motion = 1;
        config->enable_supersample_gt = 1;
        config->enable_dlaa_capture = 1;
        config->enable_multi_dlss_mode = 1;
        config->enable_scene_cut_burst = 1;
        config->scene_cut_burst_n = 8;
        config->enable_static_frame_trigger = 1;
        config->static_min_period_seconds = 300;
        break;
    case OSS_CAPTURE_MODE_LITE:
    default:
        config->mode = OSS_CAPTURE_MODE_LITE;
        config->capture_stride_seconds = 80.0;
        config->burst_n = 2u;
        config->stride_seconds = 80.0;
        config->short_burst_n = 2;
        config->short_stride_seconds = 80.0;
        config->long_burst_n = 60;
        config->long_stride_seconds = 1800.0;
        config->two_tier_enabled = 1;
        config->capture_hr_on_t0 = 1;
        config->capture_hr_on_tplus = 1;
        config->enable_static_frame_trigger = 1;
        config->static_min_period_seconds = 600;
        break;
    }
    return 1;
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
        "capture-mode configured game_id=%s mode=%s short=(n=%u stride=%.2fs) long=(enabled=%d n=%u stride=%.2fs capture_hr=%d)",
        config->game_id,
        oss_gaussian::capture::mode_name(config->mode),
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
    if (decision.capture && decision.burst_tier != OSS_CAPTURE_TIER_NONE) {
        std::lock_guard<std::mutex> burst_lk(oss_gaussian::capture::g_burst_mu);
        oss_gaussian::capture::g_active_burst = OssCaptureBurstFrame{};
        oss_gaussian::capture::g_active_burst.active = 1u;
        oss_gaussian::capture::g_active_burst.burst_index = 0u;
        oss_gaussian::capture::g_active_burst.burst_n = decision.burst_n;
        oss_gaussian::capture::g_active_burst.burst_tier = decision.burst_tier;
        oss_gaussian::capture::g_active_burst.capture_hr = decision.capture_hr_on_t0;
        oss_gaussian::capture::g_active_burst.capture_hr_on_t0 = decision.capture_hr_on_t0;
        oss_gaussian::capture::g_active_burst.capture_hr_on_tplus = decision.capture_hr_on_tplus;
        oss_gaussian::capture::g_active_burst.capture_mode = decision.capture_mode;
        oss_gaussian::capture::g_active_burst.supersample_gt = decision.supersample_gt;
        std::strncpy(
            oss_gaussian::capture::g_active_burst.burst_tier_name,
            decision.burst_tier_name,
            sizeof(oss_gaussian::capture::g_active_burst.burst_tier_name) - 1u);
        std::strncpy(
            oss_gaussian::capture::g_active_burst.capture_mode_name,
            decision.capture_mode_name,
            sizeof(oss_gaussian::capture::g_active_burst.capture_mode_name) - 1u);
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
        out->capture_hr =
            (out->burst_index == 0u)
                ? oss_gaussian::capture::g_active_burst.capture_hr_on_t0
                : oss_gaussian::capture::g_active_burst.capture_hr_on_tplus;
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
