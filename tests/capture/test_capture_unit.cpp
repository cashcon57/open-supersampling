#include "../../oss/gaussian/interception/oss_capture.h"

#include <cassert>
#include <cstdint>
#include <vector>

static OssCaptureCandidate candidate(double t) {
    OssCaptureCandidate c{};
    c.frame_index = static_cast<uint64_t>(t);
    c.timestamp_seconds = t;
    c.seconds_since_last_candidate = 80.0;
    c.seconds_since_previous_candidate = 20.0;
    c.motion_mean_magnitude_px = 3.0f;
    c.perceptual_hash_64 = 0x123456789abcdef0ull ^ static_cast<uint64_t>(t);
    return c;
}

int main() {
    OssCaptureConfig cfg = oss_capture_default_config();
    assert(cfg.mode == OSS_CAPTURE_MODE_LITE);
    assert(cfg.short_burst_n == 2);
    assert(cfg.short_stride_seconds == 80.0);
    assert(cfg.long_burst_n == 60);
    assert(cfg.long_stride_seconds == 1800.0);
    assert(cfg.capture_depth == 1);
    assert(cfg.capture_motion == 1);
    assert(cfg.capture_normals == 1);

    assert(oss_capture_apply_mode_preset(&cfg, OSS_CAPTURE_MODE_TRICKLE) == 1);
    assert(cfg.capture_hr_on_t0 == 1);
    assert(cfg.capture_hr_on_tplus == 0);
    assert(cfg.enable_static_frame_trigger == 1);
    assert(cfg.enable_opportunistic_pair == 1);
    assert(cfg.capture_depth == 1);
    assert(cfg.capture_motion == 1);
    assert(cfg.capture_normals == 1);

    assert(oss_capture_apply_mode_preset(&cfg, OSS_CAPTURE_MODE_REGULAR) == 1);
    assert(cfg.short_burst_n == 4);
    assert(cfg.short_stride_seconds == 40.0);
    assert(cfg.long_stride_seconds == 600.0);
    assert(cfg.capture_albedo == 1);
    assert(cfg.capture_roughness == 1);

    assert(oss_capture_apply_mode_preset(&cfg, OSS_CAPTURE_MODE_INSANE) == 1);
    assert(cfg.long_burst_n == 240);
    assert(cfg.long_stride_seconds == 300.0);
    assert(cfg.capture_metallic == 1);
    assert(cfg.capture_emissive == 1);
    assert(cfg.fp32_depth_motion == 1);
    assert(cfg.enable_supersample_gt == 1);

    cfg = oss_capture_default_config();
    cfg.two_tier_enabled = 0;
    cfg.stride_seconds = 80.0;
    cfg.burst_n = 4;
    cfg.short_burst_n = 4;
    cfg.short_stride_seconds = 80.0;
    cfg.enable_static_frame_trigger = 0;
    cfg.max_motion_bucket_samples = 1;

    oss_gaussian::capture::CaptureSampler sampler(cfg);

    assert(sampler.Consider(candidate(0.0)).capture == 1);
    OssCaptureCandidate early = candidate(2.0);
    assert(sampler.Consider(early).rule == OSS_CAPTURE_RULE_TEMPORAL_STRIDE);

    oss_gaussian::capture::CaptureSampler bad_sampler(cfg);
    OssCaptureCandidate bad_gbuf = candidate(100.0);
    bad_gbuf.depth_degenerate = 1;
    assert(bad_sampler.Consider(bad_gbuf).rule == OSS_CAPTURE_RULE_GBUFFER_SANITY);

    oss_gaussian::capture::CaptureSampler loading_sampler(cfg);
    OssCaptureCandidate loading = candidate(100.0);
    loading.seconds_since_previous_candidate = 31.0;
    assert(loading_sampler.Consider(loading).rule == OSS_CAPTURE_RULE_POST_LOADING_GUARD);

    oss_gaussian::capture::CaptureSampler accept_sampler(cfg);
    OssCaptureCandidate accepted = candidate(100.0);
    OssCaptureDecision accepted_decision = accept_sampler.Consider(accepted);
    assert(accepted_decision.capture == 1);
    assert(accepted_decision.burst_n == 4);
    assert(accepted_decision.burst_tier == OSS_CAPTURE_TIER_SHORT);
    assert(accepted_decision.capture_hr == 1);
    assert(accepted_decision.capture_mode == OSS_CAPTURE_MODE_LITE);
    assert(accepted_decision.burst_uuid[0] != '\0');

    OssCaptureCandidate duplicate = candidate(200.0);
    duplicate.perceptual_hash_64 = accepted.perceptual_hash_64;
    assert(accept_sampler.Consider(duplicate).rule == OSS_CAPTURE_RULE_MOTION_BUCKET);

    cfg.max_motion_bucket_samples = 8;
    oss_gaussian::capture::CaptureSampler dedup_sampler(cfg);
    assert(dedup_sampler.Consider(accepted).capture == 1);
    assert(dedup_sampler.Consider(duplicate).rule == OSS_CAPTURE_RULE_PERCEPTUAL_DEDUP);

    cfg.burst_n = 3;
    cfg.short_burst_n = 3;
    cfg.max_motion_bucket_samples = 8;
    assert(oss_capture_configure(&cfg) == 1);
    OssCaptureCandidate api_candidate = candidate(160.0);
    OssCaptureDecision api_decision = oss_capture_consider_candidate(&api_candidate);
    assert(api_decision.capture == 1);
    assert(api_decision.burst_n == 3);
    assert(api_decision.burst_tier == OSS_CAPTURE_TIER_SHORT);

    OssCaptureBurstFrame burst_frame{};
    assert(oss_capture_consume_present_burst(&burst_frame) == 1);
    assert(burst_frame.burst_index == 0);
    assert(burst_frame.burst_n == 3);
    assert(burst_frame.burst_tier == OSS_CAPTURE_TIER_SHORT);
    assert(burst_frame.capture_hr == 1);
    assert(oss_capture_consume_present_burst(&burst_frame) == 1);
    assert(burst_frame.burst_index == 1);
    assert(oss_capture_consume_present_burst(&burst_frame) == 1);
    assert(burst_frame.burst_index == 2);
    assert(oss_capture_consume_present_burst(&burst_frame) == 0);

    OssCaptureConfig two_tier = oss_capture_default_config();
    two_tier.two_tier_enabled = 1;
    two_tier.short_burst_n = 2;
    two_tier.short_stride_seconds = 80.0;
    two_tier.long_burst_n = 60;
    two_tier.long_stride_seconds = 1800.0;
    two_tier.long_capture_hr = 0;
    two_tier.max_motion_bucket_samples = 8;
    two_tier.enable_static_frame_trigger = 0;
    oss_gaussian::capture::CaptureSampler two_tier_sampler(two_tier);

    OssCaptureCandidate both_due = candidate(1800.0);
    both_due.perceptual_hash_64 = 0x0101010101010101ull;
    OssCaptureDecision long_decision = two_tier_sampler.Consider(both_due);
    assert(long_decision.capture == 1);
    assert(long_decision.burst_tier == OSS_CAPTURE_TIER_LONG);
    assert(long_decision.burst_n == 60);
    assert(long_decision.capture_hr == 0);

    OssCaptureCandidate short_due_after_long = candidate(1880.0);
    short_due_after_long.perceptual_hash_64 = 0xfefefefefefefefeull;
    OssCaptureDecision short_decision = two_tier_sampler.Consider(short_due_after_long);
    assert(short_decision.capture == 1);
    assert(short_decision.burst_tier == OSS_CAPTURE_TIER_SHORT);
    assert(short_decision.burst_n == 2);
    assert(short_decision.capture_hr == 1);

    OssCaptureConfig trickle = oss_capture_default_config();
    oss_capture_apply_mode_preset(&trickle, OSS_CAPTURE_MODE_TRICKLE);
    trickle.max_motion_bucket_samples = 8;
    oss_gaussian::capture::CaptureSampler trickle_sampler(trickle);

    OssCaptureCandidate not_settled = candidate(300.0);
    not_settled.motion_mean_magnitude_px = 0.1f;
    not_settled.motion_below_threshold_seconds = 1.0;
    assert(trickle_sampler.Consider(not_settled).capture == 0);

    OssCaptureCandidate settled = candidate(301.0);
    settled.motion_mean_magnitude_px = 0.1f;
    settled.motion_below_threshold_seconds = 2.0;
    settled.perceptual_hash_64 = 0x1111111111111111ull;
    OssCaptureDecision static_decision = trickle_sampler.Consider(settled);
    assert(static_decision.capture == 1);
    assert(static_decision.burst_tier == OSS_CAPTURE_TIER_NONE);
    assert(static_decision.capture_hr == 1);
    assert(static_decision.capture_mode == OSS_CAPTURE_MODE_TRICKLE);

    OssCaptureCandidate moving = candidate(304.0);
    moving.motion_mean_magnitude_px = 3.0f;
    moving.perceptual_hash_64 = 0xeeeeeeeeeeeeeeeeull;
    OssCaptureDecision pair_decision = trickle_sampler.Consider(moving);
    assert(pair_decision.capture == 1);
    assert(pair_decision.burst_tier == OSS_CAPTURE_TIER_SHORT);
    assert(pair_decision.burst_n == 2);
    assert(pair_decision.capture_hr_on_t0 == 1);
    assert(pair_decision.capture_hr_on_tplus == 0);

    OssCaptureConfig lite_static = oss_capture_default_config();
    lite_static.two_tier_enabled = 0;
    lite_static.short_stride_seconds = 1000.0;
    lite_static.max_motion_bucket_samples = 8;
    oss_gaussian::capture::CaptureSampler lite_static_sampler(lite_static);
    OssCaptureCandidate lite_initial = candidate(0.0);
    lite_initial.perceptual_hash_64 = 0x2222222222222222ull;
    assert(lite_static_sampler.Consider(lite_initial).burst_tier == OSS_CAPTURE_TIER_SHORT);
    OssCaptureCandidate lite_settled = candidate(600.0);
    lite_settled.motion_mean_magnitude_px = 0.1f;
    lite_settled.motion_below_threshold_seconds = 2.0;
    lite_settled.perceptual_hash_64 = 0x3333333333333333ull;
    OssCaptureDecision lite_static_decision = lite_static_sampler.Consider(lite_settled);
    assert(lite_static_decision.capture == 1);
    assert(lite_static_decision.burst_tier == OSS_CAPTURE_TIER_NONE);
    assert(lite_static_decision.capture_mode == OSS_CAPTURE_MODE_LITE);

    std::vector<uint8_t> image(8 * 8 * 3, 0);
    for (size_t i = 0; i < image.size(); i += 3) {
        image[i] = static_cast<uint8_t>(i);
        image[i + 1] = static_cast<uint8_t>(255u - (i & 255u));
        image[i + 2] = 64;
    }
    const uint64_t hash_a = oss_capture_phash64_rgb8(image.data(), 8, 8, 8 * 3);
    const uint64_t hash_b = oss_capture_phash64_rgb8(image.data(), 8, 8, 8 * 3);
    assert(hash_a == hash_b);
    assert(oss_capture_hamming_distance64(hash_a, hash_b) == 0);

    return 0;
}
