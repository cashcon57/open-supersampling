#include "../../oss/gaussian/interception/oss_capture.h"

#include <cassert>
#include <cstdint>
#include <vector>

static OssCaptureCandidate candidate(double t) {
    OssCaptureCandidate c{};
    c.frame_index = static_cast<uint64_t>(t);
    c.timestamp_seconds = t;
    c.seconds_since_last_candidate = 20.0;
    c.seconds_since_previous_candidate = 20.0;
    c.motion_mean_magnitude_px = 3.0f;
    c.perceptual_hash_64 = 0x123456789abcdef0ull ^ static_cast<uint64_t>(t);
    return c;
}

int main() {
    OssCaptureConfig cfg = oss_capture_default_config();
    cfg.capture_stride_seconds = 20.0;
    cfg.max_motion_bucket_samples = 1;

    oss_gaussian::capture::CaptureSampler sampler(cfg);

    OssCaptureCandidate early = candidate(1.0);
    early.seconds_since_last_candidate = 2.0;
    assert(sampler.Consider(early).rule == OSS_CAPTURE_RULE_TEMPORAL_STRIDE);

    OssCaptureCandidate bad_gbuf = candidate(20.0);
    bad_gbuf.depth_degenerate = 1;
    assert(sampler.Consider(bad_gbuf).rule == OSS_CAPTURE_RULE_GBUFFER_SANITY);

    OssCaptureCandidate loading = candidate(40.0);
    loading.seconds_since_previous_candidate = 31.0;
    assert(sampler.Consider(loading).rule == OSS_CAPTURE_RULE_POST_LOADING_GUARD);

    OssCaptureCandidate accepted = candidate(60.0);
    assert(sampler.Consider(accepted).capture == 1);

    OssCaptureCandidate duplicate = candidate(80.0);
    duplicate.perceptual_hash_64 = accepted.perceptual_hash_64;
    assert(sampler.Consider(duplicate).rule == OSS_CAPTURE_RULE_MOTION_BUCKET);

    cfg.max_motion_bucket_samples = 8;
    oss_gaussian::capture::CaptureSampler dedup_sampler(cfg);
    assert(dedup_sampler.Consider(accepted).capture == 1);
    assert(dedup_sampler.Consider(duplicate).rule == OSS_CAPTURE_RULE_PERCEPTUAL_DEDUP);

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
