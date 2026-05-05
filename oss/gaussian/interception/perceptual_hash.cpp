#include "oss_capture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

namespace {

constexpr double kPi = 3.1415926535897932384626433832795;

double sample_gray(const uint8_t* rgb, uint32_t width, uint32_t height, uint32_t stride, uint32_t x, uint32_t y) {
    const uint8_t* px = rgb + static_cast<size_t>(y) * stride + static_cast<size_t>(x) * 3u;
    return 0.2126 * static_cast<double>(px[0]) +
           0.7152 * static_cast<double>(px[1]) +
           0.0722 * static_cast<double>(px[2]);
}

} // namespace

extern "C" {

uint64_t oss_capture_phash64_rgb8(const uint8_t* rgb, uint32_t width, uint32_t height, uint32_t stride_bytes) {
    if (!rgb || width == 0 || height == 0 || stride_bytes < width * 3u) {
        return 0;
    }

    double small[8][8]{};
    for (uint32_t oy = 0; oy < 8; ++oy) {
        for (uint32_t ox = 0; ox < 8; ++ox) {
            const uint32_t x0 = (ox * width) / 8u;
            const uint32_t x1 = std::max<uint32_t>(((ox + 1u) * width) / 8u, x0 + 1u);
            const uint32_t y0 = (oy * height) / 8u;
            const uint32_t y1 = std::max<uint32_t>(((oy + 1u) * height) / 8u, y0 + 1u);
            double acc = 0.0;
            uint32_t count = 0;
            for (uint32_t y = y0; y < std::min(y1, height); ++y) {
                for (uint32_t x = x0; x < std::min(x1, width); ++x) {
                    acc += sample_gray(rgb, width, height, stride_bytes, x, y);
                    ++count;
                }
            }
            small[oy][ox] = acc / static_cast<double>(std::max<uint32_t>(count, 1));
        }
    }

    double dct[8][8]{};
    for (uint32_t v = 0; v < 8; ++v) {
        for (uint32_t u = 0; u < 8; ++u) {
            double sum = 0.0;
            for (uint32_t y = 0; y < 8; ++y) {
                for (uint32_t x = 0; x < 8; ++x) {
                    sum += small[y][x] *
                        std::cos(((2.0 * x + 1.0) * u * kPi) / 16.0) *
                        std::cos(((2.0 * y + 1.0) * v * kPi) / 16.0);
                }
            }
            dct[v][u] = sum;
        }
    }

    double values[64]{};
    double mean = 0.0;
    uint32_t n = 0;
    for (uint32_t v = 0; v < 8; ++v) {
        for (uint32_t u = 0; u < 8; ++u) {
            if (u == 0 && v == 0) {
                continue;
            }
            values[n++] = dct[v][u];
            mean += dct[v][u];
        }
    }
    mean /= static_cast<double>(n);

    uint64_t hash = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (values[i] > mean) {
            hash |= (uint64_t{1} << i);
        }
    }
    return hash;
}

uint32_t oss_capture_hamming_distance64(uint64_t a, uint64_t b) {
#if defined(_MSC_VER)
    return static_cast<uint32_t>(__popcnt64(a ^ b));
#else
    return static_cast<uint32_t>(__builtin_popcountll(a ^ b));
#endif
}

} // extern "C"
