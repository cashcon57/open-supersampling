// OSS-Gaussian Metal compute renderer — Sprint 7 / T7.M.2 skeleton.
//
// Tile-based top-K 2D Gaussian rasterizer, port target of the vendored
// Image-GS CUDA kernel. Tile size MUST match `oss.gaussian.renderer.TILE_SIZE`
// (16) and `oss.gaussian.network.DEFAULT_TILE_SIZE` (16).
//
// Build (manual, see Makefile):
//     xcrun -sdk macosx metal -c rasterizer.metal -o rasterizer.air
//     xcrun -sdk macosx metallib rasterizer.air -o rasterizer.metallib
//
// Sprint 7 prep ships the kernel signature only. The body is intentionally
// stubbed; T7.M.2 ports the CUDA logic. See
// `docs/superpowers/plans/2026-05-01-gaussian-sprint-7-plan.md` § Track M.

#include <metal_stdlib>
using namespace metal;

// TILE_SIZE / THREADS_PER_TILE are documented constants used by the threadgroup
// dispatch on the host side; the kernel itself receives them via the dispatch
// configuration. Defined as macros (not `constant constexpr`) so the empty
// Sprint 7 prep kernel compiles cleanly under `-Werror,-Wunused-const-variable`.
#define TILE_SIZE        (16u)
#define THREADS_PER_TILE (TILE_SIZE * TILE_SIZE)  // 256

// Per-Gaussian record matches `GaussianBatch` in oss/gaussian/renderer/rasterizer.py.
// Layout chosen to be 16-byte aligned so the threadgroup load is one vectorized
// fetch per Gaussian on M-series GPUs.
struct Gaussian {
    float2 xy;       // pixel-space center
    float2 scale;    // per-axis sigma (positive)
    float  rot;      // radians
    float  pad0;     // align to 32 bytes
    float4 feat;     // up to 4-channel feature vector (RGB + alpha or aux)
};

struct DispatchParams {
    uint  num_gaussians;
    uint  out_h;
    uint  out_w;
    uint  feat_dim;     // 1, 3, or 4 — must be ≤ 4 in this kernel
    uint  topk;
    uint  pad0;
    uint  pad1;
    uint  pad2;
};

// Tile-based top-K rasterizer. One threadgroup per output tile; one thread
// per output pixel. Cooperatively walks the per-tile Gaussian list using
// simdgroup ballot intrinsics.
kernel void gaussian_rasterize_tile(
    device const Gaussian*       gaussians  [[ buffer(0) ]],
    device const uint*           tile_index [[ buffer(1) ]],   // CSR-like per-tile Gaussian-id list
    device const uint*           tile_starts[[ buffer(2) ]],   // length = num_tiles + 1
    device       float4*         out_image  [[ buffer(3) ]],   // (out_h * out_w) RGBA-packed
    constant DispatchParams&     params     [[ buffer(4) ]],
    uint2  gid                              [[ thread_position_in_grid ]],
    uint2  tid                              [[ thread_position_in_threadgroup ]],
    uint2  tg                               [[ threadgroup_position_in_grid ]],
    uint   sg_lane                          [[ thread_index_in_simdgroup ]],
    uint   sg_size                          [[ threads_per_simdgroup ]])
{
    // TODO: T7.M.2 — port the CUDA tile rasterizer body here.
    //
    // Reference algorithm (mirrors vendored Image-GS / gsplat CUDA path):
    //  1. Per-tile shared memory holds up to TOPK Gaussian records.
    //  2. Threadgroup cooperatively loads the next batch of candidate Gaussians
    //     for this tile from `gaussians[tile_index[tile_starts[tile_id] + i]]`.
    //  3. Each thread evaluates the Gaussian's 2D quadratic form at its pixel:
    //         q = dx^2 * a + 2 * dx * dy * b + dy^2 * d
    //         w = exp(-0.5 * q)
    //     and accumulates `w * feat` into a thread-local FP32 register.
    //  4. After the Gaussian list is exhausted, top-K normalization (if
    //     requested) divides by the sum of weights.
    //  5. Final value is written to `out_image[gid.y * params.out_w + gid.x]`.
    //
    // Subgroup notes for the port:
    //  - M-series simd width is 32 (NOT 64 as on RDNA). Use `sg_size`/`sg_lane`
    //    rather than hardcoding.
    //  - Use `simd_ballot()` + `simd_shuffle()` for per-Gaussian skip masks
    //    (skip Gaussians whose bounding box doesn't overlap this tile).
    //
    // The empty body below makes this kernel valid MSL so Sprint 7 prep can
    // verify the toolchain end-to-end without committing a real port.
    if (gid.x >= params.out_w || gid.y >= params.out_h) {
        return;
    }
    out_image[gid.y * params.out_w + gid.x] = float4(0.0);
}
