// ============================================================================
//  ray_generate.cu — Wavefront Stage 1: Primary Ray Generation
// ============================================================================
//  Compiled as a regular CUDA TU (linked into the exe, NOT PTX).
//  Launched once per frame before the bounce loop.
//
//  One GPU thread per path slot  (idx = pixel index = slot index when we run
//  one path per pixel per frame and accumulate across frames).
//
//  Responsibilities
//  ────────────────
//  1. Seed the per-path RNG  (same formula as __raygen__rg so noise matches).
//  2. Generate the primary camera ray with sub-pixel jitter (AA).
//  3. Write the initial path state into the SoA.
//  4. Append the slot index to wf.rayQueue so extend knows which paths need
//     a ray cast:
//         int slot = atomicAdd(wf.rayCount, 1);
//         wf.rayQueue[slot] = idx;
//
//  See docs/WAVEFRONT_GUIDE.md §"Stage 1 – Generate" for full details.
// ============================================================================

#include "wavefront/wavefront_buffers.h"
#include "rt_cuda_math.cuh"
#include "rt_rng.cuh"
#include <cstdint>

// ---------------------------------------------------------------------------
//  wfGenerate — device kernel  (one thread per path slot)
// ---------------------------------------------------------------------------
__global__ void wfGenerate(
    WavefrontSoA wf,
    CameraData   camera,
    int          width,
    int          height,
    uint32_t     frameIndex)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= wf.maxPaths) return;

    const int px = idx % width;
    const int py = idx / width;

    // ── 1. Seed RNG ──────────────────────────────────────────────────────
    // Port the same seeding as __raygen__rg in optix_programs.cu.
    // TODO:
    //   RngState rng = rngInit(idx, frameIndex);

    // ── 2. Build primary camera ray ──────────────────────────────────────
    // Apply sub-pixel jitter for anti-aliasing.
    // Port generatePrimaryRay() from optix_programs.cu — it uses the
    // CameraData fields (origin, forward, right, up, fovYRadians, aspect).
    // TODO:
    //   float jx = rngNextFloat01(rng) - 0.5f;
    //   float jy = rngNextFloat01(rng) - 0.5f;
    //   Ray ray = generatePrimaryRay(px, py, width, height, camera, jx, jy);

    // ── 3. Initialise path state ─────────────────────────────────────────
    // TODO:
    //   wf.rayOrigin[idx]       = ray.origin;
    //   wf.rayDir[idx]          = ray.direction;
    //   wf.throughput[idx]      = {1.f, 1.f, 1.f};
    //   wf.radiance[idx]        = {0.f, 0.f, 0.f};
    //   wf.eta[idx]             = 1.f;
    //   wf.brdfPDF[idx]         = 0.f;
    //   wf.specularBounce[idx]  = 1;   // treat camera ray as "specular" for MIS
    //   wf.bounceCount[idx]     = 0;
    //   wf.pixelIndex[idx]      = static_cast<uint32_t>(idx);
    //   wf.rngState[idx]        = rng.state;

    // ── 4. Enqueue for extend ────────────────────────────────────────────
    // TODO:
    //   int slot = atomicAdd(wf.rayCount, 1);
    //   wf.rayQueue[slot] = idx;
}

// ---------------------------------------------------------------------------
//  launchWfGenerate — host wrapper called from cudaRenderWavefront()
// ---------------------------------------------------------------------------
void launchWfGenerate(
    const WavefrontSoA& wf,
    const CameraData&   camera,
    int                 width,
    int                 height,
    uint32_t            frameIndex)
{
    const int threads = 256;
    const int blocks  = (wf.maxPaths + threads - 1) / threads;
    wfGenerate<<<blocks, threads>>>(wf, camera, width, height, frameIndex);
}
