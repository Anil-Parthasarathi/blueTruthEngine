// ============================================================================
//  ray_generate.cu — Wavefront Stage 1: Primary Ray Generation
// ============================================================================
//  Compiled as a regular CUDA TU (linked into the exe, NOT PTX).
//  Launched once per frame before the bounce loop.
//
//  One GPU thread per path slot  (idx = pixel index = slot index: the
//  wavefront traces one path per pixel per frame and accumulates across
//  frames — vs the megakernel's SAMPLES_PER_PIXEL paths per frame).
//
//  Responsibilities
//  ────────────────
//  1. Seed the per-path RNG — same formula as __raygen__rg:
//         RngState rng = makeRng(hashUint(idx ^ (frameIndex * 0x9e3779b9u)));
//     (rt_rng.cuh; there is no rngInit()).
//  2. Generate the primary camera ray with sub-pixel jitter (AA) using the
//     shared generatePrimaryRay() from rt_camera.cuh — the same function the
//     megakernel calls.
//  3. Write the initial path state into the SoA (identical to the megakernel's
//     PathState init: throughput 1, radiance 0, eta 1, brdfPDF 0,
//     specularBounce = 1 so a camera ray that directly hits an emitter gets
//     full radiance, bounceCount 0).
//  4. Append the slot index to the CURRENT ray queue so extend knows which
//     paths need a ray cast:
//         int slot = atomicAdd(wf.rayCount, 1);
//         wf.rayQueue[slot] = idx;
//     (Only shade writes to rayQueueNext; generate fills the current queue.)
//
//  See docs/WAVEFRONT_GUIDE.md §"Stage 1 – Generate" for full details.
// ============================================================================

#include "wavefront/wavefront_buffers.h"
#include "rt_cuda_math.cuh"
#include "rt_rng.cuh"
#include "rt_camera.cuh"        // shared generatePrimaryRay
#include "rt_constants.cuh"
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
    RngState rng = makeRng(hashUint(static_cast<uint32_t>(idx) ^ (frameIndex * 0x9e3779b9u)));

    // ── 2. Build primary camera ray ──────────────────────────────────────
    const float jx = rngNextFloat01(rng) - 0.5f;
    const float jy = rngNextFloat01(rng) - 0.5f;
    Ray primaryRay = generatePrimaryRay(px, py, width, height, camera, jx, jy);

    // ── 3. Initialise path state ─────────────────────────────────────────
    wf.rayOrigin[idx]       = primaryRay.origin;
    wf.rayDir[idx]          = primaryRay.direction;
    wf.throughput[idx]      = {1.f, 1.f, 1.f};
    wf.radiance[idx]        = {0.f, 0.f, 0.f};
    wf.eta[idx]             = 1.f;
    wf.brdfPDF[idx]         = 0.f;
    wf.specularBounce[idx]  = 1;
    wf.bounceCount[idx]     = 0;
    wf.pixelIndex[idx]      = static_cast<uint32_t>(idx);
    wf.rngState[idx]        = rng.state;

    // ── 4. Enqueue for extend ────────────────────────────────────────────
    const int slot = atomicAdd(wf.rayCount, 1);
    wf.rayQueue[slot] = idx;

    (void)px; (void)py; (void)frameIndex;
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
