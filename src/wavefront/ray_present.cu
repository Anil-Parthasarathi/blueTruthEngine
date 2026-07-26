// ============================================================================
//  ray_present.cu — Wavefront Stage 5: Accumulate + Present
// ============================================================================
//  Compiled as a regular CUDA TU (linked into the exe, NOT PTX).
//  Launched ONCE per frame, after the bounce loop has fully drained (every
//  path terminated and all its shadow contributions were atomically added
//  into wf.radiance by connect).
//
//  Why a separate stage?  Accumulating inside shade would race with connect:
//  connect runs AFTER shade each bounce, so a path that emits NEE shadow rays
//  and then dies (miss / Russian roulette) in the same shade call would
//  snapshot its radiance into the accumulation buffer BEFORE those shadow
//  contributions land — silently dropping direct lighting.  Once the loop
//  exits, wf.radiance[idx] is final (one path per pixel per frame), so a
//  single terminal pass is both correct and simpler.
//
//  One GPU thread per path slot (idx == pixel index, wf.pixelIndex[idx]).
//
//  Responsibilities
//  ────────────────
//  1. Welford-blend the path's final radiance into the linear accumulation
//     buffer using the shared accumulateWelford() from rt_path_integrator.cuh
//     (same formula as the megakernel; scene.frameIndex = samples so far):
//         Float3 blended = accumulateWelford(scene.accumBuffer,
//                              wf.pixelIndex[idx], wf.radiance[idx],
//                              static_cast<uint32_t>(scene.frameIndex));
//  2. Gamma-encode the running mean with the shared packPixelGamma() and
//     write it into the mapped PBO:
//         framebuffer[wf.pixelIndex[idx]] = packPixelGamma(blended);
//
//  (If the denoiser is enabled the host runs denoiseAndPresent() afterwards,
//  overwriting the PBO with the denoised tonemap — same as the megakernel.)
// ============================================================================

#include "wavefront/wavefront_buffers.h"
#include "rt_cuda_math.cuh"
#include "rt_path_integrator.cuh"   // accumulateWelford, packPixelGamma
#include <cstdint>

// ---------------------------------------------------------------------------
//  wfPresent — device kernel  (one thread per path slot / pixel)
// ---------------------------------------------------------------------------
__global__ void wfPresent(
    WavefrontSoA wf,
    WfSceneView  scene,
    uint32_t*    framebuffer,
    int          width,
    int          height)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= wf.maxPaths) return;

    const uint32_t pixelIndex = wf.pixelIndex[idx];

    const Float3 radiance = wf.radiance[idx];
    const uint32_t frameIndex = static_cast<uint32_t>(scene.frameIndex);
    const Float3 blended = accumulateWelford(scene.accumBuffer, pixelIndex, radiance, frameIndex);

    framebuffer[pixelIndex] = packPixelGamma(blended);

    (void)scene; (void)framebuffer; (void)width; (void)height;
}

// ---------------------------------------------------------------------------
//  launchWfPresent — host wrapper called from cudaRenderWavefront()
// ---------------------------------------------------------------------------
void launchWfPresent(
    const WavefrontSoA& wf,
    const WfSceneView&  scene,
    uint32_t*           framebuffer,
    int                 width,
    int                 height)
{
    const int threads = 256;
    const int blocks  = (wf.maxPaths + threads - 1) / threads;
    wfPresent<<<blocks, threads>>>(wf, scene, framebuffer, width, height);
}
