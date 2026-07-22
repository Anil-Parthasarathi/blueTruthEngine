// ============================================================================
//  ray_shade.cu — Wavefront Stage 3: Shading / Bounce Preparation
// ============================================================================
//  Compiled as a regular CUDA TU (linked into the exe, NOT PTX).
//  Launched once per bounce, after extend has filled the hit buffer.
//
//  One GPU thread per entry in the CURRENT ray queue
//  (queueSlot in [0, activeCount)).
//
//  Responsibilities  (per active path)
//  ─────────────────────────────────────
//  a) Miss  (wf.hitTriIndex[idx] == -1)
//       • Add background / sky colour × throughput to wf.radiance[idx].
//       • Write wf.radiance[idx] into the accumulation buffer (path done).
//       • Do NOT re-enqueue.
//
//  b) Emitter hit  (triangleEmitterFlags[triIdx] && !specularBounce)
//       • MIS-weight the emitted radiance and add to wf.radiance[idx].
//       • Continue to material hit logic (unless this IS the emitter surface
//         and you want to terminate — your choice).
//
//  c) Material hit
//       1. NEE — area emitters:
//            Sample one emitter, build a shadow ray, write it into the shadow
//            SoA and atomicAdd into wf.shadowCount:
//                int sSlot = atomicAdd(wf.shadowCount, 1);
//                wf.shadowOrigin[sSlot]  = hitPoint + epsilon * normal;
//                wf.shadowDir[sSlot]     = dirToLight;
//                wf.shadowTMax[sSlot]    = distToLight - RT_EPSILON_WF;
//                wf.shadowContrib[sSlot] = misWeightedContrib * throughput;
//                wf.shadowPathIdx[sSlot] = (uint32_t)idx;
//            (The connect stage will fire these shadow rays via optixLaunch.)
//
//       2. NEE — spotlights:
//            One shadow slot per spotlight (same pattern as above).
//
//       3. BSDF sample → new ray direction.
//            Update wf.throughput, wf.eta, wf.brdfPDF, wf.specularBounce.
//
//       4. Russian roulette.
//            If the path survives, re-enqueue for next extend:
//                int slot = atomicAdd(wf.rayCount, 1);
//                wf.rayQueue[slot] = idx;
//            If the path terminates, accumulate and do NOT re-enqueue.
//
//  See docs/WAVEFRONT_GUIDE.md §"Stage 3 – Shade" for full algorithm details.
// ============================================================================

#include "wavefront/wavefront_buffers.h"
#include "rt_cuda_math.cuh"
#include "rt_rng.cuh"
#include "rt_intersect.cuh"
#include "rt_bsdf.cuh"
#include "rt_emitter_sampling.cuh"
#include <cstdint>

static constexpr float RT_EPSILON_WF = 1e-4f;

// ---------------------------------------------------------------------------
//  Welford accumulation helper — matches the formula used in __raygen__rg.
//  Call this whenever a path terminates (miss or RR kill) to blend the new
//  path color into the progressive linear-space buffer.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void accumulateWelford(
    Float3*  accumBuffer,
    uint32_t pixelIndex,
    Float3   pathColor,
    int      frameIndex)
{
    // TODO: port the Welford update from __raygen__rg:
    //   Float3 prev = accumBuffer[pixelIndex];
    //   Float3 delta = sub3(pathColor, prev);
    //   accumBuffer[pixelIndex] = add3(prev, div3(delta, {n, n, n}))
    //   where n = frameIndex + 1.
    (void)accumBuffer; (void)pixelIndex; (void)pathColor; (void)frameIndex;
}

// ---------------------------------------------------------------------------
//  wfShade — device kernel  (one thread per active path in the queue)
// ---------------------------------------------------------------------------
__global__ void wfShade(
    WavefrontSoA wf,
    WfSceneView  scene,
    int          activeCount)
{
    const int queueSlot = blockIdx.x * blockDim.x + threadIdx.x;
    if (queueSlot >= activeCount) return;

    const int idx = wf.rayQueue[queueSlot];   // path slot index

    // Restore RNG state.
    RngState rng;
    rng.state = wf.rngState[idx];

    const int triIdx = wf.hitTriIndex[idx];

    // ── a) Miss path — ray escaped the scene ────────────────────────────
    if (triIdx < 0) {
        // TODO: add background/sky radiance × throughput to wf.radiance[idx].
        // Then terminate:
        //   accumulateWelford(scene.accumBuffer, wf.pixelIndex[idx],
        //                     wf.radiance[idx], scene.frameIndex);
        return;
    }

    // ── b) Emitter hit — check before shading (MIS) ─────────────────────
    if (scene.triangleEmitterFlags[triIdx] && wf.specularBounce[idx]) {
        // Specular bounce hit an emitter — add full emitted radiance.
        // TODO: wf.radiance[idx] = add3(wf.radiance[idx],
        //                               mul3(emission, wf.throughput[idx]));
    } else if (scene.triangleEmitterFlags[triIdx]) {
        // Diffuse bounce hit an emitter — MIS-weight it.
        // TODO: port accumulateEmitterHit() from render_kernel.cu / optix_programs.cu.
    }

    // Recover geometry from the hit buffer.
    // const Float3 hitPoint  = wf.hitPoint[idx];
    // const Float3 hitNormal = wf.hitNormal[idx];
    // const Float2 hitUV     = wf.hitUV[idx];

    // Look up the BSDF for this triangle.
    // const BsdfData& bsdf = scene.bsdfs[scene.triangleBsdfIds[triIdx]];

    // ── c-1) NEE — area emitter shadow ray ───────────────────────────────
    // Sample one area emitter, compute MIS-weighted contribution, then write
    // the shadow ray into the shadow queue instead of tracing it here.
    // TODO: port sampleAreaEmitterNEE() but replace sceneIntersectAnyHit with:
    //   int sSlot = atomicAdd(wf.shadowCount, 1);
    //   wf.shadowOrigin[sSlot]  = hitPoint + RT_EPSILON_WF * hitNormal;
    //   wf.shadowDir[sSlot]     = emitterQuery.directionToLight;
    //   wf.shadowTMax[sSlot]    = distToLight - RT_EPSILON_WF;
    //   wf.shadowContrib[sSlot] = misWeightedContrib;
    //   wf.shadowPathIdx[sSlot] = (uint32_t)idx;

    // ── c-2) NEE — spotlights ────────────────────────────────────────────
    // One shadow slot per spotlight (same pattern; loop over scene.spotlights).
    // TODO: port sampleSpotlightNEE() using shadow queue writes.

    // ── c-3) BSDF sample ────────────────────────────────────────────────
    // TODO: port bsdfSample(); update throughput, eta, brdfPDF, specularBounce.
    //   BsdfQueryRecord bsdfQ{};
    //   bsdfQ.wi = toLocalFromNormal(hitNormal, neg(wf.rayDir[idx]));
    //   bsdfSample(bsdf, bsdfQ, rng);
    //   Float3 newDir = toWorldFromNormal(hitNormal, bsdfQ.wo);
    //   wf.throughput[idx]     = mul3(wf.throughput[idx], bsdfQ.value);
    //   wf.eta[idx]           *= bsdfQ.eta;
    //   wf.brdfPDF[idx]        = bsdfQ.pdf;
    //   wf.specularBounce[idx] = (bsdfQ.sampledType == BSDF_EDeltaReflection ||
    //                             bsdfQ.sampledType == BSDF_EDeltaTransmission) ? 1 : 0;

    // ── c-4) Russian roulette ────────────────────────────────────────────
    // TODO: compute survival probability, terminate or re-enqueue.
    //   float rrProb = fminf(1.f, maxComponent(wf.throughput[idx]) * wf.eta[idx]^2);
    //   if (rngNextFloat01(rng) >= rrProb) {
    //       accumulateWelford(scene.accumBuffer, wf.pixelIndex[idx],
    //                         wf.radiance[idx], scene.frameIndex);
    //       return;   // path terminates
    //   }
    //   wf.throughput[idx] = div3(wf.throughput[idx], {rrProb,rrProb,rrProb});
    //   int slot = atomicAdd(wf.rayCount, 1);
    //   wf.rayQueue[slot] = idx;
    //   wf.rayOrigin[idx] = hitPoint + RT_EPSILON_WF * faceForward(hitNormal, neg(newDir));
    //   wf.rayDir[idx]    = newDir;
    //   wf.rngState[idx]  = rng.state;

    // Suppress unused-variable warnings while the kernel is a stub.
    (void)rng; (void)scene;
}

// ---------------------------------------------------------------------------
//  launchWfShade — host wrapper called from cudaRenderWavefront()
// ---------------------------------------------------------------------------
void launchWfShade(
    const WavefrontSoA& wf,
    const WfSceneView&  scene,
    int                 activeCount)
{
    if (activeCount <= 0) return;
    const int threads = 256;
    const int blocks  = (activeCount + threads - 1) / threads;
    wfShade<<<blocks, threads>>>(wf, scene, activeCount);
}
