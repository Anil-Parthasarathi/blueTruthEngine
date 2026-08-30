// ============================================================================
//  ray_shade.cu — Wavefront Stage 3: Shading / Bounce Preparation
// ============================================================================
//  Compiled as a regular CUDA TU (linked into the exe, NOT PTX).
//  Launched once per bounce, after extend has filled the hit buffer.
//
//  One GPU thread per entry in the CURRENT ray queue
//  (queueSlot in [0, activeCount)).
//
//  This kernel is mostly glue: it reassembles the megakernel's PathState /
//  Intersection / DirectLightingContext from the SoA (wavefront_shading.cuh)
//  and calls the exact same shared functions the megakernel calls
//  (rt_direct_lighting.cuh, rt_material.cuh, rt_path_integrator.cuh).
//  The only difference: NEE shadow rays are ENQUEUED for the connect stage
//  instead of traced inline, and terminated paths simply stop being
//  re-enqueued — the terminal present stage accumulates, NOT this kernel.
//
//  ⚠ Ordering hazard: accumulateEmitterHit reads the PREVIOUS bounce's
//  specularBounce / brdfPDF / ray origin+direction, and prepare*NEE reads the
//  previous ray direction — all of which scatterPath overwrites.  Work on a
//  local PathState loaded ONCE at the top (loadPathState) and only call
//  scatterPath after the emitter-hit and NEE steps, exactly like the
//  megakernel's traceRay does.  Store back once at the end.
//
//  Responsibilities  (per active path; mirror traceRay + the __raygen__rg loop)
//  ─────────────────────────────────────
//      const int idx = wf.rayQueue[queueSlot];
//      PathState ps  = loadPathState(wf, idx);
//      RngState  rng; rng.state = wf.rngState[idx];
//
//  a) Miss  (wf.hitTriIndex[idx] < 0)
//       • The megakernel adds nothing on a miss (no environment light);
//         optionally add a background colour × ps.throughput here.
//       • storePathState + save rng state, do NOT re-enqueue.  The present
//         stage will accumulate wf.radiance[idx] after the loop.
//
//  b) Hit
//       Intersection its              = loadIntersection(wf, idx);
//       DirectLightingContext ctx     = makeDirectLightingContext(scene);
//       BsdfData bsdf                 = loadBsdfForTriangle(ctx, scene.bsdfs,
//                                          scene.triangleBsdfIds,
//                                          its.triangleIndex, its.uv);
//
//       1. Emitter hit — if (ctx.triangleEmitterFlags[its.triangleIndex])
//              accumulateEmitterHit(ps, its, ctx);
//          (Shared function; handles both the specular case, brdfWeight = 1,
//          and the MIS-weighted diffuse case internally.)
//
//       2. NEE — only for diffuse BSDFs, same routing as the megakernel:
//          if (bsdfIsDiffuse(bsdf)) {
//              ps.specularBounce = false;
//              ShadowRayRecord sr{};
//              if (prepareAreaEmitterNEE(ps, rng, its, bsdf, ctx, sr))
//                  enqueueShadowRay(wf, sr, static_cast<uint32_t>(idx));
//              for (int si = 0; si < ctx.spotlightCount; ++si)
//                  if (prepareSpotlightNEE(ps, its, bsdf, ctx, si, sr))
//                      enqueueShadowRay(wf, sr, static_cast<uint32_t>(idx));
//          } else {
//              ps.specularBounce = true;
//          }
//          (prepare* fills sr with origin/direction/tMax and the
//          throughput-multiplied contribution; connect traces + adds it.)
//
//       3. Scatter — scatterPath(ps, rng, its, bsdf);
//          Samples the BSDF (bsdfSample returns the throughput WEIGHT; the pdf
//          comes from bsdfPdf, recorded into ps.brdfPDF for the next bounce's
//          MIS), sets ps.ray to hitPoint + bounceDir * RT_EPSILON / bounceDir,
//          multiplies ps.throughput, updates ps.eta.
//
//       4. Russian roulette — if (!russianRoulette(ps, rng)) → terminated:
//          storePathState + rng, return WITHOUT re-enqueueing.
//          (russianRoulette only kicks in after bounce 3; survivors get their
//          throughput boosted, matching fminf(maxCoeff3(...) * eta², 0.99f).)
//
//       5. Survivors: ++ps.bounceCount; storePathState(wf, idx, ps);
//          wf.rngState[idx] = rng.state; then re-enqueue into the NEXT queue:
//              const int slot = atomicAdd(wf.rayCountNext, 1);
//              wf.rayQueueNext[slot] = idx;
//          (Never write to wf.rayQueue here — it is still being read; the
//          host swaps the queue pointers after connect.)
//
//  See docs/WAVEFRONT_GUIDE.md §"Stage 3 – Shade" for full algorithm details.
// ============================================================================

#include "wavefront/wavefront_buffers.h"
#include "wavefront/wavefront_shading.cuh"  // loadPathState / loadIntersection /
                                            // makeDirectLightingContext / enqueueShadowRay
#include "rt_cuda_math.cuh"
#include "rt_rng.cuh"
#include "rt_intersect.cuh"
#include "rt_bsdf.cuh"
#include "rt_emitter_sampling.cuh"
#include "rt_constants.cuh"                 // shared RT_EPSILON / MAX_BOUNCES
#include "rt_shading_context.cuh"
#include "rt_material.cuh"                  // loadBsdfForTriangle
#include "rt_direct_lighting.cuh"           // accumulateEmitterHit, prepare*NEE
#include "rt_path_integrator.cuh"           // scatterPath, russianRoulette
#include <cstdint>

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

    PathState rikudo = loadPathState(wf, idx);

    const int triIdx = wf.hitTriIndex[idx];

    // ── a) Miss path — ray escaped the scene ────────────────────────────
    if (triIdx < 0) {
        // Accumulate environment map radiance (MIS weighted)
        const DirectLightingContext ctx = makeDirectLightingContext(scene);
        accumulateEnvMapHit(rikudo, ctx);

        storePathState(wf, idx, rikudo);
        wf.rngState[idx] = rng.state;
        return;
    }

    // ── b) Hit: rebuild megakernel inputs from the SoA ───────────────────
    const Intersection its = loadIntersection(wf, idx);
    const DirectLightingContext ctx = makeDirectLightingContext(scene);
    const BsdfData bsdf = loadBsdfForTriangle(ctx, scene.bsdfs, scene.triangleBsdfIds, its.triangleIndex, its.uv);

    // ── b-1) Emitter hit (uses the PREVIOUS bounce's MIS state in ps) ────
    if (ctx.triangleEmitterFlags[its.triangleIndex] != 0) {
        accumulateEmitterHit(rikudo, its, ctx);
    }

    // ── b-2) NEE — enqueue shadow rays instead of tracing them ───────────
    if (bsdfIsDiffuse(bsdf)) {
        rikudo.specularBounce = false;
        ShadowRayRecord sr{};

        if (prepareAreaEmitterNEE(rikudo, rng, its, bsdf, ctx, sr)){
            enqueueShadowRay(wf, sr, static_cast<uint32_t>(idx));
        }

        for (int spotIdx = 0; spotIdx < ctx.spotlightCount; spotIdx++) {
            if (prepareSpotlightNEE(rikudo, its, bsdf, ctx, spotIdx, sr)){
                enqueueShadowRay(wf, sr, static_cast<uint32_t>(idx));
            }
        }

        if (prepareEnvMapNEE(rikudo, rng, its, bsdf, ctx, sr)){
            enqueueShadowRay(wf, sr, static_cast<uint32_t>(idx));
        }
    } else {
        rikudo.specularBounce = true;
    }

    // ── b-3) BSDF sample (overwrites ps.ray/brdfPDF — must come AFTER NEE) ─
    scatterPath(rikudo, rng, its, bsdf);

    // ── b-4) Russian roulette + re-enqueue ───────────────────────────────
    if (!russianRoulette(rikudo, rng)) {
        storePathState(wf, idx, rikudo);
        wf.rngState[idx] = rng.state;
        return; // terminating path
    }

    rikudo.bounceCount++;
    storePathState(wf, idx, rikudo);
    wf.rngState[idx] = rng.state;

    const int slot = atomicAdd(wf.rayCountNext, 1);
    wf.rayQueueNext[slot] = idx;

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
