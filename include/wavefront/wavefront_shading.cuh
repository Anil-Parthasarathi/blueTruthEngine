#pragma once

// ---------------------------------------------------------------------------
//  wavefront/wavefront_shading.cuh — glue between the wavefront SoA buffers
//  and the shared shading helpers (rt_direct_lighting.cuh, rt_path_integrator.cuh).
//
//  The idea: reassemble the megakernel's PathState / Intersection /
//  DirectLightingContext from the SoA, call the exact same shared functions
//  the megakernel calls, then scatter the PathState back into the SoA.
//  wfShade then becomes almost pure glue:
//
//      PathState ps            = loadPathState(wf, idx);
//      Intersection its        = loadIntersection(wf, idx);
//      DirectLightingContext c = makeDirectLightingContext(scene);
//      ... accumulateEmitterHit / prepare*NEE / scatterPath / russianRoulette ...
//      storePathState(wf, idx, ps);
//
//  No OptiX dependency — usable from plain CUDA kernels (wfShade, wfPresent).
// ---------------------------------------------------------------------------

#include "wavefront/wavefront_buffers.h"
#include "rt_intersect.cuh"
#include "rt_emitter_sampling.cuh"
#include "rt_shading_context.cuh"

#include <cstdint>

// ---------------------------------------------------------------------------
//  PathState  ↔  SoA
// ---------------------------------------------------------------------------
__device__ __forceinline__ PathState loadPathState(const WavefrontSoA& wf, int idx)
{
    PathState ps;
    ps.ray.origin       = wf.rayOrigin[idx];
    ps.ray.direction    = wf.rayDir[idx];
    ps.throughput       = wf.throughput[idx];
    ps.accumulatedColor = wf.radiance[idx];
    ps.bounceCount      = wf.bounceCount[idx];
    ps.eta              = wf.eta[idx];
    ps.brdfPDF          = wf.brdfPDF[idx];
    ps.specularBounce   = wf.specularBounce[idx] != 0;
    ps.pixelIndex       = wf.pixelIndex[idx];
    return ps;
}

__device__ __forceinline__ void storePathState(const WavefrontSoA& wf, int idx, const PathState& ps)
{
    wf.rayOrigin[idx]      = ps.ray.origin;
    wf.rayDir[idx]         = ps.ray.direction;
    wf.throughput[idx]     = ps.throughput;
    wf.radiance[idx]       = ps.accumulatedColor;
    wf.bounceCount[idx]    = ps.bounceCount;
    wf.eta[idx]            = ps.eta;
    wf.brdfPDF[idx]        = ps.brdfPDF;
    wf.specularBounce[idx] = ps.specularBounce ? 1 : 0;
    wf.pixelIndex[idx]     = ps.pixelIndex;
}

// ---------------------------------------------------------------------------
//  Intersection from the hit buffer (filled by __raygen__wf_extend).
//  `t` is not stored in the SoA — nothing in the shading code reads it
//  (hitPoint is already resolved), so it stays at its default.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Intersection loadIntersection(const WavefrontSoA& wf, int idx)
{
    Intersection its{};
    its.triangleIndex = wf.hitTriIndex[idx];
    its.hitPoint      = wf.hitPoint[idx];
    its.hitNormal     = wf.hitNormal[idx];
    its.uv            = wf.hitUV[idx];
    return its;
}

// ---------------------------------------------------------------------------
//  Context builders from WfSceneView — mirror the LaunchParams versions in
//  rt_optix_shading.cuh so the shared shading functions see identical data.
// ---------------------------------------------------------------------------
__device__ __forceinline__ EmitterSamplingData makeEmitterSampling(const WfSceneView& scene)
{
    EmitterSamplingData s{};
    s.triangles         = scene.triangles;
    s.emitters          = scene.emitters;
    s.emitterCount      = scene.emitterCount;
    s.emitterTriIndices = scene.emitterTriIndices;
    s.emitterTriCdf     = scene.emitterTriCdf;
    s.sceneEmitterCdf   = scene.sceneEmitterCdf;
    return s;
}

__device__ __forceinline__ DirectLightingContext makeDirectLightingContext(const WfSceneView& scene)
{
    DirectLightingContext ctx{};
    ctx.triangleEmitterFlags = scene.triangleEmitterFlags;
    ctx.triangleEmission     = scene.triangleEmission;
    ctx.emitterSampling      = makeEmitterSampling(scene);
    ctx.spotlights           = scene.spotlights;
    ctx.spotlightCount       = scene.spotlightCount;
    ctx.texObjects           = scene.texObjects;
    ctx.triangleMaterialIds  = scene.triangleMaterialIds;
    ctx.textureCount         = scene.textureCount;
    return ctx;
}

// ---------------------------------------------------------------------------
//  Shadow-ray enqueue: write a prepared ShadowRayRecord into the shadow SoA.
//  atomicAdd hands out contiguous slots, so connect can index the arrays
//  directly with its launch index (no separate index queue needed).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void enqueueShadowRay(const WavefrontSoA& wf,
                                                  const ShadowRayRecord& sr,
                                                  uint32_t pathIdx)
{
    const int sSlot = atomicAdd(wf.shadowCount, 1);
    if (sSlot >= wf.maxShadows) return;   // defensive; sized for the worst case
    wf.shadowOrigin[sSlot]  = sr.origin;
    wf.shadowDir[sSlot]     = sr.direction;
    wf.shadowTMax[sSlot]    = sr.tMax;
    wf.shadowContrib[sSlot] = sr.contribution;
    wf.shadowPathIdx[sSlot] = pathIdx;
}
