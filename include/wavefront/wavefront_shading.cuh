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
#include "rt_style.cuh"

#include <cstdint>

// ---------------------------------------------------------------------------
//  PathState  ↔  SoA
//
//  Note on `accumulatedColor`: in the wavefront pipeline it is a PER-BOUNCE
//  DELTA, not a running total.  Radiance produced at a vertex is flushed into
//  the style channel that vertex belongs to at the end of each shade call, so
//  the delta starts at zero on every load.  The megakernel, which keeps its
//  PathState on the stack for the whole path, still uses it as a running total.
// ---------------------------------------------------------------------------
__device__ __forceinline__ PathState loadPathState(const WavefrontSoA& wf, int idx)
{
    PathState ps;
    ps.ray.origin       = wf.rayOrigin[idx];
    ps.ray.direction    = wf.rayDir[idx];
    ps.throughput       = wf.throughput[idx];
    ps.accumulatedColor = {0.0f, 0.0f, 0.0f};
    ps.bounceCount      = wf.bounceCount[idx];
    ps.eta              = wf.eta[idx];
    ps.brdfPDF          = wf.brdfPDF[idx];
    ps.specularBounce   = wf.specularBounce[idx] != 0;
    ps.pixelIndex       = wf.pixelIndex[idx];
    ps.styleChannel     = static_cast<int>(wf.styleChannel[idx]);
    return ps;
}

__device__ __forceinline__ void storePathState(const WavefrontSoA& wf, int idx, const PathState& ps)
{
    wf.rayOrigin[idx]      = ps.ray.origin;
    wf.rayDir[idx]         = ps.ray.direction;
    wf.throughput[idx]     = ps.throughput;
    wf.bounceCount[idx]    = ps.bounceCount;
    wf.eta[idx]            = ps.eta;
    wf.brdfPDF[idx]        = ps.brdfPDF;
    wf.specularBounce[idx] = ps.specularBounce ? 1 : 0;
    wf.pixelIndex[idx]     = ps.pixelIndex;
    wf.styleChannel[idx]   = static_cast<unsigned char>(ps.styleChannel);
}

// ---------------------------------------------------------------------------
//  Channel writes.
//
//  In physical mode every wf.channel[] entry points at the same buffer, so
//  these are all writes to one place and sum to exactly the physical estimate.
//  That is what lets shade and connect stay branch-free on style mode.
// ---------------------------------------------------------------------------

// Shade owns each path slot exclusively for the duration of a launch, so no
// atomics are needed here.
__device__ __forceinline__ void addToChannel(const WavefrontSoA& wf, int ch, int idx,
                                              const Float3& v)
{
    if (v.x == 0.0f && v.y == 0.0f && v.z == 0.0f) return;
    Float3* dst = wf.channel[ch] + idx;
    dst->x += v.x;
    dst->y += v.y;
    dst->z += v.z;
}

// Connect needs atomics: one path can own several in-flight shadow rays (area
// or environment NEE plus one per spotlight), and they land in parallel.
__device__ __forceinline__ void atomicAddToChannel(const WavefrontSoA& wf, int ch, int idx,
                                                    const Float3& v)
{
    if (v.x == 0.0f && v.y == 0.0f && v.z == 0.0f) return;
    Float3* dst = wf.channel[ch] + idx;
    atomicAdd(&dst->x, v.x);
    atomicAdd(&dst->y, v.y);
    atomicAdd(&dst->z, v.z);
}

// Which channel the radiance produced at this vertex belongs to.
//
// Bounce 0 is emissive: a camera ray that directly sees a light, or escapes to
// the backdrop, is not shaded by any material and must stay unstyled.  Every
// deeper vertex feeds the path's own channel, which is how the sky reflected in
// a metal surface lands in the metal channel and gets banded with it.
__device__ __forceinline__ int vertexChannel(const PathState& ps)
{
    return (ps.bounceCount == 0) ? STYLE_CH_EMISSIVE : ps.styleChannel;
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
    s.selectCount       = scene.emitterSelectCount;
    s.envSlot           = scene.envEmitterSlot;
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
    ctx.env                  = scene.env;
    return ctx;
}

// Look up the style record for a BSDF, falling back to the neutral identity
// record when the material carries no style (styleId < 0) or the table is
// absent.  Every operator in rt_style.cuh is a no-op under the identity record,
// so a mixed scene needs no per-pixel branching.
__device__ __forceinline__ StyleData loadStyle(const WfSceneView& scene, int styleId)
{
    if (scene.styles == nullptr || styleId < 0 || styleId >= scene.styleCount)
        return styleDataIdentity();
    return scene.styles[styleId];
}

// ---------------------------------------------------------------------------
//  Primary-hit AOVs.
//
//  Written on the camera-ray vertex only, and in BOTH style modes: albedo and
//  normal are the OptiX denoiser's guide layers, so keeping them always-on
//  improves the photorealistic path as well.  Albedo and normal are progressively
//  averaged (the camera jitter means each frame sees a slightly different
//  surface point); depth, style ID and object ID are latest-frame values, which
//  is all the style operators and the debug views need.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void writeHitAOVs(const WfSceneView& scene, int idx,
                                              const Intersection& its,
                                              const BsdfData& bsdf,
                                              const Ray& cameraRay)
{
    const float n = static_cast<float>(scene.frameIndex + 1);

    if (scene.accumAlbedo) {
        const Float3 albedo = {bsdf.p0.x, bsdf.p0.y, bsdf.p0.z};
        Float3 prev = scene.accumAlbedo[idx];
        prev.x += (albedo.x - prev.x) / n;
        prev.y += (albedo.y - prev.y) / n;
        prev.z += (albedo.z - prev.z) / n;
        scene.accumAlbedo[idx] = prev;
    }

    if (scene.accumNormal) {
        Float3 prev = scene.accumNormal[idx];
        prev.x += (its.hitNormal.x - prev.x) / n;
        prev.y += (its.hitNormal.y - prev.y) / n;
        prev.z += (its.hitNormal.z - prev.z) / n;
        scene.accumNormal[idx] = prev;
    }

    if (scene.primaryDepth)
        scene.primaryDepth[idx] = distance3(its.hitPoint, cameraRay.origin);

    if (scene.primaryStyleId)
        scene.primaryStyleId[idx] = bsdf.styleId;

    if (scene.primaryObjectId)
        scene.primaryObjectId[idx] =
            scene.triangleObjectIds ? scene.triangleObjectIds[its.triangleIndex] : -1;
}

// A camera ray that escaped has no surface.  A negative depth marks the pixel as
// background, which is how present tells the backdrop apart from an emitter
// without needing a seventh accumulation buffer.
__device__ __forceinline__ void writeMissAOVs(const WfSceneView& scene, int idx)
{
    if (scene.accumAlbedo)     scene.accumAlbedo[idx]     = {0.0f, 0.0f, 0.0f};
    if (scene.accumNormal)     scene.accumNormal[idx]     = {0.0f, 0.0f, 0.0f};
    if (scene.primaryDepth)    scene.primaryDepth[idx]    = -1.0f;
    if (scene.primaryStyleId)  scene.primaryStyleId[idx]  = -1;
    if (scene.primaryObjectId) scene.primaryObjectId[idx] = -1;
}

// ---------------------------------------------------------------------------
//  Outlines as a throughput modulation.
//
//  The probe stage wrote how much of the neighbourhood around this path vertex
//  is a discontinuity into wf.edgeFactor.  Attenuating the path's throughput by
//  that coverage — and depositing the line colour in its place — is what makes a
//  line drawn inside a mirror a real line on the reflected path, which no
//  screen-space filter can produce.
//
//  In physical mode wf.edgeFactor is never allocated, so this reads null and
//  returns immediately: the photorealistic path pays nothing.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void applyOutline(const WavefrontSoA& wf,
                                              const WfSceneView& scene,
                                              int idx,
                                              PathState& ps,
                                              const BsdfData& bsdf)
{
    if (wf.edgeFactor == nullptr) return;

    const float edge = wf.edgeFactor[idx];
    if (edge <= 0.0f) return;

    const StyleData st = loadStyle(scene, bsdf.styleId);
    if (st.lineStrength <= 0.0f) return;

    const float coverage = fminf(edge * st.lineStrength, 1.0f);

    ps.accumulatedColor = add3(ps.accumulatedColor,
                               mul3(mul3(st.lineColor, coverage), ps.throughput));
    ps.throughput = mul3(ps.throughput, 1.0f - coverage);
}

// ---------------------------------------------------------------------------
//  Shadow-ray enqueue: write a prepared ShadowRayRecord into the shadow SoA.
//  atomicAdd hands out contiguous slots, so connect can index the arrays
//  directly with its launch index (no separate index queue needed).
// ---------------------------------------------------------------------------
//
//  `isPrimary` marks next-event estimation that happened at the primary hit.
//  Those contributions route to the two direct channels so the cel band ramp
//  and the hard highlight threshold each act on their own converged signal;
//  deeper ones collapse into the path's channel.  One shadow ray still serves
//  both halves — only the payload widens.
__device__ __forceinline__ void enqueueShadowRay(const WavefrontSoA& wf,
                                                  const ShadowRayRecord& sr,
                                                  uint32_t pathIdx,
                                                  bool isPrimary)
{
    const int sSlot = atomicAdd(wf.shadowCount, 1);
    if (sSlot >= wf.maxShadows) return;   // defensive; sized for the worst case
    wf.shadowOrigin[sSlot]      = sr.origin;
    wf.shadowDir[sSlot]         = sr.direction;
    wf.shadowTMax[sSlot]        = sr.tMax;
    wf.shadowContrib[sSlot]     = sr.contribution;
    wf.shadowContribSpec[sSlot] = sr.contributionSpec;
    wf.shadowPathIdx[sSlot]     = pathIdx;
    wf.shadowIsPrimary[sSlot]   = isPrimary ? 1u : 0u;
}
