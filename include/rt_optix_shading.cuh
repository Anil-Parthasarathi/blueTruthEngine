#pragma once

// ---------------------------------------------------------------------------
//  rt_optix_shading.cuh — OptiX-side shading glue.
//
//  Everything here calls optixTrace (or supports code that does), so this
//  header may ONLY be included from OptiX device programs compiled to PTX
//  (optix_programs.cu, wavefront/ray_extend.cu, wavefront/ray_connect.cu).
//  Plain CUDA kernels must use the trace-free headers instead
//  (rt_direct_lighting.cuh, rt_path_integrator.cuh, ...).
//
//  The GAS handle is always passed explicitly so nothing in this header
//  depends on the `params` launch-parameter constant.
// ---------------------------------------------------------------------------

#include <optix.h>

#include "optix_launch_params.h"
#include "render_kernel.h"
#include "rt_cuda_math.cuh"
#include "rt_intersect.cuh"
#include "rt_rng.cuh"
#include "rt_bsdf.cuh"
#include "rt_emitter_sampling.cuh"
#include "rt_constants.cuh"
#include "rt_shading_context.cuh"
#include "rt_material.cuh"
#include "rt_direct_lighting.cuh"
#include "rt_path_integrator.cuh"

#include <cstdint>

// ---------------------------------------------------------------------------
//  Payload pointer (un)packing — pass a stack pointer through 2 payload regs.
// ---------------------------------------------------------------------------
static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long uptr =
        (static_cast<unsigned long long>(i0) << 32) | static_cast<unsigned long long>(i1);
    return reinterpret_cast<void*>(uptr);
}

static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = static_cast<unsigned int>(uptr >> 32);
    i1 = static_cast<unsigned int>(uptr & 0xffffffffull);
}

static __forceinline__ __device__ float3 toF3(const Float3& v) { return make_float3(v.x, v.y, v.z); }

// ---------------------------------------------------------------------------
//  Closest-hit intersection (RT cores).
//  __closesthit__radiance / __miss__radiance fill the Intersection via the
//  packed payload pointer.
// ---------------------------------------------------------------------------
static __forceinline__ __device__ bool traceClosest(OptixTraversableHandle handle,
                                                     const Ray& ray, Intersection& its)
{
    its.triangleIndex = -1;
    unsigned int u0, u1;
    packPointer(&its, u0, u1);

    optixTrace(handle,
               toF3(ray.origin), toF3(ray.direction),
               RT_EPSILON, 1e30f, 0.0f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,
               RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE,
               u0, u1);

    return its.triangleIndex >= 0;
}

// ---------------------------------------------------------------------------
//  Occlusion test — returns true if anything blocks the segment
//  [origin, origin + dir*tMax].  __miss__shadow clears the payload on a miss.
// ---------------------------------------------------------------------------
static __forceinline__ __device__ bool traceOccluded(OptixTraversableHandle handle,
                                                      const Float3& origin,
                                                      const Float3& dir,
                                                      float tMax)
{
    unsigned int occluded = 1u; // assume blocked; miss program clears it
    optixTrace(handle,
               toF3(origin), toF3(dir),
               RT_EPSILON, tMax, 0.0f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               RAY_TYPE_SHADOW, RAY_TYPE_COUNT, RAY_TYPE_SHADOW,
               occluded);
    return occluded != 0u;
}

// ---------------------------------------------------------------------------
//  Context builders from LaunchParams (megakernel raygen).
// ---------------------------------------------------------------------------
static __forceinline__ __device__ EmitterSamplingData makeEmitterSampling(const LaunchParams& lp)
{
    EmitterSamplingData s{};
    s.triangles         = lp.triangles;
    s.emitters          = lp.emitters;
    s.emitterCount      = lp.emitterCount;
    s.emitterTriIndices = lp.emitterTriIndices;
    s.emitterTriCdf     = lp.emitterTriCdf;
    s.sceneEmitterCdf   = lp.sceneEmitterCdf;
    s.selectCount       = lp.emitterSelectCount;
    s.envSlot           = lp.envEmitterSlot;
    return s;
}

static __forceinline__ __device__ DirectLightingContext makeDirectLightingContext(const LaunchParams& lp)
{
    DirectLightingContext ctx{};
    ctx.triangleEmitterFlags = lp.triangleEmitterFlags;
    ctx.triangleEmission     = lp.triangleEmission;
    ctx.emitterSampling      = makeEmitterSampling(lp);
    ctx.spotlights           = lp.spotlights;
    ctx.spotlightCount       = lp.spotlightCount;
    ctx.texObjects           = lp.texObjects;
    ctx.triangleMaterialIds  = lp.triangleMaterialIds;
    ctx.textureCount         = lp.textureCount;
    ctx.env                  = lp.env;
    return ctx;
}

// ---------------------------------------------------------------------------
//  Megakernel NEE wrappers: prepare the shadow ray (shared math), trace it
//  inline on the RT cores, and add the contribution if unoccluded.
//  The wavefront pipeline calls the prepare* helpers directly and defers the
//  trace to the connect stage instead.
// ---------------------------------------------------------------------------
static __forceinline__ __device__ void sampleAreaEmitterNEE(
    PathState& pathRecord,
    RngState& rng,
    const Intersection& its,
    const BsdfData& bsdf,
    const DirectLightingContext& lightingCtx,
    OptixTraversableHandle handle)
{
    ShadowRayRecord shadowRay{};
    if (!prepareAreaEmitterNEE(pathRecord, rng, its, bsdf, lightingCtx, shadowRay)) {
        return;
    }

    if (traceOccluded(handle, shadowRay.origin, shadowRay.direction, shadowRay.tMax)) {
        return;
    }

    // The megakernel is physical-only, so it just sums the two halves the
    // shared NEE math produces.
    pathRecord.accumulatedColor =
        add3(pathRecord.accumulatedColor, shadowRayTotal(shadowRay));
}

static __forceinline__ __device__ void sampleSpotlightNEE(
    PathState& pathRecord,
    const Intersection& its,
    const BsdfData& bsdf,
    const DirectLightingContext& lightingCtx,
    OptixTraversableHandle handle)
{
    for (int si = 0; si < lightingCtx.spotlightCount; ++si) {
        ShadowRayRecord shadowRay{};
        if (!prepareSpotlightNEE(pathRecord, its, bsdf, lightingCtx, si, shadowRay)) {
            continue;
        }

        if (traceOccluded(handle, shadowRay.origin, shadowRay.direction, shadowRay.tMax)) {
            continue;
        }

        pathRecord.accumulatedColor =
            add3(pathRecord.accumulatedColor, shadowRayTotal(shadowRay));
    }
}

// ---------------------------------------------------------------------------
//  Main path-tracing step: intersect, shade, advance ray.
//  This is the megakernel's per-bounce body; the wavefront pipeline runs the
//  same sequence split across its stages:
//      traceClosest          → extend  (__raygen__wf_extend)
//      loadBsdfForTriangle,
//      accumulateEmitterHit,
//      prepare*NEE,
//      scatterPath           → shade   (wfShade)
//      traceOccluded + add   → connect (__raygen__wf_shadow)
// ---------------------------------------------------------------------------
static __forceinline__ __device__ bool traceRay(
    PathState& pathRecord,
    RngState& rng,
    const DirectLightingContext& lightingCtx,
    const BsdfData* bsdfs,
    const int* triangleBsdfIds,
    OptixTraversableHandle handle)
{
    Intersection its{};
    if (!traceClosest(handle, pathRecord.ray, its)) {
        // The ray escaped: pick up the environment light (or the backdrop, for
        // camera rays).  Contributes nothing when no environment is bound, which
        // is exactly the pre-environment behaviour.
        accumulateEnvMiss(pathRecord, lightingCtx);
        return false;
    }

    BsdfData bsdf = loadBsdfForTriangle(lightingCtx, bsdfs, triangleBsdfIds,
                                        its.triangleIndex, its.uv);

    // Check if hit an emitter
    if (lightingCtx.triangleEmitterFlags[its.triangleIndex] != 0) {
        accumulateEmitterHit(pathRecord, its, lightingCtx);
    }

    if (bsdfIsDiffuse(bsdf)) {
        pathRecord.specularBounce = false;
        sampleAreaEmitterNEE(pathRecord, rng, its, bsdf, lightingCtx, handle);
        sampleSpotlightNEE(pathRecord, its, bsdf, lightingCtx, handle);
    } else {
        pathRecord.specularBounce = true;
    }

    scatterPath(pathRecord, rng, its, bsdf);

    return true;
}
