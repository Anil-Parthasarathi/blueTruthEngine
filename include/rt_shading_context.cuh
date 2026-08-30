#pragma once

// ---------------------------------------------------------------------------
//  rt_shading_context.cuh — per-path state and read-only shading context
//  shared by the megakernel and the wavefront pipeline.
//
//  No OptiX dependency: includable from plain CUDA TUs and PTX TUs alike.
// ---------------------------------------------------------------------------

#include "render_kernel.h"
#include "rt_cuda_math.cuh"          // add3
#include "rt_intersect.cuh"          // Ray
#include "rt_emitter_sampling.cuh"   // EmitterSamplingData
#include "rt_environment.cuh"        // EnvLightData
#include "rt_style.cuh"              // StyleChannel

#include <cuda_runtime.h>            // cudaTextureObject_t
#include <cstdint>

// ---------------------------------------------------------------------------
//  PathState — everything a single path carries between bounces.
//  The megakernel keeps one of these on the stack; the wavefront pipeline
//  stores the same fields in the WavefrontSoA arrays and reassembles a
//  PathState per shade invocation (see wavefront/wavefront_shading.cuh).
// ---------------------------------------------------------------------------
struct PathState {
    Ray      ray;
    Float3   throughput;
    Float3   accumulatedColor;
    int      bounceCount;
    float    eta;
    float    brdfPDF;
    bool     specularBounce;
    uint32_t pixelIndex;

    // StyleChannel this path's radiance belongs to.  Set once at the primary hit
    // from the sampled BSDF lobe and then left alone, so everything a metal
    // surface reflects (or a gem refracts) stays in that surface's channel and
    // receives its operator.  The megakernel carries the field but ignores it.
    int      styleChannel;
};

// ---------------------------------------------------------------------------
//  DirectLightingContext — read-only scene data needed to shade one hit.
//  The megakernel builds it from LaunchParams (rt_optix_shading.cuh); the
//  wavefront shade kernel builds it from WfSceneView
//  (wavefront/wavefront_shading.cuh).
// ---------------------------------------------------------------------------
struct DirectLightingContext {
    const uint8_t* triangleEmitterFlags;
    const Float3* triangleEmission;
    EmitterSamplingData emitterSampling;
    const SpotlightData* spotlights;
    int spotlightCount;
    // Texture sampling: array of CUDA texture object handles (one per material),
    // and the per-triangle material index for lookup.
    const cudaTextureObject_t* texObjects;
    const int* triangleMaterialIds;
    int textureCount;
    // Environment light — universal, active in every render and style mode.
    EnvLightData env;
};

// ---------------------------------------------------------------------------
//  ShadowRayRecord — an NEE visibility test plus the radiance it carries.
//  Produced by prepareAreaEmitterNEE / prepareSpotlightNEE.
//    • Megakernel: traces it immediately (traceOccluded) and, if unoccluded,
//      adds `contribution` to PathState::accumulatedColor.
//    • Wavefront:  writes it into the shadow SoA; the connect stage traces it
//      and atomically adds `contribution` to the path's radiance.
//  `contribution` is already multiplied by the path throughput.
//
//  The contribution is carried as two halves so that direct lighting at the
//  primary hit can converge the cel body tone and the anime highlight
//  separately.  ONE shadow ray still serves both — only the payload widens.
//  In physical mode the two halves are added into the same buffer, which sums
//  to exactly the same estimate.
// ---------------------------------------------------------------------------
struct ShadowRayRecord {
    Float3 origin;
    Float3 direction;    // normalised
    float  tMax;         // distToLight - RT_EPSILON
    Float3 contribution;     // diffuse-ish half: radiance × MIS × throughput
    Float3 contributionSpec; // specular-ish half
};

// Total contribution of a shadow ray, for callers that do not care about the
// split (the megakernel, and any physical-mode-only code).
__device__ __forceinline__ Float3 shadowRayTotal(const ShadowRayRecord& sr)
{
    return add3(sr.contribution, sr.contributionSpec);
}
