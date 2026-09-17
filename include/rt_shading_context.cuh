#pragma once

// ---------------------------------------------------------------------------
//  rt_shading_context.cuh — per-path state and read-only shading context
//  shared by the megakernel and the wavefront pipeline.
//
//  No OptiX dependency: includable from plain CUDA TUs and PTX TUs alike.
// ---------------------------------------------------------------------------

#include "render_kernel.h"
#include "rt_intersect.cuh"          // Ray
#include "rt_emitter_sampling.cuh"   // EmitterSamplingData

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
    const cudaTextureObject_t* texObjectsMr; // metallic-roughness (G=rough, B=metal)
    const int* triangleMaterialIds;
    int textureCount;
    // Environment map (HDRI IBL)
    EnvMapData envMap;
    bool       hasEnvMap;
};

// ---------------------------------------------------------------------------
//  ShadowRayRecord — an NEE visibility test plus the radiance it carries.
//  Produced by prepareAreaEmitterNEE / prepareSpotlightNEE.
//    • Megakernel: traces it immediately (traceOccluded) and, if unoccluded,
//      adds `contribution` to PathState::accumulatedColor.
//    • Wavefront:  writes it into the shadow SoA; the connect stage traces it
//      and atomically adds `contribution` to the path's radiance.
//  `contribution` is already multiplied by the path throughput.
// ---------------------------------------------------------------------------
struct ShadowRayRecord {
    Float3 origin;
    Float3 direction;    // normalised
    float  tMax;         // distToLight - RT_EPSILON
    Float3 contribution; // radiance × MIS weight × throughput if unoccluded
};
