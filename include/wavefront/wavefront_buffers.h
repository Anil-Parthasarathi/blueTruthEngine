#pragma once
// ============================================================================
//  wavefront/wavefront_buffers.h
// ============================================================================
//  Structure-of-Arrays (SoA) layout for the wavefront path tracer.
//
//  This header has NO dependency on optix.h and can therefore be included
//  from both:
//    • Regular CUDA kernels  (ray_generate.cu, ray_shade.cu)
//    • OptiX device programs via optix_launch_params.h  (ray_extend.cu, etc.)
//
//  Layout overview (one slot per active path, N = width × height):
//
//    Generate  →  [ray queue filled]
//    Extend    →  [hit buffer filled]      ← optixLaunch with __raygen__wf_extend
//    Shade     →  [next ray queue + shadow queue filled]
//    Connect   →  [shadow occlusion tests] ← optixLaunch with __raygen__wf_shadow
//    (loop back to Extend until ray queue is empty)
// ============================================================================

#include "render_kernel.h"   // Float2/Float3/Float4, TriangleData, BsdfData, ...
#include <cuda_runtime.h>    // cudaTextureObject_t
#include <cstdint>

// ---------------------------------------------------------------------------
//  WavefrontSoA — all per-path state in flat device arrays.
//
//  This struct is passed by value to every wavefront kernel.  All fields are
//  raw device pointers; the host allocates them once in ensureWavefrontBuffers
//  inside render_kernel.cu and frees them in cudaCleanup.
// ---------------------------------------------------------------------------
struct WavefrontSoA {
    // ── Current-bounce ray ───────────────────────────────────────────────
    Float3*        rayOrigin;      // [maxPaths]
    Float3*        rayDir;         // [maxPaths]

    // ── Persistent path state ────────────────────────────────────────────
    Float3*        throughput;     // [maxPaths] spectral path weight
    Float3*        radiance;       // [maxPaths] accumulated color — added to
                                   //            accumBuffer on path termination
    float*         eta;            // [maxPaths] running relative IOR (for RR)
    float*         brdfPDF;        // [maxPaths] BRDF PDF of last scatter direction (MIS)
    unsigned char* specularBounce; // [maxPaths] 1 = last bounce was specular/delta
    int*           bounceCount;    // [maxPaths]
    uint32_t*      pixelIndex;     // [maxPaths] which pixel this path writes to
    uint32_t*      rngState;       // [maxPaths] xorshift32 state — see rt_rng.cuh

    // ── Hit buffer (written by extend, consumed by shade) ────────────────
    int*           hitTriIndex;    // [maxPaths] -1 == miss
    float*         hitBaryU;       // [maxPaths] optixGetTriangleBarycentrics().x
    float*         hitBaryV;       // [maxPaths] optixGetTriangleBarycentrics().y
    Float3*        hitPoint;       // [maxPaths] world-space hit position
    Float3*        hitNormal;      // [maxPaths] interpolated shading normal
    Float2*        hitUV;          // [maxPaths] interpolated texture UV

    // ── Ray & shadow queues ──────────────────────────────────────────────
    //  After each stage the next stage reads from these compacted index lists.
    //  Use  atomicAdd(wf.rayCount, 1)  to reserve a slot before writing.
    int*           rayQueue;       // [maxPaths]   path-slot indices → extend
    int*           shadowQueue;    // [maxShadows] shadow-slot indices → connect
    int*           rayCount;       // device pointer — reset to 0 at bounce start
    int*           shadowCount;    // device pointer — reset to 0 at bounce start

    // ── Shadow ray data (written by shade, read by connect) ─────────────
    Float3*        shadowOrigin;   // [maxShadows]
    Float3*        shadowDir;      // [maxShadows] (pre-normalised)
    float*         shadowTMax;     // [maxShadows] tMax for the visibility test
    Float3*        shadowContrib;  // [maxShadows] radiance × MIS weight if unoccluded
    uint32_t*      shadowPathIdx;  // [maxShadows] path slot to atomicAdd contrib into

    // ── Dimensions ───────────────────────────────────────────────────────
    int maxPaths;    // width × height  (one path slot per pixel)
    int maxShadows;  // maxPaths × (1 + spotlightCount) worst-case shadow rays/bounce
};

// ---------------------------------------------------------------------------
//  WfSceneView — read-only scene data for device kernels that shade.
//  Built from the static device pointers in render_kernel.cu and passed by
//  value to wfShade each bounce.  (Fits in constant cache — all reads are
//  uniform across a warp.)
// ---------------------------------------------------------------------------
struct WfSceneView {
    const TriangleData*        triangles;
    int                        triangleCount;
    const uint8_t*             triangleEmitterFlags;
    const Float3*              triangleEmission;
    const BsdfData*            bsdfs;
    int                        bsdfCount;
    const int*                 triangleBsdfIds;
    const int*                 triangleMaterialIds;
    const EmitterData*         emitters;
    int                        emitterCount;
    const int*                 emitterTriIndices;
    const float*               emitterTriCdf;
    const float*               sceneEmitterCdf;
    const SpotlightData*       spotlights;
    int                        spotlightCount;
    const cudaTextureObject_t* texObjects;
    int                        textureCount;
    Float3*                    accumBuffer;  // progressive accumulation target
    int                        frameIndex;  // Welford sample count
};

// ---------------------------------------------------------------------------
//  WavefrontBuffers — host-side handle that owns all device allocations.
//  Lives as a static variable inside render_kernel.cu.
// ---------------------------------------------------------------------------
struct WavefrontBuffers {
    WavefrontSoA soa;              // flat device pointers — copy into LaunchParams.wf
    bool         allocated = false;
};
