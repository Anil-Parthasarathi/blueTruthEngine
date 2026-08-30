#pragma once
// ============================================================================
//  wavefront/wavefront_buffers.h
// ============================================================================
//  Structure-of-Arrays (SoA) layout for the wavefront path tracer.
//
//  This header has NO dependency on optix.h and can therefore be included
//  from both:
//    • Regular CUDA kernels  (ray_generate.cu, ray_shade.cu, ray_present.cu)
//    • OptiX device programs via optix_launch_params.h  (ray_extend.cu, etc.)
//
//  Layout overview (one slot per path, N = width × height, one path/pixel):
//
//    Generate  →  [rayQueue filled with all N slots]
//    Extend    →  [hit buffer filled]      ← optixLaunch, __raygen__wf_extend
//    Shade     →  [rayQueueNext + shadow SoA filled]
//    Connect   →  [shadow rays traced, radiance updated] ← __raygen__wf_shadow
//    swap(rayQueue, rayQueueNext); loop back to Extend until the queue is empty
//    Present   →  [radiance → accumBuffer → PBO]  (once, after the loop)
//
//  Why TWO ray queues (ping-pong)?  Shade reads path indices out of the
//  current queue while appending survivors for the next bounce.  If it
//  appended into the SAME buffer it reads, a thread could overwrite an entry
//  another thread has not read yet (block execution order is undefined).
//  Reading from rayQueue and writing to rayQueueNext removes the race; the
//  host swaps the two pointers (and their counters) after each bounce.
// ============================================================================

#include "render_kernel.h"   // Float2/Float3/Float4, TriangleData, BsdfData, ...
#include "rt_environment.cuh" // EnvLightData
#include "rt_style.cuh"       // StyleChannel, STYLE_CH_COUNT
#include <cuda_runtime.h>    // cudaTextureObject_t
#include <cstdint>

// ---------------------------------------------------------------------------
//  WavefrontSoA — all per-path state in flat device arrays.
//
//  This struct is passed by value to every wavefront kernel.  All fields are
//  raw device pointers; the host allocates them once in ensureWavefrontBuffers
//  and frees them in cudaCleanup.
// ---------------------------------------------------------------------------
struct WavefrontSoA {
    // ── Current-bounce ray ───────────────────────────────────────────────
    Float3*        rayOrigin;      // [maxPaths]
    Float3*        rayDir;         // [maxPaths]

    // ── Persistent path state ────────────────────────────────────────────
    Float3*        throughput;     // [maxPaths] spectral path weight
    Float3*        radiance;       // [maxPaths] accumulated color — blended into
                                   //            accumBuffer by the present stage
    float*         eta;            // [maxPaths] running relative IOR (for RR)
    unsigned char* styleChannel;   // [maxPaths] StyleChannel this path feeds,
                                   //            fixed at the primary hit from the
                                   //            sampled BSDF lobe
    float*         edgeFactor;     // [maxPaths] outline coverage in [0,1] from the
                                   //            probe stage; null in physical mode
    float*         brdfPDF;        // [maxPaths] BRDF PDF of last scatter direction (MIS)
    unsigned char* specularBounce; // [maxPaths] 1 = last bounce was specular/delta
    int*           bounceCount;    // [maxPaths]
    uint32_t*      pixelIndex;     // [maxPaths] which pixel this path writes to
    uint32_t*      rngState;       // [maxPaths] xorshift32 state — see rt_rng.cuh

    // ── Hit buffer (written by extend, consumed by shade) ────────────────
    //  __closesthit__radiance interpolates the shading normal and texture UV
    //  from the barycentrics, so the raw barycentrics are not stored.
    int*           hitTriIndex;    // [maxPaths] -1 == miss
    Float3*        hitPoint;       // [maxPaths] world-space hit position
    Float3*        hitNormal;      // [maxPaths] interpolated shading normal
    Float2*        hitUV;          // [maxPaths] interpolated texture UV

    // ── Ray queues (ping-pong) & counters ────────────────────────────────
    //  rayQueue/rayCount hold THIS bounce's active path slots (read-only
    //  during shade).  Shade appends survivors to rayQueueNext with
    //      int slot = atomicAdd(wf.rayCountNext, 1);
    //      wf.rayQueueNext[slot] = idx;
    //  The host swaps the queue and counter pointers after each bounce.
    int*           rayQueue;       // [maxPaths] path-slot indices → extend
    int*           rayQueueNext;   // [maxPaths] filled by shade for next bounce
    int*           rayCount;       // device counter for rayQueue
    int*           rayCountNext;   // device counter for rayQueueNext — reset to 0
                                   // before each shade launch
    int*           shadowCount;    // device counter — reset to 0 before each shade

    // ── Shadow ray data (written by shade, read by connect) ─────────────
    //  No index queue is needed: atomicAdd(wf.shadowCount, 1) hands out
    //  contiguous slots, so connect indexes these arrays directly with its
    //  launch index in [0, shadowCount).
    Float3*        shadowOrigin;   // [maxShadows]
    Float3*        shadowDir;      // [maxShadows] (pre-normalised)
    float*         shadowTMax;     // [maxShadows] tMax for the visibility test
    Float3*        shadowContrib;  // [maxShadows] diffuse-ish NEE contribution
                                   //              (radiance × MIS × throughput)
    Float3*        shadowContribSpec; // [maxShadows] specular-ish NEE contribution
    uint32_t*      shadowPathIdx;  // [maxShadows] path slot to atomicAdd contrib into
    unsigned char* shadowIsPrimary; // [maxShadows] 1 when the NEE happened at the
                                    //              primary hit, so the two
                                    //              contributions route to the
                                    //              direct channels instead of
                                    //              collapsing into the path channel

    // ── Style channels ───────────────────────────────────────────────────
    //  Six accumulation targets, indexed by StyleChannel.  In PHYSICAL mode all
    //  six point at `radiance`, so every atomicAdd from shade and connect lands
    //  in the same buffer and sums correctly — which is what lets wfShade and
    //  ray_connect.cu keep exactly one code path with no style-mode branches.
    //  Only wfPresent branches, on a uniform scene-level flag.
    Float3*        channel[STYLE_CH_COUNT];

    // ── Dimensions ───────────────────────────────────────────────────────
    int maxPaths;    // width × height  (one path slot per pixel)
    int maxShadows;  // maxPaths × (1 + spotlightCount) worst-case shadow rays/bounce
};

// ---------------------------------------------------------------------------
//  WfSceneView — read-only scene data for device kernels that shade.
//  Built from the device pointers in renderer_state and passed by value to
//  wfShade / wfPresent each launch.  (Fits in constant cache — all reads are
//  uniform across a warp.)
// ---------------------------------------------------------------------------
struct WfSceneView {
    const TriangleData*        triangles;
    int                        triangleCount;
    const uint8_t*             triangleEmitterFlags;
    const Float3*              triangleEmission;
    const int*                 triangleObjectIds;   // per-triangle mesh index
    const BsdfData*            bsdfs;
    int                        bsdfCount;
    const int*                 triangleBsdfIds;
    const int*                 triangleMaterialIds;
    const EmitterData*         emitters;
    int                        emitterCount;
    const int*                 emitterTriIndices;
    const float*               emitterTriCdf;
    const float*               sceneEmitterCdf;
    int                        emitterSelectCount;  // emitterCount (+1 with env)
    int                        envEmitterSlot;      // == emitterCount, or -1
    const SpotlightData*       spotlights;
    int                        spotlightCount;
    const cudaTextureObject_t* texObjects;
    int                        textureCount;
    EnvLightData               env;

    // ── Stylization ──────────────────────────────────────────────────────
    const StyleData*           styles;
    int                        styleCount;
    const cudaTextureObject_t* rampTex;
    int                        rampCount;
    int                        styleModeAnime;  // uniform across the launch

    // ── Primary-hit AOVs (always on: albedo and normal feed the denoiser
    //     guide layers, so they improve the photoreal path too) ───────────
    Float3*                    accumAlbedo;
    Float3*                    accumNormal;
    float*                     accumEdge;
    float*                     primaryDepth;    // -1 when the camera ray escaped
    int*                       primaryStyleId;  // -1 when unstyled
    int*                       primaryObjectId; // -1 when the camera ray escaped

    // ── Progressive accumulation ─────────────────────────────────────────
    //  accumChannel[c] is the Welford target for channel c across frames.
    //  accumChannel[0] always aliases accumBuffer, and in physical mode so do
    //  all the others — so only five extra buffers are ever allocated, and only
    //  in anime mode.
    //
    //  readChannel[c] is what wfPresent reads: normally accumChannel[c], but the
    //  denoised copy for the smooth channels when the denoiser is on.  The
    //  direct channels are never denoised, which is what keeps cel bands razor
    //  sharp instead of being smeared by a denoiser trained on natural images.
    Float3*                    accumChannel[STYLE_CH_COUNT];
    Float3*                    readChannel[STYLE_CH_COUNT];
    Float3*                    accumBuffer;  // progressive accumulation target
    int                        frameIndex;  // Welford sample count

    // ── Presentation ─────────────────────────────────────────────────────
    CameraData                 camera;     // for the present-time rim term
    int                        width;
    int                        height;
    int                        debugView;  // DebugView; 0 = normal presentation
};

// ---------------------------------------------------------------------------
//  WavefrontBuffers — host-side handle that owns all device allocations.
//  Lives in renderer_state.cu.
// ---------------------------------------------------------------------------
struct WavefrontBuffers {
    WavefrontSoA soa;              // flat device pointers — copy into LaunchParams.wf
    bool         allocated = false;
};
