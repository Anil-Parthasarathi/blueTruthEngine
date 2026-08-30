#pragma once

// ---------------------------------------------------------------------------
//  optix_launch_params.h
// ---------------------------------------------------------------------------
//  Shared launch-parameter block for the OptiX raygen pipeline.
//
//  The host fills a device copy of this struct every frame and passes its
//  device pointer to optixLaunch().  The device module declares
//  `extern "C" __constant__ LaunchParams params;`  and OptiX populates it
//  before launching the selected raygen (__raygen__rg, __raygen__wf_extend,
//  or __raygen__wf_shadow).
//
//  Both host (render_kernel.cu, src/host/*.cu) and device (optix_programs.cu,
//  src/wavefront/ray_extend.cu, src/wavefront/ray_connect.cu) include this
//  header, so the layout MUST stay identical on both sides.
// ---------------------------------------------------------------------------

#include "render_kernel.h"             // Float3, TriangleData, BsdfData, EmitterData, ...
#include "wavefront/wavefront_buffers.h" // WavefrontSoA — no optix.h dependency

#include <optix.h>           // OptixTraversableHandle
#include <cuda_runtime.h>    // cudaTextureObject_t
#include <cstdint>

// Ray types used by the SBT.  Radiance rays gather closest-hit shading;
// shadow rays only need a miss program (occlusion test).
enum RayType : unsigned int {
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_SHADOW   = 1,
    RAY_TYPE_COUNT    = 2
};

struct LaunchParams {
    // ── Output targets ──────────────────────────────────────────────
    uint32_t* framebuffer;   // mapped PBO (RGBA8 packed), written by raygen
    Float3*   accumBuffer;   // progressive linear-space accumulation buffer
    int       width;
    int       height;
    int       launchOffsetY; // added to the launch-local Y so a frame can be
                             // rendered as several row-band tiles (TDR safety)
    uint32_t  frameIndex;    // number of samples already accumulated

    // ── Camera ──────────────────────────────────────────────────────
    CameraData camera;

    // ── Acceleration structure + geometry ───────────────────────────
    OptixTraversableHandle handle;          // top-level traversable (GAS)
    const TriangleData*    triangles;        // per-triangle vertex/normal/uv data
    int                    triangleCount;

    // ── Materials / BSDFs ───────────────────────────────────────────
    const BsdfData* bsdfs;
    int             bsdfCount;
    const int*      triangleBsdfIds;
    const int*      triangleMaterialIds;

    // ── Emission ────────────────────────────────────────────────────
    const uint8_t* triangleEmitterFlags;
    const Float3*  triangleEmission;

    // ── Mesh (area) emitters ────────────────────────────────────────
    const EmitterData* emitters;
    int                emitterCount;
    const int*         emitterTriIndices;
    const float*       emitterTriCdf;
    const float*       sceneEmitterCdf;

    // ── Spotlights ──────────────────────────────────────────────────
    const SpotlightData* spotlights;
    int                  spotlightCount;

    // ── Textures (one CUDA texture object handle per material) ───────
    const cudaTextureObject_t* texObjects;
    const cudaTextureObject_t* texObjectsMr; // metallic-roughness (G=rough, B=metal)
    int                        textureCount;

    // ── Environment map (HDRI IBL) ──────────────────────────────────
    EnvMapData envMap;
    bool       hasEnvMap;

    // ── Wavefront mode SoA ──────────────────────────────────────────
    // All pointers are null/zero in megakernel mode.
    // Populated each frame by cudaRenderWavefront() before optixLaunch.
    WavefrontSoA wf;
};

// SBT record headers.  We carry no per-record payload data (everything lives
// in LaunchParams), so the data field is a dummy to satisfy alignment.
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptyData { int unused; };
using RayGenSbtRecord   = SbtRecord<EmptyData>;
using MissSbtRecord     = SbtRecord<EmptyData>;
using HitGroupSbtRecord = SbtRecord<EmptyData>;
