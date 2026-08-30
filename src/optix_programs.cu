// ============================================================================
//  optix_programs.cu  –  OptiX program entry points (RT-core accelerated)
// ============================================================================
//  This file is compiled to PTX (see CMakeLists.txt) and loaded as an OptiX
//  module at runtime, together with the wavefront raygens in
//  src/wavefront/ray_extend.cu and src/wavefront/ray_connect.cu.
//
//  Only the OptiX entry points live here:
//    • __raygen__rg            — the megakernel path tracer (one thread/pixel)
//    • __closesthit__radiance  — fills the Intersection payload
//    • __miss__radiance        — marks a miss (triangleIndex = -1)
//    • __miss__shadow          — clears the occlusion payload
//
//  All shading logic is shared with the wavefront pipeline via headers:
//    rt_camera.cuh          — generatePrimaryRay
//    rt_material.cuh        — loadBsdfForTriangle (texture override)
//    rt_direct_lighting.cuh — accumulateEmitterHit, prepare*NEE
//    rt_path_integrator.cuh — scatterPath, russianRoulette,
//                             accumulateWelford, packPixelGamma
//    rt_optix_shading.cuh   — traceClosest/traceOccluded, NEE wrappers,
//                             traceRay (the per-bounce step)
// ============================================================================

#include <optix.h>

#include "optix_launch_params.h"
#include "render_kernel.h"
#include "rt_cuda_math.cuh"
#include "rt_intersect.cuh"
#include "rt_emitter_sampling.cuh"
#include "rt_bsdf.cuh"
#include "rt_rng.cuh"
#include "rt_constants.cuh"
#include "rt_camera.cuh"
#include "rt_shading_context.cuh"
#include "rt_material.cuh"
#include "rt_direct_lighting.cuh"
#include "rt_path_integrator.cuh"
#include "rt_optix_shading.cuh"

#include <cstdint>

// ---------------------------------------------------------------------------
//  Launch parameters (populated by OptiX from the device pointer given to
//  optixLaunch).  The variable name MUST match pipelineLaunchParamsVariableName.
//  This is the single definition shared by all TUs in the PTX module.
// ---------------------------------------------------------------------------
extern "C" __constant__ LaunchParams params;

// ===========================================================================
//  Programs
// ===========================================================================

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const int x = static_cast<int>(idx.x);
    const int y = static_cast<int>(idx.y) + params.launchOffsetY;
    if (y >= params.height) return;

    const DirectLightingContext lightingCtx = makeDirectLightingContext(params);

    const uint32_t pixelIndex =
        static_cast<uint32_t>(y) * static_cast<uint32_t>(params.width) + static_cast<uint32_t>(x);
    RngState rng = makeRng(hashUint(pixelIndex ^ (params.frameIndex * 0x9e3779b9u)));

    Float3 finalColor = {0.0f, 0.0f, 0.0f};

    for (int i = 0; i < SAMPLES_PER_PIXEL; ++i) {
        // Sub-pixel jitter: draw fresh offsets per path so SAMPLES_PER_PIXEL paths differ.
        const float jx = rngNextFloat01(rng) - 0.5f;
        const float jy = rngNextFloat01(rng) - 0.5f;

        PathState rikudo;
        rikudo.ray              = generatePrimaryRay(x, y, params.width, params.height,
                                                     params.camera, jx, jy);
        rikudo.throughput       = {1.0f, 1.0f, 1.0f};
        rikudo.accumulatedColor = {0.0f, 0.0f, 0.0f};
        rikudo.bounceCount      = 0;
        rikudo.pixelIndex       = pixelIndex;
        rikudo.eta              = 1.0f;
        rikudo.brdfPDF          = 0.0f;
        rikudo.specularBounce   = true;

        while (rikudo.bounceCount < MAX_BOUNCES) {
            if (!traceRay(rikudo, rng, lightingCtx,
                          params.bsdfs, params.triangleBsdfIds, params.handle)) {
                break;
            }

            if (!russianRoulette(rikudo, rng)) {
                break;
            }
            rikudo.bounceCount++;
        }

        finalColor = add3(finalColor, rikudo.accumulatedColor);
    }

    finalColor = div3(finalColor, static_cast<float>(SAMPLES_PER_PIXEL));

    // Welford streaming average into the linear accumulation buffer, then
    // gamma-encode the running mean into the display PBO.
    const Float3 blended =
        accumulateWelford(params.accumBuffer, pixelIndex, finalColor, params.frameIndex);

    params.framebuffer[y * params.width + x] = packPixelGamma(blended);
}

extern "C" __global__ void __closesthit__radiance()
{
    Intersection* its =
        reinterpret_cast<Intersection*>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));

    // Primitive index is GAS-local; recover the flat triangle index via the
    // instance's triangle offset (see cudaInitObjects / ObjectDesc).
    const unsigned int instId = optixGetInstanceId();
    const int triIdx = params.objectTriOffset[instId] + static_cast<int>(optixGetPrimitiveIndex());
    const float2 bary  = optixGetTriangleBarycentrics();
    const float  u     = bary.x;
    const float  v     = bary.y;
    const float  w0    = 1.0f - u - v;

    const TriangleData tri = params.triangles[triIdx];

    const float  t = optixGetRayTmax();
    const float3 O = optixGetWorldRayOrigin();
    const float3 D = optixGetWorldRayDirection();

    its->t             = t;
    its->triangleIndex = triIdx;
    // World-space hit point from the world ray (already in world).
    its->hitPoint      = { O.x + t * D.x, O.y + t * D.y, O.z + t * D.z };

    // Normals live in object-local space; transform with the inverse-transpose.
    const float3 localN = {
        w0 * tri.n0.x + u * tri.n1.x + v * tri.n2.x,
        w0 * tri.n0.y + u * tri.n1.y + v * tri.n2.y,
        w0 * tri.n0.z + u * tri.n1.z + v * tri.n2.z
    };
    const float3 worldN = optixTransformNormalFromObjectToWorldSpace(localN);
    its->hitNormal = normalize3({ worldN.x, worldN.y, worldN.z });

    its->uv = {
        w0 * tri.uv0.x + u * tri.uv1.x + v * tri.uv2.x,
        w0 * tri.uv0.y + u * tri.uv1.y + v * tri.uv2.y
    };
}

extern "C" __global__ void __miss__radiance()
{
    Intersection* its =
        reinterpret_cast<Intersection*>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));
    its->triangleIndex = -1;
}

extern "C" __global__ void __miss__shadow()
{
    // Ray reached tMax without hitting anything -> not occluded.
    optixSetPayload_0(0u);
}
