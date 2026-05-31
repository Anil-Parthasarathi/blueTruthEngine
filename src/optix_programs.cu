// ============================================================================
//  optix_programs.cu  –  OptiX raygen path tracer (RT-core accelerated)
// ============================================================================
//  This file is compiled to PTX (see CMakeLists.txt) and loaded as an OptiX
//  module at runtime.  It contains the ray generation, closest-hit and miss
//  programs.
//
//  The path-tracing / shading logic is a direct port of the previous CUDA
//  megakernel (renderKernel + traceRay in render_kernel.cu).  The ONLY changes
//  are:
//    • sceneIntersect()        -> optixTrace() closest-hit  (RT cores)
//    • sceneIntersectAnyHit()  -> optixTrace() terminate-on-first-hit
//  All BSDF / emitter / RNG / accumulation logic is reused unchanged from the
//  shared .cuh headers.
// ============================================================================

#include <optix.h>

#include "optix_launch_params.h"
#include "render_kernel.h"
#include "rt_cuda_math.cuh"
#include "rt_intersect.cuh"          // Ray + Intersection structs
#include "rt_emitter_sampling.cuh"
#include "rt_bsdf.cuh"
#include "rt_rng.cuh"

#include <cstdint>

// ---------------------------------------------------------------------------
//  Launch parameters (populated by OptiX from the device pointer given to
//  optixLaunch).  The variable name MUST match pipelineLaunchParamsVariableName.
// ---------------------------------------------------------------------------
extern "C" __constant__ LaunchParams params;

// Mirror the megakernel's path-tracing constants.
static constexpr int   SAMPLES_PER_PIXEL = 5;
static constexpr int   MAX_BOUNCES       = 200;
static constexpr float RT_EPSILON        = 1e-4f;

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
//  Mirror of PathState from the megakernel.
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
//  Closest-hit intersection (RT cores) — replaces sceneIntersect().
// ---------------------------------------------------------------------------
static __forceinline__ __device__ bool traceClosest(const Ray& ray, Intersection& its)
{
    its.triangleIndex = -1;
    unsigned int u0, u1;
    packPointer(&its, u0, u1);

    optixTrace(params.handle,
               toF3(ray.origin), toF3(ray.direction),
               RT_EPSILON, 1e30f, 0.0f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,
               RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE,
               u0, u1);

    return its.triangleIndex >= 0;
}

// ---------------------------------------------------------------------------
//  Occlusion test — replaces sceneIntersectAnyHit().
//  Returns true if anything blocks the segment [origin, origin + dir*tMax].
// ---------------------------------------------------------------------------
static __forceinline__ __device__ bool traceOccluded(const Float3& origin,
                                                      const Float3& dir,
                                                      float tMax)
{
    unsigned int occluded = 1u; // assume blocked; miss program clears it
    optixTrace(params.handle,
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
//  Shading helpers (ported verbatim from render_kernel.cu, reading `params`).
// ---------------------------------------------------------------------------
static __forceinline__ __device__ EmitterSamplingData makeEmitterSampling()
{
    EmitterSamplingData s{};
    s.triangles         = params.triangles;
    s.emitters          = params.emitters;
    s.emitterCount      = params.emitterCount;
    s.emitterTriIndices = params.emitterTriIndices;
    s.emitterTriCdf     = params.emitterTriCdf;
    s.sceneEmitterCdf   = params.sceneEmitterCdf;
    return s;
}

static __forceinline__ __device__ float convertAreaPDFtoSolidAnglePDF(
    const float lightPDFArea,
    const EmitterQueryRecord& emitterQuery)
{
    const Float3 diff = sub3(emitterQuery.hitPoint, emitterQuery.originPoint);
    const float dist2 = dot3(diff, diff);
    const Float3 negLightDir = {-emitterQuery.directionToLight.x,
                                -emitterQuery.directionToLight.y,
                                -emitterQuery.directionToLight.z};
    const float denom = fmaxf(fabsf(dot3(emitterQuery.hitNormal, negLightDir)), 1e-3f);
    return lightPDFArea * dist2 / denom;
}

static __forceinline__ __device__ void accumulateEmitterHit(
    PathState& pathRecord,
    const Intersection& its)
{
    const Float3 surfaceRadiance = params.triangleEmission[its.triangleIndex];

    EmitterQueryRecord directEmitterQuery{};
    directEmitterQuery.originPoint      = pathRecord.ray.origin;
    directEmitterQuery.hitPoint         = its.hitPoint;
    directEmitterQuery.hitNormal        = its.hitNormal;
    directEmitterQuery.directionToLight = pathRecord.ray.direction;

    float brdfWeight = 1.0f;

    if (!pathRecord.specularBounce) {
        const EmitterSamplingData es = makeEmitterSampling();
        const int emitterIdx = findEmitterIndexForTriangle(
            its.triangleIndex, es.emitters, es.emitterCount, es.emitterTriIndices);

        if (emitterIdx >= 0) {
            const float lightPDFAreaBRDF =
                emitterProbabilityEvaluatorFromMesh(es.emitters[emitterIdx]) *
                sceneGetEmitterPDF(emitterIdx, es.sceneEmitterCdf, es.emitterCount);

            const float lightPDFSolidAngleBRDF =
                convertAreaPDFtoSolidAnglePDF(lightPDFAreaBRDF, directEmitterQuery);

            const float weightDenomSumBRDF = pathRecord.brdfPDF + lightPDFSolidAngleBRDF;

            if (weightDenomSumBRDF <= 0.0f) brdfWeight = 0.0f;
            else                            brdfWeight = pathRecord.brdfPDF / weightDenomSumBRDF;
        }
    }

    const Float3 rad = checkRadiance(surfaceRadiance, directEmitterQuery);
    pathRecord.accumulatedColor = add3(
        pathRecord.accumulatedColor,
        mul3(mul3(rad, pathRecord.throughput), brdfWeight));
}

static __forceinline__ __device__ void sampleAreaEmitterNEE(
    PathState& pathRecord,
    RngState& rng,
    const Intersection& its,
    const BsdfData& bsdf)
{
    const EmitterSamplingData es = makeEmitterSampling();

    const float uLight0 = rngNextFloat01(rng);
    const float uLight1 = rngNextFloat01(rng);

    EmitterQueryRecord emitterQuery{};
    emitterQuery.originPoint = its.hitPoint;

    Float3 le = sampleGenerator(es, emitterQuery, uLight0, uLight1);

    if ((le.x <= 0.0f && le.y <= 0.0f && le.z <= 0.0f) ||
        emitterQuery.emitterPdf <= 0.0f ||
        probabilityEvaluator(emitterQuery) <= 0.0f) {
        return;
    }

    const float distToLight = distance3(emitterQuery.hitPoint, its.hitPoint);

    if (traceOccluded(its.hitPoint, emitterQuery.directionToLight, distToLight - RT_EPSILON)) {
        return;
    }

    const float geo =
        fabsf(dot3(its.hitNormal, emitterQuery.directionToLight)) *
        fabsf(dot3(emitterQuery.hitNormal, mul3(emitterQuery.directionToLight, -1.0f))) /
        (distToLight * distToLight);

    BsdfQueryRecord bsdfQueryDirect{};
    bsdfQueryDirect.wi      = toLocalFromNormal(its.hitNormal, mul3(pathRecord.ray.direction, -1.0f));
    bsdfQueryDirect.wo      = toLocalFromNormal(its.hitNormal, emitterQuery.directionToLight);
    bsdfQueryDirect.measure = BSDF_ESolidAngle;

    const Float3 fr = bsdfEval(bsdf, bsdfQueryDirect);

    const float lightPDFareaDirect      = emitterQuery.emitterPdf * emitterQuery.pdf;
    const float lightPDFSolidAngleDirect = convertAreaPDFtoSolidAnglePDF(lightPDFareaDirect, emitterQuery);
    const float brdfPDFDirect           = bsdfPdf(bsdf, bsdfQueryDirect);
    const float weightDenomSumDirect    = brdfPDFDirect + lightPDFSolidAngleDirect;

    if (weightDenomSumDirect <= 0.0f) return;

    const float lightWeight = lightPDFSolidAngleDirect / weightDenomSumDirect;

    Float3 contrib = mul3(fr, le);
    contrib = mul3(contrib, lightWeight);
    contrib = mul3(contrib, geo);
    contrib = div3(contrib, lightPDFareaDirect);

    pathRecord.accumulatedColor =
        add3(pathRecord.accumulatedColor, mul3(contrib, pathRecord.throughput));
}

static __forceinline__ __device__ void sampleSpotlightNEE(
    PathState& pathRecord,
    const Intersection& its,
    const BsdfData& bsdf)
{
    for (int si = 0; si < params.spotlightCount; ++si) {
        const SpotlightData spot = params.spotlights[si];

        const Float3 originToLight = sub3(spot.position, its.hitPoint);
        const float sqDist = dot3(originToLight, originToLight);
        if (sqDist < RT_EPSILON * RT_EPSILON) continue;
        const float dist       = sqrtf(sqDist);
        const Float3 dirToLight = div3(originToLight, dist);

        const float lightCos = dot3(mul3(dirToLight, -1.0f), spot.direction);

        Float3 spotLe;
        if (lightCos <= spot.outerConeCosine) {
            continue;
        } else if (lightCos >= spot.innerConeCosine) {
            spotLe = mul3(spot.radiance, spot.intensity / sqDist);
        } else {
            float falloff = (lightCos - spot.outerConeCosine) /
                            (spot.innerConeCosine - spot.outerConeCosine);
            float smoothedFalloff = (falloff * falloff) * (3.0f - (2.0f * falloff));
            spotLe = mul3(spot.radiance, (spot.intensity * smoothedFalloff) / sqDist);
        }

        if (traceOccluded(its.hitPoint, dirToLight, dist - RT_EPSILON)) {
            continue;
        }

        BsdfQueryRecord bsdfQuerySpot{};
        bsdfQuerySpot.wi      = toLocalFromNormal(its.hitNormal, mul3(pathRecord.ray.direction, -1.0f));
        bsdfQuerySpot.wo      = toLocalFromNormal(its.hitNormal, dirToLight);
        bsdfQuerySpot.measure = BSDF_ESolidAngle;

        const Float3 fr = bsdfEval(bsdf, bsdfQuerySpot);

        const float cosSurface = fabsf(dot3(its.hitNormal, dirToLight));
        const Float3 contrib   = mul3(mul3(mul3(fr, spotLe), cosSurface), pathRecord.throughput);
        pathRecord.accumulatedColor = add3(pathRecord.accumulatedColor, contrib);
    }
}

// ---------------------------------------------------------------------------
//  Main path-tracing step: intersect, shade, advance ray.
//  Direct port of traceRay() with the intersection swapped to optixTrace().
// ---------------------------------------------------------------------------
static __forceinline__ __device__ bool traceRay(PathState& pathRecord, RngState& rng)
{
    Intersection its{};
    if (!traceClosest(pathRecord.ray, its)) {
        return false;
    }

    const int bsdfId = params.triangleBsdfIds[its.triangleIndex];
    BsdfData bsdf = params.bsdfs[bsdfId];

    // Texture sampling (identical to megakernel).
    if (params.texObjects != nullptr && params.triangleMaterialIds != nullptr) {
        const int matId = params.triangleMaterialIds[its.triangleIndex];
        if (matId >= 0 && matId < params.textureCount) {
            const cudaTextureObject_t texObj = params.texObjects[matId];
            if (texObj != 0) {
                const float4 s = tex2D<float4>(texObj, its.uv.x, its.uv.y);
                bsdf.p0.x = s.x;
                bsdf.p0.y = s.y;
                bsdf.p0.z = s.z;
                if (bsdf.type == BSDF_Microfacet) {
                    bsdf.p1.w = 1.0f - fmaxf(bsdf.p0.x, fmaxf(bsdf.p0.y, bsdf.p0.z));
                }
            }
        }
    }

    if (params.triangleEmitterFlags[its.triangleIndex] != 0) {
        accumulateEmitterHit(pathRecord, its);
    }

    if (bsdfIsDiffuse(bsdf)) {
        pathRecord.specularBounce = false;
        sampleAreaEmitterNEE(pathRecord, rng, its, bsdf);
        sampleSpotlightNEE(pathRecord, its, bsdf);
    } else {
        pathRecord.specularBounce = true;
    }

    BsdfQueryRecord bsdfQueryIndirect{};
    bsdfQueryIndirect.wi       = toLocalFromNormal(its.hitNormal, mul3(pathRecord.ray.direction, -1.0f));
    bsdfQueryIndirect.rndExtra = rngNextFloat01(rng);

    const float u0 = rngNextFloat01(rng);
    const float u1 = rngNextFloat01(rng);
    Float3 sampleWeight = bsdfSample(bsdf, bsdfQueryIndirect, u0, u1);

    pathRecord.brdfPDF = bsdfPdf(bsdf, bsdfQueryIndirect);

    const Float3 bounceDir = normalize3(toWorldFromNormal(its.hitNormal, bsdfQueryIndirect.wo));

    pathRecord.ray.origin    = add3(its.hitPoint, mul3(bounceDir, RT_EPSILON));
    pathRecord.ray.direction = bounceDir;
    pathRecord.throughput    = mul3(pathRecord.throughput, sampleWeight);
    pathRecord.eta          *= bsdfQueryIndirect.eta;

    return true;
}

// ---------------------------------------------------------------------------
//  Primary ray generation (port of generatePrimaryRay, reads params.camera).
// ---------------------------------------------------------------------------
static __forceinline__ __device__ Ray generatePrimaryRay(int px, int py, float jx, float jy)
{
    const float fx = (static_cast<float>(px) + 0.5f + jx) / static_cast<float>(params.width);
    const float fy = (static_cast<float>(py) + 0.5f + jy) / static_cast<float>(params.height);
    const float ndcX = 2.0f * fx - 1.0f;
    const float ndcY = 2.0f * fy - 1.0f;

    const float tanHalfFovY = tanf(0.5f * params.camera.fovYRadians);
    const float sx = ndcX * params.camera.aspect * tanHalfFovY;
    const float sy = ndcY * tanHalfFovY;

    Float3 dir = add3(params.camera.forward,
                      add3(mul3(params.camera.right, sx),
                           mul3(params.camera.up, sy)));

    Ray ray;
    ray.origin    = params.camera.origin;
    ray.direction = normalize3(dir);
    return ray;
}

// ===========================================================================
//  Programs
// ===========================================================================

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const int x = static_cast<int>(idx.x);
    const int y = static_cast<int>(idx.y) + params.launchOffsetY;
    if (y >= params.height) return;

    const uint32_t pixelIndex =
        static_cast<uint32_t>(y) * static_cast<uint32_t>(params.width) + static_cast<uint32_t>(x);
    RngState rng = makeRng(hashUint(pixelIndex ^ (params.frameIndex * 0x9e3779b9u)));

    Float3 finalColor = {0.0f, 0.0f, 0.0f};

    for (int i = 0; i < SAMPLES_PER_PIXEL; ++i) {
        const float jx = rngNextFloat01(rng) - 0.5f;
        const float jy = rngNextFloat01(rng) - 0.5f;

        PathState rikudo;
        rikudo.ray              = generatePrimaryRay(x, y, jx, jy);
        rikudo.throughput       = {1.0f, 1.0f, 1.0f};
        rikudo.accumulatedColor = {0.0f, 0.0f, 0.0f};
        rikudo.bounceCount      = 0;
        rikudo.pixelIndex       = pixelIndex;
        rikudo.eta              = 1.0f;
        rikudo.brdfPDF          = 0.0f;
        rikudo.specularBounce   = true;

        while (rikudo.bounceCount < MAX_BOUNCES) {
            if (!traceRay(rikudo, rng)) {
                break;
            }

            if (rikudo.bounceCount > 3) {
                const float continuationProb =
                    fminf(maxCoeff3(rikudo.throughput) * rikudo.eta * rikudo.eta, 0.99f);
                if (rngNextFloat01(rng) >= continuationProb) {
                    break;
                }
                rikudo.throughput = div3(rikudo.throughput, continuationProb);
            }
            rikudo.bounceCount++;
        }

        finalColor = add3(finalColor, rikudo.accumulatedColor);
    }

    finalColor = div3(finalColor, static_cast<float>(SAMPLES_PER_PIXEL));

    // Welford streaming average into the linear accumulation buffer.
    const float n = static_cast<float>(params.frameIndex + 1);
    Float3 prev = params.accumBuffer[pixelIndex];
    Float3 blended = {
        prev.x + (finalColor.x - prev.x) / n,
        prev.y + (finalColor.y - prev.y) / n,
        prev.z + (finalColor.z - prev.z) / n
    };
    params.accumBuffer[pixelIndex] = blended;

    // Gamma (linear -> sRGB, gamma 2.2) applied to the running average.
    float dr = powf(fmaxf(0.0f, fminf(blended.x, 1.0f)), 1.0f / 2.2f);
    float dg = powf(fmaxf(0.0f, fminf(blended.y, 1.0f)), 1.0f / 2.2f);
    float db = powf(fmaxf(0.0f, fminf(blended.z, 1.0f)), 1.0f / 2.2f);

    uint8_t r = static_cast<uint8_t>(dr * 255.0f);
    uint8_t g = static_cast<uint8_t>(dg * 255.0f);
    uint8_t b = static_cast<uint8_t>(db * 255.0f);
    uint8_t a = 255;

    uint32_t pixel = (a << 24) | (b << 16) | (g << 8) | r;
    params.framebuffer[y * params.width + x] = pixel;
}

extern "C" __global__ void __closesthit__radiance()
{
    Intersection* its =
        reinterpret_cast<Intersection*>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));

    const int   triIdx = optixGetPrimitiveIndex();
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
    its->hitPoint      = { O.x + t * D.x, O.y + t * D.y, O.z + t * D.z };

    its->hitNormal = normalize3({
        w0 * tri.n0.x + u * tri.n1.x + v * tri.n2.x,
        w0 * tri.n0.y + u * tri.n1.y + v * tri.n2.y,
        w0 * tri.n0.z + u * tri.n1.z + v * tri.n2.z
    });

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
