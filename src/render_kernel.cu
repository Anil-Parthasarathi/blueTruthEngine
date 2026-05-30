// ============================================================================
//  render_kernel.cu  –  CUDA-GL interop + simple triangle rasteriser
// ============================================================================
//  This file owns all CUDA resources.  The host-side API declared in
//  render_kernel.h is implemented here.
//
//  Architecture for future path tracing:
//    • cudaInit()       – upload scene data (mesh, materials, BVH, …)
//    • cudaRegisterPBO() – register the GL pixel-buffer for interop
//    • cudaRender()     – launch the render kernel (swap in your tracer here)
//    • cudaCleanup()    – release everything
// ============================================================================

#include "render_kernel.h"
#include "rt_cuda_math.cuh"
#include "rt_intersect.cuh"
#include "rt_emitter_sampling.cuh"
#include "rt_bsdf.cuh"
#include "rt_rng.cuh"

#include <glad/gl.h>       // Must come before cuda_gl_interop.h (defines GLuint)
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>

static constexpr int   SAMPLES_PER_PIXEL = 5;
static constexpr int   MAX_BOUNCES   = 200;
static constexpr float RT_EPSILON    = 1e-4f;

struct PathState {
    Ray      ray;
    Float3   throughput;
    Float3   accumulatedColor;
    int      bounceCount;
    float    eta;
    float brdfPDF;
    bool specularBounce;
    uint32_t pixelIndex;
};

struct DirectLightingContext {
    const uint8_t* triangleEmitterFlags;
    const Float3* triangleEmission;
    const LinearBVHNode* bvhNodes;
    int bvhNodeCount;
    const int* bvhPrimIndices;
    EmitterSamplingData emitterSampling;
    const SpotlightData* spotlights;
    int spotlightCount;
    // Texture sampling: array of CUDA texture object handles (one per material),
    // and the per-triangle material index for lookup.
    const cudaTextureObject_t* texObjects;
    const int* triangleMaterialIds;
    int textureCount;
};

// ---------------------------------------------------------------------------
//  Error-checking helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d – %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
//  CUDA-GL interop state
// ---------------------------------------------------------------------------
static cudaGraphicsResource* s_pboResource = nullptr;

// ---------------------------------------------------------------------------
//  Progressive accumulation state
// ---------------------------------------------------------------------------
// Stores the running per-pixel linear-space average across all frames.
// Gamma correction is applied only when writing to the display PBO.
static Float3*   s_accumBuffer_d  = nullptr;
static int       s_accumWidth     = 0;
static int       s_accumHeight    = 0;
static uint32_t  s_frameIndex     = 0;  // number of samples already in the buffer

// ---------------------------------------------------------------------------
//  Device-side scene data (flattened)
// ---------------------------------------------------------------------------
static TriangleData*   s_triangles_d          = nullptr;
static int              s_triangleCount       = 0;
static Float3*         s_materials_d          = nullptr;
static int              s_materialCount       = 0;
static int*            s_triangleMaterialIds_d = nullptr;

// Camera used for ray generation (set once at startup).
__constant__ CameraData d_camera;

// Per-triangle emission (same length as s_triangles_d). Non-emissive = (0,0,0).
static Float3* s_triangleEmission_d = nullptr;
// Per-triangle: 1 = triangle belongs to an emissive mesh (scene mesh isEmitter).
static uint8_t* s_triangleEmitterFlags_d = nullptr;

// Global mesh-emitter sampling data (union of all emissive triangles for now).
static int*   s_emissiveTriIndices_d = nullptr;
static float* s_emissiveTriCdf_d     = nullptr;
static int    s_emissiveTriCount     = 0;

// Nori-like emitter table (one entry per emitter mesh)
static EmitterData* s_emitters_d = nullptr;
static int          s_emitterCount = 0;
static int*         s_emitterTriIndices_d = nullptr; // concatenated
static float*       s_emitterTriCdf_d     = nullptr; // concatenated
static float*       s_sceneEmitterCdf_d   = nullptr; // length emitterCount+1

// BSDF plumbing
static BsdfData* s_bsdfs_d = nullptr;
static int       s_bsdfCount = 0;
static int*      s_triangleBsdfIds_d = nullptr; // length triangleCount

// BVH acceleration structure (built on CPU by FastBVH, traversed on GPU)
static LinearBVHNode* s_bvhNodes_d       = nullptr;
static int            s_bvhNodeCount     = 0;
static int*           s_bvhPrimIndices_d = nullptr; // reordered triangle indices

// Spotlight point lights (handled separately from mesh emitters)
static SpotlightData* s_spotlights_d    = nullptr;
static int            s_spotlightCount  = 0;

// Per-material CUDA texture objects — one per scene material, indexed by triangleMaterialIds.
// Host-side arrays are kept for cleanup; the device array holds the opaque handles.
static std::vector<cudaArray_t>         s_cuArrays;        // pixel data on device (host handles)
static std::vector<cudaTextureObject_t> s_texObjects_h;    // texture object handles (host copy)
static cudaTextureObject_t*             s_texObjects_d = nullptr; // device array of handles
static int                              s_textureCount = 0;

// ---------------------------------------------------------------------------
//  Render kernel
// ---------------------------------------------------------------------------
//  Each thread computes one pixel.  A simple edge-function test determines
//  whether the pixel lies inside the triangle.  If it does, the pixel is
//  coloured with the uniform texture colour; otherwise a dark background
//  is written.
//
//  ► To convert this into a path tracer, replace the body of this kernel
//    with ray generation + tracing + shading.
// ---------------------------------------------------------------------------

__device__ float edgeFunction(float ax, float ay,
                               float bx, float by,
                               float cx, float cy)
{
    // Standard edge function: (B-A) × (C-A)
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

// jx, jy are sub-pixel offsets in [-0.5, 0.5] drawn from the per-frame RNG.
// With temporal accumulation each frame sees a different jitter position,
// giving both antialiasing and reduced variance as samples pile up.
__device__ __forceinline__ Ray generatePrimaryRay(int px, int py, int width, int height,
                                                   float jx, float jy)
{
    const float fx = (static_cast<float>(px) + 0.5f + jx) / static_cast<float>(width);
    const float fy = (static_cast<float>(py) + 0.5f + jy) / static_cast<float>(height);
    const float ndcX = 2.0f * fx - 1.0f;
    // Flip Y so the image isn't vertically inverted on display.
    const float ndcY = 2.0f * fy - 1.0f;

    const float tanHalfFovY = tanf(0.5f * d_camera.fovYRadians);
    const float sx = ndcX * d_camera.aspect * tanHalfFovY;
    const float sy = ndcY * tanHalfFovY;

    Float3 dir = add3(d_camera.forward,
                      add3(mul3(d_camera.right, sx),
                           mul3(d_camera.up, sy)));

    Ray ray;
    ray.origin = d_camera.origin;
    ray.direction = normalize3(dir);
    return ray;
}

__device__ __forceinline__ float convertAreaPDFtoSolidAnglePDF(
    const float lightPDFArea,
    const EmitterQueryRecord& emitterQuery)
{
    // Converting from area to solid angle is area pdf * distance^2 /
    // abs(cos(theta))
    const Float3 diff = sub3(emitterQuery.hitPoint, emitterQuery.originPoint);
    const float dist2 = dot3(diff, diff);
    const Float3 negLightDir = {-emitterQuery.directionToLight.x,
                                -emitterQuery.directionToLight.y,
                                -emitterQuery.directionToLight.z};
    const float denom = fmaxf(fabsf(dot3(emitterQuery.hitNormal, negLightDir)), 1e-3f);
    return lightPDFArea * dist2 / denom;
}

// ---------------------------------------------------------------------------
//  NEE helpers  (called from traceRay, broken out for readability)
// ---------------------------------------------------------------------------

// Handles the BRDF-sampled emitter hit: computes the MIS BRDF weight and
// accumulates the emitted radiance into the path's color.
__device__ __forceinline__ void accumulateEmitterHit(
    PathState& pathRecord,
    const Intersection& its,
    const DirectLightingContext& lightingCtx)
{
    const Float3 surfaceRadiance = lightingCtx.triangleEmission[its.triangleIndex];

    EmitterQueryRecord directEmitterQuery{};
    directEmitterQuery.originPoint      = pathRecord.ray.origin;
    directEmitterQuery.hitPoint         = its.hitPoint;
    directEmitterQuery.hitNormal        = its.hitNormal;
    directEmitterQuery.directionToLight = pathRecord.ray.direction;

    float brdfWeight = 1.0f;

    if (!pathRecord.specularBounce) {
        const int emitterIdx = findEmitterIndexForTriangle(
            its.triangleIndex,
            lightingCtx.emitterSampling.emitters,
            lightingCtx.emitterSampling.emitterCount,
            lightingCtx.emitterSampling.emitterTriIndices);

        if (emitterIdx >= 0) {
            const float lightPDFAreaBRDF =
                emitterProbabilityEvaluatorFromMesh(
                    lightingCtx.emitterSampling.emitters[emitterIdx]) *
                sceneGetEmitterPDF(emitterIdx,
                                   lightingCtx.emitterSampling.sceneEmitterCdf,
                                   lightingCtx.emitterSampling.emitterCount);

            const float lightPDFSolidAngleBRDF =
                convertAreaPDFtoSolidAnglePDF(lightPDFAreaBRDF, directEmitterQuery);

            const float weightDenomSumBRDF =
                pathRecord.brdfPDF + lightPDFSolidAngleBRDF;

            if (weightDenomSumBRDF <= 0.0f) {
                brdfWeight = 0.0f;
            } else {
                brdfWeight = pathRecord.brdfPDF / weightDenomSumBRDF;
            }
        }
    }

    const Float3 rad = checkRadiance(surfaceRadiance, directEmitterQuery);
    pathRecord.accumulatedColor = add3(
        pathRecord.accumulatedColor,
        mul3(mul3(rad, pathRecord.throughput), brdfWeight));
}

// Samples one randomly chosen area (mesh) emitter using MIS and accumulates
// its contribution into the path's color.
__device__ __forceinline__ void sampleAreaEmitterNEE(
    PathState& pathRecord,
    RngState& rng,
    const Intersection& its,
    const BsdfData& bsdf,
    const DirectLightingContext& lightingCtx)
{
    const float uLight0 = rngNextFloat01(rng);
    const float uLight1 = rngNextFloat01(rng);

    EmitterQueryRecord emitterQuery{};
    emitterQuery.originPoint = its.hitPoint;

    Float3 le = sampleGenerator(lightingCtx.emitterSampling, emitterQuery, uLight0, uLight1);

    if ((le.x <= 0.0f && le.y <= 0.0f && le.z <= 0.0f) ||
        emitterQuery.emitterPdf <= 0.0f ||
        probabilityEvaluator(emitterQuery) <= 0.0f) {
        return;
    }

    const float distToLight = distance3(emitterQuery.hitPoint, its.hitPoint);

    Ray shadowRay{};
    shadowRay.origin    = its.hitPoint;
    shadowRay.direction = emitterQuery.directionToLight;

    if (sceneIntersectAnyHit(shadowRay, lightingCtx.emitterSampling.triangles,
                             lightingCtx.bvhNodes, lightingCtx.bvhNodeCount,
                             lightingCtx.bvhPrimIndices,
                             RT_EPSILON, distToLight - RT_EPSILON)) {
        return;
    }

    const float geo =
        fabsf(dot3(its.hitNormal, emitterQuery.directionToLight)) *
        fabsf(dot3(emitterQuery.hitNormal,
                   mul3(emitterQuery.directionToLight, -1.0f))) /
        (distToLight * distToLight);

    BsdfQueryRecord bsdfQueryDirect{};
    bsdfQueryDirect.wi =
        toLocalFromNormal(its.hitNormal, mul3(pathRecord.ray.direction, -1.0f));
    bsdfQueryDirect.wo =
        toLocalFromNormal(its.hitNormal, emitterQuery.directionToLight);
    bsdfQueryDirect.measure = BSDF_ESolidAngle;

    const Float3 fr = bsdfEval(bsdf, bsdfQueryDirect);

    const float lightPDFareaDirect   = emitterQuery.emitterPdf * emitterQuery.pdf;
    const float lightPDFSolidAngleDirect =
        convertAreaPDFtoSolidAnglePDF(lightPDFareaDirect, emitterQuery);
    const float brdfPDFDirect        = bsdfPdf(bsdf, bsdfQueryDirect);
    const float weightDenomSumDirect = brdfPDFDirect + lightPDFSolidAngleDirect;

    if (weightDenomSumDirect <= 0.0f) return;

    const float lightWeight = lightPDFSolidAngleDirect / weightDenomSumDirect;

    Float3 contrib = mul3(fr, le);
    contrib = mul3(contrib, lightWeight);
    contrib = mul3(contrib, geo);
    contrib = div3(contrib, lightPDFareaDirect);

    pathRecord.accumulatedColor =
        add3(pathRecord.accumulatedColor, mul3(contrib, pathRecord.throughput));
}

// Iterates every spotlight and accumulates each one's NEE contribution.
// Point lights have a delta PDF — probabilityEvaluator returns 0 and
// checkRadiance returns 0 (BRDF rays can never accidentally hit a point).
// So there is no MIS: we use lightWeight = 1.0 for each spotlight.
__device__ __forceinline__ void sampleSpotlightNEE(
    PathState& pathRecord,
    const Intersection& its,
    const BsdfData& bsdf,
    const DirectLightingContext& lightingCtx)
{
    for (int si = 0; si < lightingCtx.spotlightCount; ++si) {
        const SpotlightData spot = lightingCtx.spotlights[si];

        // set the emitter query record
        // simply give the hitpoint as the origin point of the spot light
        // hit normal is the opposite of the direction to the light since the spot light is facing in the direction vector
        // pdf is 1 since there is only one possible point to hit
        const Float3 originToLight = sub3(spot.position, its.hitPoint);
        const float sqDist = dot3(originToLight, originToLight);
        if (sqDist < RT_EPSILON * RT_EPSILON) continue;
        const float dist     = sqrtf(sqDist);
        const Float3 dirToLight = div3(originToLight, dist);

        // get the cosine of the angle between the direction to the light and the forward direction of the spot light
        const float lightCos = dot3(mul3(dirToLight, -1.0f), spot.direction);

        // if the angle is greater than the outer cone angle, return 0
        // if the angle is greater then return the full radiance (scaled by distance)
        // otherwise interpolate by finding the falloff
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

        // Shadow ray — bound to just before the light to avoid self-intersection
        Ray shadowRay{};
        shadowRay.origin    = its.hitPoint;
        shadowRay.direction = dirToLight;
        if (sceneIntersectAnyHit(shadowRay, lightingCtx.emitterSampling.triangles,
                                 lightingCtx.bvhNodes, lightingCtx.bvhNodeCount,
                                 lightingCtx.bvhPrimIndices,
                                 RT_EPSILON, dist - RT_EPSILON)) {
            continue;
        }

        BsdfQueryRecord bsdfQuerySpot{};
        bsdfQuerySpot.wi =
            toLocalFromNormal(its.hitNormal, mul3(pathRecord.ray.direction, -1.0f));
        bsdfQuerySpot.wo      = toLocalFromNormal(its.hitNormal, dirToLight);
        bsdfQuerySpot.measure = BSDF_ESolidAngle;

        const Float3 fr = bsdfEval(bsdf, bsdfQuerySpot);

        // point light so we assume that random bounces never hit the spot light directly
        // contribution: fr * Le * cos_surface  (pdf = 1, no area term)
        const float cosSurface = fabsf(dot3(its.hitNormal, dirToLight));
        const Float3 contrib   = mul3(mul3(mul3(fr, spotLe), cosSurface), pathRecord.throughput);
        pathRecord.accumulatedColor = add3(pathRecord.accumulatedColor, contrib);
    }
}

// ---------------------------------------------------------------------------
//  Main path-tracing step: intersect, shade, advance ray
// ---------------------------------------------------------------------------
__device__ bool traceRay(
    PathState& pathRecord,
    RngState& rng,
    const DirectLightingContext& lightingCtx,
    const BsdfData* bsdfs,
    const int* triangleBsdfIds,
    int triangleCount)
{
    Intersection its{};
    if (!sceneIntersect(pathRecord.ray, lightingCtx.emitterSampling.triangles, triangleCount,
                         lightingCtx.bvhNodes, lightingCtx.bvhNodeCount, lightingCtx.bvhPrimIndices,
                         RT_EPSILON, 1e30f, its)) {
        return false;
    }

    const int bsdfId = triangleBsdfIds[its.triangleIndex];
    BsdfData bsdf = bsdfs[bsdfId];

    // Texture sampling
    if (lightingCtx.texObjects != nullptr && lightingCtx.triangleMaterialIds != nullptr) {
        const int matId = lightingCtx.triangleMaterialIds[its.triangleIndex];
        if (matId >= 0 && matId < lightingCtx.textureCount) {
            const cudaTextureObject_t texObj = lightingCtx.texObjects[matId];
            if (texObj != 0) {
                // tex2D returns normalised float4 in [0,1] (readMode = NormalizedFloat).
                // We use the values as-is (no gamma decode) so that textures and
                // XML albedo/base_color parameters live in the same convention.
                const float4 s = tex2D<float4>(texObj, its.uv.x, its.uv.y);
                bsdf.p0.x = s.x;
                bsdf.p0.y = s.y;
                bsdf.p0.z = s.z;
                // Microfacet derives ks = 1 - max(kd) from the albedo.
                // Recompute after texture override to stay energy-conserving.
                if (bsdf.type == BSDF_Microfacet) {
                    bsdf.p1.w = 1.0f - fmaxf(bsdf.p0.x, fmaxf(bsdf.p0.y, bsdf.p0.z));
                }
            }
        }
    }

    // Check if hit an emitter
    if (lightingCtx.triangleEmitterFlags[its.triangleIndex] != 0) {
        accumulateEmitterHit(pathRecord, its, lightingCtx);
    }

    if (bsdfIsDiffuse(bsdf)) {
        pathRecord.specularBounce = false;
        sampleAreaEmitterNEE(pathRecord, rng, its, bsdf, lightingCtx);
        sampleSpotlightNEE(pathRecord, its, bsdf, lightingCtx);
    } else {
        pathRecord.specularBounce = true;
    }

    BsdfQueryRecord bsdfQueryIndirect{};
    bsdfQueryIndirect.wi =
        toLocalFromNormal(its.hitNormal, mul3(pathRecord.ray.direction, -1.0f));

    bsdfQueryIndirect.rndExtra = rngNextFloat01(rng);

    const float u0 = rngNextFloat01(rng);
    const float u1 = rngNextFloat01(rng);
    Float3 sampleWeight = bsdfSample(bsdf, bsdfQueryIndirect, u0, u1);

    pathRecord.brdfPDF = bsdfPdf(bsdf, bsdfQueryIndirect);

    const Float3 bounceDir =
        normalize3(toWorldFromNormal(its.hitNormal, bsdfQueryIndirect.wo));

    pathRecord.ray.origin =
        add3(its.hitPoint, mul3(bounceDir, RT_EPSILON));
    pathRecord.ray.direction = bounceDir;
    pathRecord.throughput    = mul3(pathRecord.throughput, sampleWeight);
    pathRecord.eta *= bsdfQueryIndirect.eta;

    return true;
}

__global__ void renderKernel(uint32_t* framebuffer,
                              Float3*   accumBuffer,
                              int width, int height,
                              const TriangleData* triangles,
                              int triangleCount,
                              const Float3* materials,
                              const int* triangleMaterialIds,
                              int materialCount,
                              const uint8_t* triangleEmitterFlags,
                              const Float3* triangleEmission,
                              const EmitterData* emitters, int emitterCount,
                              const int* emitterTriIndices,
                              const float* emitterTriCdf,
                              const float* sceneEmitterCdf,
                              const BsdfData* bsdfs,
                              int bsdfCount,
                              const int* triangleBsdfIds,
                              const LinearBVHNode* bvhNodes,
                              int bvhNodeCount,
                              const int* bvhPrimIndices,
                              const SpotlightData* spotlights,
                              int spotlightCount,
                              const cudaTextureObject_t* texObjects,
                              int textureCount,
                              uint32_t frameIndex)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    DirectLightingContext lightingCtx{};
    lightingCtx.triangleEmitterFlags = triangleEmitterFlags;
    lightingCtx.triangleEmission = triangleEmission;
    lightingCtx.bvhNodes = bvhNodes;
    lightingCtx.bvhNodeCount = bvhNodeCount;
    lightingCtx.bvhPrimIndices = bvhPrimIndices;
    lightingCtx.emitterSampling = {
        triangles,
        emitters, emitterCount,
        emitterTriIndices,
        emitterTriCdf,
        sceneEmitterCdf
    };
    lightingCtx.spotlights          = spotlights;
    lightingCtx.spotlightCount      = spotlightCount;
    lightingCtx.texObjects          = texObjects;
    lightingCtx.triangleMaterialIds = triangleMaterialIds;
    lightingCtx.textureCount        = textureCount;

    const uint32_t pixelIndex =
        static_cast<uint32_t>(y) * static_cast<uint32_t>(width) + static_cast<uint32_t>(x);
    RngState rng = makeRng(hashUint(pixelIndex ^ (frameIndex * 0x9e3779b9u)));

    Float3 finalColor = {0.0f, 0.0f, 0.0f};

    for (int i = 0; i < SAMPLES_PER_PIXEL; i++) {

        // Sub-pixel jitter: draw fresh offsets per path so SAMPLES_PER_PIXEL paths differ.
        const float jx = rngNextFloat01(rng) - 0.5f;
        const float jy = rngNextFloat01(rng) - 0.5f;

        PathState rikudo;
        rikudo.ray = generatePrimaryRay(x, y, width, height, jx, jy);
        rikudo.throughput = {1.0f, 1.0f, 1.0f};
        rikudo.accumulatedColor = {0.0f, 0.0f, 0.0f};
        rikudo.bounceCount = 0;
        rikudo.pixelIndex = pixelIndex;
        rikudo.eta = 1.0f;
        rikudo.brdfPDF = 0.0f;
        rikudo.specularBounce = true;

        while (rikudo.bounceCount < MAX_BOUNCES) {

            if (!traceRay(rikudo, rng, lightingCtx, bsdfs, triangleBsdfIds,
                        triangleCount)) {
                break;
            }

            if (rikudo.bounceCount > 3) {
                const float continuationProb = fminf(
                    maxCoeff3(rikudo.throughput) * rikudo.eta * rikudo.eta,
                    0.99f);

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

    // Blend this sample into the running per-pixel average (linear space).
    // Welford / streaming average: new_avg = old_avg + (sample - old_avg) / n
    const float n = static_cast<float>(frameIndex + 1);
    Float3 prev = accumBuffer[pixelIndex];
    Float3 blended = {
        prev.x + (finalColor.x - prev.x) / n,
        prev.y + (finalColor.y - prev.y) / n,
        prev.z + (finalColor.z - prev.z) / n
    };
    accumBuffer[pixelIndex] = blended;

    // Gamma correction (linear → sRGB, γ = 2.2) applied to the accumulated
    // average, not to the raw per-frame sample.
    float dr = powf(fmaxf(0.0f, fminf(blended.x, 1.0f)), 1.0f / 2.2f);
    float dg = powf(fmaxf(0.0f, fminf(blended.y, 1.0f)), 1.0f / 2.2f);
    float db = powf(fmaxf(0.0f, fminf(blended.z, 1.0f)), 1.0f / 2.2f);

    uint8_t r = static_cast<uint8_t>(dr * 255.0f);
    uint8_t g = static_cast<uint8_t>(dg * 255.0f);
    uint8_t b = static_cast<uint8_t>(db * 255.0f);
    uint8_t a = 255;

    // Pack RGBA into a uint32 (ABGR byte order for GL_UNSIGNED_BYTE / RGBA)
    uint32_t pixel = (a << 24) | (b << 16) | (g << 8) | r;
    framebuffer[y * width + x] = pixel;
}

// ---------------------------------------------------------------------------
//  Host API implementation
// ---------------------------------------------------------------------------

void cudaInitScene(const TriangleData* triangles, int triangleCount,
                    const Float3* materials, int materialCount,
                    const int* triangleMaterialIds,
                    int /*imageWidth*/, int /*imageHeight*/)
{
    // Free previous scene buffers (if re-initialising)
    if (s_triangles_d) {
        cudaFree(s_triangles_d);
        s_triangles_d = nullptr;
    }
    if (s_materials_d) {
        cudaFree(s_materials_d);
        s_materials_d = nullptr;
    }
    if (s_triangleMaterialIds_d) {
        cudaFree(s_triangleMaterialIds_d);
        s_triangleMaterialIds_d = nullptr;
    }

    s_triangleCount = triangleCount;
    s_materialCount = materialCount;

    if (triangleCount > 0) {
        CUDA_CHECK(cudaMalloc(&s_triangles_d,
                              sizeof(TriangleData) * static_cast<size_t>(triangleCount)));
        CUDA_CHECK(cudaMemcpy(s_triangles_d, triangles,
                               sizeof(TriangleData) * static_cast<size_t>(triangleCount),
                               cudaMemcpyHostToDevice));
    }

    if (materialCount > 0) {
        CUDA_CHECK(cudaMalloc(&s_materials_d,
                              sizeof(Float3) * static_cast<size_t>(materialCount)));
        CUDA_CHECK(cudaMemcpy(s_materials_d, materials,
                               sizeof(Float3) * static_cast<size_t>(materialCount),
                               cudaMemcpyHostToDevice));
    }

    if (triangleMaterialIds && triangleCount > 0) {
        CUDA_CHECK(cudaMalloc(&s_triangleMaterialIds_d,
                              sizeof(int) * static_cast<size_t>(triangleCount)));
        CUDA_CHECK(cudaMemcpy(s_triangleMaterialIds_d, triangleMaterialIds,
                               sizeof(int) * static_cast<size_t>(triangleCount),
                               cudaMemcpyHostToDevice));
    }
}

void cudaInit(const TriangleData& tri, const Float3& color,
              int imageWidth, int imageHeight)
{
    // Wrap the old API into the new flattened scene API.
    cudaInitScene(&tri, 1, &color, 1, nullptr, imageWidth, imageHeight);
}

void cudaInitCamera(const CameraData& camera)
{
    CUDA_CHECK(cudaMemcpyToSymbol(d_camera, &camera, sizeof(CameraData)));
}

void cudaInitTriangleEmission(const Float3* triangleEmission, int triangleCount)
{
    if (s_triangleEmission_d) {
        cudaFree(s_triangleEmission_d);
        s_triangleEmission_d = nullptr;
    }

    if (!triangleEmission || triangleCount <= 0) return;

    CUDA_CHECK(cudaMalloc(&s_triangleEmission_d,
                          sizeof(Float3) * static_cast<size_t>(triangleCount)));
    CUDA_CHECK(cudaMemcpy(s_triangleEmission_d, triangleEmission,
                          sizeof(Float3) * static_cast<size_t>(triangleCount),
                          cudaMemcpyHostToDevice));
}

void cudaInitTriangleEmitterFlags(const uint8_t* flags, int triangleCount)
{
    if (s_triangleEmitterFlags_d) {
        cudaFree(s_triangleEmitterFlags_d);
        s_triangleEmitterFlags_d = nullptr;
    }

    if (!flags || triangleCount <= 0) return;

    CUDA_CHECK(cudaMalloc(&s_triangleEmitterFlags_d,
                          sizeof(uint8_t) * static_cast<size_t>(triangleCount)));
    CUDA_CHECK(cudaMemcpy(s_triangleEmitterFlags_d, flags,
                          sizeof(uint8_t) * static_cast<size_t>(triangleCount),
                          cudaMemcpyHostToDevice));
}

void cudaInitEmitters(const int* emissiveTriangleIndices,
                      const float* emissiveTriangleCdf,
                      int emissiveTriangleCount)
{
    if (s_emissiveTriIndices_d) {
        cudaFree(s_emissiveTriIndices_d);
        s_emissiveTriIndices_d = nullptr;
    }
    if (s_emissiveTriCdf_d) {
        cudaFree(s_emissiveTriCdf_d);
        s_emissiveTriCdf_d = nullptr;
    }
    s_emissiveTriCount = 0;

    if (!emissiveTriangleIndices || !emissiveTriangleCdf || emissiveTriangleCount <= 0)
        return;

    s_emissiveTriCount = emissiveTriangleCount;

    CUDA_CHECK(cudaMalloc(&s_emissiveTriIndices_d,
                          sizeof(int) * static_cast<size_t>(emissiveTriangleCount)));
    CUDA_CHECK(cudaMemcpy(s_emissiveTriIndices_d, emissiveTriangleIndices,
                          sizeof(int) * static_cast<size_t>(emissiveTriangleCount),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&s_emissiveTriCdf_d,
                          sizeof(float) * static_cast<size_t>(emissiveTriangleCount + 1)));
    CUDA_CHECK(cudaMemcpy(s_emissiveTriCdf_d, emissiveTriangleCdf,
                          sizeof(float) * static_cast<size_t>(emissiveTriangleCount + 1),
                          cudaMemcpyHostToDevice));
}

void cudaInitEmitterTable(const EmitterData* emitters, int emitterCount,
                          const int* emitterTriIndices, int emitterTriIndexCount,
                          const float* emitterTriCdf, int emitterTriCdfCount,
                          const float* sceneEmitterCdf)
{
    // Free previous
    if (s_emitters_d) { cudaFree(s_emitters_d); s_emitters_d = nullptr; }
    if (s_emitterTriIndices_d) { cudaFree(s_emitterTriIndices_d); s_emitterTriIndices_d = nullptr; }
    if (s_emitterTriCdf_d) { cudaFree(s_emitterTriCdf_d); s_emitterTriCdf_d = nullptr; }
    if (s_sceneEmitterCdf_d) { cudaFree(s_sceneEmitterCdf_d); s_sceneEmitterCdf_d = nullptr; }
    s_emitterCount = 0;

    if (!emitters || emitterCount <= 0 ||
        !emitterTriIndices || emitterTriIndexCount <= 0 ||
        !emitterTriCdf || emitterTriCdfCount <= 0 ||
        !sceneEmitterCdf)
        return;

    s_emitterCount = emitterCount;

    // Emitters table
    CUDA_CHECK(cudaMalloc(&s_emitters_d, sizeof(EmitterData) * static_cast<size_t>(emitterCount)));
    CUDA_CHECK(cudaMemcpy(s_emitters_d, emitters,
                          sizeof(EmitterData) * static_cast<size_t>(emitterCount),
                          cudaMemcpyHostToDevice));

    // Concatenated triangle indices
    CUDA_CHECK(cudaMalloc(&s_emitterTriIndices_d,
                          sizeof(int) * static_cast<size_t>(emitterTriIndexCount)));
    CUDA_CHECK(cudaMemcpy(s_emitterTriIndices_d, emitterTriIndices,
                          sizeof(int) * static_cast<size_t>(emitterTriIndexCount),
                          cudaMemcpyHostToDevice));

    // Concatenated CDFs
    CUDA_CHECK(cudaMalloc(&s_emitterTriCdf_d,
                          sizeof(float) * static_cast<size_t>(emitterTriCdfCount)));
    CUDA_CHECK(cudaMemcpy(s_emitterTriCdf_d, emitterTriCdf,
                          sizeof(float) * static_cast<size_t>(emitterTriCdfCount),
                          cudaMemcpyHostToDevice));

    // Scene-level emitter selection CDF
    CUDA_CHECK(cudaMalloc(&s_sceneEmitterCdf_d,
                          sizeof(float) * static_cast<size_t>(emitterCount + 1)));
    CUDA_CHECK(cudaMemcpy(s_sceneEmitterCdf_d, sceneEmitterCdf,
                          sizeof(float) * static_cast<size_t>(emitterCount + 1),
                          cudaMemcpyHostToDevice));
}

void cudaInitBsdfs(const BsdfData* bsdfs, int bsdfCount,
                   const int* triangleBsdfIds, int triangleCount)
{
    if (s_bsdfs_d) { cudaFree(s_bsdfs_d); s_bsdfs_d = nullptr; }
    if (s_triangleBsdfIds_d) { cudaFree(s_triangleBsdfIds_d); s_triangleBsdfIds_d = nullptr; }
    s_bsdfCount = 0;

    if (!bsdfs || bsdfCount <= 0 || !triangleBsdfIds || triangleCount <= 0)
        return;

    s_bsdfCount = bsdfCount;

    CUDA_CHECK(cudaMalloc(&s_bsdfs_d, sizeof(BsdfData) * static_cast<size_t>(bsdfCount)));
    CUDA_CHECK(cudaMemcpy(s_bsdfs_d, bsdfs,
                          sizeof(BsdfData) * static_cast<size_t>(bsdfCount),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&s_triangleBsdfIds_d, sizeof(int) * static_cast<size_t>(triangleCount)));
    CUDA_CHECK(cudaMemcpy(s_triangleBsdfIds_d, triangleBsdfIds,
                          sizeof(int) * static_cast<size_t>(triangleCount),
                          cudaMemcpyHostToDevice));
}

void cudaInitBVH(const LinearBVHNode* nodes, int nodeCount,
                 const int* primIndices, int primCount)
{
    if (s_bvhNodes_d)       { cudaFree(s_bvhNodes_d);       s_bvhNodes_d = nullptr; }
    if (s_bvhPrimIndices_d) { cudaFree(s_bvhPrimIndices_d); s_bvhPrimIndices_d = nullptr; }
    s_bvhNodeCount = 0;

    if (!nodes || nodeCount <= 0 || !primIndices || primCount <= 0) return;

    s_bvhNodeCount = nodeCount;

    CUDA_CHECK(cudaMalloc(&s_bvhNodes_d,       sizeof(LinearBVHNode) * static_cast<size_t>(nodeCount)));
    CUDA_CHECK(cudaMemcpy(s_bvhNodes_d, nodes, sizeof(LinearBVHNode) * static_cast<size_t>(nodeCount),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&s_bvhPrimIndices_d,            sizeof(int) * static_cast<size_t>(primCount)));
    CUDA_CHECK(cudaMemcpy(s_bvhPrimIndices_d, primIndices, sizeof(int) * static_cast<size_t>(primCount),
                          cudaMemcpyHostToDevice));
}

void cudaInitSpotlights(const SpotlightData* spotlights, int spotlightCount)
{
    if (s_spotlights_d) {
        cudaFree(s_spotlights_d);
        s_spotlights_d = nullptr;
    }
    s_spotlightCount = 0;

    if (!spotlights || spotlightCount <= 0) return;

    s_spotlightCount = spotlightCount;
    CUDA_CHECK(cudaMalloc(&s_spotlights_d,
                          sizeof(SpotlightData) * static_cast<size_t>(spotlightCount)));
    CUDA_CHECK(cudaMemcpy(s_spotlights_d, spotlights,
                          sizeof(SpotlightData) * static_cast<size_t>(spotlightCount),
                          cudaMemcpyHostToDevice));
}

void cudaInitTextures(const uint8_t* const* pixels,
                      const int* widths,
                      const int* heights,
                      int textureCount)
{
    // Release any previously uploaded textures.
    for (auto& texObj : s_texObjects_h) if (texObj) cudaDestroyTextureObject(texObj);
    for (auto& arr : s_cuArrays)        if (arr)    cudaFreeArray(arr);
    s_texObjects_h.clear();
    s_cuArrays.clear();
    if (s_texObjects_d) { cudaFree(s_texObjects_d); s_texObjects_d = nullptr; }
    s_textureCount = 0;

    if (!pixels || textureCount <= 0) return;

    s_cuArrays.resize(static_cast<size_t>(textureCount), nullptr);
    s_texObjects_h.resize(static_cast<size_t>(textureCount), 0);

    for (int i = 0; i < textureCount; ++i) {
        // nullptr means this material has no texture — leave the handle as 0.
        // The kernel checks texObj != 0 before sampling, so the BSDF's own
        // base_color / albedo parameters are used unchanged.
        if (!pixels[i]) {
            s_cuArrays[static_cast<size_t>(i)]     = nullptr;
            s_texObjects_h[static_cast<size_t>(i)] = 0;
            continue;
        }

        const uint8_t* srcPixels = pixels[i];
        const int w = widths[i];
        const int h = heights[i];

        // Allocate a 2-D CUDA array (RGBA8 per texel).
        const cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
        CUDA_CHECK(cudaMallocArray(&s_cuArrays[static_cast<size_t>(i)], &desc,
                                   static_cast<size_t>(w), static_cast<size_t>(h)));

        // Copy host pixels into the CUDA array (row-major, 4 bytes per pixel).
        CUDA_CHECK(cudaMemcpy2DToArray(
            s_cuArrays[static_cast<size_t>(i)], 0, 0,
            srcPixels, static_cast<size_t>(w) * 4,
            static_cast<size_t>(w) * 4, static_cast<size_t>(h),
            cudaMemcpyHostToDevice));

        // Create a texture object with bilinear filtering, UV wrap, normalised coords.
        // readMode = NormalizedFloat converts uchar4 → float4 in [0,1] automatically.
        cudaResourceDesc resDesc{};
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = s_cuArrays[static_cast<size_t>(i)];

        cudaTextureDesc texDesc{};
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;

        CUDA_CHECK(cudaCreateTextureObject(
            &s_texObjects_h[static_cast<size_t>(i)], &resDesc, &texDesc, nullptr));

        fprintf(stdout, "[texture] Uploaded material %d: %d×%d\n", i, w, h);
    }

    s_textureCount = textureCount;

    // Copy the array of texture-object handles to the device so the kernel can index them.
    CUDA_CHECK(cudaMalloc(&s_texObjects_d,
                          sizeof(cudaTextureObject_t) * static_cast<size_t>(textureCount)));
    CUDA_CHECK(cudaMemcpy(s_texObjects_d, s_texObjects_h.data(),
                          sizeof(cudaTextureObject_t) * static_cast<size_t>(textureCount),
                          cudaMemcpyHostToDevice));
}

void cudaRegisterPBO(uint32_t pbo)
{
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &s_pboResource, pbo,
        cudaGraphicsMapFlagsWriteDiscard));
}

void cudaResetAccumulation(int imageWidth, int imageHeight)
{
    // Reallocate buffer if the resolution has changed.
    if (s_accumBuffer_d && (s_accumWidth != imageWidth || s_accumHeight != imageHeight)) {
        cudaFree(s_accumBuffer_d);
        s_accumBuffer_d = nullptr;
    }
    if (!s_accumBuffer_d) {
        s_accumWidth  = imageWidth;
        s_accumHeight = imageHeight;
        CUDA_CHECK(cudaMalloc(&s_accumBuffer_d,
                              sizeof(Float3) * static_cast<size_t>(imageWidth * imageHeight)));
    }
    CUDA_CHECK(cudaMemset(s_accumBuffer_d, 0,
                          sizeof(Float3) * static_cast<size_t>(imageWidth * imageHeight)));
    s_frameIndex = 0;
}

void cudaRender(int imageWidth, int imageHeight)
{
    // Auto-initialise the accumulation buffer on the first call or if the
    // resolution has changed (e.g. window resize).
    if (!s_accumBuffer_d || s_accumWidth != imageWidth || s_accumHeight != imageHeight) {
        cudaResetAccumulation(imageWidth, imageHeight);
    }

    // Map the PBO so CUDA can write into it
    CUDA_CHECK(cudaGraphicsMapResources(1, &s_pboResource, 0));

    uint32_t* devPtr = nullptr;
    size_t    bufSize = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&devPtr), &bufSize, s_pboResource));

    // Launch kernel – 16×16 threads per block
    dim3 block(16, 16);
    dim3 grid((imageWidth  + block.x - 1) / block.x,
              (imageHeight + block.y - 1) / block.y);

    renderKernel<<<grid, block>>>(devPtr, s_accumBuffer_d,
                                   imageWidth, imageHeight,
                                   s_triangles_d, s_triangleCount,
                                   s_materials_d,
                                   s_triangleMaterialIds_d,
                                   s_materialCount,
                                   s_triangleEmitterFlags_d,
                                   s_triangleEmission_d,
                                   s_emitters_d, s_emitterCount,
                                   s_emitterTriIndices_d,
                                   s_emitterTriCdf_d,
                                   s_sceneEmitterCdf_d,
                                   s_bsdfs_d, s_bsdfCount,
                                   s_triangleBsdfIds_d,
                                   s_bvhNodes_d, s_bvhNodeCount,
                                   s_bvhPrimIndices_d,
                                   s_spotlights_d, s_spotlightCount,
                                   s_texObjects_d, s_textureCount,
                                   s_frameIndex++);
    CUDA_CHECK(cudaGetLastError());

    // Unmap so OpenGL can read the PBO
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &s_pboResource, 0));
}

void cudaCleanup()
{
    if (s_pboResource) {
        cudaGraphicsUnregisterResource(s_pboResource);
        s_pboResource = nullptr;
    }

    if (s_accumBuffer_d) {
        cudaFree(s_accumBuffer_d);
        s_accumBuffer_d = nullptr;
        s_accumWidth    = 0;
        s_accumHeight   = 0;
        s_frameIndex    = 0;
    }

    if (s_triangles_d) {
        cudaFree(s_triangles_d);
        s_triangles_d = nullptr;
        s_triangleCount = 0;
    }
    if (s_materials_d) {
        cudaFree(s_materials_d);
        s_materials_d = nullptr;
        s_materialCount = 0;
    }
    if (s_triangleMaterialIds_d) {
        cudaFree(s_triangleMaterialIds_d);
        s_triangleMaterialIds_d = nullptr;
    }

    if (s_triangleEmission_d) {
        cudaFree(s_triangleEmission_d);
        s_triangleEmission_d = nullptr;
    }

    if (s_triangleEmitterFlags_d) {
        cudaFree(s_triangleEmitterFlags_d);
        s_triangleEmitterFlags_d = nullptr;
    }

    if (s_emissiveTriIndices_d) {
        cudaFree(s_emissiveTriIndices_d);
        s_emissiveTriIndices_d = nullptr;
    }
    if (s_emissiveTriCdf_d) {
        cudaFree(s_emissiveTriCdf_d);
        s_emissiveTriCdf_d = nullptr;
    }
    s_emissiveTriCount = 0;

    if (s_emitters_d) {
        cudaFree(s_emitters_d);
        s_emitters_d = nullptr;
    }
    if (s_emitterTriIndices_d) {
        cudaFree(s_emitterTriIndices_d);
        s_emitterTriIndices_d = nullptr;
    }
    if (s_emitterTriCdf_d) {
        cudaFree(s_emitterTriCdf_d);
        s_emitterTriCdf_d = nullptr;
    }
    if (s_sceneEmitterCdf_d) {
        cudaFree(s_sceneEmitterCdf_d);
        s_sceneEmitterCdf_d = nullptr;
    }
    s_emitterCount = 0;

    if (s_bsdfs_d) {
        cudaFree(s_bsdfs_d);
        s_bsdfs_d = nullptr;
    }
    if (s_triangleBsdfIds_d) {
        cudaFree(s_triangleBsdfIds_d);
        s_triangleBsdfIds_d = nullptr;
    }
    s_bsdfCount = 0;

    if (s_bvhNodes_d) {
        cudaFree(s_bvhNodes_d);
        s_bvhNodes_d = nullptr;
    }
    if (s_bvhPrimIndices_d) {
        cudaFree(s_bvhPrimIndices_d);
        s_bvhPrimIndices_d = nullptr;
    }
    s_bvhNodeCount = 0;

    if (s_spotlights_d) {
        cudaFree(s_spotlights_d);
        s_spotlights_d = nullptr;
    }
    s_spotlightCount = 0;

    for (auto& texObj : s_texObjects_h) if (texObj) cudaDestroyTextureObject(texObj);
    for (auto& arr : s_cuArrays)        if (arr)    cudaFreeArray(arr);
    s_texObjects_h.clear();
    s_cuArrays.clear();
    if (s_texObjects_d) { cudaFree(s_texObjects_d); s_texObjects_d = nullptr; }
    s_textureCount = 0;
}
