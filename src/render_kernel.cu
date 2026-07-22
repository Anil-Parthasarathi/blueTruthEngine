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
#include "wavefront/wavefront_buffers.h"
#include "rt_cuda_math.cuh"
#include "rt_intersect.cuh"
#include "rt_emitter_sampling.cuh"
#include "rt_bsdf.cuh"
#include "rt_rng.cuh"

#include <glad/gl.h>       // Must come before cuda_gl_interop.h (defines GLuint)
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// ── OptiX (RT-core acceleration) ────────────────────────────────────
#include <cuda.h>          // CUdeviceptr / CUcontext typedefs used by the OptiX API
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>   // MUST appear in exactly one TU
#include <optix_stack_size.h>
#include "optix_launch_params.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

static constexpr int   SAMPLES_PER_PIXEL = 5;
static constexpr int   MAX_BOUNCES   = 10;
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

#define OPTIX_CHECK(call)                                                     \
    do {                                                                       \
        OptixResult res = (call);                                              \
        if (res != OPTIX_SUCCESS) {                                            \
            fprintf(stderr, "OptiX error at %s:%d – %s\n",                    \
                    __FILE__, __LINE__, optixGetErrorString(res));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Embedded PTX of src/optix_programs.cu (generated at build time, see CMake).
extern "C" const char* getOptixPtx();
extern "C" size_t      getOptixPtxSize();

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
//  OptiX state (RT-core acceleration). The path tracer now runs as an OptiX
//  raygen pipeline; the CPU-built BVH and CUDA megakernel are no longer used.
// ---------------------------------------------------------------------------
static OptixDeviceContext      s_optixContext   = nullptr;
static OptixModule             s_optixModule    = nullptr;
static OptixPipeline           s_optixPipeline  = nullptr;
static OptixProgramGroup       s_pgRaygen       = nullptr;
static OptixProgramGroup       s_pgMissRadiance = nullptr;
static OptixProgramGroup       s_pgMissShadow   = nullptr;
static OptixProgramGroup       s_pgHitRadiance  = nullptr;
static OptixShaderBindingTable s_sbt            = {};
static bool                    s_optixReady     = false;

// Geometry acceleration structure (GAS) built on-GPU from the triangle data.
static CUdeviceptr             s_gasOutputBuffer = 0;
static OptixTraversableHandle  s_gasHandle       = 0;
static Float3*                 s_gasVertices_d    = nullptr; // packed 3*triCount vertices

// Device copy of the launch parameters (uploaded each frame).
static LaunchParams* s_launchParams_d = nullptr;

// Camera staged on the host (uploaded into LaunchParams at launch time).
static CameraData s_camera_h = {};

// ── OptiX AI denoiser ───────────────────────────────────────────────
static OptixDenoiser s_denoiser            = nullptr;
static CUdeviceptr   s_denoiserState       = 0;
static size_t        s_denoiserStateSize   = 0;
static CUdeviceptr   s_denoiserScratch     = 0;
static size_t        s_denoiserScratchSize = 0;
static CUdeviceptr   s_denoiserIntensity   = 0;       // single float (HDR intensity)
static Float3*       s_denoisedBuffer_d    = nullptr; // denoiser output (linear)
static int           s_denoiserWidth       = 0;
static int           s_denoiserHeight      = 0;
static bool          s_denoiserEnabled     = false;

// ---------------------------------------------------------------------------
//  Render mode + wavefront state
// ---------------------------------------------------------------------------
static RenderMode              s_renderMode = RenderMode::Wavefront;

// Extra OptiX program groups for the wavefront extend and shadow stages.
// These are registered in ensureOptixPipeline alongside the megakernel group.
static OptixProgramGroup       s_pgWfExtend         = nullptr;
static OptixProgramGroup       s_pgWfShadow         = nullptr;

// Separate SBTs for the two wavefront raygen programs.
// The miss and hitgroup records are shared with s_sbt (same device pointers).
static OptixShaderBindingTable s_sbt_wf_extend      = {};
static OptixShaderBindingTable s_sbt_wf_shadow      = {};
// Saved device pointers so the wavefront SBT raygen records can be freed.
static CUdeviceptr             s_d_raygen_wf_extend = 0;
static CUdeviceptr             s_d_raygen_wf_shadow = 0;

// All wavefront SoA device buffers (allocated lazily in ensureWavefrontBuffers).
static WavefrontBuffers s_wf = {};

// Forward declarations for host launchers defined in src/wavefront/*.cu
void launchWfGenerate(const WavefrontSoA&, const CameraData&, int, int, uint32_t);
void launchWfShade   (const WavefrontSoA&, const WfSceneView&, int);

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
//  OptiX setup helpers
// ---------------------------------------------------------------------------
static void optixLogCallback(unsigned int level, const char* tag, const char* message, void*)
{
    fprintf(stderr, "[optix][%u][%s] %s\n", level, tag ? tag : "", message ? message : "");
}

// Lazily creates the OptiX context, module, program groups, pipeline and SBT.
// Scene-independent: called once before the first GAS build.
static void ensureOptixPipeline()
{
    if (s_optixReady) return;

    // Make sure a CUDA context exists for OptiX to attach to.
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions ctxOptions = {};
    ctxOptions.logCallbackFunction = &optixLogCallback;
    ctxOptions.logCallbackLevel    = 4;
    OPTIX_CHECK(optixDeviceContextCreate(0 /*current CUDA context*/, &ctxOptions, &s_optixContext));

    // ── Module ──────────────────────────────────────────────────────
    OptixModuleCompileOptions moduleOptions = {};
    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOptions.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOptions.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur                   = 0;
    pipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues                 = 2;
    pipelineCompileOptions.numAttributeValues               = 2;
    pipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    char   log[8192];
    size_t logSize = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(s_optixContext, &moduleOptions, &pipelineCompileOptions,
                                  getOptixPtx(), getOptixPtxSize(),
                                  log, &logSize, &s_optixModule));

    // ── Program groups ──────────────────────────────────────────────
    OptixProgramGroupOptions pgOptions = {};

    OptixProgramGroupDesc rgDesc = {};
    rgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgDesc.raygen.module            = s_optixModule;
    rgDesc.raygen.entryFunctionName = "__raygen__rg";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(s_optixContext, &rgDesc, 1, &pgOptions, log, &logSize, &s_pgRaygen));

    OptixProgramGroupDesc msDesc = {};
    msDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msDesc.miss.module            = s_optixModule;
    msDesc.miss.entryFunctionName = "__miss__radiance";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(s_optixContext, &msDesc, 1, &pgOptions, log, &logSize, &s_pgMissRadiance));

    OptixProgramGroupDesc msShadowDesc = {};
    msShadowDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msShadowDesc.miss.module            = s_optixModule;
    msShadowDesc.miss.entryFunctionName = "__miss__shadow";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(s_optixContext, &msShadowDesc, 1, &pgOptions, log, &logSize, &s_pgMissShadow));

    OptixProgramGroupDesc hgDesc = {};
    hgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hgDesc.hitgroup.moduleCH            = s_optixModule;
    hgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(s_optixContext, &hgDesc, 1, &pgOptions, log, &logSize, &s_pgHitRadiance));

    // ── Wavefront raygen program groups ─────────────────────────────
    OptixProgramGroupDesc wfExtendDesc = {};
    wfExtendDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    wfExtendDesc.raygen.module            = s_optixModule;
    wfExtendDesc.raygen.entryFunctionName = "__raygen__wf_extend";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(s_optixContext, &wfExtendDesc, 1, &pgOptions, log, &logSize, &s_pgWfExtend));

    OptixProgramGroupDesc wfShadowDesc = {};
    wfShadowDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    wfShadowDesc.raygen.module            = s_optixModule;
    wfShadowDesc.raygen.entryFunctionName = "__raygen__wf_shadow";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(s_optixContext, &wfShadowDesc, 1, &pgOptions, log, &logSize, &s_pgWfShadow));

    // ── Pipeline ────────────────────────────────────────────────────
    // All raygen programs must be in the pipeline even if not all are used
    // in every frame — OptiX validates all referenced entry functions at link time.
    OptixProgramGroup groups[] = { s_pgRaygen, s_pgWfExtend, s_pgWfShadow,
                                    s_pgMissRadiance, s_pgMissShadow, s_pgHitRadiance };

    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 2;
    logSize = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(s_optixContext, &pipelineCompileOptions, &linkOptions,
                                    groups, static_cast<unsigned int>(sizeof(groups) / sizeof(groups[0])),
                                    log, &logSize, &s_optixPipeline));

    // ── Stack sizes ─────────────────────────────────────────────────
    OptixStackSizes stackSizes = {};
    for (OptixProgramGroup pg : groups)
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stackSizes, s_optixPipeline));

    unsigned int dcFromTraversal = 0, dcFromState = 0, contStack = 0;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes,
                                           2 /*maxTraceDepth*/, 0 /*maxCCDepth*/, 0 /*maxDCDepth*/,
                                           &dcFromTraversal, &dcFromState, &contStack));
    OPTIX_CHECK(optixPipelineSetStackSize(s_optixPipeline,
                                          dcFromTraversal, dcFromState, contStack,
                                          1 /*maxTraversableGraphDepth (single GAS)*/));

    // ── Shader binding table ────────────────────────────────────────
    RayGenSbtRecord rgRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(s_pgRaygen, &rgRecord));
    CUdeviceptr d_raygen = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen), sizeof(RayGenSbtRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen), &rgRecord,
                          sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

    MissSbtRecord missRecords[RAY_TYPE_COUNT];
    OPTIX_CHECK(optixSbtRecordPackHeader(s_pgMissRadiance, &missRecords[RAY_TYPE_RADIANCE]));
    OPTIX_CHECK(optixSbtRecordPackHeader(s_pgMissShadow,   &missRecords[RAY_TYPE_SHADOW]));
    CUdeviceptr d_miss = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss), sizeof(missRecords)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss), missRecords,
                          sizeof(missRecords), cudaMemcpyHostToDevice));

    HitGroupSbtRecord hitRecords[RAY_TYPE_COUNT];
    OPTIX_CHECK(optixSbtRecordPackHeader(s_pgHitRadiance, &hitRecords[RAY_TYPE_RADIANCE]));
    // Shadow rays disable CH/AH, but still index a hitgroup record — reuse the
    // radiance hitgroup header so the SBT index is always valid.
    OPTIX_CHECK(optixSbtRecordPackHeader(s_pgHitRadiance, &hitRecords[RAY_TYPE_SHADOW]));
    CUdeviceptr d_hit = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit), sizeof(hitRecords)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hit), hitRecords,
                          sizeof(hitRecords), cudaMemcpyHostToDevice));

    s_sbt.raygenRecord                = d_raygen;
    s_sbt.missRecordBase              = d_miss;
    s_sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
    s_sbt.missRecordCount             = RAY_TYPE_COUNT;
    s_sbt.hitgroupRecordBase          = d_hit;
    s_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    s_sbt.hitgroupRecordCount         = RAY_TYPE_COUNT;

    // ── Wavefront SBTs (share miss/hit records; only the raygen differs) ─
    RayGenSbtRecord wfExtendRgRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(s_pgWfExtend, &wfExtendRgRecord));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_d_raygen_wf_extend), sizeof(RayGenSbtRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(s_d_raygen_wf_extend), &wfExtendRgRecord,
                          sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

    RayGenSbtRecord wfShadowRgRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(s_pgWfShadow, &wfShadowRgRecord));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_d_raygen_wf_shadow), sizeof(RayGenSbtRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(s_d_raygen_wf_shadow), &wfShadowRgRecord,
                          sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

    // Copy the megakernel SBT layout then swap only the raygen record.
    s_sbt_wf_extend = s_sbt;
    s_sbt_wf_extend.raygenRecord = s_d_raygen_wf_extend;

    s_sbt_wf_shadow = s_sbt;
    s_sbt_wf_shadow.raygenRecord = s_d_raygen_wf_shadow;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_launchParams_d), sizeof(LaunchParams)));

    s_optixReady = true;
    fprintf(stdout, "[optix] Pipeline ready (RT-core acceleration enabled)\n");
}

// Builds the GAS from the uploaded triangle data (s_triangles_d). The GAS uses
// RT-core hardware; primitive index == triangle index, so all per-triangle
// arrays continue to work unchanged.
static void buildGAS()
{
    if (s_triangleCount <= 0 || !s_triangles_d) return;

    const size_t vertexCount = static_cast<size_t>(s_triangleCount) * 3;

    // Pack a contiguous vertex buffer (v0,v1,v2 per triangle) from the
    // interleaved TriangleData (v0,v1,v2 sit at offset 0 of each struct).
    if (s_gasVertices_d) { cudaFree(s_gasVertices_d); s_gasVertices_d = nullptr; }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_gasVertices_d), vertexCount * sizeof(Float3)));
    CUDA_CHECK(cudaMemcpy2D(s_gasVertices_d, 3 * sizeof(Float3),
                            s_triangles_d, sizeof(TriangleData),
                            3 * sizeof(Float3), static_cast<size_t>(s_triangleCount),
                            cudaMemcpyDeviceToDevice));

    CUdeviceptr d_vertices = reinterpret_cast<CUdeviceptr>(s_gasVertices_d);

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(Float3);
    buildInput.triangleArray.numVertices         = static_cast<unsigned int>(vertexCount);
    buildInput.triangleArray.vertexBuffers       = &d_vertices;

    const unsigned int triangleFlags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    buildInput.triangleArray.flags        = triangleFlags;
    buildInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufferSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(s_optixContext, &accelOptions, &buildInput, 1, &bufferSizes));

    CUdeviceptr d_temp = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp), bufferSizes.tempSizeInBytes));

    CUdeviceptr d_outputUncompacted = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_outputUncompacted), bufferSizes.outputSizeInBytes));

    // Request the compacted size via an emit property.
    CUdeviceptr d_compactedSize = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compactedSize), sizeof(uint64_t)));
    OptixAccelEmitDesc emitDesc = {};
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = d_compactedSize;

    OptixTraversableHandle uncompactedHandle = 0;
    OPTIX_CHECK(optixAccelBuild(s_optixContext, 0 /*stream*/, &accelOptions, &buildInput, 1,
                                d_temp, bufferSizes.tempSizeInBytes,
                                d_outputUncompacted, bufferSizes.outputSizeInBytes,
                                &uncompactedHandle, &emitDesc, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    uint64_t compactedSize = 0;
    CUDA_CHECK(cudaMemcpy(&compactedSize, reinterpret_cast<void*>(d_compactedSize),
                          sizeof(uint64_t), cudaMemcpyDeviceToHost));

    if (s_gasOutputBuffer) { cudaFree(reinterpret_cast<void*>(s_gasOutputBuffer)); s_gasOutputBuffer = 0; }

    if (compactedSize > 0 && compactedSize < bufferSizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_gasOutputBuffer), compactedSize));
        OPTIX_CHECK(optixAccelCompact(s_optixContext, 0, uncompactedHandle,
                                      s_gasOutputBuffer, compactedSize, &s_gasHandle));
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(reinterpret_cast<void*>(d_outputUncompacted));
    } else {
        s_gasOutputBuffer = d_outputUncompacted;
        s_gasHandle       = uncompactedHandle;
    }

    cudaFree(reinterpret_cast<void*>(d_temp));
    cudaFree(reinterpret_cast<void*>(d_compactedSize));

    fprintf(stdout, "[optix] GAS built from %d triangles (%.2f MB)\n",
            s_triangleCount, static_cast<double>(compactedSize) / (1024.0 * 1024.0));
}

// ---------------------------------------------------------------------------
//  Tonemap a linear-space buffer (e.g. the denoiser output) into the RGBA8 PBO.
//  Uses the identical gamma 2.2 encode as __raygen__rg so denoised and
//  non-denoised frames match tonally.
// ---------------------------------------------------------------------------
__global__ void presentLinearKernel(uint32_t* framebuffer, const Float3* linear,
                                     int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    const Float3 c = linear[idx];

    float dr = powf(fmaxf(0.0f, fminf(c.x, 1.0f)), 1.0f / 2.2f);
    float dg = powf(fmaxf(0.0f, fminf(c.y, 1.0f)), 1.0f / 2.2f);
    float db = powf(fmaxf(0.0f, fminf(c.z, 1.0f)), 1.0f / 2.2f);

    uint8_t r = static_cast<uint8_t>(dr * 255.0f);
    uint8_t g = static_cast<uint8_t>(dg * 255.0f);
    uint8_t b = static_cast<uint8_t>(db * 255.0f);
    uint8_t a = 255;
    framebuffer[idx] = (a << 24) | (b << 16) | (g << 8) | r;
}

// Lazily creates / resizes the denoiser for the given resolution.
static void ensureDenoiser(int width, int height)
{
    if (s_denoiser && s_denoiserWidth == width && s_denoiserHeight == height) return;

    if (s_denoiser)          { optixDenoiserDestroy(s_denoiser); s_denoiser = nullptr; }
    if (s_denoiserState)     { cudaFree(reinterpret_cast<void*>(s_denoiserState));   s_denoiserState = 0; }
    if (s_denoiserScratch)   { cudaFree(reinterpret_cast<void*>(s_denoiserScratch)); s_denoiserScratch = 0; }
    if (s_denoisedBuffer_d)  { cudaFree(s_denoisedBuffer_d); s_denoisedBuffer_d = nullptr; }
    if (!s_denoiserIntensity) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_denoiserIntensity), sizeof(float)));

    OptixDenoiserOptions options = {};
    options.guideAlbedo  = 0;
    options.guideNormal  = 0;
    options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
    OPTIX_CHECK(optixDenoiserCreate(s_optixContext, OPTIX_DENOISER_MODEL_KIND_HDR,
                                    &options, &s_denoiser));

    OptixDenoiserSizes sizes = {};
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(s_denoiser,
                static_cast<unsigned int>(width), static_cast<unsigned int>(height), &sizes));
    s_denoiserStateSize   = sizes.stateSizeInBytes;
    s_denoiserScratchSize = sizes.withoutOverlapScratchSizeInBytes;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_denoiserState),   s_denoiserStateSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_denoiserScratch), s_denoiserScratchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_denoisedBuffer_d),
                          sizeof(Float3) * static_cast<size_t>(width) * static_cast<size_t>(height)));

    OPTIX_CHECK(optixDenoiserSetup(s_denoiser, 0,
                static_cast<unsigned int>(width), static_cast<unsigned int>(height),
                s_denoiserState, s_denoiserStateSize,
                s_denoiserScratch, s_denoiserScratchSize));

    s_denoiserWidth  = width;
    s_denoiserHeight = height;
}

// Runs the denoiser on the accumulated linear image and tonemaps the result
// into the mapped PBO (devPtr).
static void denoiseAndPresent(uint32_t* devPtr, int width, int height)
{
    ensureDenoiser(width, height);

    OptixImage2D inputImage = {};
    inputImage.data              = reinterpret_cast<CUdeviceptr>(s_accumBuffer_d);
    inputImage.width             = static_cast<unsigned int>(width);
    inputImage.height            = static_cast<unsigned int>(height);
    inputImage.rowStrideInBytes  = static_cast<unsigned int>(width * sizeof(Float3));
    inputImage.pixelStrideInBytes = static_cast<unsigned int>(sizeof(Float3));
    inputImage.format            = OPTIX_PIXEL_FORMAT_FLOAT3;

    OptixImage2D outputImage = inputImage;
    outputImage.data = reinterpret_cast<CUdeviceptr>(s_denoisedBuffer_d);

    OPTIX_CHECK(optixDenoiserComputeIntensity(s_denoiser, 0, &inputImage,
                s_denoiserIntensity, s_denoiserScratch, s_denoiserScratchSize));

    OptixDenoiserParams params = {};
    params.hdrIntensity = s_denoiserIntensity;
    params.blendFactor  = 0.0f;

    OptixDenoiserGuideLayer guideLayer = {};
    OptixDenoiserLayer layer = {};
    layer.input  = inputImage;
    layer.output = outputImage;

    OPTIX_CHECK(optixDenoiserInvoke(s_denoiser, 0, &params,
                s_denoiserState, s_denoiserStateSize,
                &guideLayer, &layer, 1, 0, 0,
                s_denoiserScratch, s_denoiserScratchSize));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    presentLinearKernel<<<grid, block>>>(devPtr, s_denoisedBuffer_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

bool cudaToggleDenoiser()
{
    s_denoiserEnabled = !s_denoiserEnabled;
    fprintf(stdout, "[denoise] %s\n", s_denoiserEnabled ? "ON" : "OFF");
    return s_denoiserEnabled;
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
    // Stage the camera on the host; it is copied into LaunchParams each frame.
    // (The OptiX module reads the camera from launch params, not a __constant__.)
    s_camera_h = camera;
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

void cudaInitBVH(const LinearBVHNode* /*nodes*/, int /*nodeCount*/,
                 const int* /*primIndices*/, int /*primCount*/)
{
    // The CPU-built LinearBVH is no longer used. Intersection now runs on the
    // RT cores via an OptiX GAS. We initialise the OptiX pipeline (once) and
    // build the GAS directly from the triangle data uploaded by cudaInitScene.
    // The previous LinearBVHNode arguments are intentionally ignored.
    ensureOptixPipeline();
    buildGAS();
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

// ---------------------------------------------------------------------------
//  Wavefront buffer allocation  (called lazily on first wavefront render)
// ---------------------------------------------------------------------------
static void ensureWavefrontBuffers(int width, int height)
{
    if (s_wf.allocated &&
        s_wf.soa.maxPaths == width * height)
        return;

    // Free old allocation if resolution changed.
    if (s_wf.allocated) {
        WavefrontSoA& w = s_wf.soa;
        cudaFree(w.rayOrigin);      cudaFree(w.rayDir);
        cudaFree(w.throughput);     cudaFree(w.radiance);
        cudaFree(w.eta);            cudaFree(w.brdfPDF);
        cudaFree(w.specularBounce); cudaFree(w.bounceCount);
        cudaFree(w.pixelIndex);     cudaFree(w.rngState);
        cudaFree(w.hitTriIndex);    cudaFree(w.hitBaryU);  cudaFree(w.hitBaryV);
        cudaFree(w.hitPoint);       cudaFree(w.hitNormal); cudaFree(w.hitUV);
        cudaFree(w.rayQueue);       cudaFree(w.shadowQueue);
        cudaFree(w.rayCount);       cudaFree(w.shadowCount);
        cudaFree(w.shadowOrigin);   cudaFree(w.shadowDir);
        cudaFree(w.shadowTMax);     cudaFree(w.shadowContrib);
        cudaFree(w.shadowPathIdx);
        s_wf.allocated = false;
    }

    WavefrontSoA& w = s_wf.soa;
    // One path slot per pixel; shadow slots must cover area NEE + all spotlights.
    w.maxPaths   = width * height;
    w.maxShadows = w.maxPaths * (1 + s_spotlightCount + 1); // +1 area, +1 slack

    const int  N = w.maxPaths;
    const int  M = w.maxShadows;

#define WF_ALLOC(ptr, T, n)  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&(ptr)), (n) * sizeof(T)))
    WF_ALLOC(w.rayOrigin,       Float3,        N);
    WF_ALLOC(w.rayDir,          Float3,        N);
    WF_ALLOC(w.throughput,      Float3,        N);
    WF_ALLOC(w.radiance,        Float3,        N);
    WF_ALLOC(w.eta,             float,         N);
    WF_ALLOC(w.brdfPDF,         float,         N);
    WF_ALLOC(w.specularBounce,  unsigned char, N);
    WF_ALLOC(w.bounceCount,     int,           N);
    WF_ALLOC(w.pixelIndex,      uint32_t,      N);
    WF_ALLOC(w.rngState,        uint32_t,      N);
    WF_ALLOC(w.hitTriIndex,     int,           N);
    WF_ALLOC(w.hitBaryU,        float,         N);
    WF_ALLOC(w.hitBaryV,        float,         N);
    WF_ALLOC(w.hitPoint,        Float3,        N);
    WF_ALLOC(w.hitNormal,       Float3,        N);
    WF_ALLOC(w.hitUV,           Float2,        N);
    WF_ALLOC(w.rayQueue,        int,           N);
    WF_ALLOC(w.shadowQueue,     int,           M);
    WF_ALLOC(w.rayCount,        int,           1);
    WF_ALLOC(w.shadowCount,     int,           1);
    WF_ALLOC(w.shadowOrigin,    Float3,        M);
    WF_ALLOC(w.shadowDir,       Float3,        M);
    WF_ALLOC(w.shadowTMax,      float,         M);
    WF_ALLOC(w.shadowContrib,   Float3,        M);
    WF_ALLOC(w.shadowPathIdx,   uint32_t,      M);
#undef WF_ALLOC

    s_wf.allocated = true;
    fprintf(stdout, "[wavefront] Buffers allocated (%dx%d, %d paths, %d shadow slots)\n",
            width, height, N, M);
}

// ---------------------------------------------------------------------------
//  Pack a WfSceneView from the current static scene state.
// ---------------------------------------------------------------------------
static WfSceneView buildSceneView(int frameIndex)
{
    WfSceneView sv = {};
    sv.triangles             = s_triangles_d;
    sv.triangleCount         = s_triangleCount;
    sv.triangleEmitterFlags  = s_triangleEmitterFlags_d;
    sv.triangleEmission      = s_triangleEmission_d;
    sv.bsdfs                 = s_bsdfs_d;
    sv.bsdfCount             = s_bsdfCount;
    sv.triangleBsdfIds       = s_triangleBsdfIds_d;
    sv.triangleMaterialIds   = s_triangleMaterialIds_d;
    sv.emitters              = s_emitters_d;
    sv.emitterCount          = s_emitterCount;
    sv.emitterTriIndices     = s_emitterTriIndices_d;
    sv.emitterTriCdf         = s_emitterTriCdf_d;
    sv.sceneEmitterCdf       = s_sceneEmitterCdf_d;
    sv.spotlights            = s_spotlights_d;
    sv.spotlightCount        = s_spotlightCount;
    sv.texObjects            = s_texObjects_d;
    sv.textureCount          = s_textureCount;
    sv.accumBuffer           = s_accumBuffer_d;
    sv.frameIndex            = frameIndex;
    return sv;
}

// ---------------------------------------------------------------------------
//  Wavefront render loop
//  Called instead of the megakernel optixLaunch block when
//  s_renderMode == RenderMode::Wavefront.
// ---------------------------------------------------------------------------
static void cudaRenderWavefront(uint32_t* devPtr, int imageWidth, int imageHeight)
{
    ensureWavefrontBuffers(imageWidth, imageHeight);

    WavefrontSoA& wf = s_wf.soa;

    // Reset both queue counters to zero before the new frame.
    const int zero = 0;
    CUDA_CHECK(cudaMemcpy(wf.rayCount,    &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(wf.shadowCount, &zero, sizeof(int), cudaMemcpyHostToDevice));

    // ── Stage 1: Generate primary rays ──────────────────────────────────
    launchWfGenerate(wf, s_camera_h, imageWidth, imageHeight, s_frameIndex);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Bounce loop ─────────────────────────────────────────────────────
    constexpr int MAX_WF_BOUNCES = 10;   // TODO: tune or expose as a constant
    for (int bounce = 0; bounce < MAX_WF_BOUNCES; ++bounce) {

        int activeCount = 0;
        CUDA_CHECK(cudaMemcpy(&activeCount, wf.rayCount, sizeof(int), cudaMemcpyDeviceToHost));
        if (activeCount <= 0) break;

        // ── Stage 2: Extend — optixLaunch with the wf_extend raygen ─────
        // Fill the wf SoA pointers into params so the raygen can access them.
        // TODO: once __raygen__wf_extend is implemented, remove the (void) guard.
        {
            LaunchParams lp = {};
            lp.handle      = s_gasHandle;
            lp.triangles   = s_triangles_d;
            lp.wf          = wf;
            CUDA_CHECK(cudaMemcpy(s_launchParams_d, &lp, sizeof(LaunchParams), cudaMemcpyHostToDevice));

            // Launch one thread per active ray.
            OPTIX_CHECK(optixLaunch(s_optixPipeline, 0 /*stream*/,
                                    reinterpret_cast<CUdeviceptr>(s_launchParams_d),
                                    sizeof(LaunchParams), &s_sbt_wf_extend,
                                    static_cast<unsigned int>(activeCount), 1, 1));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Reset queue counters for this bounce's shade output.
        CUDA_CHECK(cudaMemcpy(wf.rayCount,    &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(wf.shadowCount, &zero, sizeof(int), cudaMemcpyHostToDevice));

        // ── Stage 3: Shade ───────────────────────────────────────────────
        WfSceneView sv = buildSceneView(static_cast<int>(s_frameIndex));
        launchWfShade(wf, sv, activeCount);
        CUDA_CHECK(cudaDeviceSynchronize());

        // ── Stage 4: Connect shadow rays ─────────────────────────────────
        int shadowCount = 0;
        CUDA_CHECK(cudaMemcpy(&shadowCount, wf.shadowCount, sizeof(int), cudaMemcpyDeviceToHost));
        if (shadowCount > 0) {
            LaunchParams lp = {};
            lp.handle = s_gasHandle;
            lp.wf     = wf;
            CUDA_CHECK(cudaMemcpy(s_launchParams_d, &lp, sizeof(LaunchParams), cudaMemcpyHostToDevice));

            OPTIX_CHECK(optixLaunch(s_optixPipeline, 0 /*stream*/,
                                    reinterpret_cast<CUdeviceptr>(s_launchParams_d),
                                    sizeof(LaunchParams), &s_sbt_wf_shadow,
                                    static_cast<unsigned int>(shadowCount), 1, 1));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // TODO: Once wfShade writes directly to s_accumBuffer_d via accumulateWelford,
    // this step is done automatically inside the shade kernel on path termination.
    // If you prefer a dedicated present kernel that reads wf.radiance[] and blends
    // into s_accumBuffer_d, add it here — mirror the Welford update from __raygen__rg.
    (void)devPtr;
}

void cudaRender(int imageWidth, int imageHeight)
{
    // Auto-initialise the accumulation buffer on the first call or if the
    // resolution has changed (e.g. window resize).
    if (!s_accumBuffer_d || s_accumWidth != imageWidth || s_accumHeight != imageHeight) {
        cudaResetAccumulation(imageWidth, imageHeight);
    }

    // ── Dispatch to the active render path ─────────────────────────────
    if (s_renderMode == RenderMode::Wavefront) {
        // Map PBO briefly — wavefront will write to accumBuffer internally,
        // and the present step (TODO) will tonemap to the PBO.
        // For now we still need to map/unmap so OpenGL gets a valid buffer.
        CUDA_CHECK(cudaGraphicsMapResources(1, &s_pboResource, 0));
        uint32_t* devPtr = nullptr;
        size_t    bufSize = 0;
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&devPtr), &bufSize, s_pboResource));

        cudaRenderWavefront(devPtr, imageWidth, imageHeight);

        ++s_frameIndex;
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &s_pboResource, 0));
        return;
    }

    // Map the PBO so CUDA can write into it
    CUDA_CHECK(cudaGraphicsMapResources(1, &s_pboResource, 0));

    uint32_t* devPtr = nullptr;
    size_t    bufSize = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&devPtr), &bufSize, s_pboResource));

    // Fill the launch parameters and upload to the device.
    LaunchParams lp = {};
    lp.framebuffer          = devPtr;
    lp.accumBuffer          = s_accumBuffer_d;
    lp.width                = imageWidth;
    lp.height               = imageHeight;
    lp.launchOffsetY        = 0;
    lp.frameIndex           = s_frameIndex;
    lp.camera               = s_camera_h;
    lp.handle               = s_gasHandle;
    lp.triangles            = s_triangles_d;
    lp.triangleCount        = s_triangleCount;
    lp.bsdfs                = s_bsdfs_d;
    lp.bsdfCount            = s_bsdfCount;
    lp.triangleBsdfIds      = s_triangleBsdfIds_d;
    lp.triangleMaterialIds  = s_triangleMaterialIds_d;
    lp.triangleEmitterFlags = s_triangleEmitterFlags_d;
    lp.triangleEmission     = s_triangleEmission_d;
    lp.emitters             = s_emitters_d;
    lp.emitterCount         = s_emitterCount;
    lp.emitterTriIndices    = s_emitterTriIndices_d;
    lp.emitterTriCdf        = s_emitterTriCdf_d;
    lp.sceneEmitterCdf      = s_sceneEmitterCdf_d;
    lp.spotlights           = s_spotlights_d;
    lp.spotlightCount       = s_spotlightCount;
    lp.texObjects           = s_texObjects_d;
    lp.textureCount         = s_textureCount;

    // Render the frame in horizontal row-band tiles. Each optixLaunch covers
    // only TILE_HEIGHT scanlines so that no single GPU command runs long enough
    // to trip the Windows display-driver watchdog (TDR). Output is identical to
    // a single full-frame launch — only the launch granularity changes.
    constexpr int TILE_HEIGHT = 64;
    for (int y0 = 0; y0 < imageHeight; y0 += TILE_HEIGHT) {
        const int rows = (imageHeight - y0 < TILE_HEIGHT) ? (imageHeight - y0) : TILE_HEIGHT;

        lp.launchOffsetY = y0;
        CUDA_CHECK(cudaMemcpy(s_launchParams_d, &lp, sizeof(LaunchParams), cudaMemcpyHostToDevice));

        OPTIX_CHECK(optixLaunch(s_optixPipeline, 0 /*stream*/,
                                reinterpret_cast<CUdeviceptr>(s_launchParams_d),
                                sizeof(LaunchParams), &s_sbt,
                                static_cast<unsigned int>(imageWidth),
                                static_cast<unsigned int>(rows), 1));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ++s_frameIndex;

    // Optional: denoise the accumulated image and tonemap into the PBO,
    // overwriting the raygen's direct write.
    if (s_denoiserEnabled) {
        denoiseAndPresent(devPtr, imageWidth, imageHeight);
    }

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

    // ── OptiX teardown ──────────────────────────────────────────────
    if (s_sbt.raygenRecord)     { cudaFree(reinterpret_cast<void*>(s_sbt.raygenRecord));     s_sbt.raygenRecord = 0; }
    if (s_sbt.missRecordBase)   { cudaFree(reinterpret_cast<void*>(s_sbt.missRecordBase));   s_sbt.missRecordBase = 0; }
    if (s_sbt.hitgroupRecordBase){ cudaFree(reinterpret_cast<void*>(s_sbt.hitgroupRecordBase)); s_sbt.hitgroupRecordBase = 0; }
    if (s_launchParams_d)       { cudaFree(s_launchParams_d); s_launchParams_d = nullptr; }
    if (s_gasOutputBuffer)      { cudaFree(reinterpret_cast<void*>(s_gasOutputBuffer)); s_gasOutputBuffer = 0; }
    if (s_gasVertices_d)        { cudaFree(s_gasVertices_d); s_gasVertices_d = nullptr; }
    s_gasHandle = 0;

    if (s_denoiser)         { optixDenoiserDestroy(s_denoiser); s_denoiser = nullptr; }
    if (s_denoiserState)    { cudaFree(reinterpret_cast<void*>(s_denoiserState));   s_denoiserState = 0; }
    if (s_denoiserScratch)  { cudaFree(reinterpret_cast<void*>(s_denoiserScratch)); s_denoiserScratch = 0; }
    if (s_denoiserIntensity){ cudaFree(reinterpret_cast<void*>(s_denoiserIntensity)); s_denoiserIntensity = 0; }
    if (s_denoisedBuffer_d) { cudaFree(s_denoisedBuffer_d); s_denoisedBuffer_d = nullptr; }
    s_denoiserWidth = 0; s_denoiserHeight = 0;

    // ── Wavefront SBT records and program groups ────────────────────
    if (s_d_raygen_wf_extend) { cudaFree(reinterpret_cast<void*>(s_d_raygen_wf_extend)); s_d_raygen_wf_extend = 0; }
    if (s_d_raygen_wf_shadow) { cudaFree(reinterpret_cast<void*>(s_d_raygen_wf_shadow)); s_d_raygen_wf_shadow = 0; }
    if (s_pgWfExtend)  { optixProgramGroupDestroy(s_pgWfExtend); s_pgWfExtend = nullptr; }
    if (s_pgWfShadow)  { optixProgramGroupDestroy(s_pgWfShadow); s_pgWfShadow = nullptr; }

    // ── Wavefront SoA buffers ────────────────────────────────────────
    if (s_wf.allocated) {
        WavefrontSoA& w = s_wf.soa;
        cudaFree(w.rayOrigin);      cudaFree(w.rayDir);
        cudaFree(w.throughput);     cudaFree(w.radiance);
        cudaFree(w.eta);            cudaFree(w.brdfPDF);
        cudaFree(w.specularBounce); cudaFree(w.bounceCount);
        cudaFree(w.pixelIndex);     cudaFree(w.rngState);
        cudaFree(w.hitTriIndex);    cudaFree(w.hitBaryU);  cudaFree(w.hitBaryV);
        cudaFree(w.hitPoint);       cudaFree(w.hitNormal); cudaFree(w.hitUV);
        cudaFree(w.rayQueue);       cudaFree(w.shadowQueue);
        cudaFree(w.rayCount);       cudaFree(w.shadowCount);
        cudaFree(w.shadowOrigin);   cudaFree(w.shadowDir);
        cudaFree(w.shadowTMax);     cudaFree(w.shadowContrib);
        cudaFree(w.shadowPathIdx);
        s_wf.allocated = false;
    }

    if (s_optixPipeline)  { optixPipelineDestroy(s_optixPipeline);   s_optixPipeline = nullptr; }
    if (s_pgRaygen)       { optixProgramGroupDestroy(s_pgRaygen);    s_pgRaygen = nullptr; }
    if (s_pgMissRadiance) { optixProgramGroupDestroy(s_pgMissRadiance); s_pgMissRadiance = nullptr; }
    if (s_pgMissShadow)   { optixProgramGroupDestroy(s_pgMissShadow);   s_pgMissShadow = nullptr; }
    if (s_pgHitRadiance)  { optixProgramGroupDestroy(s_pgHitRadiance);  s_pgHitRadiance = nullptr; }
    if (s_optixModule)    { optixModuleDestroy(s_optixModule);       s_optixModule = nullptr; }
    if (s_optixContext)   { optixDeviceContextDestroy(s_optixContext); s_optixContext = nullptr; }
    s_optixReady = false;
}

// ---------------------------------------------------------------------------
//  Render mode API
// ---------------------------------------------------------------------------
void cudaSetRenderMode(RenderMode mode)
{
    if (mode == s_renderMode) return;
    s_renderMode = mode;
    cudaResetAccumulation(s_accumWidth, s_accumHeight);
    fprintf(stdout, "[render] Mode switched to %s\n",
            mode == RenderMode::Wavefront ? "Wavefront" : "Megakernel");
}

RenderMode cudaGetRenderMode()
{
    return s_renderMode;
}
