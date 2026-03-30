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

static constexpr int   LIGHT_SAMPLES = 30;
static constexpr int   MAX_BOUNCES   = 10;

struct PathState {
    Ray      ray;
    Float3   throughput;
    Float3   accumulatedColor;
    int      bounceCount;
    uint32_t pixelIndex;
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

// Direct illumination for a diffuse hit.
// Writes into pathState.accumulatedColor (scaled by pathState.throughput).
__device__ void handleDiffuse(
    PathState&             rikudo,
    const Intersection&    sceneIntersection,
    const BsdfData&        bsdf,
    RngState&              rng,
    const TriangleData*    triangles,
    const Float3*          triangleEmission,
    const EmitterData*     emitters,       int emitterCount,
    const int*             emitterTriIndices,
    const float*           emitterTriCdf,
    const float*           sceneEmitterCdf,
    const LinearBVHNode*   bvhNodes,       int bvhNodeCount,
    const int*             bvhPrimIndices){

        // If the hit triangle is emissive, show direct emission
        if (triangleEmission) {
            const Float3 e = triangleEmission[sceneIntersection.triangleIndex];
            if (e.x > 0.0f || e.y > 0.0f || e.z > 0.0f) {
                rikudo.accumulatedColor = add3(rikudo.accumulatedColor, mul3(e, rikudo.throughput));
            }
        }

        Float3 directLight = {0.0f, 0.0f, 0.0f};

        for (int i = 0; i < LIGHT_SAMPLES; i++) {
            float uLight0 = rngNextFloat01(rng);
            float uLight1 = rngNextFloat01(rng);

            RandomEmitterSample chosenEmitter = sampleRandomEmitter(
                triangles,
                emitters, emitterCount,
                emitterTriIndices,
                emitterTriCdf,
                sceneEmitterCdf,
                uLight0, uLight1);

            float distToLight = distance3(sceneIntersection.hitPoint, chosenEmitter.mesh.position);
            Float3 directionToLight = div3(sub3(chosenEmitter.mesh.position, sceneIntersection.hitPoint), distToLight);

            Float3 le = emitters[chosenEmitter.emitterIndex].radiance;
            
            // Cast shadow ray to check if the light is visible
            Ray shadowRay;
            shadowRay.origin = sceneIntersection.hitPoint;
            shadowRay.direction = directionToLight;

            if (sceneIntersectAnyHit(shadowRay, triangles,
                                     bvhNodes, bvhNodeCount, bvhPrimIndices,
                                     1e-4f, distToLight - 1e-3f)) {
                continue;
            }

            float cosSurface = fmaxf(0.0f, dot3(sceneIntersection.hitNormal, directionToLight));
            float cosLight   = fmaxf(0.0f, dot3(chosenEmitter.mesh.normal, mul3(directionToLight, -1.0f)));
            float geo = cosSurface * cosLight / (distToLight * distToLight);

            BsdfQueryRecord bRec{};
            bRec.wi = toLocalFromNormal(sceneIntersection.hitNormal, mul3(rikudo.ray.direction, -1.0f));
            bRec.wo = toLocalFromNormal(sceneIntersection.hitNormal, directionToLight);
            bRec.measure = BSDF_ESolidAngle;

            Float3 fr = bsdfEval(bsdf, bRec);

            Float3 sampleColor = div3(mul3(le, mul3(fr, geo)), chosenEmitter.mesh.pdf * chosenEmitter.emitterPdf);

            directLight = add3(directLight, sampleColor);

        }

        Float3 averageDirectLight = div3(directLight, LIGHT_SAMPLES);

        rikudo.accumulatedColor = add3(rikudo.accumulatedColor, mul3(averageDirectLight, rikudo.throughput));
    }

// Specular bounce for a dielectric/mirror hit.
// Updates pathState.ray and pathState.throughput for the next bounce.
// Never applies Russian roulette — delta BSDFs have deterministic throughput
// so RR only adds variance (dark holes) without any benefit.
__device__ bool handleSpecular(
    PathState&             rikudo,
    const Intersection&    sceneIntersection,
    const BsdfData&        bsdf,
    RngState&              rng){

        BsdfQueryRecord bRec{};
        bRec.wi = toLocalFromNormal(sceneIntersection.hitNormal, mul3(rikudo.ray.direction, -1.0f));

        Float3 sampleWeight = bsdfSample(bsdf, bRec, rngNextFloat01(rng), rngNextFloat01(rng));

        Float3 newDir = normalize3(toWorldFromNormal(sceneIntersection.hitNormal, bRec.wo));

        rikudo.ray.origin    = add3(sceneIntersection.hitPoint, mul3(newDir, 1e-4f));
        rikudo.ray.direction = newDir;
        rikudo.throughput    = mul3(rikudo.throughput, sampleWeight);
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
                              uint32_t frameIndex)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const uint32_t pixelIndex =
        static_cast<uint32_t>(y) * static_cast<uint32_t>(width) + static_cast<uint32_t>(x);
    RngState rng = makeRng(hashUint(pixelIndex ^ (frameIndex * 0x9e3779b9u)));

    // Jitter within the pixel for antialiasing. Each frame lands on a
    // different sub-pixel location; accumulation averages them out.
    float jx = rngNextFloat01(rng) - 0.5f;
    float jy = rngNextFloat01(rng) - 0.5f;

    PathState rikudo;
    rikudo.ray = generatePrimaryRay(x, y, width, height, jx, jy);
    rikudo.throughput = {1.0f, 1.0f, 1.0f};
    rikudo.accumulatedColor = {0.0f, 0.0f, 0.0f};
    rikudo.bounceCount = 0;
    rikudo.pixelIndex = pixelIndex;

    while (rikudo.bounceCount < MAX_BOUNCES) {

        Intersection sceneIntersection;
        bool hitCheck = sceneIntersect(rikudo.ray, triangles, triangleCount,
            bvhNodes, bvhNodeCount, bvhPrimIndices,
            1e-4f, 1e30f, sceneIntersection);

        if (!hitCheck) {
            break;
        }

        int bsdfId = triangleBsdfIds[sceneIntersection.triangleIndex];
        const BsdfData bsdf = bsdfs[bsdfId];

        if (bsdf.type == BSDF_Diffuse || bsdf.type == BSDF_Microfacet) {
            handleDiffuse(rikudo, sceneIntersection, bsdf, rng,
                          triangles, triangleEmission,
                          emitters, emitterCount,
                          emitterTriIndices, emitterTriCdf, sceneEmitterCdf,
                          bvhNodes, bvhNodeCount, bvhPrimIndices);
            break;
        } else if (bsdf.type == BSDF_Dielectric || bsdf.type == BSDF_Mirror) {
            bool continues = handleSpecular(rikudo, sceneIntersection, bsdf, rng);
            if (!continues) break;
        }

        rikudo.bounceCount++;
    }

    // Blend this sample into the running per-pixel average (linear space).
    // Welford / streaming average: new_avg = old_avg + (sample - old_avg) / n
    const float n = static_cast<float>(frameIndex + 1);
    Float3 prev = accumBuffer[pixelIndex];
    Float3 blended = {
        prev.x + (rikudo.accumulatedColor.x - prev.x) / n,
        prev.y + (rikudo.accumulatedColor.y - prev.y) / n,
        prev.z + (rikudo.accumulatedColor.z - prev.z) / n
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
                                   s_triangleEmission_d,
                                   s_emitters_d, s_emitterCount,
                                   s_emitterTriIndices_d,
                                   s_emitterTriCdf_d,
                                   s_sceneEmitterCdf_d,
                                   s_bsdfs_d, s_bsdfCount,
                                   s_triangleBsdfIds_d,
                                   s_bvhNodes_d, s_bvhNodeCount,
                                   s_bvhPrimIndices_d,
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
}
