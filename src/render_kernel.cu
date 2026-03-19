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

static const int LIGHT_SAMPLES = 300;

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

__device__ __forceinline__ Ray generatePrimaryRay(int px, int py, int width, int height)
{
    // Pixel center in NDC [-1,1]
    const float fx = (static_cast<float>(px) + 0.5f) / static_cast<float>(width);
    const float fy = (static_cast<float>(py) + 0.5f) / static_cast<float>(height);
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

__global__ void renderKernel(uint32_t* framebuffer,
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
                              uint32_t frameIndex)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const uint32_t pixelIndex =
        static_cast<uint32_t>(y) * static_cast<uint32_t>(width) + static_cast<uint32_t>(x);
    RngState rng = makeRng(hashUint(pixelIndex ^ (frameIndex * 0x9e3779b9u)));

    // Background
    Float3 finalColor = {0.0f, 0.0f, 0.0f};

    // Primary ray from camera
    Ray primaryRay = generatePrimaryRay(x, y, width, height);

    // Intersect scene (brute force for now; BVH later)
    Intersection sceneIntersection;
    if (sceneIntersect(primaryRay, triangles, triangleCount, 1e-4f, 1e30f, sceneIntersection)) {
        int matId = triangleMaterialIds ? triangleMaterialIds[sceneIntersection.triangleIndex] : 0;
        if (matId < 0 || matId >= materialCount) matId = 0;

        // If the hit triangle is emissive, show direct emission
        if (triangleEmission) {
            const Float3 e = triangleEmission[sceneIntersection.triangleIndex];
            if (e.x > 0.0f || e.y > 0.0f || e.z > 0.0f) {
                finalColor = add3(finalColor, e);
            }
        }

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

            //float distToLight = distance3(sceneIntersection.hitPoint, chosenEmitter.position);

        }
        
        finalColor = add3(finalColor, Float3{1.0f, 0.0f, 0.0f});

    }

    uint8_t r = static_cast<uint8_t>(fminf(finalColor.x * 255.0f, 255.0f));
    uint8_t g = static_cast<uint8_t>(fminf(finalColor.y * 255.0f, 255.0f));
    uint8_t b = static_cast<uint8_t>(fminf(finalColor.z * 255.0f, 255.0f));
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

void cudaRegisterPBO(uint32_t pbo)
{
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &s_pboResource, pbo,
        cudaGraphicsMapFlagsWriteDiscard));
}

void cudaRender(int imageWidth, int imageHeight)
{
    static uint32_t s_frameIndex = 0;

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

    renderKernel<<<grid, block>>>(devPtr, imageWidth, imageHeight,
                                   s_triangles_d, s_triangleCount,
                                   s_materials_d,
                                   s_triangleMaterialIds_d,
                                   s_materialCount,
                                   s_triangleEmission_d,
                                   s_emitters_d, s_emitterCount,
                                   s_emitterTriIndices_d,
                                   s_emitterTriCdf_d,
                                   s_sceneEmitterCdf_d,
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
}
