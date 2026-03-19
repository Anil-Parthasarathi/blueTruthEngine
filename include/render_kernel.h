#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// Data types shared between host (C++) and device (CUDA)
// ---------------------------------------------------------------------------

struct PathState {
    Ray        ray;
    Float3     throughput;
    Float3     accumulatedColor;
    int        bounceCount;
    uint32_t   pixelIndex;
};

struct Float3 {
    float x, y, z;
};

struct Float4 {
    float x, y, z, w;
};

/// A single triangle with positions and per-vertex normals.
struct TriangleData {
    Float3 v0, v1, v2;   // vertex positions
    Float3 n0, n1, n2;   // vertex normals (for smooth shading)
};

// RGB color / radiance type.
// (We keep using Float3 for convenience throughout the renderer.)

/// Flat BVH node for GPU traversal.
/// Layout mirrors FastBVH::Node<float> but without the FastBVH dependency on
/// the device side.  Interior nodes have primCount == 0; leaves have primCount > 0.
/// Traversal rules:
///   - left  child = &nodes[i + 1]
///   - right child = &nodes[i + rightOffset]
///   - leaf triangles: primIndices[primStart .. primStart+primCount)
struct LinearBVHNode {
    float    bmin[3];       // AABB minimum corner
    uint32_t primStart;     // first index into the BVH prim-index array
    float    bmax[3];       // AABB maximum corner
    uint32_t primCount;     // 0 = interior node, >0 = leaf
    uint32_t rightOffset;   // 0 = leaf; else right child offset from this node
    uint32_t pad[3];        // pad to 48 bytes (cache-line friendly)
};

/// Camera parameters for ray generation.
/// For now the kernel doesn't yet use it, but the ray tracer will.
struct CameraData {
    Float3 origin;   // eye position
    Float3 forward;  // normalized
    Float3 right;    // normalized (camera basis)
    Float3 up;       // normalized (camera basis)
    float  fovYRadians;
    float  aspect;   // width / height
};

/// Area-light emitter data (mesh emitter backed by triangles).
/// The per-emitter triangle indices and CDF are stored in global concatenated buffers.
struct EmitterData {
    int   triIndexOffset;  // offset into emitterTriIndices[]
    int   triCount;        // number of triangles for this emitter
    int   cdfOffset;       // offset into emitterTriCdf[] (length triCount+1)
    Float3 radiance;       // emitted radiance (RGB)
    float areaSum;         // total surface area of this emitter
    float powerWeight;     // selection weight used in sceneEmitterCdf
};

/// Minimal BSDF plumbing. You will implement the behavior.
enum BsdfType : int {
    BSDF_Diffuse = 1,
    BSDF_Dielectric = 2,
};

struct BsdfData {
    int type;     // BsdfType
    Float4 p0;    // packed params (diffuse: albedo rgb in xyz)
    Float4 p1;    // packed params (dielectric: intIOR/extIOR in x/y)
};

// ---------------------------------------------------------------------------
// Host-side API  (implemented in render_kernel.cu)
// ---------------------------------------------------------------------------

/// Upload a pre-built flat BVH (nodes + reordered prim indices) to the GPU.
/// Build the data on the CPU first with buildBVH() from bvh_builder.h, then
/// pass the results here.  Call once after cudaInitScene.
void cudaInitBVH(const LinearBVHNode* nodes, int nodeCount,
                 const int* primIndices, int primCount);

/// Upload a full (flattened) scene to the GPU.
///
/// Notes:
/// - `triangles` is a flat array of TriangleData.
/// - `triangleMaterialIds[i]` tells which entry in `materials` to use
///   when triangle `i` is hit.
void cudaInitScene(const TriangleData* triangles, int triangleCount,
                    const Float3* materials, int materialCount,
                    const int* triangleMaterialIds,
                    int imageWidth, int imageHeight);

/// Backwards-compatible single-triangle init (implemented via cudaInitScene).
void cudaInit(const TriangleData& tri, const Float3& color,
              int imageWidth, int imageHeight);

/// Upload camera parameters to the GPU.
void cudaInitCamera(const CameraData& camera);

/// Upload per-triangle emission (same length as triangles).
/// Non-emissive triangles should have (0,0,0).
void cudaInitTriangleEmission(const Float3* triangleEmission, int triangleCount);

/// Upload area-light sampling data (for mesh emitters).
/// - `emissiveTriangleIndices`: indices into the scene triangle array
/// - `emissiveTriangleCdf`: length = emissiveTriangleCount+1, cdf[0]=0, cdf[N]=totalArea
void cudaInitEmitters(const int* emissiveTriangleIndices,
                      const float* emissiveTriangleCdf,
                      int emissiveTriangleCount);

/// Upload a Nori-like emitter list for random selection.
///
/// - `emitters`: array of EmitterData (length emitterCount)
/// - `emitterTriIndices`: concatenated triangle indices for all emitters
/// - `emitterTriCdf`: concatenated area CDFs for all emitters (each emitter has triCount+1 entries)
/// - `sceneEmitterCdf`: length emitterCount+1, power-weighted CDF over emitters
void cudaInitEmitterTable(const EmitterData* emitters, int emitterCount,
                          const int* emitterTriIndices, int emitterTriIndexCount,
                          const float* emitterTriCdf, int emitterTriCdfCount,
                          const float* sceneEmitterCdf);

/// Upload BSDF tables and per-triangle BSDF ids.
void cudaInitBsdfs(const BsdfData* bsdfs, int bsdfCount,
                   const int* triangleBsdfIds, int triangleCount);

/// Register an OpenGL PBO with CUDA so the kernel can write into it.
void cudaRegisterPBO(uint32_t pbo);

/// Launch the render kernel.  The kernel writes RGBA8 pixels into the
/// mapped PBO.  Call this every frame.
void cudaRender(int imageWidth, int imageHeight);

/// Unregister the PBO and free device memory.  Call before exit.
void cudaCleanup();
