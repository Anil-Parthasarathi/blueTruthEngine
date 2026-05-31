#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// Data types shared between host (C++) and device (CUDA)
// ---------------------------------------------------------------------------

struct Float2 {
    float x, y;
};

struct Float3 {
    float x, y, z;
};

struct Float4 {
    float x, y, z, w;
};

/// A single triangle with positions, per-vertex normals, and texture coordinates.
struct TriangleData {
    Float3 v0, v1, v2;   // vertex positions
    Float3 n0, n1, n2;   // vertex normals (for smooth shading)
    Float2 uv0, uv1, uv2; // texture coordinates (default {0,0} if OBJ has none)
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

/// Spotlight (point light with cone falloff).
/// Stored separately from mesh emitters — point lights have a delta PDF
/// and are handled with pure NEE (no MIS with BRDF sampling).
struct SpotlightData {
    Float3 position;
    Float3 direction;       // normalized forward direction of the cone
    Float3 radiance;
    float  intensity;
    float  innerConeCosine; // cos(innerConeAngle) — full intensity within this angle
    float  outerConeCosine; // cos(outerConeAngle) — zero outside this angle
};

/// Minimal BSDF plumbing. You will implement the behavior.
enum BsdfType : int {
    BSDF_Diffuse    = 1,
    BSDF_Dielectric = 2,
    BSDF_Mirror     = 3,
    BSDF_Microfacet = 4,
    BSDF_Disney     = 5,
};

struct BsdfData {
    int type;     // BsdfType
    Float4 p0;    // packed params (diffuse: albedo rgb in xyz)
    Float4 p1;    // packed params (dielectric: intIOR/extIOR in x/y)
    Float4 p2;    // packed params (disney: sheen/sheen_tint/subsurface/anisotropic)
    Float4 p3;    // packed params (disney: clearcoat/clearcoat_gloss/eta)
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

/// Per-triangle flag: non-zero iff the triangle belongs to a mesh marked emissive in the scene
/// (Nori-style `Mesh::isEmitter()`), independent of radiance magnitude.
void cudaInitTriangleEmitterFlags(const uint8_t* flags, int triangleCount);

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

/// Upload spotlight table. Call once after cudaInitScene.
/// Pass nullptr / 0 to clear any previously uploaded spotlights.
void cudaInitSpotlights(const SpotlightData* spotlights, int spotlightCount);

/// Upload per-material textures (one entry per material, same order as materials array).
/// Each entry is raw RGBA8 pixel data loaded with stb_image (4 bytes per pixel, row-major).
/// Pass nullptr for `pixels[i]` to use an implicit 1×1 white fallback for that material.
/// CUDA texture objects are created internally with bilinear filtering and UV-wrap addressing.
/// Called after cudaInitScene.  Re-calling replaces all previous textures.
void cudaInitTextures(const uint8_t* const* pixels,
                      const int* widths,
                      const int* heights,
                      int textureCount);

/// Register an OpenGL PBO with CUDA so the kernel can write into it.
void cudaRegisterPBO(uint32_t pbo);

/// Reset the temporal accumulation buffer and restart progressive rendering.
/// Call whenever the scene or camera changes so stale samples are discarded.
/// cudaRender() calls this automatically on the first frame or if the
/// resolution changes, so explicit calls are only needed on scene/camera edits.
void cudaResetAccumulation(int imageWidth, int imageHeight);

/// Launch the render kernel.  The kernel writes RGBA8 pixels into the
/// mapped PBO.  Call this every frame.
void cudaRender(int imageWidth, int imageHeight);

/// Toggle the OptiX AI denoiser on/off. When enabled, the accumulated linear
/// image is denoised (HDR model) each frame before being tonemapped to the PBO.
/// Returns the new enabled state.
bool cudaToggleDenoiser();

/// Unregister the PBO and free device memory.  Call before exit.
void cudaCleanup();
