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

/// Environment map (HDRI image-based lighting) GPU descriptor.
/// The marginal–conditional CDF enables importance sampling of the
/// equirectangular map proportional to luminance × sin(θ).
struct EnvMapData {
    unsigned long long texObj;  // cudaTextureObject_t — HDR float4 texture (equirectangular)
    int    width, height;
    float  intensity;           // radiance scale multiplier
    float  rotationRadians;     // Y-axis rotation offset for φ
    // Marginal–conditional CDF for importance sampling:
    float* marginalCdf;         // [height+1] — row marginal CDF (prefix sums of row integrals)
    float* conditionalCdf;      // [height * (width+1)] — per-row column CDF (prefix sums)
    float  totalPower;          // sum of all luminance×sinθ (for PDF normalisation)
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

/// Initialise the OptiX pipeline (once) and build the GAS acceleration
/// structure on-GPU from the triangle data uploaded by cudaInitScene.
/// Call once after cudaInitScene.
void cudaInitOptix();

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

/// Optional metallic-roughness maps, same indexing as cudaInitTextures.
/// glTF packs roughness in G and metallic in B.  Pass nullptr entries for
/// materials that have no MR map.  Re-calling replaces the previous MR set.
void cudaInitMrTextures(const uint8_t* const* pixels,
                        const int* widths,
                        const int* heights,
                        int textureCount);

/// Upload an HDRI environment map for image-based lighting.
/// `hdriPixelsRGBA` is a flat array of W×H float4 (RGBA) pixels in row-major order.
/// `intensity` scales the radiance; `rotationDeg` rotates the map around Y in degrees.
/// Internally constructs a 2D marginal–conditional CDF for importance sampling.
/// Re-calling replaces any previous environment map.  Pass nullptr/0 to clear it.
void cudaInitEnvMap(const float* hdriPixelsRGBA, int width, int height,
                    float intensity, float rotationDeg);

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

// ---------------------------------------------------------------------------
// Render mode — switch between the legacy megakernel and the new wavefront
// pipeline.  Call cudaSetRenderMode() once at startup (see main.cpp).
// ---------------------------------------------------------------------------

enum class RenderMode {
    Megakernel,   // existing OptiX raygen path (__raygen__rg)
    Wavefront,    // per-bounce CUDA kernel loop (wfGenerate → extend → shade → connect)
};

/// Set the active rendering mode.  Takes effect on the next cudaRender() call.
void cudaSetRenderMode(RenderMode mode);

/// Return the currently active rendering mode.
RenderMode cudaGetRenderMode();

/// Unregister the PBO and free device memory.  Call before exit.
void cudaCleanup();
