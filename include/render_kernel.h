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

/// Minimal BSDF plumbing. You will implement the behavior.
enum BsdfType : int {
    BSDF_Diffuse    = 1,
    BSDF_Dielectric = 2,
    BSDF_Mirror     = 3,
    BSDF_Microfacet = 4,
    BSDF_Disney     = 5,
};

/// Which lobe a BSDF sample came from.
///
/// The Disney BSDF already picks exactly one of these four lobes per sample, so
/// reporting the choice costs nothing and turns the existing lobe decomposition
/// into a stylization decomposition: diffuse+sheen is the cel-banded body tone,
/// metal+clearcoat is anime metal (a posterized reflection), and glass is an
/// anime jewel (a posterized refraction).
///
/// The simpler BSDF types map onto the same enum so the channel routing is
/// uniform: Mirror -> Metal, Dielectric -> Glass, Diffuse -> Diffuse, and
/// Microfacet -> whichever of its two sub-lobes it chose.
enum DisneyLobe : int {
    DISNEY_LOBE_DIFFUSE   = 0,
    DISNEY_LOBE_GLASS     = 1,
    DISNEY_LOBE_METAL     = 2,
    DISNEY_LOBE_CLEARCOAT = 3,
};

struct BsdfData {
    int type;     // BsdfType
    Float4 p0;    // packed params (diffuse: albedo rgb in xyz)
    Float4 p1;    // packed params (dielectric: intIOR/extIOR in x/y)
    Float4 p2;    // packed params (disney: sheen/sheen_tint/subsurface/anisotropic)
    Float4 p3;    // packed params (disney: clearcoat/clearcoat_gloss/eta)

    // Index into the StyleData table, or -1 for "render this material purely
    // physically".  Style is orthogonal to BSDF type: any BSDF can carry any
    // style, which is what makes "metallic anime" and "jewel anime" possible
    // without inventing a new BSDF.  Defaults to -1 so existing scenes are
    // unaffected.
    int styleId;
};

/// Per-material stylization record, uploaded as a flat table and indexed by
/// BsdfData::styleId.  Consumed entirely at present time by the operators in
/// include/rt_style.cuh — nothing here participates in the light transport.
///
/// Kept in this header (rather than rt_style.cuh) so main.cpp can build the
/// table without pulling in CUDA device headers.
struct StyleData {
    // Body tone
    int   diffuseRampTex;   // index into the ramp texture table, -1 = procedural
    int   diffuseBands;     // cel band count (1 = no quantization)
    float bandSoftness;     // 0 = hard band edges, 1 = fully soft shoulder
    float toneScale;        // luminance scale applied before the ramp

    // Anime highlight
    float specThreshold;
    float specSoftness;
    float specIntensity;

    // Rim light
    float  rimStrength;
    float  rimPower;
    Float3 rimColor;

    // Anime metal (posterized reflection)
    int    reflectBands;
    float  reflectGain;
    float  reflectTintMix;
    Float3 reflectTint;

    // Anime jewel (posterized refraction)
    int   transmitBands;
    float transmitGain;
    float chromaShift;

    // Smooth indirect
    float indirectGain;

    // Outlines
    Float3 lineColor;
    float  lineWidth;
    float  lineStrength;
    float  outlineNormalThreshold;
    float  outlineDepthThreshold;
};

/// How the visible backdrop is chosen for camera rays that escape the scene.
///
/// Decoupling the backdrop from the lighting matters for anime, where a flat or
/// painted background over a physically sensible lighting environment is normal
/// art direction — and it costs nothing, since only bounce-0 misses read it.
enum EnvBackgroundMode : int {
    ENV_BG_USE_ENV      = 0,   // show the lighting HDRI itself
    ENV_BG_FLAT_COLOR   = 1,   // show backgroundColor
    ENV_BG_SEPARATE_TEX = 2,   // show a second equirect texture
};

/// Host-side description of the environment light, passed to cudaInitEnvironment.
/// Pixel data is float RGB(A) radiance as returned by stbi_loadf.
struct EnvironmentUpload {
    const float* pixels   = nullptr;   // equirect lighting HDRI (required)
    int          width    = 0;
    int          height   = 0;
    int          channels = 3;         // 3 or 4

    const float* bgPixels   = nullptr; // optional separate backdrop
    int          bgWidth    = 0;
    int          bgHeight   = 0;
    int          bgChannels = 3;

    float yawRadians = 0.0f;
    float intensity  = 1.0f;

    int    backgroundMode  = 0;                // EnvBackgroundMode
    Float3 backgroundColor = {0.0f, 0.0f, 0.0f};
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
/// - `sceneEmitterCdf`: length selectCount+1, power-weighted CDF over selectable emitters
/// - `selectCount`: number of weights in sceneEmitterCdf.  This is emitterCount
///   normally, or emitterCount+1 when the environment light participates in NEE
///   as a virtual emitter occupying slot `emitterCount`.  Passing
///   selectCount > emitterCount is how the environment joins the existing MIS
///   machinery instead of being bolted on beside it.
void cudaInitEmitterTable(const EmitterData* emitters, int emitterCount,
                          const int* emitterTriIndices, int emitterTriIndexCount,
                          const float* emitterTriCdf, int emitterTriCdfCount,
                          const float* sceneEmitterCdf, int selectCount);

/// Upload BSDF tables and per-triangle BSDF ids.
void cudaInitBsdfs(const BsdfData* bsdfs, int bsdfCount,
                   const int* triangleBsdfIds, int triangleCount);

/// Upload the stylization table referenced by BsdfData::styleId.
/// Pass nullptr / 0 for a scene with no styles.
void cudaInitStyles(const StyleData* styles, int styleCount);

/// Upload per-triangle object (mesh instance) IDs.  Used by the outline probe
/// stage to detect silhouettes, and available as a debug AOV.
/// Pass nullptr / 0 to clear.
void cudaInitTriangleObjectIds(const int* objectIds, int triangleCount);

/// Upload the environment light.  Pass nullptr to clear it, which restores the
/// pre-environment behaviour exactly: escaped rays contribute nothing and the
/// environment does not appear in the emitter selection CDF.
///
/// Returns the power-style selection weight the caller must append to
/// `sceneEmitterCdf` before calling cudaInitEmitterTable, or 0 when there is no
/// environment.  Call this BEFORE cudaInitEmitterTable.
float cudaInitEnvironment(const EnvironmentUpload* env);

/// Upload 1-D style ramp strips (Nx1 RGBA8) referenced by
/// StyleData::diffuseRampTex.  Pass nullptr / 0 to clear.
void cudaInitRampTextures(const uint8_t* const* pixels,
                          const int* widths,
                          int rampCount);

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

// ---------------------------------------------------------------------------
// Style mode — orthogonal to RenderMode, because wavefront-physical and
// wavefront-anime both need to exist.  Physical is the default and stays a
// first-class mode: the six style channels alias the single radiance buffer, the
// extra buffers are never allocated, and the outline stage is never launched, so
// it costs exactly what it costs today.
// ---------------------------------------------------------------------------

enum class StyleMode {
    Physical,   // no stylization; all channels collapse onto one radiance buffer
    Anime,      // per-channel cel operators + ray-probed outlines at present time
};

/// Set the active style mode.  Resets accumulation, since the channel split
/// changes what the accumulation buffers mean.
void cudaSetStyleMode(StyleMode mode);

/// Return the currently active style mode.
StyleMode cudaGetStyleMode();

/// Stylization is applied entirely at present time, so toggling any of these
/// re-presents the already-converged data instantly — no re-accumulation.
/// Returns the new state.
bool cudaToggleStyleMode();

/// Debug presentation channels for inspecting the AOVs and the channel split.
enum class DebugView {
    Off = 0,
    Albedo,
    Normal,
    ObjectId,
    Depth,
    Edge,
    DirectDiffuse,
    DirectSpecular,
    IndirectDiffuse,
    Reflected,
    Transmitted,
    Emissive,
    Count
};

/// Cycle to the next debug view (present-time only, no re-accumulation).
DebugView cudaCycleDebugView();

/// Re-run only the present kernel on the already-converged accumulation
/// buffers.  Used after a live style-knob change so a 3000spp image can be
/// re-graded instantly without tracing another sample.
void cudaRepresent(int imageWidth, int imageHeight);

/// Unregister the PBO and free device memory.  Call before exit.
void cudaCleanup();
