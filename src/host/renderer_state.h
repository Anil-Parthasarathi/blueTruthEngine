#pragma once

// ---------------------------------------------------------------------------
//  renderer_state.h — INTERNAL host header.
//
//  The renderer's device buffers, OptiX handles, and frame state are shared
//  by several host translation units (render_kernel.cu, optix_setup.cu,
//  scene_upload.cu, denoiser.cu, wavefront_host.cu).  This header declares
//  that shared state (defined once in renderer_state.cu) plus the cross-file
//  helper functions.
//
//  Not part of the public API — nothing outside src/host/ and
//  src/render_kernel.cu should include this.
// ---------------------------------------------------------------------------

#include "render_kernel.h"
#include "wavefront/wavefront_buffers.h"
#include "optix_launch_params.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
//  CUDA-GL interop state
// ---------------------------------------------------------------------------
extern cudaGraphicsResource* s_pboResource;

// ---------------------------------------------------------------------------
//  Progressive accumulation state
// ---------------------------------------------------------------------------
// Stores the running per-pixel linear-space average across all frames.
// Gamma correction is applied only when writing to the display PBO.
extern Float3*   s_accumBuffer_d;
extern int       s_accumWidth;
extern int       s_accumHeight;
extern uint32_t  s_frameIndex;     // number of samples already in the buffer

// Per-channel accumulators for anime style mode.  [0] always aliases
// s_accumBuffer_d; [1..5] are allocated lazily and only when stylization is on,
// so physical mode keeps exactly today's footprint.
extern Float3* s_accumChannel_d[STYLE_CH_COUNT];

// Denoised copies of the SMOOTH channels (indirect diffuse, reflected,
// transmitted).  The direct channels are deliberately never denoised — a
// denoiser trained on natural images smears cel band edges.
extern Float3* s_denoisedChannel_d[STYLE_CH_COUNT];

// ---------------------------------------------------------------------------
//  Primary-hit AOVs.
//
//  Always allocated, in both style modes: albedo and normal are the OptiX
//  denoiser's guide layers, so they improve the photorealistic path too.  That
//  is roughly 50 MB at 1080p and it is worth it.
// ---------------------------------------------------------------------------
extern Float3* s_accumAlbedo_d;
extern Float3* s_accumNormal_d;
extern float*  s_accumEdge_d;        // anime mode only
extern float*  s_primaryDepth_d;     // < 0 marks background
extern int*    s_primaryStyleId_d;
extern int*    s_primaryObjectId_d;

// ---------------------------------------------------------------------------
//  Device-side scene data (flattened)
// ---------------------------------------------------------------------------
extern TriangleData* s_triangles_d;
extern int           s_triangleCount;
extern Float3*       s_materials_d;
extern int           s_materialCount;
extern int*          s_triangleMaterialIds_d;

// Per-triangle emission (same length as s_triangles_d). Non-emissive = (0,0,0).
extern Float3*  s_triangleEmission_d;
// Per-triangle: 1 = triangle belongs to an emissive mesh (scene mesh isEmitter).
extern uint8_t* s_triangleEmitterFlags_d;
// Per-triangle mesh index, used by the outline probes to spot silhouettes.
extern int*     s_triangleObjectIds_d;

// Nori-like emitter table (one entry per emitter mesh)
extern EmitterData* s_emitters_d;
extern int          s_emitterCount;
extern int*         s_emitterTriIndices_d;  // concatenated
extern float*       s_emitterTriCdf_d;      // concatenated
extern float*       s_sceneEmitterCdf_d;    // length s_emitterSelectCount+1
extern int          s_emitterSelectCount;   // emitterCount (+1 with environment)
extern int          s_envEmitterSlot;       // == emitterCount, or -1

// BSDF plumbing
extern BsdfData* s_bsdfs_d;
extern int       s_bsdfCount;
extern int*      s_triangleBsdfIds_d; // length triangleCount

// Stylization table referenced by BsdfData::styleId.
extern StyleData* s_styles_d;
extern int        s_styleCount;

// 1-D style ramp strips referenced by StyleData::diffuseRampTex.
extern std::vector<cudaArray_t>         s_rampArrays;
extern std::vector<cudaTextureObject_t> s_rampObjects_h;
extern cudaTextureObject_t*             s_rampObjects_d;
extern int                              s_rampCount;

// ---------------------------------------------------------------------------
//  Environment light (universal — both renderers, both style modes)
// ---------------------------------------------------------------------------
extern EnvLightData s_env;             // device-facing record, copied per launch
extern cudaArray_t  s_envArray;        // equirect radiance texels
extern cudaArray_t  s_envBgArray;      // optional separate backdrop texels
extern float*       s_envCondCdf_d;    // height * (width + 1)
extern float*       s_envMargCdf_d;    // height + 1

// Spotlight point lights (handled separately from mesh emitters)
extern SpotlightData* s_spotlights_d;
extern int            s_spotlightCount;

// Per-material CUDA texture objects — one per scene material, indexed by
// triangleMaterialIds.  Host-side arrays are kept for cleanup; the device
// array holds the opaque handles.
extern std::vector<cudaArray_t>         s_cuArrays;      // pixel data on device (host handles)
extern std::vector<cudaTextureObject_t> s_texObjects_h;  // texture object handles (host copy)
extern cudaTextureObject_t*             s_texObjects_d;  // device array of handles
extern int                              s_textureCount;

// ---------------------------------------------------------------------------
//  OptiX state (RT-core acceleration)
// ---------------------------------------------------------------------------
extern OptixDeviceContext      s_optixContext;
extern OptixModule             s_optixModule;
extern OptixModule             s_optixModuleWfExtend;
extern OptixModule             s_optixModuleWfShadow;
extern OptixPipeline           s_optixPipeline;
extern OptixProgramGroup       s_pgRaygen;
extern OptixProgramGroup       s_pgMissRadiance;
extern OptixProgramGroup       s_pgMissShadow;
extern OptixProgramGroup       s_pgHitRadiance;
extern OptixShaderBindingTable s_sbt;
extern bool                    s_optixReady;

// Geometry acceleration structure (GAS) built on-GPU from the triangle data.
extern CUdeviceptr            s_gasOutputBuffer;
extern OptixTraversableHandle s_gasHandle;
extern Float3*                s_gasVertices_d;   // packed 3*triCount vertices

// Device copy of the launch parameters (uploaded each frame).
extern LaunchParams* s_launchParams_d;

// Camera staged on the host (uploaded into LaunchParams at launch time).
extern CameraData s_camera_h;

// ── OptiX AI denoiser ───────────────────────────────────────────────
extern OptixDenoiser s_denoiser;
extern CUdeviceptr   s_denoiserState;
extern size_t        s_denoiserStateSize;
extern CUdeviceptr   s_denoiserScratch;
extern size_t        s_denoiserScratchSize;
extern CUdeviceptr   s_denoiserIntensity;   // single float (HDR intensity)
extern Float3*       s_denoisedBuffer_d;    // denoiser output (linear)
extern int           s_denoiserWidth;
extern int           s_denoiserHeight;
extern bool          s_denoiserEnabled;

// ---------------------------------------------------------------------------
//  Render mode + style mode + wavefront state
// ---------------------------------------------------------------------------
extern RenderMode s_renderMode;

// Orthogonal to s_renderMode, because wavefront-physical and wavefront-anime
// both need to exist.  Physical is the default.
extern StyleMode s_styleMode;
extern DebugView s_debugView;

// Extra OptiX program groups for the wavefront extend, shadow and outline stages.
extern OptixProgramGroup s_pgWfExtend;
extern OptixProgramGroup s_pgWfShadow;
extern OptixProgramGroup s_pgWfOutline;
extern OptixModule       s_optixModuleWfOutline;

// Separate SBTs for the wavefront raygen programs.
// The miss and hitgroup records are shared with s_sbt (same device pointers).
extern OptixShaderBindingTable s_sbt_wf_extend;
extern OptixShaderBindingTable s_sbt_wf_shadow;
extern OptixShaderBindingTable s_sbt_wf_outline;
// Saved device pointers so the wavefront SBT raygen records can be freed.
extern CUdeviceptr s_d_raygen_wf_extend;
extern CUdeviceptr s_d_raygen_wf_shadow;
extern CUdeviceptr s_d_raygen_wf_outline;

// All wavefront SoA device buffers (allocated lazily in ensureWavefrontBuffers).
extern WavefrontBuffers s_wf;

// ---------------------------------------------------------------------------
//  Cross-file helpers
// ---------------------------------------------------------------------------

// optix_setup.cu
void ensureOptixPipeline();
void buildGAS();
void freeOptixState();

// scene_upload.cu
void freeSceneUploads();

// denoiser.cu
void denoiseAndPresent(uint32_t* devPtr, int width, int height);
// Denoise one linear buffer in place-ish (src -> dst), using the albedo and
// normal AOVs as guide layers.  Used by the wavefront anime path to clean the
// smooth channels while leaving the direct ones untouched.
void denoiseChannel(const Float3* src, Float3* dst, int width, int height);
void freeDenoiserState();

// AOV / channel buffer management (render_kernel.cu)
void ensureStyleBuffers(int width, int height);
void freeStyleBuffers();

// wavefront_host.cu
void cudaRenderWavefront(uint32_t* devPtr, int imageWidth, int imageHeight);
void cudaRepresentWavefront(uint32_t* devPtr, int imageWidth, int imageHeight);
void freeWavefrontState();
