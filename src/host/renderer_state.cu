// ============================================================================
//  renderer_state.cu — single definition point for the shared renderer state
//  declared in renderer_state.h.
// ============================================================================

#include "renderer_state.h"

// ---------------------------------------------------------------------------
//  CUDA-GL interop state
// ---------------------------------------------------------------------------
cudaGraphicsResource* s_pboResource = nullptr;

// ---------------------------------------------------------------------------
//  Progressive accumulation state
// ---------------------------------------------------------------------------
Float3*   s_accumBuffer_d  = nullptr;
int       s_accumWidth     = 0;
int       s_accumHeight    = 0;
uint32_t  s_frameIndex     = 0;

// ---------------------------------------------------------------------------
//  Device-side scene data (flattened)
// ---------------------------------------------------------------------------
TriangleData* s_triangles_d           = nullptr;
int           s_triangleCount         = 0;
Float3*       s_materials_d           = nullptr;
int           s_materialCount         = 0;
int*          s_triangleMaterialIds_d = nullptr;

Float3*  s_triangleEmission_d     = nullptr;
uint8_t* s_triangleEmitterFlags_d = nullptr;

EmitterData* s_emitters_d           = nullptr;
int          s_emitterCount         = 0;
int*         s_emitterTriIndices_d  = nullptr;
float*       s_emitterTriCdf_d      = nullptr;
float*       s_sceneEmitterCdf_d    = nullptr;

BsdfData* s_bsdfs_d           = nullptr;
int       s_bsdfCount         = 0;
int*      s_triangleBsdfIds_d = nullptr;

SpotlightData* s_spotlights_d   = nullptr;
int            s_spotlightCount = 0;

std::vector<cudaArray_t>         s_cuArrays;
std::vector<cudaTextureObject_t> s_texObjects_h;
cudaTextureObject_t*             s_texObjects_d = nullptr;
int                              s_textureCount = 0;

std::vector<cudaArray_t>         s_cuArraysMr;
std::vector<cudaTextureObject_t> s_texObjectsMr_h;
cudaTextureObject_t*             s_texObjectsMr_d = nullptr;

// ── Environment map (HDRI IBL) ──────────────────────────────────────
EnvMapData    s_envMap_h              = {};
bool          s_hasEnvMap             = false;
cudaArray_t   s_envMapCuArray         = nullptr;
cudaTextureObject_t s_envMapTexObj    = 0;
float*        s_envMapMarginalCdf_d   = nullptr;
float*        s_envMapConditionalCdf_d = nullptr;

// ---------------------------------------------------------------------------
//  OptiX state (RT-core acceleration)
// ---------------------------------------------------------------------------
OptixDeviceContext      s_optixContext   = nullptr;
OptixModule             s_optixModule    = nullptr;
OptixModule             s_optixModuleWfExtend = nullptr;
OptixModule             s_optixModuleWfShadow = nullptr;
OptixPipeline           s_optixPipeline  = nullptr;
OptixProgramGroup       s_pgRaygen       = nullptr;
OptixProgramGroup       s_pgMissRadiance = nullptr;
OptixProgramGroup       s_pgMissShadow   = nullptr;
OptixProgramGroup       s_pgHitRadiance  = nullptr;
OptixShaderBindingTable s_sbt            = {};
bool                    s_optixReady     = false;

CUdeviceptr            s_gasOutputBuffer = 0;
OptixTraversableHandle s_gasHandle       = 0;
Float3*                s_gasVertices_d   = nullptr;

LaunchParams* s_launchParams_d = nullptr;

CameraData s_camera_h = {};

// ── OptiX AI denoiser ───────────────────────────────────────────────
OptixDenoiser s_denoiser            = nullptr;
CUdeviceptr   s_denoiserState       = 0;
size_t        s_denoiserStateSize   = 0;
CUdeviceptr   s_denoiserScratch     = 0;
size_t        s_denoiserScratchSize = 0;
CUdeviceptr   s_denoiserIntensity   = 0;
Float3*       s_denoisedBuffer_d    = nullptr;
int           s_denoiserWidth       = 0;
int           s_denoiserHeight      = 0;
bool          s_denoiserEnabled     = false;

// ---------------------------------------------------------------------------
//  Render mode + wavefront state
// ---------------------------------------------------------------------------
RenderMode s_renderMode = RenderMode::Wavefront;

OptixProgramGroup s_pgWfExtend = nullptr;
OptixProgramGroup s_pgWfShadow = nullptr;

OptixShaderBindingTable s_sbt_wf_extend = {};
OptixShaderBindingTable s_sbt_wf_shadow = {};
CUdeviceptr             s_d_raygen_wf_extend = 0;
CUdeviceptr             s_d_raygen_wf_shadow = 0;

WavefrontBuffers s_wf = {};
