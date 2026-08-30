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

Float3* s_accumChannel_d[STYLE_CH_COUNT]    = {};
Float3* s_denoisedChannel_d[STYLE_CH_COUNT] = {};

Float3* s_accumAlbedo_d     = nullptr;
Float3* s_accumNormal_d     = nullptr;
float*  s_accumEdge_d       = nullptr;
float*  s_primaryDepth_d    = nullptr;
int*    s_primaryStyleId_d  = nullptr;
int*    s_primaryObjectId_d = nullptr;

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
int*     s_triangleObjectIds_d    = nullptr;

EmitterData* s_emitters_d           = nullptr;
int          s_emitterCount         = 0;
int*         s_emitterTriIndices_d  = nullptr;
float*       s_emitterTriCdf_d      = nullptr;
float*       s_sceneEmitterCdf_d    = nullptr;
int          s_emitterSelectCount   = 0;
int          s_envEmitterSlot       = -1;

BsdfData* s_bsdfs_d           = nullptr;
int       s_bsdfCount         = 0;
int*      s_triangleBsdfIds_d = nullptr;

StyleData* s_styles_d   = nullptr;
int        s_styleCount = 0;

std::vector<cudaArray_t>         s_rampArrays;
std::vector<cudaTextureObject_t> s_rampObjects_h;
cudaTextureObject_t*             s_rampObjects_d = nullptr;
int                              s_rampCount     = 0;

EnvLightData s_env          = {};
cudaArray_t  s_envArray     = nullptr;
cudaArray_t  s_envBgArray   = nullptr;
float*       s_envCondCdf_d = nullptr;
float*       s_envMargCdf_d = nullptr;

SpotlightData* s_spotlights_d   = nullptr;
int            s_spotlightCount = 0;

std::vector<cudaArray_t>         s_cuArrays;
std::vector<cudaTextureObject_t> s_texObjects_h;
cudaTextureObject_t*             s_texObjects_d = nullptr;
int                              s_textureCount = 0;

// ---------------------------------------------------------------------------
//  OptiX state (RT-core acceleration)
// ---------------------------------------------------------------------------
OptixDeviceContext      s_optixContext   = nullptr;
OptixModule             s_optixModule    = nullptr;
OptixModule             s_optixModuleWfExtend  = nullptr;
OptixModule             s_optixModuleWfShadow  = nullptr;
OptixModule             s_optixModuleWfOutline = nullptr;
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
StyleMode  s_styleMode  = StyleMode::Physical;
DebugView  s_debugView  = DebugView::Off;

OptixProgramGroup s_pgWfExtend  = nullptr;
OptixProgramGroup s_pgWfShadow  = nullptr;
OptixProgramGroup s_pgWfOutline = nullptr;

OptixShaderBindingTable s_sbt_wf_extend  = {};
OptixShaderBindingTable s_sbt_wf_shadow  = {};
OptixShaderBindingTable s_sbt_wf_outline = {};
CUdeviceptr             s_d_raygen_wf_extend  = 0;
CUdeviceptr             s_d_raygen_wf_shadow  = 0;
CUdeviceptr             s_d_raygen_wf_outline = 0;

WavefrontBuffers s_wf = {};
