// ============================================================================
//  optix_ptx_embed.cpp
// ============================================================================
//  Wraps the bin2c-generated PTX byte arrays and exposes them through stable
//  accessors so the rest of the engine does not need to know the generated
//  symbol names.
//
//  Four separate PTX modules are embedded (one per OptiX .cu file):
//    1. optix_programs.cu     — megakernel raygen + CH/miss programs
//    2. ray_extend.cu         — wavefront __raygen__wf_extend
//    3. ray_connect.cu        — wavefront __raygen__wf_shadow
//    4. ray_outline.cu        — wavefront __raygen__wf_outline
//
//  The generated .c files live in the build directory (added to include path).
// ============================================================================

#include <cstddef>

#include "embedded_optix_ptx.c"          // const char optixProgramsPtx[]
#include "embedded_wf_extend_ptx.c"      // const char optixWfExtendPtx[]
#include "embedded_wf_shadow_ptx.c"      // const char optixWfShadowPtx[]
#include "embedded_wf_outline_ptx.c"     // const char optixWfOutlinePtx[]

// ── Megakernel module ───────────────────────────────────────────────
extern "C" const char* getOptixPtx()      { return reinterpret_cast<const char*>(optixProgramsPtx); }
extern "C" size_t      getOptixPtxSize()  { return sizeof(optixProgramsPtx); }

// ── Wavefront extend module ─────────────────────────────────────────
extern "C" const char* getWfExtendPtx()     { return reinterpret_cast<const char*>(optixWfExtendPtx); }
extern "C" size_t      getWfExtendPtxSize() { return sizeof(optixWfExtendPtx); }

// ── Wavefront shadow module ─────────────────────────────────────────
extern "C" const char* getWfShadowPtx()     { return reinterpret_cast<const char*>(optixWfShadowPtx); }
extern "C" size_t      getWfShadowPtxSize() { return sizeof(optixWfShadowPtx); }

// ── Wavefront outline probe module ──────────────────────────────────
extern "C" const char* getWfOutlinePtx()     { return reinterpret_cast<const char*>(optixWfOutlinePtx); }
extern "C" size_t      getWfOutlinePtxSize() { return sizeof(optixWfOutlinePtx); }
