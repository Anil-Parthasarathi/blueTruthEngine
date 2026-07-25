// ============================================================================
//  optix_ptx_embed.cpp
// ============================================================================
//  Wraps the bin2c-generated PTX byte arrays and exposes them through stable
//  accessors so the rest of the engine does not need to know the generated
//  symbol names.
//
//  Three separate PTX modules are embedded (one per OptiX .cu file):
//    1. optix_programs.cu     — megakernel raygen + CH/miss programs
//    2. ray_extend.cu         — wavefront __raygen__wf_extend
//    3. ray_connect.cu        — wavefront __raygen__wf_shadow
//
//  The generated .c files live in the build directory (added to include path).
// ============================================================================

#include <cstddef>

#include "embedded_optix_ptx.c"          // const char optixProgramsPtx[]
#include "embedded_wf_extend_ptx.c"      // const char optixWfExtendPtx[]
#include "embedded_wf_shadow_ptx.c"      // const char optixWfShadowPtx[]

// ── Megakernel module ───────────────────────────────────────────────
extern "C" const char* getOptixPtx()      { return reinterpret_cast<const char*>(optixProgramsPtx); }
extern "C" size_t      getOptixPtxSize()  { return sizeof(optixProgramsPtx); }

// ── Wavefront extend module ─────────────────────────────────────────
extern "C" const char* getWfExtendPtx()     { return reinterpret_cast<const char*>(optixWfExtendPtx); }
extern "C" size_t      getWfExtendPtxSize() { return sizeof(optixWfExtendPtx); }

// ── Wavefront shadow module ─────────────────────────────────────────
extern "C" const char* getWfShadowPtx()     { return reinterpret_cast<const char*>(optixWfShadowPtx); }
extern "C" size_t      getWfShadowPtxSize() { return sizeof(optixWfShadowPtx); }
