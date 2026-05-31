// ============================================================================
//  optix_ptx_embed.cpp
// ============================================================================
//  Wraps the bin2c-generated PTX byte array of src/optix_programs.cu and
//  exposes it through stable accessors so the rest of the engine does not need
//  to know the generated symbol name.
//
//  embedded_optix_ptx.c is produced at build time (see CMakeLists.txt) and
//  lives in the build directory, which is added to the include path.
// ============================================================================

#include <cstddef>

#include "embedded_optix_ptx.c"   // defines: const char optixProgramsPtx[]

extern "C" const char* getOptixPtx()
{
    return reinterpret_cast<const char*>(optixProgramsPtx);
}

extern "C" size_t getOptixPtxSize()
{
    return sizeof(optixProgramsPtx);
}
