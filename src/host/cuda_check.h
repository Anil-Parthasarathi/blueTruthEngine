#pragma once

// ---------------------------------------------------------------------------
//  cuda_check.h — error-checking helpers shared by all host translation units.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d – %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define OPTIX_CHECK(call)                                                     \
    do {                                                                       \
        OptixResult res = (call);                                              \
        if (res != OPTIX_SUCCESS) {                                            \
            fprintf(stderr, "OptiX error at %s:%d – %s\n",                    \
                    __FILE__, __LINE__, optixGetErrorString(res));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
