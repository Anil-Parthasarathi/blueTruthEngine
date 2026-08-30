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

// After an async kernel / optixLaunch: sync, then surface the sticky error
// with a stage name.  OptiX faults often do not fail cudaDeviceSynchronize
// itself and only appear at the next CUDA kernel — which is why they were
// showing up at wfAccumulate.
#define CUDA_STAGE_CHECK(stage)                                               \
    do {                                                                       \
        cudaError_t syncErr = cudaDeviceSynchronize();                         \
        if (syncErr != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error after %s at %s:%d – %s\n",            \
                    (stage), __FILE__, __LINE__, cudaGetErrorString(syncErr)); \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
        cudaError_t lastErr = cudaGetLastError();                              \
        if (lastErr != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error after %s at %s:%d – %s\n",            \
                    (stage), __FILE__, __LINE__, cudaGetErrorString(lastErr)); \
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
