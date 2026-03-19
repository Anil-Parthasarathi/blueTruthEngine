#pragma once

#include <cstdint>

// Small, fast RNG utilities for CUDA kernels.
// Intended for path tracing sampling (next1D/next2D style).

__device__ __forceinline__ uint32_t hashUint(uint32_t x)
{
    // PCG-style integer hash (good diffusion, cheap).
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

struct RngState {
    uint32_t state;
};

__device__ __forceinline__ RngState makeRng(uint32_t seed)
{
    RngState r{};
    r.state = seed ? seed : 1u;
    return r;
}

__device__ __forceinline__ uint32_t rngNextU32(RngState& r)
{
    // xorshift32 (very small state, very fast)
    uint32_t x = r.state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    r.state = x;
    return x;
}

__device__ __forceinline__ float rngNextFloat01(RngState& r)
{
    // 24-bit mantissa uniform in [0,1).
    const uint32_t u = rngNextU32(r) >> 8;
    return static_cast<float>(u) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ void rngNextFloat2(RngState& r, float& u0, float& u1)
{
    u0 = rngNextFloat01(r);
    u1 = rngNextFloat01(r);
}

