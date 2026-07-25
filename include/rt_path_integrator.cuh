#pragma once

// ---------------------------------------------------------------------------
//  rt_path_integrator.cuh — per-bounce path bookkeeping shared by the
//  megakernel and the wavefront pipeline: BSDF scattering, Russian roulette,
//  progressive accumulation, and display encoding.
// ---------------------------------------------------------------------------

#include "render_kernel.h"
#include "rt_cuda_math.cuh"
#include "rt_intersect.cuh"
#include "rt_rng.cuh"
#include "rt_bsdf.cuh"
#include "rt_constants.cuh"
#include "rt_shading_context.cuh"

#include <cstdint>

// ---------------------------------------------------------------------------
//  Sample the BSDF for the next bounce and advance the path: sets the new ray
//  (origin offset by RT_EPSILON along the bounce direction), multiplies the
//  throughput by the sample weight, and records brdfPDF for the next bounce's
//  emitter-hit MIS.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void scatterPath(
    PathState& pathRecord,
    RngState& rng,
    const Intersection& its,
    const BsdfData& bsdf)
{
    BsdfQueryRecord bsdfQueryIndirect{};
    bsdfQueryIndirect.wi       = toLocalFromNormal(its.hitNormal, mul3(pathRecord.ray.direction, -1.0f));
    bsdfQueryIndirect.rndExtra = rngNextFloat01(rng);

    const float u0 = rngNextFloat01(rng);
    const float u1 = rngNextFloat01(rng);
    Float3 sampleWeight = bsdfSample(bsdf, bsdfQueryIndirect, u0, u1);

    pathRecord.brdfPDF = bsdfPdf(bsdf, bsdfQueryIndirect);

    const Float3 bounceDir = normalize3(toWorldFromNormal(its.hitNormal, bsdfQueryIndirect.wo));

    pathRecord.ray.origin    = add3(its.hitPoint, mul3(bounceDir, RT_EPSILON));
    pathRecord.ray.direction = bounceDir;
    pathRecord.throughput    = mul3(pathRecord.throughput, sampleWeight);
    pathRecord.eta          *= bsdfQueryIndirect.eta;
}

// ---------------------------------------------------------------------------
//  Russian roulette (only after bounce 3).  Returns true if the path survives;
//  survivors have their throughput divided by the continuation probability to
//  stay unbiased.  Returns false when the path should terminate.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool russianRoulette(PathState& pathRecord, RngState& rng)
{
    if (pathRecord.bounceCount > 3) {
        const float continuationProb =
            fminf(maxCoeff3(pathRecord.throughput) * pathRecord.eta * pathRecord.eta, 0.99f);
        if (rngNextFloat01(rng) >= continuationProb) {
            return false;
        }
        pathRecord.throughput = div3(pathRecord.throughput, continuationProb);
    }
    return true;
}

// ---------------------------------------------------------------------------
//  Welford streaming average into the linear accumulation buffer:
//      new_avg = old_avg + (sample - old_avg) / n,   n = frameIndex + 1
//  Returns the blended running mean (useful for the display encode below).
// ---------------------------------------------------------------------------
__device__ __forceinline__ Float3 accumulateWelford(
    Float3* accumBuffer,
    uint32_t pixelIndex,
    Float3 sampleColor,
    uint32_t frameIndex)
{
    const float n = static_cast<float>(frameIndex + 1);
    Float3 prev = accumBuffer[pixelIndex];
    Float3 blended = {
        prev.x + (sampleColor.x - prev.x) / n,
        prev.y + (sampleColor.y - prev.y) / n,
        prev.z + (sampleColor.z - prev.z) / n
    };
    accumBuffer[pixelIndex] = blended;
    return blended;
}

// ---------------------------------------------------------------------------
//  Gamma correction (linear → sRGB, γ = 2.2) and RGBA8 packing
//  (ABGR byte order for GL_UNSIGNED_BYTE / RGBA).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t packPixelGamma(Float3 linear)
{
    float dr = powf(fmaxf(0.0f, fminf(linear.x, 1.0f)), 1.0f / 2.2f);
    float dg = powf(fmaxf(0.0f, fminf(linear.y, 1.0f)), 1.0f / 2.2f);
    float db = powf(fmaxf(0.0f, fminf(linear.z, 1.0f)), 1.0f / 2.2f);

    uint8_t r = static_cast<uint8_t>(dr * 255.0f);
    uint8_t g = static_cast<uint8_t>(dg * 255.0f);
    uint8_t b = static_cast<uint8_t>(db * 255.0f);
    uint8_t a = 255;

    return (a << 24) | (b << 16) | (g << 8) | r;
}
