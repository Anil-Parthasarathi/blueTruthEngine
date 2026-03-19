#pragma once
// Main BSDF header — include this everywhere.
// Defines shared types/constants, then pulls in per-material implementations.

#include "render_kernel.h"
#include "rt_cuda_math.cuh"

#include <cmath>

// ── Shared constants ─────────────────────────────────────────────────
static constexpr float RT_PI     = 3.14159265358979323846f;
static constexpr float RT_INV_PI = 1.0f / RT_PI;

// ── Shared types ─────────────────────────────────────────────────────
enum BsdfMeasure : int {
    BSDF_ESolidAngle = 1,
};

struct BsdfQueryRecord {
    Float3 wi;      // incident direction  (local frame)
    Float3 wo;      // outgoing direction  (local frame, set by sample())
    int    measure; // BsdfMeasure
    float  eta;     // relative IOR        (set by sample())
};

__device__ __forceinline__ float cosThetaLocal(const Float3& w) { return w.z; }
__device__ __forceinline__ Float3 make3(float x, float y, float z) { return {x, y, z}; }
__device__ __forceinline__ Float3 hadamard3(const Float3& a, const Float3& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

// ── Per-material implementations ─────────────────────────────────────
#include "rt_bsdf_diffuse.cuh"
#include "rt_bsdf_dielectric.cuh"

// ── Dispatchers ───────────────────────────────────────────────────────
__device__ __forceinline__ Float3 bsdfEval(const BsdfData& b, const BsdfQueryRecord& rec)
{
    if (b.type == BSDF_Diffuse)    return bsdfEvalDiffuse(b, rec);
    if (b.type == BSDF_Dielectric) return bsdfEvalDielectric(b, rec);
    return {0.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ float bsdfPdf(const BsdfData& b, const BsdfQueryRecord& rec)
{
    if (b.type == BSDF_Diffuse)    return bsdfPdfDiffuse(rec);
    if (b.type == BSDF_Dielectric) return bsdfPdfDielectric(rec);
    return 0.0f;
}

__device__ __forceinline__ Float3 bsdfSample(const BsdfData& b, BsdfQueryRecord& rec,
                                              float u1, float u2, float* outPdf = nullptr)
{
    if (b.type == BSDF_Diffuse)    return bsdfSampleDiffuse(b, rec, u1, u2, outPdf);
    if (b.type == BSDF_Dielectric) return bsdfSampleDielectric(b, rec, u1, u2, outPdf);
    if (outPdf) *outPdf = 0.0f;
    return {0.0f, 0.0f, 0.0f};
}

// True for delta BSDFs (dirac reflection/refraction) — skip direct light sampling for these.
__device__ __forceinline__ bool bsdfIsDelta(const BsdfData& b)
{
    return b.type == BSDF_Dielectric;
}
