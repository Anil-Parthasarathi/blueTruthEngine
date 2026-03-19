#pragma once

#include "render_kernel.h"
#include "rt_cuda_math.cuh"

#include <cmath>

// Minimal Nori-like BSDF query plumbing for CUDA kernels.
// All directions here are assumed to be in the *local shading frame*:
// - +Z is the surface normal direction
// - cosTheta(w) == w.z

static constexpr float RT_PI      = 3.14159265358979323846f;
static constexpr float RT_INV_PI  = 1.0f / RT_PI;

enum BsdfMeasure : int {
    BSDF_ESolidAngle = 1,
};

struct BsdfQueryRecord {
    Float3 wi;       // incident direction (local)
    Float3 wo;       // outgoing direction (local) [set by sample()]
    int    measure;  // BsdfMeasure
    float  eta;      // relative IOR (set by sample())
};

__device__ __forceinline__ float cosThetaLocal(const Float3& w) { return w.z; }

__device__ __forceinline__ Float3 make3(float x, float y, float z) { return {x, y, z}; }

__device__ __forceinline__ Float3 hadamard3(const Float3& a, const Float3& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ __forceinline__ Float3 bsdfDiffuseAlbedo(const BsdfData& b)
{
    // Packed in p0.xyz
    return {b.p0.x, b.p0.y, b.p0.z};
}

// Warp::squareToCosineHemisphere
__device__ __forceinline__ Float3 squareToCosineHemisphere(float u1, float u2)
{
    // Concentric mapping is slightly better, but this is already fine and matches Nori’s intent.
    const float r   = sqrtf(fmaxf(u1, 0.0f));
    const float phi = 2.0f * RT_PI * u2;
    const float x = r * cosf(phi);
    const float y = r * sinf(phi);
    const float z = sqrtf(fmaxf(0.0f, 1.0f - u1));
    return {x, y, z};
}

// Diffuse eval (Lambert): albedo / pi
__device__ __forceinline__ Float3 bsdfEvalDiffuse(const BsdfData& bsdf,
                                                  const BsdfQueryRecord& bRec)
{
    if (bRec.measure != BSDF_ESolidAngle ||
        cosThetaLocal(bRec.wi) <= 0.0f ||
        cosThetaLocal(bRec.wo) <= 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }

    const Float3 albedo = bsdfDiffuseAlbedo(bsdf);
    return mul3(albedo, RT_INV_PI);
}

// Diffuse pdf: cos(theta) / pi
__device__ __forceinline__ float bsdfPdfDiffuse(const BsdfQueryRecord& bRec)
{
    if (bRec.measure != BSDF_ESolidAngle ||
        cosThetaLocal(bRec.wi) <= 0.0f ||
        cosThetaLocal(bRec.wo) <= 0.0f) {
        return 0.0f;
    }
    return RT_INV_PI * cosThetaLocal(bRec.wo);
}

// Diffuse sample: cosine hemisphere, returns albedo (like Nori)
__device__ __forceinline__ Float3 bsdfSampleDiffuse(const BsdfData& bsdf,
                                                    BsdfQueryRecord& bRec,
                                                    float u1, float u2,
                                                    float* outPdf = nullptr)
{
    if (cosThetaLocal(bRec.wi) <= 0.0f) {
        if (outPdf) *outPdf = 0.0f;
        return {0.0f, 0.0f, 0.0f};
    }

    bRec.measure = BSDF_ESolidAngle;
    bRec.wo = squareToCosineHemisphere(u1, u2);
    bRec.eta = 1.0f;

    if (outPdf) {
        *outPdf = RT_INV_PI * cosThetaLocal(bRec.wo);
    }

    return bsdfDiffuseAlbedo(bsdf);
}

// Dispatcher helpers (diffuse implemented, dielectric left for you).
__device__ __forceinline__ bool bsdfIsDiffuse(const BsdfData& b)
{
    return b.type == BSDF_Diffuse;
}

__device__ __forceinline__ Float3 bsdfEval(const BsdfData& b, const BsdfQueryRecord& rec)
{
    if (b.type == BSDF_Diffuse) return bsdfEvalDiffuse(b, rec);
    return {0.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ float bsdfPdf(const BsdfData& b, const BsdfQueryRecord& rec)
{
    if (b.type == BSDF_Diffuse) return bsdfPdfDiffuse(rec);
    return 0.0f;
}

