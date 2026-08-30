#pragma once
// Main BSDF header — include this everywhere.
// Defines shared types/constants, then pulls in per-material implementations.

#include "render_kernel.h"
#include "rt_cuda_math.cuh"
#include "rt_style.cuh"      // StyleChannel — for the lobe → channel mapping

#include <cmath>

// ── Shared constants ─────────────────────────────────────────────────
static constexpr float RT_PI     = 3.14159265358979323846f;
static constexpr float RT_INV_PI = 1.0f / RT_PI;

// ── Shared types ─────────────────────────────────────────────────────
enum BsdfMeasure : int {
    BSDF_ESolidAngle = 1,
};

struct BsdfQueryRecord {
    Float3 wi;       // incident direction  (local frame)
    Float3 wo;       // outgoing direction  (local frame, set by sample())
    int    measure;  // BsdfMeasure
    float  eta;      // relative IOR        (set by sample())
    float  rndExtra; // extra 1D sample for lobe selection (set by integrator)
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
#include "rt_bsdf_mirror.cuh"
#include "rt_bsdf_microfacet.cuh"
#include "rt_bsdf_disney.cuh"

// ── Dispatchers ───────────────────────────────────────────────────────
__device__ __forceinline__ Float3 bsdfEval(const BsdfData& b, const BsdfQueryRecord& rec)
{
    if (b.type == BSDF_Diffuse)    return bsdfEvalDiffuse(b, rec);
    if (b.type == BSDF_Dielectric) return bsdfEvalDielectric(b, rec);
    if (b.type == BSDF_Mirror)     return bsdfEvalMirror(b, rec);
    if (b.type == BSDF_Microfacet) return bsdfEvalMicrofacet(b, rec);
    if (b.type == BSDF_Disney)     return bsdfEvalDisney(b, rec);
    return {0.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ float bsdfPdf(const BsdfData& b, const BsdfQueryRecord& rec)
{
    if (b.type == BSDF_Diffuse)    return bsdfPdfDiffuse(rec);
    if (b.type == BSDF_Dielectric) return bsdfPdfDielectric(rec);
    if (b.type == BSDF_Mirror)     return bsdfPdfMirror(rec);
    if (b.type == BSDF_Microfacet) return bsdfPdfMicrofacet(b, rec);
    if (b.type == BSDF_Disney)     return bsdfPdfDisney(b, rec);
    return 0.0f;
}

// Sample the BSDF.  `outLobe`, when non-null, reports which DisneyLobe the
// sample came from so the integrator can route the rest of the path into the
// matching style channel.  Every BSDF type maps onto the same four-lobe enum:
// Mirror is a metal reflection, Dielectric is glass, Diffuse is diffuse, and
// Microfacet reports metal or diffuse depending on the sub-lobe it picked.
__device__ __forceinline__ Float3 bsdfSample(const BsdfData& b, BsdfQueryRecord& rec,
                                              float u1, float u2, float* outPdf = nullptr,
                                              int* outLobe = nullptr)
{
    if (b.type == BSDF_Diffuse) {
        if (outLobe) *outLobe = DISNEY_LOBE_DIFFUSE;
        return bsdfSampleDiffuse(b, rec, u1, u2, outPdf);
    }
    if (b.type == BSDF_Dielectric) {
        if (outLobe) *outLobe = DISNEY_LOBE_GLASS;
        return bsdfSampleDielectric(b, rec, u1, u2, outPdf);
    }
    if (b.type == BSDF_Mirror) {
        if (outLobe) *outLobe = DISNEY_LOBE_METAL;
        return bsdfSampleMirror(b, rec, u1, u2, outPdf);
    }
    if (b.type == BSDF_Microfacet) {
        // bsdfSampleMicrofacet splits u1 against ks: below ks it takes the
        // Beckmann specular lobe, above it the cosine diffuse lobe.
        if (outLobe) *outLobe = (u1 < microfacetKs(b)) ? DISNEY_LOBE_METAL : DISNEY_LOBE_DIFFUSE;
        return bsdfSampleMicrofacet(b, rec, u1, u2, outPdf);
    }
    if (b.type == BSDF_Disney)     return bsdfSampleDisney(b, rec, u1, u2, outPdf, outLobe);
    if (outPdf) *outPdf = 0.0f;
    if (outLobe) *outLobe = DISNEY_LOBE_DIFFUSE;
    return {0.0f, 0.0f, 0.0f};
}

// Split the BSDF evaluation into the halves the direct-lighting channels want.
// Only the Disney BSDF has a real decomposition; for the others the whole
// response goes to whichever half matches its character, so a scene mixing
// material types still routes sensibly.
__device__ __forceinline__ void bsdfEvalSplit(const BsdfData& b, const BsdfQueryRecord& rec,
                                               Float3& outDiffuseish, Float3& outSpecularish)
{
    if (b.type == BSDF_Disney) {
        bsdfEvalDisneySplit(b, rec, outDiffuseish, outSpecularish);
        return;
    }

    const Float3 total = bsdfEval(b, rec);

    if (b.type == BSDF_Microfacet) {
        // Microfacet's eval is kd/pi plus a scalar specular term; approximate the
        // split by the same ks weighting used for sampling.
        const float ks = microfacetKs(b);
        outSpecularish = mul3(total, ks);
        outDiffuseish  = mul3(total, 1.0f - ks);
        return;
    }

    // Mirror and Dielectric are delta BSDFs whose eval is zero, so this reduces
    // to zero for both halves; Diffuse is entirely diffuse-ish.
    outDiffuseish  = total;
    outSpecularish = {0.0f, 0.0f, 0.0f};
}

// Map a sampled lobe onto the radiance channel the path should feed from here on.
__device__ __forceinline__ int styleChannelForLobe(int lobe)
{
    if (lobe == DISNEY_LOBE_GLASS)                                   return STYLE_CH_TRANSMITTED;
    if (lobe == DISNEY_LOBE_METAL || lobe == DISNEY_LOBE_CLEARCOAT)  return STYLE_CH_REFLECTED;
    return STYLE_CH_INDIRECT_DIFFUSE;
}

// True for delta BSDFs (dirac reflection/refraction) — skip direct light sampling for these.
__device__ __forceinline__ bool bsdfIsDelta(const BsdfData& b)
{
    return b.type == BSDF_Dielectric || b.type == BSDF_Mirror;
}

// Nori-like: non-delta BSDFs that participate in NEE / MIS (diffuse + microfacet).
__device__ __forceinline__ bool bsdfIsDiffuse(const BsdfData& b)
{
    return b.type == BSDF_Diffuse || b.type == BSDF_Microfacet || b.type == BSDF_Disney;
}
