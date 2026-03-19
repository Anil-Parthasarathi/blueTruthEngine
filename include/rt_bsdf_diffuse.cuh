#pragma once
// Diffuse (Lambertian) BSDF implementation.
// Included by rt_bsdf.cuh — shared types (BsdfQueryRecord, RT_PI, etc.)
// are defined there and must already be visible.

__device__ __forceinline__ Float3 bsdfDiffuseAlbedo(const BsdfData& b)
{
    return {b.p0.x, b.p0.y, b.p0.z};
}

__device__ __forceinline__ Float3 squareToCosineHemisphere(float u1, float u2)
{
    const float r   = sqrtf(fmaxf(u1, 0.0f));
    const float phi = 2.0f * RT_PI * u2;
    const float x   = r * cosf(phi);
    const float y   = r * sinf(phi);
    const float z   = sqrtf(fmaxf(0.0f, 1.0f - u1));
    return {x, y, z};
}

// eval: albedo / pi
__device__ __forceinline__ Float3 bsdfEvalDiffuse(const BsdfData& bsdf,
                                                   const BsdfQueryRecord& bRec)
{
    if (bRec.measure != BSDF_ESolidAngle ||
        cosThetaLocal(bRec.wi) <= 0.0f   ||
        cosThetaLocal(bRec.wo) <= 0.0f)
        return {0.0f, 0.0f, 0.0f};

    return mul3(bsdfDiffuseAlbedo(bsdf), RT_INV_PI);
}

// pdf: cos(theta) / pi
__device__ __forceinline__ float bsdfPdfDiffuse(const BsdfQueryRecord& bRec)
{
    if (bRec.measure != BSDF_ESolidAngle ||
        cosThetaLocal(bRec.wi) <= 0.0f   ||
        cosThetaLocal(bRec.wo) <= 0.0f)
        return 0.0f;

    return RT_INV_PI * cosThetaLocal(bRec.wo);
}

// sample: cosine-weighted hemisphere, returns albedo weight
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
    bRec.wo      = squareToCosineHemisphere(u1, u2);
    bRec.eta     = 1.0f;

    if (outPdf) *outPdf = RT_INV_PI * cosThetaLocal(bRec.wo);

    return bsdfDiffuseAlbedo(bsdf);
}
