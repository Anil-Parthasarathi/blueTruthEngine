#pragma once
// Perfect specular mirror BSDF.
// Included by rt_bsdf.cuh — shared types must already be visible.
//
// p0.xyz = reflectance tint (default white {1,1,1}).

__device__ __forceinline__ Float3 mirrorReflectance(const BsdfData& b)
{
    return {b.p0.x, b.p0.y, b.p0.z};
}

__device__ __forceinline__ Float3 bsdfEvalMirror(const BsdfData& /*b*/,
                                                  const BsdfQueryRecord& /*rec*/)
{
    return {0.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ float bsdfPdfMirror(const BsdfQueryRecord& /*rec*/)
{
    return 0.0f;
}

__device__ __forceinline__ Float3 bsdfSampleMirror(const BsdfData& b,
                                                    BsdfQueryRecord& rec,
                                                    float /*u1*/, float /*u2*/,
                                                    float* outPdf = nullptr)
{
    rec.wo      = {-rec.wi.x, -rec.wi.y, rec.wi.z};
    rec.eta     = 1.0f;
    rec.measure = BSDF_ESolidAngle;
    if (outPdf) *outPdf = 1.0f;
    return mirrorReflectance(b);
}
