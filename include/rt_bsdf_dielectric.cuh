#pragma once
// Ideal dielectric (glass) BSDF — direct CUDA port of Nori's Dielectric class.
// Included by rt_bsdf.cuh — shared types must already be visible.
//
// IOR packing in BsdfData:
//   intIOR = b.p1.x   (interior / transmitted side)
//   extIOR = b.p1.y   (exterior / incident side, usually 1.0 for air)

__device__ __forceinline__ float dielectricIntIOR(const BsdfData& b) { return b.p1.x; }
__device__ __forceinline__ float dielectricExtIOR(const BsdfData& b) { return b.p1.y; }

// fresnel(cosThetaI, extIOR, intIOR) — exact port of Nori's fresnel()
__device__ __forceinline__ float fresnelDielectric(float cosThetaI, float etaI, float etaT)
{
    if (cosThetaI < 0.0f) {
        float tmp = etaI; etaI = etaT; etaT = tmp;
        cosThetaI = -cosThetaI;
    }

    float sinThetaT = (etaI / etaT) * sqrtf(fmaxf(0.0f, 1.0f - cosThetaI * cosThetaI));

    if (sinThetaT >= 1.0f)
        return 1.0f; // total internal reflection

    float cosThetaT = sqrtf(fmaxf(0.0f, 1.0f - sinThetaT * sinThetaT));

    float Rs = (etaI * cosThetaI - etaT * cosThetaT) / (etaI * cosThetaI + etaT * cosThetaT);
    float Rp = (etaT * cosThetaI - etaI * cosThetaT) / (etaT * cosThetaI + etaI * cosThetaT);

    return (Rs * Rs + Rp * Rp) * 0.5f;
}

__device__ __forceinline__ Float3 reflectLocal(const Float3& wi)
{
    return {-wi.x, -wi.y, wi.z};
}

// eval: delta BSDF — always zero
__device__ __forceinline__ Float3 bsdfEvalDielectric(const BsdfData& /*bsdf*/,
                                                      const BsdfQueryRecord& /*bRec*/)
{
    return {0.0f, 0.0f, 0.0f};
}

// pdf: delta BSDF — always zero
__device__ __forceinline__ float bsdfPdfDielectric(const BsdfQueryRecord& /*bRec*/)
{
    return 0.0f;
}

__device__ __forceinline__ Float3 bsdfSampleDielectric(const BsdfData& bsdf,
                                                        BsdfQueryRecord& bRec,
                                                        float u1, float /*u2*/,
                                                        float* outPdf = nullptr)
{
    bRec.measure = BSDF_ESolidAngle;
    if (outPdf) *outPdf = 1.0f;

    Float3 dielectricResult = {1.0f, 1.0f, 1.0f};  // Color3f(1.0f)

    float refRafSpark = u1;                          // sample.x()
    float cosThetaI   = cosThetaLocal(bRec.wi);      // bRec.wi.z()

    float fresnelVal = fresnelDielectric(cosThetaI,
                                         dielectricExtIOR(bsdf),
                                         dielectricIntIOR(bsdf));

    if (refRafSpark < fresnelVal) {
        // reflection event
        bRec.wo  = reflectLocal(bRec.wi);
        bRec.eta = 1.0f;
    } else {
        // refraction event
        float  indexRatio;
        Float3 normal = {0.0f, 0.0f, 1.0f};

        if (cosThetaI > 0) {
            indexRatio = dielectricExtIOR(bsdf) / dielectricIntIOR(bsdf);
        } else {
            indexRatio = dielectricIntIOR(bsdf) / dielectricExtIOR(bsdf);
            normal     = {0.0f, 0.0f, -1.0f};   // normal = -normal
        }

        cosThetaI = fabsf(cosThetaI);            // abs(cosThetaI)

        float rootTerm = 1.0f - (indexRatio * indexRatio) * (1.0f - cosThetaI * cosThetaI);

        // bRec.wo = -indexRatio * bRec.wi + (indexRatio * cosThetaI - sqrt(rootTerm)) * normal
        float diaScale = indexRatio * cosThetaI - sqrtf(rootTerm);
        bRec.wo = {
            -indexRatio * bRec.wi.x + diaScale * normal.x,
            -indexRatio * bRec.wi.y + diaScale * normal.y,
            -indexRatio * bRec.wi.z + diaScale * normal.z
        };
        bRec.eta = 1.0f / indexRatio;

        // dielectricResult = Color3f(indexRatio * indexRatio)
        float w = indexRatio * indexRatio;
        dielectricResult = {w, w, w};
    }

    return dielectricResult;
}
