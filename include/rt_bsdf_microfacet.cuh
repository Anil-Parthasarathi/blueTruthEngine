#pragma once
// Microfacet BRDF implementation.
// Direct CUDA translation of the provided Nori Microfacet logic/structure.

// p0.xyz = kd
// p1.x   = alpha
// p1.y   = intIOR
// p1.z   = extIOR
// p1.w   = ks (precomputed as 1 - max(kd))

static constexpr float AZIMUTH = 1.0f / (2.0f * RT_PI);

__device__ __forceinline__ Float3 microfacetKd(const BsdfData& b)
{
    return {b.p0.x, b.p0.y, b.p0.z};
}

__device__ __forceinline__ float microfacetAlpha(const BsdfData& b) { return b.p1.x; }
__device__ __forceinline__ float microfacetIntIOR(const BsdfData& b) { return b.p1.y; }
__device__ __forceinline__ float microfacetExtIOR(const BsdfData& b) { return b.p1.z; }
__device__ __forceinline__ float microfacetKs(const BsdfData& b) { return b.p1.w; }

__device__ __forceinline__ float tanThetaMicrofacet(const Float3& w)
{
    const float ct = cosThetaLocal(w);
    const float sin2Theta = fmaxf(0.0f, 1.0f - ct * ct);
    const float cos2Theta = ct * ct;
    if (cos2Theta <= 0.0f) return 0.0f;
    return sqrtf(sin2Theta / cos2Theta);
}

__device__ __forceinline__ float squareToCosineHemispherePdfMicrofacet(const Float3& v)
{
    if (v.z <= 0.0f) return 0.0f;
    return RT_INV_PI * v.z;
}

__device__ __forceinline__ Float3 squareToCosineHemisphereMicrofacet(float u1, float u2)
{
    const float r   = sqrtf(fmaxf(u1, 0.0f));
    const float phi = 2.0f * RT_PI * u2;
    const float x   = r * cosf(phi);
    const float y   = r * sinf(phi);
    const float z   = sqrtf(fmaxf(0.0f, 1.0f - u1));
    return {x, y, z};
}

__device__ __forceinline__ Float3 squareToBeckmann(float u1, float u2, float alpha)
{
    const float phi = 2.0f * RT_PI * u2;
    const float tan2Theta = -alpha * alpha * logf(fmaxf(1.0e-6f, 1.0f - u1));
    const float cosTheta = 1.0f / sqrtf(1.0f + tan2Theta);
    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    return {sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta};
}

__device__ __forceinline__ float squareToBeckmannPdf(const Float3& m, float alpha)
{
    if (m.z <= 0.0f) return 0.0f;
    const float tanTheta = tanThetaMicrofacet(m);
    const float alpha2 = alpha * alpha;
    const float cosTheta = m.z;
    const float cosTheta3 = cosTheta * cosTheta * cosTheta;
    return AZIMUTH * (2.0f * expf(-1.0f * tanTheta * tanTheta / alpha2)) / (alpha2 * cosTheta3);
}

__device__ __forceinline__ float shadowMaskOperator(const BsdfData& bsdf,
                                                     const Float3& wv,
                                                     const Float3& wh)
{
    const float c = dot3(wv, wh) / wv.z;

    if (c <= 0.0f) {
        return 0.0f;
    }

    const float b = 1.0f / (microfacetAlpha(bsdf) * tanThetaMicrofacet(wv));

    if (b < 1.6f) {
        return ((3.535f * b) + 2.181f * b * b) / (1.0f + (2.276f * b) + 2.577f * b * b);
    }
    else {
        return 1.0f;
    }
}

__device__ __forceinline__ Float3 bsdfEvalMicrofacet(const BsdfData& bsdf,
                                                      const BsdfQueryRecord& bRec)
{
    if (bRec.measure != BSDF_ESolidAngle
        || cosThetaLocal(bRec.wi) <= 0.0f
        || cosThetaLocal(bRec.wo) <= 0.0f)
        return {0.0f, 0.0f, 0.0f};

    const Float3 kd = microfacetKd(bsdf);
    const Float3 diffuseTerm = {kd.x * RT_INV_PI, kd.y * RT_INV_PI, kd.z * RT_INV_PI};

    const Float3 wh = normalize3(add3(bRec.wi, bRec.wo));

    const float tanValH = tanThetaMicrofacet(wh);
    const float cosValH = cosThetaLocal(wh);

    const float beckmanLongitudinalNumerator = 2.0f * expf(-1.0f * tanValH * tanValH / (microfacetAlpha(bsdf) * microfacetAlpha(bsdf)));
    const float beckmanLongitudinalDenominator = microfacetAlpha(bsdf) * microfacetAlpha(bsdf) * cosValH * cosValH * cosValH;

    const float microfacetDistribution = AZIMUTH * beckmanLongitudinalNumerator / beckmanLongitudinalDenominator;

    const float fresnelVal = fresnelDielectric(dot3(wh, bRec.wi), microfacetExtIOR(bsdf), microfacetIntIOR(bsdf));

    const float shadowMask = shadowMaskOperator(bsdf, bRec.wi, wh) * shadowMaskOperator(bsdf, bRec.wo, wh);

    const float cosValI = cosThetaLocal(bRec.wi);
    const float cosValO = cosThetaLocal(bRec.wo);

    const float energyConservationFactor = 4.0f * cosValI * cosValH * cosValO;

    const float spec = microfacetKs(bsdf) * (microfacetDistribution * fresnelVal * shadowMask) / energyConservationFactor;
    return {diffuseTerm.x + spec, diffuseTerm.y + spec, diffuseTerm.z + spec};
}

__device__ __forceinline__ float bsdfPdfMicrofacet(const BsdfData& bsdf,
                                                    const BsdfQueryRecord& bRec)
{
    if (bRec.measure != BSDF_ESolidAngle
        || cosThetaLocal(bRec.wi) <= 0.0f
        || cosThetaLocal(bRec.wo) <= 0.0f)
        return 0.0f;

    const float cosineHemispherePdf = squareToCosineHemispherePdfMicrofacet(bRec.wo);

    const Float3 wh = normalize3(add3(bRec.wi, bRec.wo));

    const float beckmannPdf = squareToBeckmannPdf(wh, microfacetAlpha(bsdf));

    const float jacobian = 1.0f / (4.0f * dot3(wh, bRec.wo));

    return microfacetKs(bsdf) * beckmannPdf * jacobian + (1.0f - microfacetKs(bsdf)) * cosineHemispherePdf;
}

__device__ __forceinline__ Float3 bsdfSampleMicrofacet(const BsdfData& bsdf,
                                                        BsdfQueryRecord& bRec,
                                                        float u1, float u2,
                                                        float* outPdf = nullptr)
{
    if (bRec.wi.z <= 0.0f) {
        if (outPdf) *outPdf = 0.0f;
        return {0.0f, 0.0f, 0.0f};
    }

    float sampleX = u1;

    size_t reflectionType = 0;
    if (sampleX < microfacetKs(bsdf)) {
        reflectionType = 0;
        sampleX = sampleX / microfacetKs(bsdf);
    } else {
        reflectionType = 1;
        sampleX = (sampleX - microfacetKs(bsdf)) / (1.0f - microfacetKs(bsdf));
    }

    const float sx = sampleX;
    const float sy = u2;

    bRec.measure = BSDF_ESolidAngle;
    bRec.eta = 1.0f;

    if (reflectionType == 0) {
        const Float3 normalSample = squareToBeckmann(sx, sy, microfacetAlpha(bsdf));
        bRec.wo = sub3(mul3(normalSample, 2.0f * dot3(bRec.wi, normalSample)), bRec.wi);
    }
    else {
        bRec.wo = squareToCosineHemisphereMicrofacet(sx, sy);
    }

    if (bRec.wo.z <= 0.0f) {
        if (outPdf) *outPdf = 0.0f;
        return {0.0f, 0.0f, 0.0f};
    }

    const float p = bsdfPdfMicrofacet(bsdf, bRec);
    if (outPdf) *outPdf = p;
    if (p <= 0.0f) return {0.0f, 0.0f, 0.0f};

    return mul3(bsdfEvalMicrofacet(bsdf, bRec), cosThetaLocal(bRec.wo) / p);
}
