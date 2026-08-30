#pragma once
// Disney principled BSDF — direct CUDA port of the Nori Disney class.
// Included by rt_bsdf.cuh — shared types (BsdfQueryRecord, RT_PI, etc.)
// are defined there and must already be visible.
//
// Parameter packing in BsdfData:
//   p0.xyz = base_color           p0.w = roughness
//   p1.x   = metallic             p1.y = specular
//   p1.z   = specular_transmission p1.w = specular_tint
//   p2.x   = sheen                p2.y = sheen_tint
//   p2.z   = subsurface           p2.w = anisotropic
//   p3.x   = clearcoat            p3.y = clearcoat_gloss
//   p3.z   = eta

// ── Parameter accessors ──────────────────────────────────────────────
__device__ __forceinline__ Float3 disneyBaseColor(const BsdfData& b) { return {b.p0.x, b.p0.y, b.p0.z}; }
__device__ __forceinline__ float disneyRoughness(const BsdfData& b)             { return b.p0.w; }
__device__ __forceinline__ float disneyMetallic(const BsdfData& b)              { return b.p1.x; }
__device__ __forceinline__ float disneySpecular(const BsdfData& b)              { return b.p1.y; }
__device__ __forceinline__ float disneySpecularTransmission(const BsdfData& b)  { return b.p1.z; }
__device__ __forceinline__ float disneySpecularTint(const BsdfData& b)          { return b.p1.w; }
__device__ __forceinline__ float disneySheen(const BsdfData& b)                 { return b.p2.x; }
__device__ __forceinline__ float disneySheenTint(const BsdfData& b)             { return b.p2.y; }
__device__ __forceinline__ float disneySubsurface(const BsdfData& b)            { return b.p2.z; }
__device__ __forceinline__ float disneyAnisotropic(const BsdfData& b)           { return b.p2.w; }
__device__ __forceinline__ float disneyClearcoat(const BsdfData& b)             { return b.p3.x; }
__device__ __forceinline__ float disneyClearcoatGloss(const BsdfData& b)        { return b.p3.y; }
__device__ __forceinline__ float disneyEta(const BsdfData& b)                   { return b.p3.z; }

// Component-wise sqrt of a color (used by the glass lobe).
__device__ __forceinline__ Float3 disneySqrt3(const Float3& c)
{
    return {sqrtf(fmaxf(0.0f, c.x)), sqrtf(fmaxf(0.0f, c.y)), sqrtf(fmaxf(0.0f, c.z))};
}

// Forward declarations of the top-level eval/pdf (lobe sample helpers use them).
__device__ __forceinline__ Float3 bsdfEvalDisney(const BsdfData& bsdf, const BsdfQueryRecord& bRec);
__device__ __forceinline__ float  bsdfPdfDisney(const BsdfData& bsdf, const BsdfQueryRecord& bRec);

// ############DIFFUSE###########################################################

__device__ __forceinline__ float diffuseFresnelNinety(const BsdfData& bsdf, float hDotWo) {
    return 0.5f + (2.0f * disneyRoughness(bsdf) * hDotWo * hDotWo);
}

__device__ __forceinline__ float diffuseFresnel(float cosTheta, float dfNinety) {
    return 1.0f + ((dfNinety - 1.0f) * powf(1.0f - cosTheta, 5));
}

__device__ __forceinline__ float subsurfaceFresnelNinety(const BsdfData& bsdf, float hDotWo) {
    return disneyRoughness(bsdf) * hDotWo * hDotWo;
}

__device__ __forceinline__ float subsurfaceFresnel(float cosTheta, float ssNinety) {
    return 1.0f + ((ssNinety - 1.0f) * powf(1.0f - cosTheta, 5));
}

/// Evaluate the BRDF for the given pair of directions
__device__ __forceinline__ Float3 evalDiffuseLobe(const BsdfData& bsdf, const BsdfQueryRecord& bRec) {

    // originally I had this as absolute values, but that resulted in a big issue where
    // spec transmission above 0 led to diffuse lobe blowing up into fireflies everywhere
    // before that, i had forgotten to have the guard here at all
    // both had the same issue I think, it didn't stop the invalid rays
    float cosThetaIn = cosThetaLocal(bRec.wi);
    float cosThetaOut = cosThetaLocal(bRec.wo);

    // check to make sure rays arent underneath
    if (cosThetaIn <= 0.0f || cosThetaOut <= 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }

    Float3 wh = normalize3(add3(bRec.wi, bRec.wo));

    float hDotWo = fabsf(dot3(wh, bRec.wo));

    float dfNinety = diffuseFresnelNinety(bsdf, hDotWo);
    float ssNinety = subsurfaceFresnelNinety(bsdf, hDotWo);

    Float3 colorOverPi = div3(disneyBaseColor(bsdf), RT_PI);

    Float3 baseDiffuse =
        mul3(colorOverPi,
            diffuseFresnel(cosThetaIn, dfNinety) *
            diffuseFresnel(cosThetaOut, dfNinety) *
            cosThetaOut);

    // Clamp the denominator to help prevent fireflies
    float ssDenom = fmaxf(1e-2f, cosThetaIn + cosThetaOut);

    Float3 subsurface =
        mul3(colorOverPi,
            1.25f *
            (
                subsurfaceFresnel(cosThetaIn, ssNinety) *
                subsurfaceFresnel(cosThetaOut, ssNinety) *
                ((1.0f / ssDenom) - 0.5f) + 0.5f
            ) *
            cosThetaOut);

    // Return the weighted sum of diffuse and subsurface components based on the parameter for subsurface
    return add3(mul3(baseDiffuse, 1.0f - disneySubsurface(bsdf)),
                mul3(subsurface, disneySubsurface(bsdf)));
}

/// Evaluate the sampling density of \ref sample() wrt. solid angles
__device__ __forceinline__ float pdfDiffuseLobe(const BsdfQueryRecord& bRec) {

    // standard cosine hemisphere sampling similar to diffuse.cpp

    if (bRec.measure != BSDF_ESolidAngle
        || cosThetaLocal(bRec.wi) <= 0
        || cosThetaLocal(bRec.wo) <= 0)
        return 0.0f;


    /* Importance sampling density wrt. solid angles:
       cos(theta) / pi.

       Note that the directions in 'bRec' are in local coordinates,
       so Frame::cosTheta() actually just returns the 'z' component.
    */
    return RT_INV_PI * cosThetaLocal(bRec.wo);
}

/// Sample the BRDF
__device__ __forceinline__ Float3 sampleDiffuseLobe(const BsdfData& bsdf, BsdfQueryRecord& bRec, float u1, float u2) {

    // Standard cosine hemisphere sampling, basing on diffuse.cpp

    if (cosThetaLocal(bRec.wi) <= 0)
        return {0.0f, 0.0f, 0.0f};

    bRec.measure = BSDF_ESolidAngle;

    /* Warp a uniformly distributed sample on [0,1]^2
       to a direction on a cosine-weighted hemisphere */
    bRec.wo = squareToCosineHemisphere(u1, u2);

    /* Relative index of refraction: no change */
    bRec.eta = 1.0f;

    /* eval() / pdf() * cos(theta) = albedo. There
       is no need to call these functions. */
    return div3(evalDiffuseLobe(bsdf, bRec), pdfDiffuseLobe(bRec));
}

// ############METAL###########################################################

__device__ __forceinline__ float schlickFresnel(float eta) {
    float numerator = eta - 1.0f;
    float denominator = eta + 1.0f;

    return (numerator * numerator) / (denominator * denominator);
}

__device__ __forceinline__ Float3 fresnelMetal(const BsdfData& bsdf, float hDotWo) {

    Float3 baseColor = disneyBaseColor(bsdf);

    float lumi = 0.3f * baseColor.x + 0.6f * baseColor.y + 0.1f * baseColor.z;

    Float3 tint = {1.0f, 1.0f, 1.0f};

    if (lumi > 0){
        tint = div3(baseColor, lumi);
    }

    float specTint = disneySpecularTint(bsdf);
    Float3 spec = {(1.0f - specTint) + (specTint * tint.x),
                   (1.0f - specTint) + (specTint * tint.y),
                   (1.0f - specTint) + (specTint * tint.z)};

    float coef = disneySpecular(bsdf) * schlickFresnel(disneyEta(bsdf)) * (1.0f - disneyMetallic(bsdf));
    Float3 clr = {(coef * spec.x) + (baseColor.x * disneyMetallic(bsdf)),
                  (coef * spec.y) + (baseColor.y * disneyMetallic(bsdf)),
                  (coef * spec.z) + (baseColor.z * disneyMetallic(bsdf))};

    float schlickTerm = powf(1.0f - hDotWo, 5);
    return {clr.x + ((1.0f - clr.x) * schlickTerm),
            clr.y + ((1.0f - clr.y) * schlickTerm),
            clr.z + ((1.0f - clr.z) * schlickTerm)};

}

__device__ __forceinline__ float groundGlassXDistribution(Float3 wh, float ax, float ay) {

    float denominator = RT_PI * ax * ay *
        powf(((wh.x * wh.x) / (ax * ax)) + ((wh.y * wh.y) / (ay * ay)) + (wh.z * wh.z), 2);

    return 1.0f / denominator;
}

__device__ __forceinline__ float lambdaFunc(Float3 w, float ax, float ay) {

    float axw = w.x * ax;
    float ayw = w.y * ay;

    float numerator = sqrtf(1 + (((axw * axw) + (ayw * ayw)) / fmaxf(1e-8f, (w.z * w.z)))) - 1.0f;

    float lambda =  numerator / 2.0f;

    return 1.0f / (1.0f + lambda);
}

__device__ __forceinline__ float shadowMaskOperatorDisney(const Float3& wi, const Float3& wo, float ax, float ay) {

    return lambdaFunc(wi, ax, ay) * lambdaFunc(wo, ax, ay);

}

/// Evaluate the BRDF for the given pair of directions
__device__ __forceinline__ Float3 evalMetalLobe(const BsdfData& bsdf, const BsdfQueryRecord& bRec) {

    float cosThetaIn = cosThetaLocal(bRec.wi);

    // check to make sure rays arent underneath
    if (cosThetaIn <= 0.0f || cosThetaLocal(bRec.wo) <= 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }

    Float3 wh = normalize3(add3(bRec.wi, bRec.wo));

    float hDotWo = fabsf(dot3(wh, bRec.wo));

    float aspect = sqrtf(1.0f - (0.9f * disneyAnisotropic(bsdf)));

    // Clamps the alpha to a minimum above 0 to avoid perfect mirrors
    // (this way all materials are solid angle distributions)
    float ax = fmaxf(0.0001, (disneyRoughness(bsdf) * disneyRoughness(bsdf)) / aspect);
    float ay = fmaxf(0.0001, (disneyRoughness(bsdf) * disneyRoughness(bsdf)) * aspect);

    Float3 fm = fresnelMetal(bsdf, hDotWo);

    float dm = groundGlassXDistribution(wh, ax, ay);

    float gm = shadowMaskOperatorDisney(bRec.wi, bRec.wo, ax, ay);

    return div3(mul3(fm, dm * gm), (4 * cosThetaIn));

}

/// Evaluate the sampling density of \ref sample() wrt. solid angles
__device__ __forceinline__ float pdfMetalLobe(const BsdfData& bsdf, const BsdfQueryRecord& bRec) {

    float cosThetaIn = cosThetaLocal(bRec.wi);

    // check to make sure rays arent underneath
    if (cosThetaIn <= 0.0f || cosThetaLocal(bRec.wo) <= 0.0f) {
        return 0.0f;
    }

    Float3 wh = normalize3(add3(bRec.wi, bRec.wo));

    float aspect = sqrtf(1.0f - (0.9f * disneyAnisotropic(bsdf)));

    // Clamps the alpha to a minimum above 0 to avoid perfect mirrors
    // (this way all materials are solid angle distributions)
    float ax = fmaxf(0.0001, (disneyRoughness(bsdf) * disneyRoughness(bsdf)) / aspect);
    float ay = fmaxf(0.0001, (disneyRoughness(bsdf) * disneyRoughness(bsdf)) * aspect);

    float dm = groundGlassXDistribution(wh, ax, ay);

    float gm = lambdaFunc(bRec.wi, ax, ay);

    float hDotWi = fabsf(dot3(wh, bRec.wi));
    float hDotWo = fabsf(dot3(wh, bRec.wo));

    return (dm * gm * hDotWi) / (cosThetaIn * 4.0f * hDotWo);

}

/// Sample the BRDF
__device__ __forceinline__ Float3 sampleMetalLobe(const BsdfData& bsdf, BsdfQueryRecord& bRec, float u1, float u2) {

    // check to make sure rays arent underneath
    if (cosThetaLocal(bRec.wi) <= 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }

    bRec.measure = BSDF_ESolidAngle;
    bRec.eta = 1.0f;

    float aspect = sqrtf(1.0f - (0.9f * disneyAnisotropic(bsdf)));

    // Clamps the alpha to a minimum above 0 to avoid perfect mirrors
    // (this way all materials are solid angle distributions)
    float ax = fmaxf(0.0001, (disneyRoughness(bsdf) * disneyRoughness(bsdf)) / aspect);
    float ay = fmaxf(0.0001, (disneyRoughness(bsdf) * disneyRoughness(bsdf)) * aspect);

    // Sample a new outgooing direction with heitz

    Float3 vh = normalize3(Float3{ax * bRec.wi.x, ay * bRec.wi.y, bRec.wi.z});

    float pl = (vh.x * vh.x) + (vh.y * vh.y);

    Float3 orthOne = {1.0f, 0.0f, 0.0f};

    if (pl > 0.0f){
        orthOne = div3(Float3{-1.0f * vh.y, vh.x, 0.0f}, sqrtf(pl));
    }

    Float3 orthTwo = cross3(vh, orthOne);

    float radius = sqrtf(u1);
    float phi = 2.0f * RT_PI * u2;

    float t1 = radius * cosf(phi);
    float t2 = radius * sinf(phi);

    float skew = 0.5f * (1.0f + vh.z);
    t2 = (1.0f - skew) * sqrtf(1.0f - (t1 * t1)) + (skew * t2);

    Float3 ph = add3(add3(mul3(orthOne, t1), mul3(orthTwo, t2)),
                     mul3(vh, sqrtf(fmaxf(0.0f, 1 - (t1 * t1) - (t2 * t2)))));

    Float3 randH = normalize3(Float3{ax * ph.x, ay * ph.y, fmaxf(0.0f, ph.z)});

    bRec.wo = sub3(mul3(randH, 2.0f * dot3(bRec.wi, randH)), bRec.wi);

    // check to make sure rays arent underneath
    if (cosThetaLocal(bRec.wo) <= 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }

    // eval and pdf math cancel out to make the rest simple

    float hDotWo = fabsf(dot3(randH, bRec.wo));

    Float3 fm = fresnelMetal(bsdf, hDotWo);

    float gm = lambdaFunc(bRec.wo, ax, ay);

    return mul3(fm, gm);
}

// ############CLEARCOAT###########################################################

__device__ __forceinline__ float fresnelClearcoat(float hDotWo) {

    // note: disney hard codes the ior to 1.5 for everything
    // this is not technically physically accurate but apparently its good enough

    // This is what math would have occured, but I realized its always 0.04
    //float r = pow((1.5f - 1.0f), 2) / pow(1.5 + 1.0f, 2);
    float r = 0.04;

    return r + ((1.0f - r) * powf(1.0f - hDotWo, 5));
}

__device__ __forceinline__ float distributionClearcoat(const BsdfData& bsdf, Float3 wh) {

    float isotropicRoughness = ((1.0f - disneyClearcoatGloss(bsdf)) * 0.1f) + (disneyClearcoatGloss(bsdf) * 0.001f);

    float irSquared = isotropicRoughness * isotropicRoughness;

    float numerator = irSquared - 1.0f;

    float denominator = RT_PI * logf(irSquared) * (1.0f + ((irSquared - 1.0f) * (wh.z * wh.z)));

    return numerator / denominator;

}

__device__ __forceinline__ float shadowMaskOperatorClearcoat(const Float3& wi, const Float3& wo) {
    return lambdaFunc(wi, 0.25, 0.25) * lambdaFunc(wo, 0.25, 0.25);
}

/// Evaluate the BRDF for the given pair of directions
__device__ __forceinline__ Float3 evalClearcoatLobe(const BsdfData& bsdf, const BsdfQueryRecord& bRec) {

    float cosThetaIn = cosThetaLocal(bRec.wi);

    // check to make sure rays arent underneath
    if (cosThetaIn <= 0.0f || cosThetaLocal(bRec.wo) <= 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }

    Float3 wh = normalize3(add3(bRec.wi, bRec.wo));

    float hDotWo = fabsf(dot3(wh, bRec.wo));

    float fc = fresnelClearcoat(hDotWo);

    float dc = distributionClearcoat(bsdf, wh);

    float gc = shadowMaskOperatorClearcoat(bRec.wi, bRec.wo);

    float denominator = 4.0f * cosThetaIn;

    float clearcoat = (fc * dc * gc) / denominator;
    return {clearcoat, clearcoat, clearcoat};

}

/// Evaluate the sampling density of \ref sample() wrt. solid angles
__device__ __forceinline__ float pdfClearcoatLobe(const BsdfData& bsdf, const BsdfQueryRecord& bRec) {

    float cosThetaIn = cosThetaLocal(bRec.wi);

    // check to make sure rays arent underneath
    if (cosThetaIn <= 0.0f || cosThetaLocal(bRec.wo) <= 0.0f) {
        return 0.0f;
    }

    Float3 wh = normalize3(add3(bRec.wi, bRec.wo));

    float dc = distributionClearcoat(bsdf, wh);

    float hDotWo = fabsf(dot3(wh, bRec.wo));

    return (dc * cosThetaLocal(wh)) / (4.0f * hDotWo);

}

/// Sample the BRDF
__device__ __forceinline__ Float3 sampleClearcoatLobe(const BsdfData& bsdf, BsdfQueryRecord& bRec, float u1, float u2) {

    // check to make sure rays arent underneath
    if (cosThetaLocal(bRec.wi) <= 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }

    bRec.measure = BSDF_ESolidAngle;

    float isotropicRoughness = ((1.0f - disneyClearcoatGloss(bsdf)) * 0.1f) + (disneyClearcoatGloss(bsdf) * 0.001f);
    float irSquared = isotropicRoughness * isotropicRoughness;

    float azimuth = 2.0f * RT_PI * u1;

    float cosElevationSquared = (1.0f - powf(irSquared, 1.0f - u2)) / (1.0f - irSquared);

    float sinElevation = sqrtf(1.0f - cosElevationSquared);

    float hz = sqrtf(cosElevationSquared);
    float hy = sinElevation * sinf(azimuth);
    float hx = sinElevation * cosf(azimuth);

    Float3 wh = normalize3(Float3{hx, hy, hz});

    bRec.wo = sub3(mul3(wh, 2.0f * dot3(bRec.wi, wh)), bRec.wi);
    bRec.eta = 1.0f;

    // check to make sure rays arent underneath
    if (cosThetaLocal(bRec.wo) <= 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }

    return div3(evalClearcoatLobe(bsdf, bRec), pdfClearcoatLobe(bsdf, bRec));

}

// ############GLASS###########################################################

__device__ __forceinline__ float fresnelGlass(Float3 wh, float hDotWi, float hDotWo, float etahDotWi, float etahDotWo) {

    hDotWi = fabsf(hDotWi);
    hDotWo = fabsf(hDotWo);
    etahDotWi = fabsf(etahDotWi);
    etahDotWo = fabsf(etahDotWo);

    float rsNumerator = hDotWi - etahDotWo;
    float rsDenominator = hDotWi + etahDotWo;

    float rpNumerator = etahDotWi - hDotWo;
    float rpDenominator = etahDotWi + hDotWo;

    float rs = rsNumerator / rsDenominator;
    float rp = rpNumerator / rpDenominator;

    return 0.5f * ((rs * rs) + (rp * rp));
}

/// Evaluate the BRDF for the given pair of directions
__device__ __forceinline__ Float3 evalGlassLobe(const BsdfData& bsdf, const BsdfQueryRecord& bRec) {

    float cosThetaIn = cosThetaLocal(bRec.wi);
    float cosThetaOut = cosThetaLocal(bRec.wo);

    float eta = disneyEta(bsdf);

    if (cosThetaIn < 0.0f) {
        eta = 1.0f / eta;
    }

    float inNout = cosThetaIn * cosThetaOut;

    Float3 wh;

    if (inNout > 0.0f) {
        wh = normalize3(add3(bRec.wi, bRec.wo));
    } else {
        wh = mul3(normalize3(add3(bRec.wi, mul3(bRec.wo, eta))), -1.0f);
    }

    if (cosThetaLocal(wh) < 0.0f) {
        wh = mul3(wh, -1.0f);
    }

    float hDotWi = dot3(wh, bRec.wi);
    float hDotWo = dot3(wh, bRec.wo);

    float etahDotWi = eta * hDotWi;
    float etahDotWo = eta * hDotWo;

    float fg = fresnelGlass(wh, hDotWi, hDotWo, etahDotWi, etahDotWo);

    float aspect = sqrtf(1.0f - (0.9f * disneyAnisotropic(bsdf)));

    // Clamps the alpha to a minimum above 0 to avoid perfect mirrors
    // (this way all materials are solid angle distributions)
    float ax = fmaxf(0.0001, (disneyRoughness(bsdf) * disneyRoughness(bsdf)) / aspect);
    float ay = fmaxf(0.0001, (disneyRoughness(bsdf) * disneyRoughness(bsdf)) * aspect);

    float dg = groundGlassXDistribution(wh, ax, ay);

    float gg = shadowMaskOperatorDisney(bRec.wi, bRec.wo, ax, ay);

    if (inNout > 0.0f) {
        Float3 numerator = mul3(disneyBaseColor(bsdf), fg * dg * gg);
        float denominator = 4.0f * fabsf(cosThetaIn);

        return div3(numerator, denominator);
    }
    else {
        Float3 numerator = mul3(disneySqrt3(disneyBaseColor(bsdf)), (1.0f - fg) * dg * gg * fabsf(hDotWi * hDotWo));
        float denominator = fabsf(cosThetaIn) * fmaxf(powf(hDotWi + etahDotWo, 2), 1e-4f);

        return div3(numerator, denominator);
    }

}

/// Evaluate the sampling density of \ref sample() wrt. solid angles
__device__ __forceinline__ float pdfGlassLobe(const BsdfData& bsdf, const BsdfQueryRecord& bRec) {

    float cosThetaIn = cosThetaLocal(bRec.wi);
    float cosThetaOut = cosThetaLocal(bRec.wo);

    float eta = disneyEta(bsdf);

    if (cosThetaIn < 0.0f) {
        eta = 1.0f / eta;
    }

    float inNout = cosThetaIn * cosThetaOut;

    Float3 wh;

    if (inNout > 0.0f) {
        wh = normalize3(add3(bRec.wi, bRec.wo));
    } else {
        wh = mul3(normalize3(add3(bRec.wi, mul3(bRec.wo, eta))), -1.0f);
    }

    if (cosThetaLocal(wh) < 0.0f) {
        wh = mul3(wh, -1.0f);
    }

    float hDotWi = dot3(wh, bRec.wi);
    float hDotWo = dot3(wh, bRec.wo);

    float fg = fresnelGlass(wh, hDotWi, hDotWo, eta * hDotWi, eta * hDotWo);

    float aspect = sqrtf(1.0f - (0.9f * disneyAnisotropic(bsdf)));

    // Clamps the alpha to a minimum above 0 to avoid perfect mirrors
    // (this way all materials are solid angle distributions)
    float ax = fmaxf(0.0001, (disneyRoughness(bsdf) * disneyRoughness(bsdf)) / aspect);
    float ay = fmaxf(0.0001, (disneyRoughness(bsdf) * disneyRoughness(bsdf)) * aspect);

    float dg = groundGlassXDistribution(wh, ax, ay);

    float gg = lambdaFunc(bRec.wi, ax, ay);

    float commonPDF =  (dg * gg * fabsf(hDotWi)) / fabsf(cosThetaIn);

    if (inNout > 0.0f) {
        return (commonPDF * fg) / (4.0f * fabsf(hDotWo));
    }
    else {

        float jacobian = (eta * eta * fabsf(hDotWo)) / fmaxf(powf(hDotWi + (eta * hDotWo), 2), 1e-4f);

        return (1.0f - fg) * commonPDF * jacobian;
    }

}

/// Sample the BRDF
__device__ __forceinline__ Float3 sampleGlassLobe(const BsdfData& bsdf, BsdfQueryRecord& bRec, float u1, float u2) {

    float cosThetaIn = cosThetaLocal(bRec.wi);

    bRec.measure = BSDF_ESolidAngle;

    float aspect = sqrtf(1.0f - (0.9f * disneyAnisotropic(bsdf)));
    float ax = fmaxf(0.0001, (disneyRoughness(bsdf) * disneyRoughness(bsdf)) / aspect);
    float ay = fmaxf(0.0001, (disneyRoughness(bsdf) * disneyRoughness(bsdf)) * aspect);

    float eta = disneyEta(bsdf);
    if (cosThetaIn < 0.0f) {
        eta = 1.0f / eta;
    }

    Float3 wi_upper = (cosThetaIn < 0.0f) ? mul3(bRec.wi, -1.0f) : bRec.wi;

    // Sample a new outgooing direction with heitz

    Float3 vh = normalize3(Float3{ax * wi_upper.x, ay * wi_upper.y, wi_upper.z});

    float pl = (vh.x * vh.x) + (vh.y * vh.y);

    Float3 orthOne = {1.0f, 0.0f, 0.0f};

    if (pl > 0.0f){
        orthOne = div3(Float3{-1.0f * vh.y, vh.x, 0.0f}, sqrtf(pl));
    }

    Float3 orthTwo = cross3(vh, orthOne);

    float radius = sqrtf(u1);
    float phi = 2.0f * RT_PI * u2;

    float t1 = radius * cosf(phi);
    float t2 = radius * sinf(phi);

    float skew = 0.5f * (1.0f + vh.z);
    t2 = (1.0f - skew) * sqrtf(1.0f - (t1 * t1)) + (skew * t2);

    Float3 ph = add3(add3(mul3(orthOne, t1), mul3(orthTwo, t2)),
                     mul3(vh, sqrtf(fmaxf(0.0f, 1 - (t1 * t1) - (t2 * t2)))));

    Float3 randH = normalize3(Float3{ax * ph.x, ay * ph.y, fmaxf(0.0f, ph.z)});

    if (cosThetaIn < 0.0f){
        randH = mul3(randH, -1.0f);
    }

    float hDotWi = dot3(randH, bRec.wi);

    float sinSqThetaT = (1.0f - hDotWi * hDotWi) / (eta * eta);
    float hDotWt = sqrtf(fmaxf(0.0f, 1.0f - sinSqThetaT));

    float fg = fresnelGlass(randH, hDotWi, hDotWt, eta * hDotWi, eta * hDotWt);

    if (bRec.rndExtra < fg || sinSqThetaT >= 1.0f) {

        // reflection

        bRec.wo = sub3(mul3(randH, 2.0f * dot3(bRec.wi, randH)), bRec.wi);
        bRec.eta = 1.0f;

    }
    else {

        // refraction

        bRec.wo = sub3(mul3(randH, (hDotWi / eta) - hDotWt), div3(bRec.wi, eta));
        bRec.wo = normalize3(bRec.wo);

        bRec.eta = eta;

    }

    float pdf = pdfGlassLobe(bsdf, bRec);

    if (pdf == 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }

    return div3(evalGlassLobe(bsdf, bRec), pdf);

}

// ############SHEEN###########################################################

/// Evaluate the BRDF for the given pair of directions
__device__ __forceinline__ Float3 evalSheenLobe(const BsdfData& bsdf, const BsdfQueryRecord& bRec) {

    float cosThetaIn = cosThetaLocal(bRec.wi);
    float cosThetaOut = cosThetaLocal(bRec.wo);

    // check to make sure rays arent underneath
    if (cosThetaIn <= 0.0f || cosThetaOut <= 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }

    Float3 wh = normalize3(add3(bRec.wi, bRec.wo));

    float hDotWo = fabsf(dot3(wh, bRec.wo));

    Float3 baseColor = disneyBaseColor(bsdf);

    float lumi = 0.3f * baseColor.x + 0.6f * baseColor.y + 0.1f * baseColor.z;

    Float3 tint = {1.0f, 1.0f, 1.0f};

    if (lumi > 0){
        tint = div3(baseColor, lumi);
    }

    float sheenTint = disneySheenTint(bsdf);
    Float3 sheen = {(1.0f - sheenTint) + (sheenTint * tint.x),
                    (1.0f - sheenTint) + (sheenTint * tint.y),
                    (1.0f - sheenTint) + (sheenTint * tint.z)};

    return mul3(sheen, powf(1.0f - hDotWo, 5) * cosThetaOut);
}

/// Evaluate the sampling density of \ref sample() wrt. solid angles
__device__ __forceinline__ float pdfSheenLobe(const BsdfQueryRecord& bRec) {

    // standard cosine hemisphere sampling similar to diffuse.cpp

    if (bRec.measure != BSDF_ESolidAngle
        || cosThetaLocal(bRec.wi) <= 0
        || cosThetaLocal(bRec.wo) <= 0)
        return 0.0f;


    /* Importance sampling density wrt. solid angles:
       cos(theta) / pi.

       Note that the directions in 'bRec' are in local coordinates,
       so Frame::cosTheta() actually just returns the 'z' component.
    */
    return RT_INV_PI * cosThetaLocal(bRec.wo);

}

/// Sample the BRDF
__device__ __forceinline__ Float3 sampleSheenLobe(const BsdfData& bsdf, BsdfQueryRecord& bRec, float u1, float u2) {

    // Standard cosine hemisphere sampling, basing on diffuse.cpp

    if (cosThetaLocal(bRec.wi) <= 0)
        return {0.0f, 0.0f, 0.0f};

    bRec.measure = BSDF_ESolidAngle;

    /* Warp a uniformly distributed sample on [0,1]^2
       to a direction on a cosine-weighted hemisphere */
    bRec.wo = squareToCosineHemisphere(u1, u2);

    /* Relative index of refraction: no change */
    bRec.eta = 1.0f;

    /* eval() / pdf() * cos(theta) = albedo. There
       is no need to call these functions. */
    return div3(evalSheenLobe(bsdf, bRec), pdfSheenLobe(bRec));

}

// ############DISNEY TOP-LEVEL###################################################

/// Evaluate the BRDF, keeping the diffuse-ish and specular-ish halves apart.
///
/// This is the same arithmetic bsdfEvalDisney has always done — the five lobe
/// terms were already computed separately before being summed.  Splitting the
/// sum is what lets next-event estimation at the primary hit feed the cel body
/// tone and the hard anime highlight as two independently converged channels,
/// without a single change to the transport.
///
///   diffuse-ish  = fd + fs                (base diffuse + sheen)
///   specular-ish = fm + fc + fg           (metal + clearcoat + glass)
///
/// bsdfEvalDisney is implemented in terms of this function so the split can
/// never drift away from the total.
__device__ __forceinline__ void bsdfEvalDisneySplit(const BsdfData& bsdf,
                                                     const BsdfQueryRecord& bRec,
                                                     Float3& outDiffuseish,
                                                     Float3& outSpecularish) {

    outDiffuseish  = {0.0f, 0.0f, 0.0f};
    outSpecularish = {0.0f, 0.0f, 0.0f};

    // if the ray is inside of object then only do glass

    float cosThetaIn = cosThetaLocal(bRec.wi);

    if (cosThetaIn <= 0.0f) {

        // Only glass
        if (disneySpecularTransmission(bsdf) <= 0.0f) return;

        outSpecularish = evalGlassLobe(bsdf, bRec);
        return;

    }
    else {

        float minMetal = (1.0f - disneyMetallic(bsdf));
        float minSpecTransmission = (1.0f - disneySpecularTransmission(bsdf));

        Float3 fd = mul3(evalDiffuseLobe(bsdf, bRec), minMetal * minSpecTransmission);
        Float3 fs = mul3(evalSheenLobe(bsdf, bRec), minMetal * disneySheen(bsdf));
        Float3 fm = mul3(evalMetalLobe(bsdf, bRec), (1.0f - (disneySpecularTransmission(bsdf) * minMetal)));
        Float3 fc = mul3(evalClearcoatLobe(bsdf, bRec), 0.25f * disneyClearcoat(bsdf));
        Float3 fg = mul3(evalGlassLobe(bsdf, bRec), minMetal * disneySpecularTransmission(bsdf));

        outDiffuseish  = add3(fd, fs);
        outSpecularish = add3(add3(fm, fc), fg);

    }

}

/// Evaluate the BRDF for the given pair of directions
__device__ __forceinline__ Float3 bsdfEvalDisney(const BsdfData& bsdf, const BsdfQueryRecord& bRec) {

    Float3 diffuseish, specularish;
    bsdfEvalDisneySplit(bsdf, bRec, diffuseish, specularish);
    return add3(diffuseish, specularish);

}

/// Evaluate the sampling density of \ref sample() wrt. solid angles
__device__ __forceinline__ float bsdfPdfDisney(const BsdfData& bsdf, const BsdfQueryRecord& bRec) {

    float minMetal = (1.0f - disneyMetallic(bsdf));
    float minSpecTransmission = (1.0f - disneySpecularTransmission(bsdf));

    float diffuseWeight = minMetal * minSpecTransmission;
    float metallicWeight = 1.0f - (disneySpecularTransmission(bsdf) * minMetal);
    float glassWeight = minMetal * disneySpecularTransmission(bsdf);
    float clearcoatWeight = 0.25f * disneyClearcoat(bsdf);

    float weightSum = diffuseWeight + glassWeight + metallicWeight + clearcoatWeight;

    float cosThetaIn = cosThetaLocal(bRec.wi);

    if (cosThetaIn <= 0) {

        if (disneySpecularTransmission(bsdf) <= 0.0f) return 0.0f;

        return pdfGlassLobe(bsdf, bRec);

    }
    else {

        float probDiffuse = pdfDiffuseLobe(bRec);
        float probMetal = pdfMetalLobe(bsdf, bRec);
        float probGlass = pdfGlassLobe(bsdf, bRec);
        float probClearcoat = pdfClearcoatLobe(bsdf, bRec);

        return (diffuseWeight * probDiffuse + glassWeight * probGlass + metallicWeight * probMetal + clearcoatWeight * probClearcoat) / weightSum;
    }
}

/// Sample the BRDF.
///
/// `outLobe` optionally reports which DisneyLobe was chosen.  The path tracer
/// uses it at the primary hit to decide which radiance channel the rest of the
/// path feeds, which is how a metal surface's reflections end up posterized as
/// anime metal and a gem's refractions as an anime jewel.
__device__ __forceinline__ Float3 bsdfSampleDisney(const BsdfData& bsdf, BsdfQueryRecord& bRec, float u1, float u2, float* outPdf = nullptr, int* outLobe = nullptr) {

    // Pick a random lobe to sample out of the 5

    // if the ray is inside of object then only do glass

    float cosThetaIn = cosThetaLocal(bRec.wi);

    if (cosThetaIn <= 0.0f) {

        if (disneySpecularTransmission(bsdf) <= 0.0f) {
            if (outPdf) *outPdf = 0.0f;
            if (outLobe) *outLobe = DISNEY_LOBE_GLASS;
            return {0.0f, 0.0f, 0.0f};
        }

        if (outLobe) *outLobe = DISNEY_LOBE_GLASS;
        sampleGlassLobe(bsdf, bRec, u1, u2);

    }
    else {

        float minMetal = (1.0f - disneyMetallic(bsdf));

        float diffuseWeight = minMetal * (1.0f - disneySpecularTransmission(bsdf));
        float specularWeight = minMetal * disneySpecularTransmission(bsdf);
        float metallicWeight = 1.0f - (disneySpecularTransmission(bsdf) * minMetal);
        float clearcoatWeight = 0.25f * disneyClearcoat(bsdf);

        float weightSum = diffuseWeight + specularWeight + metallicWeight + clearcoatWeight;

        float rndChoice = bRec.rndExtra;

        float diffuseFrac  = diffuseWeight  / weightSum;
        float specularFrac = specularWeight / weightSum;
        float metallicFrac = metallicWeight / weightSum;

        if (rndChoice < diffuseFrac) {

            if (outLobe) *outLobe = DISNEY_LOBE_DIFFUSE;
            sampleDiffuseLobe(bsdf, bRec, u1, u2);

        }
        else if (rndChoice < diffuseFrac + specularFrac) {

            if (outLobe) *outLobe = DISNEY_LOBE_GLASS;
            bRec.rndExtra = (specularFrac > 0.0f) ? (rndChoice - diffuseFrac) / specularFrac : 0.0f;
            sampleGlassLobe(bsdf, bRec, u1, u2);

        }
        else if (rndChoice < diffuseFrac + specularFrac + metallicFrac) {

            if (outLobe) *outLobe = DISNEY_LOBE_METAL;
            sampleMetalLobe(bsdf, bRec, u1, u2);

        }
        else {

            if (outLobe) *outLobe = DISNEY_LOBE_CLEARCOAT;
            sampleClearcoatLobe(bsdf, bRec, u1, u2);

        }
    }

    float disneyPDF = bsdfPdfDisney(bsdf, bRec);

    if (outPdf) *outPdf = disneyPDF;

    if (disneyPDF < 1e-7f) {
        return {0.0f, 0.0f, 0.0f};
    }
    else {
        return div3(bsdfEvalDisney(bsdf, bRec), disneyPDF);
    }

}
