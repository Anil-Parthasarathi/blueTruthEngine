/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob
*/

#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/warp.h>
#include <nori/dpdf.h>
#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN

class Disney : public BSDF {
public:
    Disney(const PropertyList &propList) {

        d_base_color = propList.getColor("base_color", Color3f(0.5f));
        d_roughness = propList.getFloat("roughness", 0.1f);
        d_metallic = propList.getFloat("metallic", 0.0f);
        d_specular = propList.getFloat("specular", 0.5f);
        d_specular_transmission = propList.getFloat("specular_transmission", 0.0f);
        d_specular_tint = propList.getFloat("specular_tint", 0.0f);
        d_sheen = propList.getFloat("sheen", 0.0f);
        d_sheen_tint = propList.getFloat("sheen_tint", 0.5f);
        d_subsurface = propList.getFloat("subsurface", 0.0f);
        d_anisotropic = propList.getFloat("anisotropic", 0.0f);
        d_clearcoat = propList.getFloat("clearcoat", 0.0f);
        d_clearcoat_gloss = propList.getFloat("clearcoat_gloss", 1.0f);
        d_eta = propList.getFloat("eta", 1.5f);
    }

    /// Evaluate the BRDF for the given pair of directions
    Color3f eval(const BSDFQueryRecord &bRec) const {

        // if the ray is inside of object then only do glass

        float cosThetaIn = Frame::cosTheta(bRec.wi);

        if (cosThetaIn <= 0.0f) {

            // Only glass 
            if (d_specular_transmission <= 0.0f) return Color3f(0.0f);

            return evalGlassLobe(bRec);

        }
        else {

            float minMetal = (1.0f - d_metallic);
            float minSpecTransmission = (1.0f - d_specular_transmission);

            Color3f fd = evalDiffuseLobe(bRec) * minMetal * minSpecTransmission;
            Color3f fs = evalSheenLobe(bRec) * minMetal * d_sheen;
            Color3f fm = evalMetalLobe(bRec) * (1.0f - (d_specular_transmission * minMetal));
            Color3f fc = evalClearcoatLobe(bRec) * 0.25f * d_clearcoat;
            Color3f fg = evalGlassLobe(bRec) * minMetal * d_specular_transmission;

            return fd + fm + fg + fc + fs;

        }

    }

    /// Evaluate the sampling density of \ref sample() wrt. solid angles
    float pdf(const BSDFQueryRecord &bRec) const {

        float minMetal = (1.0f - d_metallic);
        float minSpecTransmission = (1.0f - d_specular_transmission);

        float diffuseWeight = minMetal * minSpecTransmission;
        float metallicWeight = 1.0f - (d_specular_transmission * minMetal);
        float glassWeight = minMetal * d_specular_transmission;
        float clearcoatWeight = 0.25f * d_clearcoat;

        float weightSum = diffuseWeight + glassWeight + metallicWeight + clearcoatWeight;

        float cosThetaIn = Frame::cosTheta(bRec.wi);

        if (cosThetaIn <= 0) {

            if (d_specular_transmission <= 0.0f) return 0.0f;

            return pdfGlassLobe(bRec);

        }
        else {

            float probDiffuse = pdfDiffuseLobe(bRec);
            float probMetal = pdfMetalLobe(bRec);
            float probGlass = pdfGlassLobe(bRec);
            float probClearcoat = pdfClearcoatLobe(bRec);

            return (diffuseWeight * probDiffuse + glassWeight * probGlass + metallicWeight * probMetal + clearcoatWeight * probClearcoat) / weightSum;
        }
    }

    /// Sample the BRDF
    Color3f sample(BSDFQueryRecord &bRec, const Point2f &_sample) const {

        // Pick a random lobe to sample out of the 5

        // if the ray is inside of object then only do glass

        float cosThetaIn = Frame::cosTheta(bRec.wi);

        if (cosThetaIn <= 0.0f) {

            if (d_specular_transmission <= 0.0f) return Color3f(0.0f);

            sampleGlassLobe(bRec, _sample);

        }
        else {

            float minMetal = (1.0f - d_metallic);

            float diffuseWeight = minMetal * (1.0f - d_specular_transmission);
            float specularWeight = minMetal * d_specular_transmission;
            float metallicWeight = 1.0f - (d_specular_transmission * minMetal);
            float clearcoatWeight = 0.25f * d_clearcoat;

            float weightSum = diffuseWeight + specularWeight + metallicWeight + clearcoatWeight;

            float rndChoice = bRec.rndExtra;

            float diffuseFrac  = diffuseWeight  / weightSum;
            float specularFrac = specularWeight / weightSum;
            float metallicFrac = metallicWeight / weightSum;

            if (rndChoice < diffuseFrac) {

                sampleDiffuseLobe(bRec, _sample);

            }
            else if (rndChoice < diffuseFrac + specularFrac) {

                bRec.rndExtra = (specularFrac > 0.0f) ? (rndChoice - diffuseFrac) / specularFrac : 0.0f;
                sampleGlassLobe(bRec, _sample);

            }
            else if (rndChoice < diffuseFrac + specularFrac + metallicFrac) {

                sampleMetalLobe(bRec, _sample);

            }
            else {

                sampleClearcoatLobe(bRec, _sample);

            }
        }

        float disneyPDF = pdf(bRec);

        if (disneyPDF < 1e-7f) {
            return Color3f(0.0f);
        }
        else {
            return eval(bRec) / disneyPDF;
        }

    }

    // ############DIFFUSE###########################################################

    float diffuseFresnelNinety(float hDotWo) const {
        return 0.5f + (2.0f * d_roughness * hDotWo * hDotWo);
    }

    float diffuseFresnel(float cosTheta, float dfNinety) const {
        return 1.0f + ((dfNinety - 1.0f) * pow(1.0f - cosTheta, 5));
    }

    float subsurfaceFresnelNinety(float hDotWo) const {
        return d_roughness * hDotWo * hDotWo;
    }

    float subsurfaceFresnel(float cosTheta, float ssNinety) const {
        return 1.0f + ((ssNinety - 1.0f) * pow(1.0f - cosTheta, 5));
    }

    /// Evaluate the BRDF for the given pair of directions
    Color3f evalDiffuseLobe(const BSDFQueryRecord &bRec) const {

        // originally I had this as absolute values, but that resulted in a big issue where
        // spec transmission above 0 led to diffuse lobe blowing up into fireflies everywhere
        // before that, i had forgotten to have the guard here at all
        // both had the same issue I think, it didn't stop the invalid rays 
        float cosThetaIn = Frame::cosTheta(bRec.wi);
        float cosThetaOut = Frame::cosTheta(bRec.wo);

        // check to make sure rays arent underneath
        if (cosThetaIn <= 0.0f || cosThetaOut <= 0.0f) {
            return Color3f(0.0f);
        }

        Vector3f wh = (bRec.wi + bRec.wo).normalized();

        float hDotWo = abs(wh.dot(bRec.wo));

        float dfNinety = diffuseFresnelNinety(hDotWo);
        float ssNinety = subsurfaceFresnelNinety(hDotWo);

        Color3f colorOverPi = d_base_color / M_PI;

        Color3f baseDiffuse = 
            colorOverPi * 
            diffuseFresnel(cosThetaIn, dfNinety) * 
            diffuseFresnel(cosThetaOut, dfNinety) * 
            cosThetaOut;

        // Clamp the denominator to help prevent fireflies
        float ssDenom = fmax(1e-2f, cosThetaIn + cosThetaOut);

        Color3f subsurface = 
            1.25f * 
            colorOverPi * 
            (
                subsurfaceFresnel(cosThetaIn, ssNinety) * 
                subsurfaceFresnel(cosThetaOut, ssNinety) * 
                ((1.0f / ssDenom) - 0.5f) + 0.5f
            ) *
            cosThetaOut;

        // Return the weighted sum of diffuse and subsurface components based on the parameter for subsurface    
        return (1.0f - d_subsurface) * baseDiffuse + d_subsurface * subsurface;
    }

    /// Evaluate the sampling density of \ref sample() wrt. solid angles
    float pdfDiffuseLobe(const BSDFQueryRecord &bRec) const {

        // standard cosine hemisphere sampling similar to diffuse.cpp

        if (bRec.measure != ESolidAngle
            || Frame::cosTheta(bRec.wi) <= 0
            || Frame::cosTheta(bRec.wo) <= 0)
            return 0.0f;


        /* Importance sampling density wrt. solid angles:
           cos(theta) / pi.

           Note that the directions in 'bRec' are in local coordinates,
           so Frame::cosTheta() actually just returns the 'z' component.
        */
        return INV_PI * Frame::cosTheta(bRec.wo);
    }

    /// Sample the BRDF
    Color3f sampleDiffuseLobe(BSDFQueryRecord &bRec, const Point2f &_sample) const {

        // Standard cosine hemisphere sampling, basing on diffuse.cpp

        if (Frame::cosTheta(bRec.wi) <= 0)
            return Color3f(0.0f);

        bRec.measure = ESolidAngle;

        /* Warp a uniformly distributed sample on [0,1]^2
           to a direction on a cosine-weighted hemisphere */
        bRec.wo = Warp::squareToCosineHemisphere(_sample);

        /* Relative index of refraction: no change */
        bRec.eta = 1.0f;

        /* eval() / pdf() * cos(theta) = albedo. There
           is no need to call these functions. */
        return evalDiffuseLobe(bRec) / pdfDiffuseLobe(bRec);
    }

    // ############METAL###########################################################

    float schlickFresnel(float eta) const {
        float numerator = eta - 1.0f;
        float denominator = eta + 1.0f;

        return (numerator * numerator) / (denominator * denominator);
    }

    Color3f fresnelMetal(float hDotWo) const {

        float lumi = 0.3f * d_base_color.x() + 0.6f * d_base_color.y() + 0.1f * d_base_color.z();

        Color3f tint = Color3f(1.0f, 1.0f, 1.0f);

        if (lumi > 0){
            tint = d_base_color / lumi;
        } 

        Color3f spec = (1.0f - d_specular_tint) + (d_specular_tint * tint);

        Color3f clr = (d_specular * schlickFresnel(d_eta) * (1.0f - d_metallic) * spec) + (d_base_color * d_metallic);

        return clr + ((1.0f - clr) * pow(1.0f - hDotWo, 5));

    }

    float groundGlassXDistribution(Vector3f wh, float ax, float ay) const {

        float denominator = M_PI * ax * ay * 
            pow(((wh.x() * wh.x()) / (ax * ax)) + ((wh.y() * wh.y()) / (ay * ay)) + (wh.z() * wh.z()), 2);

        return 1.0f / denominator;
    }

    float lambdaFunc(Vector3f w, float ax, float ay) const{

        float axw = w.x() * ax;
        float ayw = w.y() * ay;

        float numerator = sqrt(1 + (((axw * axw) + (ayw * ayw)) / fmax(1e-8f, (w.z() * w.z())))) - 1.0f;

        float lambda =  numerator / 2.0f;

        return 1.0f / (1.0f + lambda);
    }

    float shadowMaskOperator(const Vector3f &wi, const Vector3f &wo, float ax, float ay) const {
        
        return lambdaFunc(wi, ax, ay) * lambdaFunc(wo, ax, ay);

    }

    /// Evaluate the BRDF for the given pair of directions
    Color3f evalMetalLobe(const BSDFQueryRecord &bRec) const {

        float cosThetaIn = Frame::cosTheta(bRec.wi);

        // check to make sure rays arent underneath
        if (cosThetaIn <= 0.0f || Frame::cosTheta(bRec.wo) <= 0.0f) {
            return Color3f(0.0f);
        }

        Vector3f wh = (bRec.wi + bRec.wo).normalized();

        float hDotWo = abs(wh.dot(bRec.wo));

        float aspect = sqrt(1.0f - (0.9f * d_anisotropic));

        // Clamps the alpha to a minimum above 0 to avoid perfect mirrors
        // (this way all materials are solid angle distributions)
        float ax = fmax(0.0001, (d_roughness * d_roughness) / aspect);
        float ay = fmax(0.0001, (d_roughness * d_roughness) * aspect);

        Color3f fm = fresnelMetal(hDotWo);

        float dm = groundGlassXDistribution(wh, ax, ay);

        float gm = shadowMaskOperator(bRec.wi, bRec.wo, ax, ay);

        return (fm * dm * gm) / (4 * cosThetaIn);

    }

    /// Evaluate the sampling density of \ref sample() wrt. solid angles
    float pdfMetalLobe(const BSDFQueryRecord &bRec) const {
        
        float cosThetaIn = Frame::cosTheta(bRec.wi);

        // check to make sure rays arent underneath
        if (cosThetaIn <= 0.0f || Frame::cosTheta(bRec.wo) <= 0.0f) {
            return 0.0f;
        }

        Vector3f wh = (bRec.wi + bRec.wo).normalized();

        float aspect = sqrt(1.0f - (0.9f * d_anisotropic));

        // Clamps the alpha to a minimum above 0 to avoid perfect mirrors
        // (this way all materials are solid angle distributions)
        float ax = fmax(0.0001, (d_roughness * d_roughness) / aspect);
        float ay = fmax(0.0001, (d_roughness * d_roughness) * aspect);

        float dm = groundGlassXDistribution(wh, ax, ay);

        float gm = lambdaFunc(bRec.wi, ax, ay);

        float hDotWi = abs(wh.dot(bRec.wi));
        float hDotWo = abs(wh.dot(bRec.wo));
        
        return (dm * gm * hDotWi) / (cosThetaIn * 4.0f * hDotWo);

    }

    /// Sample the BRDF
    Color3f sampleMetalLobe(BSDFQueryRecord &bRec, const Point2f &_sample) const {

        // check to make sure rays arent underneath
        if (Frame::cosTheta(bRec.wi) <= 0.0f) {
            return Color3f(0.0f);
        }

        bRec.measure = ESolidAngle;
        bRec.eta = 1.0f;

        float aspect = sqrt(1.0f - (0.9f * d_anisotropic));

        // Clamps the alpha to a minimum above 0 to avoid perfect mirrors
        // (this way all materials are solid angle distributions)
        float ax = fmax(0.0001, (d_roughness * d_roughness) / aspect);
        float ay = fmax(0.0001, (d_roughness * d_roughness) * aspect);

        // Sample a new outgooing direction with heitz

        Vector3f vh = Vector3f(ax * bRec.wi.x(), ay * bRec.wi.y(), bRec.wi.z()).normalized();

        float pl = (vh.x() * vh.x()) + (vh.y() * vh.y());

        Vector3f orthOne = Vector3f(1.0f, 0.0f, 0.0f);

        if (pl > 0.0f){
            orthOne = Vector3f(-1.0f * vh.y(), vh.x(), 0.0f) / sqrt(pl);
        }

        Vector3f orthTwo = vh.cross(orthOne);

        float radius = sqrt(_sample.x());
        float phi = 2.0f * M_PI * _sample.y();

        float t1 = radius * cos(phi);
        float t2 = radius * sin(phi);

        float skew = 0.5f * (1.0f + vh.z());
        t2 = (1.0f - skew) * sqrt(1.0f - (t1 * t1)) + (skew * t2);

        Vector3f ph = (orthOne * t1) + (orthTwo * t2) + sqrt(fmax(0.0f, 1 - (t1 * t1) - (t2 * t2))) * vh;

        Vector3f randH = Vector3f(ax * ph.x(), ay * ph.y(), fmax(0.0f, ph.z())).normalized();

        bRec.wo = (2.0f * bRec.wi.dot(randH) * randH) - bRec.wi;

        // check to make sure rays arent underneath
        if (Frame::cosTheta(bRec.wo) <= 0.0f) {
            return Color3f(0.0f);
        }

        // eval and pdf math cancel out to make the rest simple

        float hDotWo = abs(randH.dot(bRec.wo));

        Color3f fm = fresnelMetal(hDotWo);

        float gm = lambdaFunc(bRec.wo, ax, ay);

        return fm * gm;
    }

    // ############CLEARCOAT###########################################################

    float fresnelClearcoat(float hDotWo) const {
        
        // note: disney hard codes the ior to 1.5 for everything
        // this is not technically physically accurate but apparently its good enough
        
        // This is what math would have occured, but I realized its always 0.04
        //float r = pow((1.5f - 1.0f), 2) / pow(1.5 + 1.0f, 2);
        float r = 0.04;

        return r + ((1.0f - r) * pow(1.0f - hDotWo, 5));
    }

    float distributionClearcoat(Vector3f wh) const{

        float isotropicRoughness = ((1.0f - d_clearcoat_gloss) * 0.1f) + (d_clearcoat_gloss * 0.001f);

        float irSquared = isotropicRoughness * isotropicRoughness;

        float numerator = irSquared - 1.0f;

        float denominator = M_PI * log(irSquared) * (1.0f + ((irSquared - 1.0f) * (wh.z() * wh.z())));

        return numerator / denominator;

    }

    float shadowMaskOperatorClearcoat(const Vector3f &wi, const Vector3f &wo) const{
        return lambdaFunc(wi, 0.25, 0.25) * lambdaFunc(wo, 0.25, 0.25);
    }

    /// Evaluate the BRDF for the given pair of directions
    Color3f evalClearcoatLobe(const BSDFQueryRecord &bRec) const {

        float cosThetaIn = Frame::cosTheta(bRec.wi);

        // check to make sure rays arent underneath
        if (cosThetaIn <= 0.0f || Frame::cosTheta(bRec.wo) <= 0.0f) {
            return Color3f(0.0f);
        }

        Vector3f wh = (bRec.wi + bRec.wo).normalized();

        float hDotWo = abs(wh.dot(bRec.wo));

        float fc = fresnelClearcoat(hDotWo);

        float dc = distributionClearcoat(wh);

        float gc = shadowMaskOperatorClearcoat(bRec.wi, bRec.wo);

        float denominator = 4.0f * cosThetaIn;

        return (fc * dc * gc) / denominator;

    }

    /// Evaluate the sampling density of \ref sample() wrt. solid angles
    float pdfClearcoatLobe(const BSDFQueryRecord &bRec) const {

        float cosThetaIn = Frame::cosTheta(bRec.wi);

        // check to make sure rays arent underneath
        if (cosThetaIn <= 0.0f || Frame::cosTheta(bRec.wo) <= 0.0f) {
            return 0.0f;
        }

        Vector3f wh = (bRec.wi + bRec.wo).normalized();

        float dc = distributionClearcoat(wh);

        float hDotWo = abs(wh.dot(bRec.wo));
        
        return (dc * Frame::cosTheta(wh)) / (4.0f * hDotWo);

    }

    /// Sample the BRDF
    Color3f sampleClearcoatLobe(BSDFQueryRecord &bRec, const Point2f &_sample) const {

        // check to make sure rays arent underneath
        if (Frame::cosTheta(bRec.wi) <= 0.0f) {
            return Color3f(0.0f);
        }

        bRec.measure = ESolidAngle;

        float isotropicRoughness = ((1.0f - d_clearcoat_gloss) * 0.1f) + (d_clearcoat_gloss * 0.001f);
        float irSquared = isotropicRoughness * isotropicRoughness;

        float azimuth = 2.0f * M_PI * _sample.x();

        float cosElevationSquared = (1.0f - pow(irSquared, 1.0f - _sample.y())) / (1.0f - irSquared);

        float sinElevation = sqrt(1.0f - cosElevationSquared);

        float hz = sqrt(cosElevationSquared);
        float hy = sinElevation * sin(azimuth);
        float hx = sinElevation * cos(azimuth);

        Vector3f wh = Vector3f(hx, hy, hz).normalized();

        bRec.wo = (2.0f * bRec.wi.dot(wh) * wh) - bRec.wi;
        bRec.eta = 1.0f;

        // check to make sure rays arent underneath
        if (Frame::cosTheta(bRec.wo) <= 0.0f) {
            return Color3f(0.0f);
        }

        return evalClearcoatLobe(bRec) / pdfClearcoatLobe(bRec);

    }

    // ############GLASS###########################################################

    float fresnelGlass(Vector3f wh, float hDotWi, float hDotWo, float etahDotWi, float etahDotWo) const {

        hDotWi = std::abs(hDotWi);
        hDotWo = std::abs(hDotWo);
        etahDotWi = std::abs(etahDotWi);
        etahDotWo = std::abs(etahDotWo);

        float rsNumerator = hDotWi - etahDotWo;
        float rsDenominator = hDotWi + etahDotWo;

        float rpNumerator = etahDotWi - hDotWo;
        float rpDenominator = etahDotWi + hDotWo;

        float rs = rsNumerator / rsDenominator;
        float rp = rpNumerator / rpDenominator;

        return 0.5f * ((rs * rs) + (rp * rp));
    }

    /// Evaluate the BRDF for the given pair of directions
    Color3f evalGlassLobe(const BSDFQueryRecord &bRec) const {

        float cosThetaIn = Frame::cosTheta(bRec.wi);
        float cosThetaOut = Frame::cosTheta(bRec.wo);

        float eta = d_eta;

        if (cosThetaIn < 0.0f) {
            eta = 1.0f / eta;
        }

        float inNout = cosThetaIn * cosThetaOut;

        Vector3f wh;

        if (inNout > 0.0f) {
            wh = (bRec.wi + bRec.wo).normalized();
        } else {
            wh = -1.0f *(bRec.wi + (eta * bRec.wo)).normalized();
        }

        if (Frame::cosTheta(wh) < 0.0f) {
            wh = -1.0f * wh;
        }

        float hDotWi = wh.dot(bRec.wi);
        float hDotWo = wh.dot(bRec.wo);

        float etahDotWi = eta * hDotWi;
        float etahDotWo = eta * hDotWo;

        float fg = fresnelGlass(wh, hDotWi, hDotWo, etahDotWi, etahDotWo);

        float aspect = sqrt(1.0f - (0.9f * d_anisotropic));

        // Clamps the alpha to a minimum above 0 to avoid perfect mirrors
        // (this way all materials are solid angle distributions)
        float ax = fmax(0.0001, (d_roughness * d_roughness) / aspect);
        float ay = fmax(0.0001, (d_roughness * d_roughness) * aspect);

        float dg = groundGlassXDistribution(wh, ax, ay);

        float gg = shadowMaskOperator(bRec.wi, bRec.wo, ax, ay);

        if (inNout > 0.0f) {
            Color3f numerator = d_base_color * fg * dg * gg;
            float denominator = 4.0f * abs(cosThetaIn);

            return numerator / denominator;
        }
        else {
            Color3f numerator = sqrt(d_base_color) * (1.0f - fg) * dg * gg * abs(hDotWi * hDotWo);
            float denominator = abs(cosThetaIn) * fmax(pow(hDotWi + etahDotWo, 2), 1e-4f);

            return numerator / denominator;
        }

    }

    /// Evaluate the sampling density of \ref sample() wrt. solid angles
    float pdfGlassLobe(const BSDFQueryRecord &bRec) const {

        float cosThetaIn = Frame::cosTheta(bRec.wi);
        float cosThetaOut = Frame::cosTheta(bRec.wo);

        float eta = d_eta;

        if (cosThetaIn < 0.0f) {
            eta = 1.0f / eta;
        }

        float inNout = cosThetaIn * cosThetaOut;

        Vector3f wh;

        if (inNout > 0.0f) {
            wh = (bRec.wi + bRec.wo).normalized();
        } else {
            wh = -1.0f *(bRec.wi + (eta * bRec.wo)).normalized();
        }

        if (Frame::cosTheta(wh) < 0.0f) {
            wh = -1.0f * wh;
        }

        float hDotWi = wh.dot(bRec.wi);
        float hDotWo = wh.dot(bRec.wo);

        float fg = fresnelGlass(wh, hDotWi, hDotWo, eta * hDotWi, eta * hDotWo);

        float aspect = sqrt(1.0f - (0.9f * d_anisotropic));

        // Clamps the alpha to a minimum above 0 to avoid perfect mirrors
        // (this way all materials are solid angle distributions)
        float ax = fmax(0.0001, (d_roughness * d_roughness) / aspect);
        float ay = fmax(0.0001, (d_roughness * d_roughness) * aspect);

        float dg = groundGlassXDistribution(wh, ax, ay);

        float gg = lambdaFunc(bRec.wi, ax, ay);

        float commonPDF =  (dg * gg * abs(hDotWi)) / abs(cosThetaIn);

        if (inNout > 0.0f) {
            return (commonPDF * fg) / (4.0f * abs(hDotWo));
        }
        else {

            float jacobian = (eta * eta * abs(hDotWo)) / fmax(pow(hDotWi + (eta * hDotWo), 2), 1e-4f);

            return (1.0f - fg) * commonPDF * jacobian;
        }

    }

    /// Sample the BRDF
    Color3f sampleGlassLobe(BSDFQueryRecord &bRec, const Point2f &_sample) const {

        float cosThetaIn = Frame::cosTheta(bRec.wi);

        bRec.measure = ESolidAngle;

        float aspect = sqrt(1.0f - (0.9f * d_anisotropic));
        float ax = fmax(0.0001, (d_roughness * d_roughness) / aspect);
        float ay = fmax(0.0001, (d_roughness * d_roughness) * aspect);

        float eta = d_eta;
        if (cosThetaIn < 0.0f) {
            eta = 1.0f / eta;
        }

        Vector3f wi_upper = (cosThetaIn < 0.0f) ? -bRec.wi : bRec.wi;

        // Sample a new outgooing direction with heitz

        Vector3f vh = Vector3f(ax * wi_upper.x(), ay * wi_upper.y(), wi_upper.z()).normalized();

        float pl = (vh.x() * vh.x()) + (vh.y() * vh.y());

        Vector3f orthOne = Vector3f(1.0f, 0.0f, 0.0f);

        if (pl > 0.0f){
            orthOne = Vector3f(-1.0f * vh.y(), vh.x(), 0.0f) / sqrt(pl);
        }

        Vector3f orthTwo = vh.cross(orthOne);

        float radius = sqrt(_sample.x());
        float phi = 2.0f * M_PI * _sample.y();

        float t1 = radius * cos(phi);
        float t2 = radius * sin(phi);

        float skew = 0.5f * (1.0f + vh.z());
        t2 = (1.0f - skew) * sqrt(1.0f - (t1 * t1)) + (skew * t2);

        Vector3f ph = (orthOne * t1) + (orthTwo * t2) + sqrt(fmax(0.0f, 1 - (t1 * t1) - (t2 * t2))) * vh;

        Vector3f randH = Vector3f(ax * ph.x(), ay * ph.y(), fmax(0.0f, ph.z())).normalized();

        if (cosThetaIn < 0.0f){
            randH = -randH;
        } 

        float hDotWi = randH.dot(bRec.wi);

        float sinSqThetaT = (1.0f - hDotWi * hDotWi) / (eta * eta);
        float hDotWt = sqrt(fmax(0.0f, 1.0f - sinSqThetaT));

        float fg = fresnelGlass(randH, hDotWi, hDotWt, eta * hDotWi, eta * hDotWt);

        if (bRec.rndExtra < fg || sinSqThetaT >= 1.0f) {

            // reflection

            bRec.wo = (2.0f * bRec.wi.dot(randH) * randH) - bRec.wi;
            bRec.eta = 1.0f;

        }
        else {

            // refraction

            bRec.wo = ((hDotWi / eta) - hDotWt) * randH - (bRec.wi / eta);
            bRec.wo = bRec.wo.normalized();

            bRec.eta = eta;

        }

        float pdf = pdfGlassLobe(bRec);

        if (pdf == 0.0f) {
            return Color3f(0.0f);
        }

        return evalGlassLobe(bRec) / pdf;

    }

    // ############SHEEN###########################################################

    /// Evaluate the BRDF for the given pair of directions
    Color3f evalSheenLobe(const BSDFQueryRecord &bRec) const {

        float cosThetaIn = Frame::cosTheta(bRec.wi);
        float cosThetaOut = Frame::cosTheta(bRec.wo);

        // check to make sure rays arent underneath
        if (cosThetaIn <= 0.0f || cosThetaOut <= 0.0f) {
            return Color3f(0.0f);
        }

        Vector3f wh = (bRec.wi + bRec.wo).normalized();

        float hDotWo = abs(wh.dot(bRec.wo));

        float lumi = 0.3f * d_base_color.x() + 0.6f * d_base_color.y() + 0.1f * d_base_color.z();

        Color3f tint = Color3f(1.0f, 1.0f, 1.0f);

        if (lumi > 0){
            tint = d_base_color / lumi;
        }

        Color3f sheen = (1.0f - d_sheen_tint) + (d_sheen_tint * tint);

        return sheen * pow(1.0f - hDotWo, 5) * cosThetaOut;
    }

    /// Evaluate the sampling density of \ref sample() wrt. solid angles
    float pdfSheenLobe(const BSDFQueryRecord &bRec) const {

        // standard cosine hemisphere sampling similar to diffuse.cpp

        if (bRec.measure != ESolidAngle
            || Frame::cosTheta(bRec.wi) <= 0
            || Frame::cosTheta(bRec.wo) <= 0)
            return 0.0f;


        /* Importance sampling density wrt. solid angles:
           cos(theta) / pi.

           Note that the directions in 'bRec' are in local coordinates,
           so Frame::cosTheta() actually just returns the 'z' component.
        */
        return INV_PI * Frame::cosTheta(bRec.wo);

    }

    /// Sample the BRDF
    Color3f sampleSheenLobe(BSDFQueryRecord &bRec, const Point2f &_sample) const {

        // Standard cosine hemisphere sampling, basing on diffuse.cpp

        if (Frame::cosTheta(bRec.wi) <= 0)
            return Color3f(0.0f);

        bRec.measure = ESolidAngle;

        /* Warp a uniformly distributed sample on [0,1]^2
           to a direction on a cosine-weighted hemisphere */
        bRec.wo = Warp::squareToCosineHemisphere(_sample);

        /* Relative index of refraction: no change */
        bRec.eta = 1.0f;

        /* eval() / pdf() * cos(theta) = albedo. There
           is no need to call these functions. */
        return evalSheenLobe(bRec) / pdfSheenLobe(bRec);

    }

    bool isDiffuse() const {
        return true;
    }

    std::string toString() const {
        return tfm::format(
            "Disney[\n"
            "  base_color = %s,\n"  
            "  roughness = %f,\n"
            "  metallic = %f,\n"
            "  specular = %f,\n"
            "  specular_transmission = %f,\n"
            "  specular_tint = %f,\n"
            "  sheen = %f,\n"
            "  sheen_tint = %f,\n"
            "  subsurface = %f,\n"
            "  anisotropic = %f,\n"
            "  clearcoat = %f,\n"
            "  clearcoat_gloss = %f\n" 
                "  eta = %f\n"
            "]",
            d_base_color.toString(),
            d_roughness,
            d_metallic,
            d_specular,
            d_specular_transmission,
            d_specular_tint,
            d_sheen,
            d_sheen_tint,
            d_subsurface,
            d_anisotropic,
            d_clearcoat,
            d_clearcoat_gloss,
            d_eta
        );
    }
private:
    Color3f d_base_color;
    float d_roughness;
    float d_metallic;
    float d_specular;
    float d_specular_transmission;
    float d_specular_tint;
    float d_sheen;
    float d_sheen_tint;
    float d_subsurface;
    float d_anisotropic;
    float d_clearcoat;
    float d_clearcoat_gloss;
    float d_eta;    
};

NORI_REGISTER_CLASS(Disney, "disney");
NORI_NAMESPACE_END