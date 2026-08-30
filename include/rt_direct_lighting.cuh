#pragma once

// ---------------------------------------------------------------------------
//  rt_direct_lighting.cuh — emitter-hit MIS and next-event estimation (NEE),
//  shared by the megakernel and the wavefront pipeline.
//
//  The NEE functions are split so both pipelines can reuse the math while
//  resolving visibility differently:
//
//    prepareAreaEmitterNEE / prepareSpotlightNEE
//        Pure: sample the light, evaluate the BSDF, compute the MIS-weighted
//        contribution, and fill a ShadowRayRecord.  NO ray is traced here.
//
//    Megakernel:  sampleAreaEmitterNEE / sampleSpotlightNEE wrappers in
//                 rt_optix_shading.cuh call prepare*, trace the shadow ray
//                 inline (traceOccluded), and add the contribution.
//    Wavefront:   wfShade calls prepare* and writes the ShadowRayRecord into
//                 the shadow SoA; the connect stage traces it later.
//
//  Note: compared to the original inline code, the visibility test now runs
//  AFTER the contribution math (occluded samples do a little extra arithmetic).
//  The rendered image is identical — the deferred-shadow-ray wavefront design
//  requires the contribution to be precomputed anyway.
// ---------------------------------------------------------------------------

#include "render_kernel.h"
#include "rt_cuda_math.cuh"
#include "rt_intersect.cuh"
#include "rt_rng.cuh"
#include "rt_bsdf.cuh"
#include "rt_emitter_sampling.cuh"
#include "rt_constants.cuh"
#include "rt_shading_context.cuh"
#include "rt_env_map.cuh"

__device__ __forceinline__ float convertAreaPDFtoSolidAnglePDF(
    const float lightPDFArea,
    const EmitterQueryRecord& emitterQuery)
{
    // Converting from area to solid angle is area pdf * distance^2 /
    // abs(cos(theta))
    const Float3 diff = sub3(emitterQuery.hitPoint, emitterQuery.originPoint);
    const float dist2 = dot3(diff, diff);
    const Float3 negLightDir = {-emitterQuery.directionToLight.x,
                                -emitterQuery.directionToLight.y,
                                -emitterQuery.directionToLight.z};
    const float denom = fmaxf(fabsf(dot3(emitterQuery.hitNormal, negLightDir)), 1e-3f);
    return lightPDFArea * dist2 / denom;
}

// ---------------------------------------------------------------------------
//  Handles the BRDF-sampled emitter hit: computes the MIS BRDF weight and
//  accumulates the emitted radiance into the path's color.  Trace-free, so it
//  is shared as-is by both pipelines.
//
//  Reads the PREVIOUS bounce's MIS state (specularBounce, brdfPDF) and the
//  PREVIOUS ray (origin/direction) — in the wavefront shade kernel, call this
//  BEFORE scatterPath overwrites them.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void accumulateEmitterHit(
    PathState& pathRecord,
    const Intersection& its,
    const DirectLightingContext& lightingCtx)
{
    const Float3 surfaceRadiance = lightingCtx.triangleEmission[its.triangleIndex];

    EmitterQueryRecord directEmitterQuery{};
    directEmitterQuery.originPoint      = pathRecord.ray.origin;
    directEmitterQuery.hitPoint         = its.hitPoint;
    directEmitterQuery.hitNormal        = its.hitNormal;
    directEmitterQuery.directionToLight = pathRecord.ray.direction;

    float brdfWeight = 1.0f;

    if (!pathRecord.specularBounce) {
        const EmitterSamplingData& es = lightingCtx.emitterSampling;
        const int emitterIdx = findEmitterIndexForTriangle(
            its.triangleIndex, es.emitters, es.emitterCount, es.emitterTriIndices);

        if (emitterIdx >= 0) {
            const float lightPDFAreaBRDF =
                emitterProbabilityEvaluatorFromMesh(es.emitters[emitterIdx]) *
                sceneGetEmitterPDF(emitterIdx, es.sceneEmitterCdf, es.emitterCount);

            const float lightPDFSolidAngleBRDF =
                convertAreaPDFtoSolidAnglePDF(lightPDFAreaBRDF, directEmitterQuery);

            const float weightDenomSumBRDF = pathRecord.brdfPDF + lightPDFSolidAngleBRDF;

            if (weightDenomSumBRDF <= 0.0f) brdfWeight = 0.0f;
            else                            brdfWeight = pathRecord.brdfPDF / weightDenomSumBRDF;
        }
    }

    const Float3 rad = checkRadiance(surfaceRadiance, directEmitterQuery);
    pathRecord.accumulatedColor = add3(
        pathRecord.accumulatedColor,
        mul3(mul3(rad, pathRecord.throughput), brdfWeight));
}

// ---------------------------------------------------------------------------
//  Samples one randomly chosen area (mesh) emitter using MIS and fills
//  `shadowRay` with the visibility test plus the throughput-multiplied
//  contribution to add if the ray is unoccluded.
//  Returns false when the sample carries no energy (the original early-outs).
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool prepareAreaEmitterNEE(
    PathState& pathRecord,
    RngState& rng,
    const Intersection& its,
    const BsdfData& bsdf,
    const DirectLightingContext& lightingCtx,
    ShadowRayRecord& shadowRay)
{
    const EmitterSamplingData& es = lightingCtx.emitterSampling;

    const float uLight0 = rngNextFloat01(rng);
    const float uLight1 = rngNextFloat01(rng);

    EmitterQueryRecord emitterQuery{};
    emitterQuery.originPoint = its.hitPoint;

    Float3 le = sampleGenerator(es, emitterQuery, uLight0, uLight1);

    if ((le.x <= 0.0f && le.y <= 0.0f && le.z <= 0.0f) ||
        emitterQuery.emitterPdf <= 0.0f ||
        probabilityEvaluator(emitterQuery) <= 0.0f) {
        return false;
    }

    const float distToLight = distance3(emitterQuery.hitPoint, its.hitPoint);

    const float geo =
        fabsf(dot3(its.hitNormal, emitterQuery.directionToLight)) *
        fabsf(dot3(emitterQuery.hitNormal, mul3(emitterQuery.directionToLight, -1.0f))) /
        (distToLight * distToLight);

    BsdfQueryRecord bsdfQueryDirect{};
    bsdfQueryDirect.wi      = toLocalFromNormal(its.hitNormal, mul3(pathRecord.ray.direction, -1.0f));
    bsdfQueryDirect.wo      = toLocalFromNormal(its.hitNormal, emitterQuery.directionToLight);
    bsdfQueryDirect.measure = BSDF_ESolidAngle;

    const Float3 fr = bsdfEval(bsdf, bsdfQueryDirect);

    const float lightPDFareaDirect       = emitterQuery.emitterPdf * emitterQuery.pdf;
    const float lightPDFSolidAngleDirect = convertAreaPDFtoSolidAnglePDF(lightPDFareaDirect, emitterQuery);
    const float brdfPDFDirect            = bsdfPdf(bsdf, bsdfQueryDirect);
    const float weightDenomSumDirect     = brdfPDFDirect + lightPDFSolidAngleDirect;

    if (weightDenomSumDirect <= 0.0f) return false;

    const float lightWeight = lightPDFSolidAngleDirect / weightDenomSumDirect;

    Float3 contrib = mul3(fr, le);
    contrib = mul3(contrib, lightWeight);
    contrib = mul3(contrib, geo);
    contrib = div3(contrib, lightPDFareaDirect);

    shadowRay.origin       = its.hitPoint;
    shadowRay.direction    = emitterQuery.directionToLight;
    shadowRay.tMax         = distToLight - RT_EPSILON;
    shadowRay.contribution = mul3(contrib, pathRecord.throughput);
    return true;
}

// ---------------------------------------------------------------------------
//  Computes ONE spotlight's NEE contribution and fills `shadowRay`.
//  Point lights have a delta PDF — probabilityEvaluator returns 0 and
//  checkRadiance returns 0 (BRDF rays can never accidentally hit a point).
//  So there is no MIS: we use lightWeight = 1.0 for each spotlight.
//
//  Callers loop over lightingCtx.spotlightCount and call this per spotlight
//  (see sampleSpotlightNEE in rt_optix_shading.cuh for the megakernel loop).
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool prepareSpotlightNEE(
    const PathState& pathRecord,
    const Intersection& its,
    const BsdfData& bsdf,
    const DirectLightingContext& lightingCtx,
    int spotlightIndex,
    ShadowRayRecord& shadowRay)
{
    const SpotlightData spot = lightingCtx.spotlights[spotlightIndex];

    // set the emitter query record
    // simply give the hitpoint as the origin point of the spot light
    // hit normal is the opposite of the direction to the light since the spot light is facing in the direction vector
    // pdf is 1 since there is only one possible point to hit
    const Float3 originToLight = sub3(spot.position, its.hitPoint);
    const float sqDist = dot3(originToLight, originToLight);
    if (sqDist < RT_EPSILON * RT_EPSILON) return false;
    const float dist        = sqrtf(sqDist);
    const Float3 dirToLight = div3(originToLight, dist);

    // get the cosine of the angle between the direction to the light and the forward direction of the spot light
    const float lightCos = dot3(mul3(dirToLight, -1.0f), spot.direction);

    // if the angle is greater than the outer cone angle, return 0
    // if the angle is greater then return the full radiance (scaled by distance)
    // otherwise interpolate by finding the falloff
    Float3 spotLe;
    if (lightCos <= spot.outerConeCosine) {
        return false;
    } else if (lightCos >= spot.innerConeCosine) {
        spotLe = mul3(spot.radiance, spot.intensity / sqDist);
    } else {
        float falloff = (lightCos - spot.outerConeCosine) /
                        (spot.innerConeCosine - spot.outerConeCosine);
        float smoothedFalloff = (falloff * falloff) * (3.0f - (2.0f * falloff));
        spotLe = mul3(spot.radiance, (spot.intensity * smoothedFalloff) / sqDist);
    }

    BsdfQueryRecord bsdfQuerySpot{};
    bsdfQuerySpot.wi      = toLocalFromNormal(its.hitNormal, mul3(pathRecord.ray.direction, -1.0f));
    bsdfQuerySpot.wo      = toLocalFromNormal(its.hitNormal, dirToLight);
    bsdfQuerySpot.measure = BSDF_ESolidAngle;

    const Float3 fr = bsdfEval(bsdf, bsdfQuerySpot);

    // point light so we assume that random bounces never hit the spot light directly
    // contribution: fr * Le * cos_surface  (pdf = 1, no area term)
    const float cosSurface = fabsf(dot3(its.hitNormal, dirToLight));

    // Shadow ray — bound to just before the light to avoid self-intersection
    shadowRay.origin       = its.hitPoint;
    shadowRay.direction    = dirToLight;
    shadowRay.tMax         = dist - RT_EPSILON;
    shadowRay.contribution = mul3(mul3(mul3(fr, spotLe), cosSurface), pathRecord.throughput);
    return true;
}

// ---------------------------------------------------------------------------
//  Environment map hit: called when a BSDF-sampled ray MISSES all geometry.
//  Applies the MIS BRDF weight (balance heuristic) to the env map radiance.
//  For specular bounces, weight = 1 (BRDF PDF is a delta, env PDF is finite).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void accumulateEnvMapHit(
    PathState& pathRecord,
    const DirectLightingContext& lightingCtx)
{
    if (!lightingCtx.hasEnvMap) return;

    const Float3 envRadiance = envMapEval(lightingCtx.envMap, pathRecord.ray.direction);

    float brdfWeight = 1.0f;

    if (!pathRecord.specularBounce) {
        const float envPdf = envMapPdf(lightingCtx.envMap, pathRecord.ray.direction);
        const float sumPdf = pathRecord.brdfPDF + envPdf;
        if (sumPdf > 0.0f)
            brdfWeight = pathRecord.brdfPDF / sumPdf;
        else
            brdfWeight = 0.0f;
    }

    pathRecord.accumulatedColor = add3(
        pathRecord.accumulatedColor,
        mul3(mul3(envRadiance, pathRecord.throughput), brdfWeight));
}

// ---------------------------------------------------------------------------
//  NEE for the environment map: importance-sample a direction from the env
//  map CDF, evaluate the BSDF, apply MIS light weight, and fill the shadow
//  ray.  The caller traces the shadow ray to test visibility.
//  Returns false if the sample carries no energy (early exit).
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool prepareEnvMapNEE(
    PathState& pathRecord,
    RngState& rng,
    const Intersection& its,
    const BsdfData& bsdf,
    const DirectLightingContext& lightingCtx,
    ShadowRayRecord& shadowRay)
{
    if (!lightingCtx.hasEnvMap) return false;

    const float u0 = rngNextFloat01(rng);
    const float u1 = rngNextFloat01(rng);

    EnvMapSampleResult envSample = envMapSample(lightingCtx.envMap, u0, u1);

    if (envSample.pdf <= 0.0f) return false;
    if (envSample.radiance.x <= 0.0f && envSample.radiance.y <= 0.0f && envSample.radiance.z <= 0.0f)
        return false;

    // Check that the sampled direction is on the correct hemisphere
    const float cosAtSurface = dot3(its.hitNormal, envSample.direction);
    if (cosAtSurface <= 0.0f) return false;

    // Evaluate BSDF for the sampled env map direction
    BsdfQueryRecord bsdfQuery{};
    bsdfQuery.wi      = toLocalFromNormal(its.hitNormal, mul3(pathRecord.ray.direction, -1.0f));
    bsdfQuery.wo      = toLocalFromNormal(its.hitNormal, envSample.direction);
    bsdfQuery.measure = BSDF_ESolidAngle;

    const Float3 fr = bsdfEval(bsdf, bsdfQuery);

    // MIS: balance heuristic — weight = lightPdf / (lightPdf + brdfPdf)
    const float brdfPdf = bsdfPdf(bsdf, bsdfQuery);
    const float sumPdf  = brdfPdf + envSample.pdf;
    if (sumPdf <= 0.0f) return false;
    const float lightWeight = envSample.pdf / sumPdf;

    // contribution = fr * Le * cos(θ) / envPdf * weight * throughput
    Float3 contrib = mul3(fr, envSample.radiance);
    contrib = mul3(contrib, cosAtSurface);
    contrib = div3(contrib, envSample.pdf);
    contrib = mul3(contrib, lightWeight);

    shadowRay.origin       = its.hitPoint;
    shadowRay.direction    = envSample.direction;
    shadowRay.tMax         = 1e30f - RT_EPSILON;   // env map is infinitely far away
    shadowRay.contribution = mul3(contrib, pathRecord.throughput);
    return true;
}
