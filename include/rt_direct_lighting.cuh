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
#include "rt_environment.cuh"
#include "rt_constants.cuh"
#include "rt_shading_context.cuh"

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
            // Note the selection probability uses emitterSelectCount, not
            // emitterCount: when an environment light is present it occupies an
            // extra slot in the same CDF, which lowers every mesh emitter's
            // chance of being picked.  Using the wrong count here would
            // overestimate the light pdf and darken emitters.
            const float lightPDFAreaBRDF =
                emitterProbabilityEvaluatorFromMesh(es.emitters[emitterIdx]) *
                sceneGetEmitterPDF(emitterIdx, es.sceneEmitterCdf, emitterSelectCount(es));

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
//  Accumulates the environment light along a ray that escaped the scene.
//
//  Mirrors accumulateEmitterHit's balance-heuristic weighting, but with
//  envPdf(dir) times the environment's selection probability standing in for the
//  mesh emitter's solid-angle pdf.  Keeping this in a shared header means the
//  megakernel and the wavefront pipeline weight escaped rays identically, so the
//  megakernel stays usable as the reference for environment MIS.
//
//  Camera rays (bounceCount == 0) read the BACKDROP instead of the lighting
//  environment.  A flat or painted backdrop over a physically sensible lighting
//  environment is the normal anime art direction, and separating the two costs
//  nothing.  Camera rays have specularBounce = true, so they take the full
//  contribution with no MIS weight, which is correct: no NEE strategy could
//  have generated them.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void accumulateEnvMiss(
    PathState& pathRecord,
    const DirectLightingContext& lightingCtx)
{
    const EnvLightData& env = lightingCtx.env;
    if (!env.enabled) return;

    const Float3 dir = pathRecord.ray.direction;

    if (pathRecord.bounceCount == 0) {
        const Float3 bg = envEvalBackground(env, dir);
        pathRecord.accumulatedColor =
            add3(pathRecord.accumulatedColor, mul3(bg, pathRecord.throughput));
        return;
    }

    const Float3 radiance = envEval(env, dir);

    float brdfWeight = 1.0f;

    const EmitterSamplingData& es = lightingCtx.emitterSampling;
    if (!pathRecord.specularBounce && es.envSlot >= 0) {
        const float selectPdf =
            sceneGetEmitterPDF(es.envSlot, es.sceneEmitterCdf, emitterSelectCount(es));
        const float lightPdf = envPdf(env, dir) * selectPdf;
        const float denom    = pathRecord.brdfPDF + lightPdf;

        if (denom <= 0.0f) brdfWeight = 0.0f;
        else               brdfWeight = pathRecord.brdfPDF / denom;
    }

    pathRecord.accumulatedColor = add3(
        pathRecord.accumulatedColor,
        mul3(mul3(radiance, pathRecord.throughput), brdfWeight));
}

// ---------------------------------------------------------------------------
//  Environment NEE for an ALREADY-SELECTED environment slot.
//
//  Works directly in solid angle (there is no emitter surface to convert from),
//  but follows the same contribution convention as the mesh path so the two
//  agree on any given surface:
//      contribution = fr * Le * |cos(n, wi)| * misWeight / pdf
//  with pdf including the discrete selection probability.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool prepareEnvNEE(
    PathState& pathRecord,
    const Intersection& its,
    const BsdfData& bsdf,
    const DirectLightingContext& lightingCtx,
    float selectPdf,
    float u0, float u1,
    ShadowRayRecord& shadowRay)
{
    const EnvLightData& env = lightingCtx.env;
    if (!env.enabled || selectPdf <= 0.0f) return false;

    const EnvSample sample = envSample(env, u0, u1);
    if (sample.pdf <= 0.0f) return false;
    if (sample.radiance.x <= 0.0f && sample.radiance.y <= 0.0f && sample.radiance.z <= 0.0f)
        return false;

    BsdfQueryRecord bsdfQueryEnv{};
    bsdfQueryEnv.wi      = toLocalFromNormal(its.hitNormal, mul3(pathRecord.ray.direction, -1.0f));
    bsdfQueryEnv.wo      = toLocalFromNormal(its.hitNormal, sample.direction);
    bsdfQueryEnv.measure = BSDF_ESolidAngle;

    Float3 frDiffuse, frSpecular;
    bsdfEvalSplit(bsdf, bsdfQueryEnv, frDiffuse, frSpecular);

    const float lightPdf = sample.pdf * selectPdf;
    const float brdfPdfEnv = bsdfPdf(bsdf, bsdfQueryEnv);
    const float denom      = brdfPdfEnv + lightPdf;
    if (denom <= 0.0f) return false;

    const float lightWeight = lightPdf / denom;
    const float cosSurface  = fabsf(dot3(its.hitNormal, sample.direction));

    // Everything except the BSDF, so the two halves share the work.
    Float3 common = mul3(sample.radiance, lightWeight * cosSurface);
    common = div3(common, lightPdf);
    common = mul3(common, pathRecord.throughput);

    shadowRay.origin           = its.hitPoint;
    shadowRay.direction        = sample.direction;
    shadowRay.tMax             = RT_ENV_TMAX;
    shadowRay.contribution     = mul3(frDiffuse,  common);
    shadowRay.contributionSpec = mul3(frSpecular, common);
    return true;
}

// ---------------------------------------------------------------------------
//  Mesh-emitter NEE for an ALREADY-SELECTED emitter slot.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool prepareMeshEmitterNEE(
    PathState& pathRecord,
    const Intersection& its,
    const BsdfData& bsdf,
    const DirectLightingContext& lightingCtx,
    int emitterIndex,
    float selectPdf,
    float u0, float u1,
    ShadowRayRecord& shadowRay)
{
    const EmitterSamplingData& es = lightingCtx.emitterSampling;

    EmitterQueryRecord emitterQuery{};
    emitterQuery.originPoint = its.hitPoint;

    Float3 le = sampleGeneratorForEmitter(es, emitterIndex, selectPdf, emitterQuery, u0, u1);

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

    // Split rather than summed, so direct lighting at the primary hit can feed
    // the cel body tone and the anime highlight as separate converged channels.
    // The two halves add up to exactly what bsdfEval would have returned.
    Float3 frDiffuse, frSpecular;
    bsdfEvalSplit(bsdf, bsdfQueryDirect, frDiffuse, frSpecular);

    const float lightPDFareaDirect       = emitterQuery.emitterPdf * emitterQuery.pdf;
    const float lightPDFSolidAngleDirect = convertAreaPDFtoSolidAnglePDF(lightPDFareaDirect, emitterQuery);
    const float brdfPDFDirect            = bsdfPdf(bsdf, bsdfQueryDirect);
    const float weightDenomSumDirect     = brdfPDFDirect + lightPDFSolidAngleDirect;

    if (weightDenomSumDirect <= 0.0f) return false;

    const float lightWeight = lightPDFSolidAngleDirect / weightDenomSumDirect;

    Float3 common = mul3(le, lightWeight);
    common = mul3(common, geo);
    common = div3(common, lightPDFareaDirect);
    common = mul3(common, pathRecord.throughput);

    const float tMax = distToLight - RT_EPSILON;
    // optixTrace with tmin > tmax is undefined and can illegal-access.
    // This happens when NEE samples a point on the same (or a very nearby)
    // emitter the path is already sitting on — e.g. camera rays that hit
    // the area light and then sample it again.
    if (tMax <= RT_EPSILON) return false;

    shadowRay.origin           = its.hitPoint;
    shadowRay.direction        = emitterQuery.directionToLight;
    shadowRay.tMax             = tMax;
    shadowRay.contribution     = mul3(frDiffuse,  common);
    shadowRay.contributionSpec = mul3(frSpecular, common);
    return true;
}

// ---------------------------------------------------------------------------
//  Samples ONE randomly chosen emitter using MIS and fills `shadowRay` with the
//  visibility test plus the throughput-multiplied contribution to add if the ray
//  is unoccluded.  Returns false when the sample carries no energy.
//
//  The environment light participates as a virtual slot in the same selection
//  CDF as the mesh emitters, so this one call covers every light in the scene
//  and there is no separate environment NEE strategy to keep in sync.
//
//  Only two RNG draws are used, unchanged from before the environment existed:
//  sampleCdfReuse remaps the uniform to its position within the chosen bin, and
//  the chosen light reuses it for its own internal sampling.
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

    float uLight0 = rngNextFloat01(rng);
    float uLight1 = rngNextFloat01(rng);

    const EmitterSelection sel = selectSceneEmitter(es, uLight0);
    if (sel.slot < 0) return false;

    if (sel.isEnv) {
        return prepareEnvNEE(pathRecord, its, bsdf, lightingCtx,
                             sel.selectPdf, uLight0, uLight1, shadowRay);
    }

    return prepareMeshEmitterNEE(pathRecord, its, bsdf, lightingCtx,
                                 sel.slot, sel.selectPdf, uLight0, uLight1, shadowRay);
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

    Float3 frDiffuse, frSpecular;
    bsdfEvalSplit(bsdf, bsdfQuerySpot, frDiffuse, frSpecular);

    // point light so we assume that random bounces never hit the spot light directly
    // contribution: fr * Le * cos_surface  (pdf = 1, no area term)
    const float cosSurface = fabsf(dot3(its.hitNormal, dirToLight));

    const Float3 common = mul3(mul3(spotLe, cosSurface), pathRecord.throughput);

    const float tMax = dist - RT_EPSILON;
    if (tMax <= RT_EPSILON) return false;

    // Shadow ray — bound to just before the light to avoid self-intersection
    shadowRay.origin           = its.hitPoint;
    shadowRay.direction        = dirToLight;
    shadowRay.tMax             = tMax;
    shadowRay.contribution     = mul3(frDiffuse,  common);
    shadowRay.contributionSpec = mul3(frSpecular, common);
    return true;
}
