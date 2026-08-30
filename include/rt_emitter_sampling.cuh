#pragma once

#include "render_kernel.h"
#include "rt_cuda_math.cuh"

#include <cmath>

struct MeshSample {
    Float3 position;
    Float3 normal;
    float  pdf;          // area PDF (1 / totalArea) when sampling proportional to area
    int    triangleIndex;
};

struct EmitterQueryRecord {
    Float3 originPoint;
    Float3 hitPoint;
    Float3 hitNormal;
    float  pdf;
    float  emitterPdf;
    Float3 directionToLight;
};

struct EmitterSamplingData {
    const TriangleData* triangles;
    const EmitterData* emitters;
    int emitterCount;
    const int* emitterTriIndices;
    const float* emitterTriCdf;
    const float* sceneEmitterCdf;

    // The environment light joins the emitter selection as one extra virtual
    // slot rather than being bolted on beside it, so a single discrete draw
    // still chooses among every light in the scene and the MIS weights stay
    // consistent.  `selectCount` is the number of weights in sceneEmitterCdf:
    // emitterCount normally, emitterCount + 1 when the environment is present.
    // `envSlot` is that extra index, or -1 when there is no environment.
    //
    // Keeping these as plain ints (rather than an EnvLightData pointer) keeps
    // this header free of any environment dependency: the actual environment
    // sampling lives in rt_direct_lighting.cuh.
    int selectCount;
    int envSlot;
};

// Number of selectable emitters, environment included.
__device__ __forceinline__ int emitterSelectCount(const EmitterSamplingData& s)
{
    return (s.selectCount > 0) ? s.selectCount : s.emitterCount;
}

__device__ __forceinline__ int sampleCdfReuse(
    const float* cdf, int count, float& u, float& outPdf)
{
    const float sum = cdf[count];
    if (sum <= 0.0f || count <= 0) {
        outPdf = 0.0f;
        u = 0.0f;
        return 0;
    }

    float x = u * sum;

    int lo = 0, hi = count;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (cdf[mid] <= x) lo = mid;
        else hi = mid;
    }
    int idx = lo;

    float c0 = cdf[idx];
    float c1 = cdf[idx + 1];
    float denom = c1 - c0;
    if (denom <= 0.0f) {
        u = 0.0f;
        outPdf = 0.0f;
        return idx;
    }

    u = (x - c0) / denom;
    outPdf = denom / sum;
    return idx;
}

__device__ __forceinline__ MeshSample sampleMeshEmitter(
    const TriangleData* triangles,
    const int* emitterTriangleIndices,
    const float* emitterTriangleCdf,
    int emitterTriangleCount,
    float u0, float u1)
{
    MeshSample result{};
    result.position = {0.0f, 0.0f, 0.0f};
    result.normal   = {0.0f, 0.0f, 0.0f};
    result.pdf      = 0.0f;
    result.triangleIndex = -1;

    if (emitterTriangleCount <= 0) return result;

    float triPickPdf = 0.0f;
    int localIdx = sampleCdfReuse(emitterTriangleCdf, emitterTriangleCount, u0, triPickPdf);
    int triIdx = emitterTriangleIndices[localIdx];
    const TriangleData tri = triangles[triIdx];

    float rootTerm = sqrtf(fmaxf(0.0f, 1.0f - u0));
    float alpha = 1.0f - rootTerm;
    float beta  = u1 * rootTerm;
    float gamma = 1.0f - alpha - beta;

    result.position = add3(add3(mul3(tri.v0, alpha), mul3(tri.v1, beta)), mul3(tri.v2, gamma));
    result.normal = normalize3(cross3(sub3(tri.v1, tri.v0), sub3(tri.v2, tri.v0)));

    const float totalArea = emitterTriangleCdf[emitterTriangleCount];
    if (totalArea > 0.0f) result.pdf = 1.0f / totalArea;

    result.triangleIndex = triIdx;
    return result;
}

struct RandomEmitterSample {
    int emitterIndex;
    float emitterPdf;
    MeshSample mesh;
};

// Result of the single discrete draw that chooses which light to sample.
struct EmitterSelection {
    int   slot;       // mesh emitter index, or envSlot, or -1 if nothing selectable
    float selectPdf;  // discrete probability of having chosen this slot
    bool  isEnv;      // true when `slot` is the virtual environment slot
};

// Choose one light (mesh emitter or environment) proportional to the scene
// emitter CDF.  `u` is remapped in place to the position within the chosen bin,
// so the caller can reuse it for the light's own internal sampling without
// spending another RNG draw — the same CDF-reuse chaining sampleMeshEmitter
// already relies on.
__device__ __forceinline__ EmitterSelection selectSceneEmitter(
    const EmitterSamplingData& sampling, float& u)
{
    EmitterSelection sel{};
    sel.slot      = -1;
    sel.selectPdf = 0.0f;
    sel.isEnv     = false;

    const int n = emitterSelectCount(sampling);
    if (n <= 0 || !sampling.sceneEmitterCdf) return sel;

    float pdf = 0.0f;
    const int slot = sampleCdfReuse(sampling.sceneEmitterCdf, n, u, pdf);
    if (pdf <= 0.0f) return sel;

    sel.slot      = slot;
    sel.selectPdf = pdf;
    sel.isEnv     = (sampling.envSlot >= 0 && slot == sampling.envSlot);
    return sel;
}

__device__ __forceinline__ RandomEmitterSample sampleRandomEmitter(
    const EmitterSamplingData& sampling,
    float u0, float u1);

__device__ __forceinline__ Float3 checkRadiance(
    const Float3& radiance,
    const EmitterQueryRecord& emitterQuery)
{
    const Float3 negDir = {-emitterQuery.directionToLight.x,
                           -emitterQuery.directionToLight.y,
                           -emitterQuery.directionToLight.z};
    const float cosTheta = dot3(emitterQuery.hitNormal, negDir);

    if (cosTheta <= 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }
    return radiance;
}

__device__ __forceinline__ float probabilityEvaluator(const EmitterQueryRecord& emitterQuery)
{
    return emitterQuery.pdf;
}

__device__ __forceinline__ int findEmitterIndexForTriangle(
    int triIdx,
    const EmitterData* emitters, int emitterCount,
    const int* emitterTriIndices)
{
    if (!emitters || !emitterTriIndices || emitterCount <= 0 || triIdx < 0)
        return -1;

    for (int e = 0; e < emitterCount; ++e) {
        const EmitterData& E = emitters[e];
        for (int j = 0; j < E.triCount; ++j) {
            if (emitterTriIndices[E.triIndexOffset + j] == triIdx)
                return e;
        }
    }
    return -1;
}

// Nori Scene::getEmitterPDF-style: discrete probability of selecting emitter `emitterIdx`.
__device__ __forceinline__ float sceneGetEmitterPDF(
    int emitterIdx,
    const float* sceneEmitterCdf, int emitterCount)
{
    if (emitterIdx < 0 || emitterCount <= 0 || !sceneEmitterCdf)
        return 0.0f;

    const float sum = sceneEmitterCdf[emitterCount];
    if (sum <= 0.0f)
        return 0.0f;

    const float w = sceneEmitterCdf[emitterIdx + 1] - sceneEmitterCdf[emitterIdx];
    return w / sum;
}

// Mesh area PDF (1 / total emitter area), matches uniform sampling in sampleMeshEmitter.
__device__ __forceinline__ float emitterProbabilityEvaluatorFromMesh(
    const EmitterData& emitter)
{
    if (emitter.areaSum <= 0.0f)
        return 0.0f;
    return 1.0f / emitter.areaSum;
}

// Sample a point on an ALREADY-CHOSEN mesh emitter and fill the query record.
// Split out from sampleGenerator so the environment can share the one discrete
// emitter-selection draw without duplicating it.
__device__ __forceinline__ Float3 sampleGeneratorForEmitter(
    const EmitterSamplingData& sampling,
    int emitterIndex,
    float selectPdf,
    EmitterQueryRecord& emitterQuery,
    float u0, float u1)
{
    emitterQuery.emitterPdf = 0.0f;
    if (emitterIndex < 0 || emitterIndex >= sampling.emitterCount || !sampling.emitters)
        return {0.0f, 0.0f, 0.0f};

    const EmitterData e = sampling.emitters[emitterIndex];
    const MeshSample mesh = sampleMeshEmitter(
        sampling.triangles,
        sampling.emitterTriIndices + e.triIndexOffset,
        sampling.emitterTriCdf + e.cdfOffset,
        e.triCount,
        u0, u1);

    if (mesh.triangleIndex < 0) return {0.0f, 0.0f, 0.0f};

    emitterQuery.hitPoint = mesh.position;
    emitterQuery.hitNormal = mesh.normal;
    emitterQuery.pdf = mesh.pdf;
    emitterQuery.directionToLight = normalize3(sub3(emitterQuery.hitPoint, emitterQuery.originPoint));
    emitterQuery.emitterPdf = selectPdf;

    return checkRadiance(e.radiance, emitterQuery);
}

__device__ __forceinline__ Float3 sampleGenerator(
    const EmitterSamplingData& sampling,
    EmitterQueryRecord& emitterQuery,
    float u0, float u1)
{
    emitterQuery.emitterPdf = 0.0f;
    if (sampling.emitterCount <= 0) return {0.0f, 0.0f, 0.0f};

    RandomEmitterSample chosen = sampleRandomEmitter(
        sampling,
        u0, u1);

    if (chosen.emitterIndex < 0) return {0.0f, 0.0f, 0.0f};

    emitterQuery.hitPoint = chosen.mesh.position;
    emitterQuery.hitNormal = chosen.mesh.normal;
    emitterQuery.pdf = chosen.mesh.pdf;
    emitterQuery.directionToLight = normalize3(sub3(emitterQuery.hitPoint, emitterQuery.originPoint));
    emitterQuery.emitterPdf = chosen.emitterPdf;

    const Float3 radiance = sampling.emitters[chosen.emitterIndex].radiance;
    return checkRadiance(radiance, emitterQuery);
}

__device__ __forceinline__ RandomEmitterSample sampleRandomEmitter(
    const EmitterSamplingData& sampling,
    float u0, float u1)
{
    RandomEmitterSample out{};
    out.emitterIndex = -1;
    out.emitterPdf = 0.0f;
    out.mesh = {};

    if (sampling.emitterCount <= 0) return out;

    const EmitterSelection sel = selectSceneEmitter(sampling, u0);
    if (sel.slot < 0 || sel.isEnv) return out;

    out.emitterIndex = sel.slot;
    out.emitterPdf = sel.selectPdf;

    const EmitterData e = sampling.emitters[sel.slot];
    out.mesh = sampleMeshEmitter(
        sampling.triangles,
        sampling.emitterTriIndices + e.triIndexOffset,
        sampling.emitterTriCdf + e.cdfOffset,
        e.triCount,
        u0, u1);

    return out;
}

