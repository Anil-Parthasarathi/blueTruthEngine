#pragma once

#include "render_kernel.h"
#include "rt_cuda_math.cuh"

#include <cmath>

struct Intersection {
    Float3 hitPoint = {0.0f, 0.0f, 0.0f};
    Float3 hitNormal = {0.0f, 0.0f, 0.0f};
    float t = -1.0f;
    int triangleIndex = -1;
};

struct Ray {
    Float3 origin;
    Float3 direction;
};

// Möller–Trumbore ray/triangle intersection.
__device__ __forceinline__ bool rayTriangleIntersect(
    const Float3& ro, const Float3& rd,
    const TriangleData& tri,
    float tMin, float tMax,
    float& t, float& u, float& v)
{
    const Float3 e1 = sub3(tri.v1, tri.v0);
    const Float3 e2 = sub3(tri.v2, tri.v0);
    const Float3 pvec = cross3(rd, e2);
    const float det = dot3(e1, pvec);

    const float eps = 1e-8f;
    if (fabsf(det) < eps) return false;

    const float invDet = 1.0f / det;
    const Float3 tvec = sub3(ro, tri.v0);

    u = dot3(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    const Float3 qvec = cross3(tvec, e1);

    v = dot3(rd, qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) return false;

    const float tHit = dot3(e2, qvec) * invDet;
    if (tHit < tMin || tHit > tMax) return false;

    t = tHit;
    return true;
}

__device__ __forceinline__ bool sceneIntersect(
    const Ray& ray,
    const TriangleData* tris, int triCount,
    float tMin, float tMax,
    Intersection& out)
{
    bool hit = false;
    float closest = tMax;

    out.hitPoint = {0.0f, 0.0f, 0.0f};
    out.hitNormal = {0.0f, 0.0f, 0.0f};
    out.t = tMax;
    out.triangleIndex = -1;

    for (int i = 0; i < triCount; ++i) {
        float t, u, v;
        if (!rayTriangleIntersect(ray.origin, ray.direction, tris[i], tMin, closest, t, u, v))
            continue;

        hit = true;
        closest = t;

        const TriangleData& tri = tris[i];
        out.hitPoint = add3(ray.origin, mul3(ray.direction, t));
        out.hitNormal = normalize3(cross3(sub3(tri.v1, tri.v0), sub3(tri.v2, tri.v0)));
        out.t = t;
        out.triangleIndex = i;
    }

    return hit;
}

