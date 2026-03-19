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

// Slab test for ray/AABB intersection.
// Pass precomputed invDir (1/ray.direction) to avoid repeated divisions.
__device__ __forceinline__ bool rayAABBIntersect(
    const Float3& ro, const Float3& invDir,
    const float* bmin, const float* bmax,
    float tMin, float tMax)
{
    float t0, t1;

    t0 = (bmin[0] - ro.x) * invDir.x;
    t1 = (bmax[0] - ro.x) * invDir.x;
    if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
    tMin = fmaxf(tMin, t0);
    tMax = fminf(tMax, t1);

    t0 = (bmin[1] - ro.y) * invDir.y;
    t1 = (bmax[1] - ro.y) * invDir.y;
    if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
    tMin = fmaxf(tMin, t0);
    tMax = fminf(tMax, t1);

    t0 = (bmin[2] - ro.z) * invDir.z;
    t1 = (bmax[2] - ro.z) * invDir.z;
    if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
    tMin = fmaxf(tMin, t0);
    tMax = fminf(tMax, t1);

    return tMin <= tMax;
}

// Any-hit variant — returns true as soon as one blocker is found.
// Use this for shadow rays: no need to find the closest hit.
__device__ __forceinline__ bool sceneIntersectAnyHit(
    const Ray& ray,
    const TriangleData* tris,
    const LinearBVHNode* bvhNodes, int bvhNodeCount,
    const int* bvhPrimIndices,
    float tMin, float tMax)
{
    const Float3 invDir = { 1.0f / ray.direction.x,
                             1.0f / ray.direction.y,
                             1.0f / ray.direction.z };

    int stack[64];
    int stackTop = 0;
    stack[stackTop++] = 0;

    while (stackTop > 0) {
        const int idx = stack[--stackTop];
        if (idx < 0 || idx >= bvhNodeCount) continue;

        const LinearBVHNode& node = bvhNodes[idx];

        if (!rayAABBIntersect(ray.origin, invDir, node.bmin, node.bmax, tMin, tMax))
            continue;

        if (node.primCount > 0) {
            for (uint32_t p = node.primStart; p < node.primStart + node.primCount; ++p) {
                float t, u, v;
                if (rayTriangleIntersect(ray.origin, ray.direction,
                                         tris[bvhPrimIndices[p]], tMin, tMax, t, u, v))
                    return true; // blocked — no need to go further
            }
        } else {
            if (stackTop < 63) stack[stackTop++] = idx + (int)node.rightOffset;
            if (stackTop < 63) stack[stackTop++] = idx + 1;
        }
    }

    return false;
}

__device__ __forceinline__ bool sceneIntersect(
    const Ray& ray,
    const TriangleData* tris, int /*triCount*/,
    const LinearBVHNode* bvhNodes, int bvhNodeCount,
    const int* bvhPrimIndices,
    float tMin, float tMax,
    Intersection& out)
{
    out.hitPoint      = {0.0f, 0.0f, 0.0f};
    out.hitNormal     = {0.0f, 0.0f, 0.0f};
    out.t             = tMax;
    out.triangleIndex = -1;

    const Float3 invDir = { 1.0f / ray.direction.x,
                             1.0f / ray.direction.y,
                             1.0f / ray.direction.z };

    bool  hit     = false;
    float closest = tMax;

    // Iterative BVH traversal using a small per-thread stack.
    int stack[64];
    int stackTop = 0;
    stack[stackTop++] = 0; // root

    while (stackTop > 0) {
        const int idx = stack[--stackTop];
        if (idx < 0 || idx >= bvhNodeCount) continue;

        const LinearBVHNode& node = bvhNodes[idx];

        if (!rayAABBIntersect(ray.origin, invDir, node.bmin, node.bmax, tMin, closest))
            continue;

        if (node.primCount > 0) {
            // Leaf — test each triangle
            for (uint32_t p = node.primStart; p < node.primStart + node.primCount; ++p) {
                const int triIdx = bvhPrimIndices[p];
                float t, u, v;
                if (!rayTriangleIntersect(ray.origin, ray.direction, tris[triIdx], tMin, closest, t, u, v))
                    continue;

                hit     = true;
                closest = t;

                const TriangleData& tri = tris[triIdx];
                out.hitPoint      = add3(ray.origin, mul3(ray.direction, t));
                out.t             = t;
                out.triangleIndex = triIdx;

                // Interpolate per-vertex normals for smooth shading.
                const float w0 = 1.0f - u - v;
                out.hitNormal = normalize3({
                    w0 * tri.n0.x + u * tri.n1.x + v * tri.n2.x,
                    w0 * tri.n0.y + u * tri.n1.y + v * tri.n2.y,
                    w0 * tri.n0.z + u * tri.n1.z + v * tri.n2.z
                });
            }
        } else {
            // Interior — push right child first so left is popped first
            if (stackTop < 63) stack[stackTop++] = idx + (int)node.rightOffset;
            if (stackTop < 63) stack[stackTop++] = idx + 1;
        }
    }

    return hit;
}

