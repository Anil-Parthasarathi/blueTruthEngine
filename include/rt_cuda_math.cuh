#pragma once

#include "render_kernel.h" // Float3, TriangleData

#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ Float3 sub3(const Float3& a, const Float3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __forceinline__ Float3 add3(const Float3& a, const Float3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __forceinline__ Float3 mul3(const Float3& a, float s)
{
    return {a.x * s, a.y * s, a.z * s};
}

__device__ __forceinline__ Float3 mul3(const Float3& a, const Float3& b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ __forceinline__ Float3 div3(const Float3& a, float s)
{
    if (s == 0.0f) return {0.0f, 0.0f, 0.0f};
    const float inv = 1.0f / s;
    return {a.x * inv, a.y * inv, a.z * inv};
}

__device__ __forceinline__ float dot3(const Float3& a, const Float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ Float3 cross3(const Float3& a, const Float3& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__device__ __forceinline__ Float3 normalize3(const Float3& v)
{
    const float len2 = dot3(v, v);
    if (len2 <= 0.0f) return {0.0f, 0.0f, 0.0f};
    const float invLen = rsqrtf(len2);
    return mul3(v, invLen);
}

__device__ __forceinline__ float length3(const Float3& v)
{
    return sqrtf(fmaxf(0.0f, dot3(v, v)));
}

__device__ __forceinline__ float distance3(const Float3& a, const Float3& b)
{
    return length3(sub3(a, b));
}

// ---------------------------------------------------------------------------
// Local shading frame helpers (+Z aligns with the provided normal)
// ---------------------------------------------------------------------------

struct Frame3 {
    Float3 t; // tangent
    Float3 b; // bitangent
    Float3 n; // normal (local +Z)
};

__device__ __forceinline__ Frame3 makeFrameFromNormal(const Float3& normal)
{
    Frame3 f{};
    f.n = normalize3(normal);

    // Pick a helper vector that is not (nearly) parallel to n to avoid
    // degeneracy in the cross product.
    const Float3 a = (fabsf(f.n.z) < 0.999f) ? Float3{0.0f, 0.0f, 1.0f}
                                            : Float3{1.0f, 0.0f, 0.0f};

    f.t = normalize3(cross3(a, f.n));
    f.b = cross3(f.n, f.t);
    return f;
}

// Convert a world-space direction into local shading space.
// Assumes local +Z corresponds to the frame normal.
__device__ __forceinline__ Float3 toLocal(const Frame3& f, const Float3& vWorld)
{
    return {dot3(vWorld, f.t), dot3(vWorld, f.b), dot3(vWorld, f.n)};
}

// Convert a local shading direction into world space.
__device__ __forceinline__ Float3 toWorld(const Frame3& f, const Float3& vLocal)
{
    return add3(add3(mul3(f.t, vLocal.x), mul3(f.b, vLocal.y)), mul3(f.n, vLocal.z));
}

// Convenience wrappers: build the frame from a normal automatically.
__device__ __forceinline__ Float3 toLocalFromNormal(
    const Float3& normal, const Float3& vWorld)
{
    Frame3 f = makeFrameFromNormal(normal);
    return toLocal(f, vWorld);
}

__device__ __forceinline__ Float3 toWorldFromNormal(
    const Float3& normal, const Float3& vLocal)
{
    Frame3 f = makeFrameFromNormal(normal);
    return toWorld(f, vLocal);
}

