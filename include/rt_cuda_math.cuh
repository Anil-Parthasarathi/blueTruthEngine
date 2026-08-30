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

__device__ __forceinline__ float maxCoeff3(const Float3& v)
{
    return fmaxf(v.x, fmaxf(v.y, v.z));
}

// ---------------------------------------------------------------------------
// Object-to-world transforms (row-major 3x4 affine — see ObjectTransform)
// ---------------------------------------------------------------------------

// Full affine transform of a position.
__device__ __forceinline__ Float3 transformPoint34(const ObjectTransform& t, const Float3& p)
{
    return {
        t.m[0] * p.x + t.m[1] * p.y + t.m[2]  * p.z + t.m[3],
        t.m[4] * p.x + t.m[5] * p.y + t.m[6]  * p.z + t.m[7],
        t.m[8] * p.x + t.m[9] * p.y + t.m[10] * p.z + t.m[11]
    };
}

// Transform a direction/normal by the inverse-transpose of the linear 3x3.
// Correct for non-uniform scale; for pure rotation it reduces to R*n.
__device__ __forceinline__ Float3 transformNormal34(const ObjectTransform& t, const Float3& n)
{
    // Inverse of 3x3 linear part via adjugate / det, then transpose-multiply
    // is equivalent to multiplying by the cofactor matrix (no divide by det
    // needed before normalize).
    const float a00 = t.m[0], a01 = t.m[1], a02 = t.m[2];
    const float a10 = t.m[4], a11 = t.m[5], a12 = t.m[6];
    const float a20 = t.m[8], a21 = t.m[9], a22 = t.m[10];

    // Cofactors of A^T = cofactors laid out transposed = columns of adj(A)
    const Float3 out = {
        (a11 * a22 - a12 * a21) * n.x + (a02 * a21 - a01 * a22) * n.y + (a01 * a12 - a02 * a11) * n.z,
        (a12 * a20 - a10 * a22) * n.x + (a00 * a22 - a02 * a20) * n.y + (a02 * a10 - a00 * a12) * n.z,
        (a10 * a21 - a11 * a20) * n.x + (a01 * a20 - a00 * a21) * n.y + (a00 * a11 - a01 * a10) * n.z
    };
    return normalize3(out);
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

