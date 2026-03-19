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

