#pragma once

// ---------------------------------------------------------------------------
//  rt_camera.cuh — primary-ray generation shared by the megakernel raygen
//  (__raygen__rg) and the wavefront generate stage (wfGenerate).
// ---------------------------------------------------------------------------

#include "render_kernel.h"
#include "rt_cuda_math.cuh"
#include "rt_intersect.cuh"   // Ray

// jx, jy are sub-pixel offsets in [-0.5, 0.5] drawn from the per-path RNG.
// With temporal accumulation each path sees a different jitter position,
// giving both antialiasing and reduced variance as samples pile up.
__device__ __forceinline__ Ray generatePrimaryRay(int px, int py, int width, int height,
                                                   const CameraData& camera,
                                                   float jx, float jy)
{
    const float fx = (static_cast<float>(px) + 0.5f + jx) / static_cast<float>(width);
    const float fy = (static_cast<float>(py) + 0.5f + jy) / static_cast<float>(height);
    const float ndcX = 2.0f * fx - 1.0f;
    // Flip Y so the image isn't vertically inverted on display.
    const float ndcY = 2.0f * fy - 1.0f;

    const float tanHalfFovY = tanf(0.5f * camera.fovYRadians);
    const float sx = ndcX * camera.aspect * tanHalfFovY;
    const float sy = ndcY * tanHalfFovY;

    Float3 dir = add3(camera.forward,
                      add3(mul3(camera.right, sx),
                           mul3(camera.up, sy)));

    Ray ray;
    ray.origin    = camera.origin;
    ray.direction = normalize3(dir);
    return ray;
}
