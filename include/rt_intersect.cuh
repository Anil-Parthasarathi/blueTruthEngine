#pragma once

// ---------------------------------------------------------------------------
//  rt_intersect.cuh — Ray / Intersection types shared by all render paths.
//
//  Traversal itself runs on the RT cores via OptiX (see rt_optix_shading.cuh);
//  the old CPU-BVH software traversal was removed together with the CUDA
//  megakernel.
// ---------------------------------------------------------------------------

#include "render_kernel.h"

struct Intersection {
    Float3 hitPoint  = {0.0f, 0.0f, 0.0f};
    Float3 hitNormal = {0.0f, 0.0f, 0.0f};
    Float2 uv        = {0.0f, 0.0f};  // barycentrically interpolated texture coordinate
    float  t         = -1.0f;
    int    triangleIndex = -1;
};

struct Ray {
    Float3 origin;
    Float3 direction;
};
