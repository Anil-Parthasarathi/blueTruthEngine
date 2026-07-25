// ============================================================================
//  ray_extend.cu — Wavefront Stage 2: Ray Extension  (OptiX raygen)
// ============================================================================
//  COMPILATION: This file is compiled to PTX as part of the `optix_programs`
//  OBJECT library (see CMakeLists.txt).  It is loaded at runtime alongside
//  optix_programs.cu via the single embedded PTX array.
//
//  The host-side optixLaunch call for this stage lives in
//  src/host/wavefront_host.cu (cudaRenderWavefront), using the wavefront SBT
//  record  s_sbt_wf_extend.
//
//  What this stage does
//  ────────────────────
//  For each slot in the current ray queue (launch index = queue slot):
//    1. Read the path-slot index:     idx = params.wf.rayQueue[launchIdx.x]
//    2. Read the ray from the SoA:    params.wf.rayOrigin[idx] / rayDir[idx]
//    3. Trace a closest-hit ray with the shared traceClosest() from
//       rt_optix_shading.cuh — the same helper the megakernel's traceRay uses.
//       It handles the payload-pointer packing and fills an Intersection.
//    4. Write hit results into the hit buffer so wfShade can read them:
//           params.wf.hitTriIndex[idx] = its.triangleIndex;   // -1 == miss
//           params.wf.hitPoint[idx]    = its.hitPoint;
//           params.wf.hitNormal[idx]   = its.hitNormal;       // interpolated
//           params.wf.hitUV[idx]       = its.uv;              // interpolated
//       (No barycentrics: __closesthit__radiance already consumes them to
//       interpolate hitNormal and uv, so the SoA does not store them.)
//
//  The existing __closesthit__radiance and __miss__radiance programs (in
//  optix_programs.cu) handle the payload write, so you do not need new hit/miss
//  programs — traceClosest uses RAY_TYPE_RADIANCE and the shared SBT records.
//
//  See docs/WAVEFRONT_GUIDE.md §"Stage 2 – Extend" for full details.
// ============================================================================

#include <optix.h>
#include "optix_launch_params.h"   // LaunchParams → params.wf (WavefrontSoA)
#include "rt_intersect.cuh"        // Ray / Intersection structs
#include "rt_cuda_math.cuh"
#include "rt_optix_shading.cuh"    // traceClosest (+ packPointer/unpackPointer/toF3)

// The launch-params constant is defined once in optix_programs.cu; we just
// re-declare it here so this TU can access it.
extern "C" __constant__ LaunchParams params;

// ---------------------------------------------------------------------------
//  __raygen__wf_extend
//  Launched with width = activeRayCount, height = 1, depth = 1.
// ---------------------------------------------------------------------------
extern "C" __global__ void __raygen__wf_extend()
{
    const uint32_t queueSlot = optixGetLaunchIndex().x;

    // get path slot from the ray queue
    const int idx = params.wf.rayQueue[queueSlot];

    // read the ray from the SoA
    Ray extendRay;
    extendRay.origin = params.wf.rayOrigin[idx];
    extendRay.direction = params.wf.rayDir[idx];

    // trace the ray 
    Intersection its;
    traceClosest(params.handle, extendRay, its);

    // write hit results to the hit buffer (miss ⇒ its.triangleIndex == -1)
    params.wf.hitTriIndex[idx] = its.triangleIndex;
    params.wf.hitPoint[idx]    = its.hitPoint;
    params.wf.hitNormal[idx]   = its.hitNormal;
    params.wf.hitUV[idx]       = its.uv;

    (void)queueSlot;
}
