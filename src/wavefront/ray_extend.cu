// ============================================================================
//  ray_extend.cu — Wavefront Stage 2: Ray Extension  (OptiX raygen)
// ============================================================================
//  COMPILATION: This file is compiled to PTX as part of the `optix_programs`
//  OBJECT library (see CMakeLists.txt).  It is loaded at runtime alongside
//  optix_programs.cu via the single embedded PTX array.
//
//  The host-side optixLaunch call for this stage lives in render_kernel.cu
//  (function cudaRenderWavefront) because it needs access to the static OptiX
//  pipeline and the wavefront SBT record  s_sbt_wf_extend.
//
//  What this stage does
//  ────────────────────
//  For each slot in the current ray queue (launch index = queue slot):
//    1. Read the path-slot index:     idx = params.wf.rayQueue[launchIdx.x]
//    2. Read the ray from the SoA:    org = params.wf.rayOrigin[idx]
//                                     dir = params.wf.rayDir[idx]
//    3. Trace a closest-hit ray with optixTrace (RT cores).
//       Use the same payload-pointer trick as traceClosest() in optix_programs.cu.
//    4. Write hit results into the hit buffer so wfShade can read them:
//           params.wf.hitTriIndex[idx] = its.triangleIndex;  // -1 == miss
//           params.wf.hitBaryU[idx]    = its.baryU;
//           params.wf.hitBaryV[idx]    = its.baryV;
//           params.wf.hitPoint[idx]    = its.hitPoint;
//           params.wf.hitNormal[idx]   = its.hitNormal;
//           params.wf.hitUV[idx]       = its.hitUV;
//
//  The existing __closesthit__radiance and __miss__radiance programs (in
//  optix_programs.cu) handle the payload write, so you do not need new hit/miss
//  programs — just launch with RAY_TYPE_RADIANCE and the same SBT hit/miss
//  records as the megakernel.
//
//  See docs/WAVEFRONT_GUIDE.md §"Stage 2 – Extend" for full details.
// ============================================================================

#include <optix.h>
#include "optix_launch_params.h"   // LaunchParams → params.wf (WavefrontSoA)
#include "rt_intersect.cuh"        // Intersection struct
#include "rt_cuda_math.cuh"

// The launch-params constant is defined once in optix_programs.cu; we just
// re-declare it here so this TU can access it.
extern "C" __constant__ LaunchParams params;

// Pointer packing helpers — identical to the ones in optix_programs.cu.
// Because both files compile into the same PTX module these can be static
// __forceinline__ without conflict.
static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long uptr =
        (static_cast<unsigned long long>(i0) << 32) | static_cast<unsigned long long>(i1);
    return reinterpret_cast<void*>(uptr);
}
static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = static_cast<unsigned int>(uptr >> 32);
    i1 = static_cast<unsigned int>(uptr & 0xffffffffull);
}

static __forceinline__ __device__ float3 toF3(const Float3& v)
{ return make_float3(v.x, v.y, v.z); }

static constexpr float RT_EPSILON_EXTEND = 1e-4f;

// ---------------------------------------------------------------------------
//  __raygen__wf_extend
//  Launched with width = activeRayCount, height = 1, depth = 1.
// ---------------------------------------------------------------------------
extern "C" __global__ void __raygen__wf_extend()
{
    const uint32_t queueSlot = optixGetLaunchIndex().x;

    // TODO: get path slot from the ray queue
    //   int idx = params.wf.rayQueue[queueSlot];

    // TODO: read ray from SoA
    //   Float3 org = params.wf.rayOrigin[idx];
    //   Float3 dir = params.wf.rayDir[idx];

    // TODO: allocate an Intersection on the stack, pack pointer, call optixTrace.
    //   Intersection its{};
    //   unsigned int p0, p1;
    //   packPointer(&its, p0, p1);
    //   optixTrace(
    //       params.handle,
    //       toF3(org), toF3(dir),
    //       RT_EPSILON_EXTEND, 1e16f, 0.f,
    //       255u, OPTIX_RAY_FLAG_NONE,
    //       RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE,
    //       p0, p1);

    // TODO: write hit results to the hit buffer
    //   params.wf.hitTriIndex[idx] = its.triangleIndex;
    //   params.wf.hitBaryU[idx]    = its.baryU;
    //   params.wf.hitBaryV[idx]    = its.baryV;
    //   params.wf.hitPoint[idx]    = its.hitPoint;
    //   params.wf.hitNormal[idx]   = its.hitNormal;
    //   params.wf.hitUV[idx]       = its.hitUV;

    (void)queueSlot;
}
