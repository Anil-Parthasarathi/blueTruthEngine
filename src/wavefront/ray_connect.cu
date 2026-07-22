// ============================================================================
//  ray_connect.cu — Wavefront Stage 4: Shadow / Occlusion Connection (OptiX raygen)
// ============================================================================
//  COMPILATION: Compiled to PTX as part of the `optix_programs` OBJECT library
//  (same module as ray_extend.cu and optix_programs.cu).
//
//  The host-side optixLaunch for this stage lives in render_kernel.cu
//  (cudaRenderWavefront), using the static SBT record  s_sbt_wf_shadow.
//
//  What this stage does
//  ────────────────────
//  For each slot in the current shadow queue (launch index = shadow queue slot):
//    1. Read the shadow-slot index: sIdx = params.wf.shadowQueue[launchIdx.x]
//    2. Read the shadow ray:   org  = params.wf.shadowOrigin[sIdx]
//                              dir  = params.wf.shadowDir[sIdx]
//                              tMax = params.wf.shadowTMax[sIdx]
//    3. Trace an any-hit / terminate-on-first-hit shadow ray:
//           OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
//           OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT     |
//           OPTIX_RAY_FLAG_DISABLE_ANYHIT
//       Use RAY_TYPE_SHADOW so __miss__shadow fires on a miss.
//       Payload register 0: set to 1u before trace (= occluded); __miss__shadow
//       sets it to 0u.
//    4. If NOT occluded (payload0 == 0u):
//           uint32_t pathIdx = params.wf.shadowPathIdx[sIdx];
//           atomicAdd(&params.wf.radiance[pathIdx].x, contrib.x);
//           atomicAdd(&params.wf.radiance[pathIdx].y, contrib.y);
//           atomicAdd(&params.wf.radiance[pathIdx].z, contrib.z);
//
//  The existing __miss__shadow program in optix_programs.cu already sets
//  payload0 = 0u on a miss, so no new miss program is needed.
//
//  See docs/WAVEFRONT_GUIDE.md §"Stage 4 – Connect" for full algorithm details.
// ============================================================================

#include <optix.h>
#include "optix_launch_params.h"
#include "rt_cuda_math.cuh"

extern "C" __constant__ LaunchParams params;

static __forceinline__ __device__ float3 toF3(const Float3& v)
{ return make_float3(v.x, v.y, v.z); }

static constexpr float RT_EPSILON_CONNECT = 1e-4f;

// ---------------------------------------------------------------------------
//  __raygen__wf_shadow
//  Launched with width = activeShadowCount, height = 1, depth = 1.
// ---------------------------------------------------------------------------
extern "C" __global__ void __raygen__wf_shadow()
{
    const uint32_t queueSlot = optixGetLaunchIndex().x;

    // TODO: get shadow slot index from the shadow queue
    //   int sIdx = params.wf.shadowQueue[queueSlot];

    // TODO: read shadow ray from SoA
    //   Float3   org     = params.wf.shadowOrigin[sIdx];
    //   Float3   dir     = params.wf.shadowDir[sIdx];
    //   float    tMax    = params.wf.shadowTMax[sIdx];
    //   Float3   contrib = params.wf.shadowContrib[sIdx];
    //   uint32_t pathIdx = params.wf.shadowPathIdx[sIdx];

    // TODO: trace shadow ray (any-hit / terminate on first hit).
    //   unsigned int occluded = 1u;   // assume occluded; miss program clears to 0
    //   optixTrace(
    //       params.handle,
    //       toF3(org), toF3(dir),
    //       RT_EPSILON_CONNECT, tMax, 0.f,
    //       255u,
    //       OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
    //       OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT     |
    //       OPTIX_RAY_FLAG_DISABLE_ANYHIT,
    //       RAY_TYPE_SHADOW, RAY_TYPE_COUNT, RAY_TYPE_SHADOW,
    //       occluded);

    // TODO: accumulate if unoccluded
    //   if (occluded == 0u) {
    //       atomicAdd(&params.wf.radiance[pathIdx].x, contrib.x);
    //       atomicAdd(&params.wf.radiance[pathIdx].y, contrib.y);
    //       atomicAdd(&params.wf.radiance[pathIdx].z, contrib.z);
    //   }

    (void)queueSlot;
}
