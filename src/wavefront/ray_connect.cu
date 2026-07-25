// ============================================================================
//  ray_connect.cu — Wavefront Stage 4: Shadow / Occlusion Connection (OptiX raygen)
// ============================================================================
//  COMPILATION: Compiled to PTX as part of the `optix_programs` OBJECT library
//  (same module as ray_extend.cu and optix_programs.cu).
//
//  The host-side optixLaunch for this stage lives in
//  src/host/wavefront_host.cu (cudaRenderWavefront), using the wavefront SBT
//  record  s_sbt_wf_shadow.
//
//  What this stage does
//  ────────────────────
//  There is NO shadow index queue: shade hands out contiguous slots via
//  atomicAdd(wf.shadowCount, 1), so the launch index IS the shadow slot.
//
//  For each shadow slot (launch index in [0, shadowCount)):
//    1. Read the shadow ray:   org  = params.wf.shadowOrigin[sIdx]
//                              dir  = params.wf.shadowDir[sIdx]
//                              tMax = params.wf.shadowTMax[sIdx]
//    2. Test occlusion with the shared traceOccluded() from
//       rt_optix_shading.cuh (terminate-on-first-hit, RAY_TYPE_SHADOW;
//       __miss__shadow clears the payload — same helper the megakernel's NEE
//       wrappers use).
//    3. If NOT occluded, add the precomputed contribution to the owning
//       path's radiance.  atomicAdd is required: one path can own several
//       in-flight shadow rays (area NEE + one per spotlight):
//           const uint32_t pathIdx = params.wf.shadowPathIdx[sIdx];
//           atomicAdd(&params.wf.radiance[pathIdx].x, contrib.x);
//           atomicAdd(&params.wf.radiance[pathIdx].y, contrib.y);
//           atomicAdd(&params.wf.radiance[pathIdx].z, contrib.z);
//
//  wf.shadowContrib[sIdx] is already MIS-weighted and throughput-multiplied
//  (see prepareAreaEmitterNEE / prepareSpotlightNEE in rt_direct_lighting.cuh)
//  — this stage does no shading math at all.
//
//  See docs/WAVEFRONT_GUIDE.md §"Stage 4 – Connect" for full algorithm details.
// ============================================================================

#include <optix.h>
#include "optix_launch_params.h"
#include "rt_cuda_math.cuh"
#include "rt_optix_shading.cuh"    // traceOccluded (+ toF3)

// The launch-params constant is defined once in optix_programs.cu; we just
// re-declare it here so this TU can access it.
extern "C" __constant__ LaunchParams params;

// ---------------------------------------------------------------------------
//  __raygen__wf_shadow
//  Launched with width = activeShadowCount, height = 1, depth = 1.
// ---------------------------------------------------------------------------
extern "C" __global__ void __raygen__wf_shadow()
{
    const uint32_t sIdx = optixGetLaunchIndex().x;   // launch index == shadow slot

    // read the shadow ray from the SoA
    const Float3 origin = params.wf.shadowOrigin[sIdx];
    const Float3 direction = params.wf.shadowDir[sIdx];
    const float tMax = params.wf.shadowTMax[sIdx];
    const Float3 contribution = params.wf.shadowContrib[sIdx];
    const uint32_t pathIdx = params.wf.shadowPathIdx[sIdx];

    // occlusion test via the shared helper (returns true if blocked)
    if (traceOccluded(params.handle, origin, direction, tMax)) return;

    // add the contribution to the owning path
    atomicAdd(&params.wf.radiance[pathIdx].x, contribution.x);
    atomicAdd(&params.wf.radiance[pathIdx].y, contribution.y);
    atomicAdd(&params.wf.radiance[pathIdx].z, contribution.z);

    (void)sIdx;
}
