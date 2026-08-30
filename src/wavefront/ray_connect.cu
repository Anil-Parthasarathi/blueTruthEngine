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
#include "rt_optix_shading.cuh"                // traceOccluded (+ toF3)
#include "wavefront/wavefront_shading.cuh"     // atomicAddToChannel
#include "rt_constants.cuh"                    // RT_EPSILON

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

    if (params.wf.shadowOrigin == nullptr) return;
    if (static_cast<int>(sIdx) >= params.wf.maxShadows) return;

    // read the shadow ray from the SoA
    const Float3 origin = params.wf.shadowOrigin[sIdx];
    const Float3 direction = params.wf.shadowDir[sIdx];
    const float tMax = params.wf.shadowTMax[sIdx];
    if (tMax <= RT_EPSILON) return;
    const Float3 contribDiffuse = params.wf.shadowContrib[sIdx];
    const Float3 contribSpecular = params.wf.shadowContribSpec[sIdx];
    const uint32_t pathIdx = params.wf.shadowPathIdx[sIdx];
    const bool isPrimary = params.wf.shadowIsPrimary[sIdx] != 0;

    // occlusion test via the shared helper (returns true if blocked)
    if (traceOccluded(params.handle, origin, direction, tMax)) return;

    // Direct lighting at the primary hit is the signal the cel operators act on,
    // so its two halves converge in their own channels: the diffuse-ish half
    // becomes the banded body tone, the specular-ish half the hard highlight.
    // Deeper next-event estimation is indirect illumination arriving through
    // whatever lobe the primary hit sampled, so it collapses into that path's
    // channel and gets that surface's operator.
    //
    // In physical mode all channel pointers alias wf.radiance, so both branches
    // add into the same buffer and sum to exactly the physical estimate — which
    // is why this kernel needs no style-mode branch.
    if (isPrimary) {
        atomicAddToChannel(params.wf, STYLE_CH_DIRECT_DIFFUSE,  pathIdx, contribDiffuse);
        atomicAddToChannel(params.wf, STYLE_CH_DIRECT_SPECULAR, pathIdx, contribSpecular);
    } else {
        int ch = static_cast<int>(params.wf.styleChannel[pathIdx]);
        if (ch < 0 || ch >= STYLE_CH_COUNT) ch = STYLE_CH_INDIRECT_DIFFUSE;
        atomicAddToChannel(params.wf, ch, pathIdx, contribDiffuse);
        atomicAddToChannel(params.wf, ch, pathIdx, contribSpecular);
    }
}
