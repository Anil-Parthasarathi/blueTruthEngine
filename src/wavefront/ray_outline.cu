// ============================================================================
//  ray_outline.cu — Wavefront outline probe stage (OptiX raygen)
// ============================================================================
//  COMPILATION: compiled to PTX in its own OBJECT library (see CMakeLists.txt),
//  loaded as its own OptiX module, and launched from
//  src/host/wavefront_host.cu with the SBT record s_sbt_wf_outline.
//
//  ANIME STYLE MODE ONLY.  The host skips this launch entirely in physical mode
//  and leaves wf.edgeFactor unallocated, so the photorealistic path pays nothing
//  — not a device branch, just an absent launch.
//
//  What this stage does
//  ────────────────────
//  Outlines here are not a screen-space filter.  They are found with rays, at
//  every path vertex, and applied as a modulation of path THROUGHPUT in wfShade.
//  That distinction is the whole point: a line found on a reflected path is a
//  real line on that path, so it appears correctly INSIDE a mirror, drawn from
//  the mirror's point of view rather than smeared from the camera's. Raster
//  cannot do this.
//
//  Per active path, after extend has recorded the centre hit:
//    1. Build an orthonormal basis around the ray direction.
//    2. Shoot OUTLINE_PROBE_COUNT probes at an angular offset of
//       lineWidth / hitDistance, so the line has a roughly constant screen
//       thickness regardless of how far away the surface is.
//    3. Classify each probe against the centre hit:
//         • different object ID          -> silhouette
//         • large relative depth jump    -> occluding contour
//         • normals diverge past a       -> crease
//           threshold
//    4. Write edge = hits / probes into wf.edgeFactor[idx].
//
//  Camera jitter antialiases the resulting lines across accumulation for free.
// ============================================================================

#include <optix.h>
#include "optix_launch_params.h"
#include "rt_intersect.cuh"
#include "rt_cuda_math.cuh"
#include "rt_constants.cuh"        // OUTLINE_MAX_DEPTH / OUTLINE_PROBE_COUNT
#include "rt_style.cuh"
#include "rt_optix_shading.cuh"    // traceClosest

// The launch-params constant is defined once per module; re-declare it here so
// this TU can access it.
extern "C" __constant__ LaunchParams params;

static constexpr float kOutlineTwoPi = 6.28318530717958647692f;

// Style record for the surface a probe centre landed on, or the neutral identity
// record when the material carries no style.
static __forceinline__ __device__ StyleData outlineStyleForTriangle(int triIdx)
{
    if (triIdx < 0 || params.bsdfs == nullptr || params.triangleBsdfIds == nullptr)
        return styleDataIdentity();

    const int bsdfId = params.triangleBsdfIds[triIdx];
    if (bsdfId < 0 || bsdfId >= params.bsdfCount) return styleDataIdentity();

    const int styleId = params.bsdfs[bsdfId].styleId;
    if (params.styles == nullptr || styleId < 0 || styleId >= params.styleCount)
        return styleDataIdentity();

    return params.styles[styleId];
}

static __forceinline__ __device__ int objectIdOf(int triIdx)
{
    if (triIdx < 0 || params.triangleObjectIds == nullptr) return -1;
    return params.triangleObjectIds[triIdx];
}

// ---------------------------------------------------------------------------
//  __raygen__wf_outline
//  Launched with width = activeRayCount, height = 1, depth = 1 — the same queue
//  extend just consumed, so the centre hit for each path is already resolved.
// ---------------------------------------------------------------------------
extern "C" __global__ void __raygen__wf_outline()
{
    const uint32_t queueSlot = optixGetLaunchIndex().x;
    const int idx = params.wf.rayQueue[queueSlot];

    if (params.wf.edgeFactor == nullptr) return;

    params.wf.edgeFactor[idx] = 0.0f;

    // Lines deeper than a couple of bounces are not perceptible, and each extra
    // level costs OUTLINE_PROBE_COUNT rays per path.
    if (params.wf.bounceCount[idx] > OUTLINE_MAX_DEPTH) return;

    const int centreTri = params.wf.hitTriIndex[idx];
    if (centreTri < 0) return;   // escaped rays have no surface to outline

    const StyleData st = outlineStyleForTriangle(centreTri);
    if (st.lineStrength <= 0.0f || st.lineWidth <= 0.0f) return;

    const Float3 origin    = params.wf.rayOrigin[idx];
    const Float3 direction = params.wf.rayDir[idx];

    const Float3 centreHit    = params.wf.hitPoint[idx];
    const Float3 centreNormal = params.wf.hitNormal[idx];
    const int    centreObject = objectIdOf(centreTri);

    const float centreDist = distance3(centreHit, origin);
    if (centreDist <= RT_EPSILON) return;

    // Constant apparent thickness: a wider angular offset for near surfaces, a
    // narrower one for distant ones.
    const float probeAngle = st.lineWidth / centreDist;

    // Orthonormal basis around the ray so the probes fan out evenly.
    const Frame3 basis = makeFrameFromNormal(direction);

    int edgeHits = 0;

    for (int p = 0; p < OUTLINE_PROBE_COUNT; ++p) {
        const float phi = (kOutlineTwoPi * static_cast<float>(p)) /
                          static_cast<float>(OUTLINE_PROBE_COUNT);

        const Float3 offset = add3(mul3(basis.t, probeAngle * cosf(phi)),
                                   mul3(basis.b, probeAngle * sinf(phi)));

        Ray probe;
        probe.origin    = origin;
        probe.direction = normalize3(add3(direction, offset));

        Intersection probeIts{};
        if (!traceClosest(params.handle, probe, probeIts)) {
            // The probe left the scene while the centre hit geometry: silhouette.
            ++edgeHits;
            continue;
        }

        // Silhouette: the probe landed on a different object.
        if (objectIdOf(probeIts.triangleIndex) != centreObject) {
            ++edgeHits;
            continue;
        }

        // Occluding contour: a large depth step within the same object.
        const float probeDist = distance3(probeIts.hitPoint, origin);
        if (fabsf(probeDist - centreDist) > st.outlineDepthThreshold * centreDist) {
            ++edgeHits;
            continue;
        }

        // Crease: the surface folds sharply between the centre and the probe.
        if (dot3(probeIts.hitNormal, centreNormal) < st.outlineNormalThreshold) {
            ++edgeHits;
        }
    }

    params.wf.edgeFactor[idx] =
        static_cast<float>(edgeHits) / static_cast<float>(OUTLINE_PROBE_COUNT);
}
