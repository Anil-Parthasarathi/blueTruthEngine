// ============================================================================
//  ray_present.cu — Wavefront Stage 5: Accumulate, then Present
// ============================================================================
//  Compiled as a regular CUDA TU (linked into the exe, NOT PTX).
//  Launched ONCE per frame, after the bounce loop has fully drained (every
//  path terminated and all its shadow contributions were atomically added
//  into the style channels by connect).
//
//  Why a separate stage?  Accumulating inside shade would race with connect:
//  connect runs AFTER shade each bounce, so a path that emits NEE shadow rays
//  and then dies (miss / Russian roulette) in the same shade call would
//  snapshot its radiance into the accumulation buffer BEFORE those shadow
//  contributions land — silently dropping direct lighting.  Once the loop
//  exits, the channels are final (one path per pixel per frame), so a single
//  terminal pass is both correct and simpler.
//
//  Why TWO kernels?  Because the denoiser has to run in between:
//
//      wfAccumulate   Welford-blend this frame's channels into the persistent
//                     per-channel accumulators.
//      (host)         denoise the SMOOTH channels only.
//      wfPresent      apply the style operators to the converged (and where
//                     appropriate denoised) channels and write the PBO.
//
//  Everything stylistic happens in wfPresent, on converged data.  Two
//  consequences worth stating plainly:
//
//    • Quantizing a CONVERGED signal turns a physically correct soft penumbra
//      into a hard anime shadow whose shape is exactly right — including from
//      off-screen occluders and through multi-bounce transport.  Quantizing
//      per-sample instead would make band edges dither and converge to a blur.
//
//    • Restyling costs one kernel launch.  A 3000spp image can be re-graded
//      instantly with no re-convergence, which is both the best demo and by far
//      the best authoring loop.
//
//  In PHYSICAL mode every channel pointer aliases the one radiance buffer, so
//  wfPresent takes its early-out branch and behaves exactly as before.  The
//  branch is on a scene-level uniform, so there is no warp divergence.
// ============================================================================

#include "wavefront/wavefront_buffers.h"
#include "wavefront/wavefront_shading.cuh"   // loadStyle
#include "rt_cuda_math.cuh"
#include "rt_camera.cuh"                     // generatePrimaryRay (rim view dir)
#include "rt_style.cuh"
#include "rt_path_integrator.cuh"            // accumulateWelford, packPixelGamma
#include <cstdint>

// ---------------------------------------------------------------------------
//  wfAccumulate — Welford-blend this frame's channels into the accumulators.
//  One thread per path slot / pixel.
// ---------------------------------------------------------------------------
__global__ void wfAccumulate(
    WavefrontSoA wf,
    WfSceneView  scene)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= wf.maxPaths) return;

    if (!wf.pixelIndex || !wf.radiance) return;

    const uint32_t pixelIndex = wf.pixelIndex[idx];
    if (pixelIndex >= static_cast<uint32_t>(wf.maxPaths)) return;

    const uint32_t frameIndex = static_cast<uint32_t>(scene.frameIndex);

    if (!scene.styleModeAnime) {
        // All six channel pointers alias wf.radiance, so one blend is the whole
        // image — identical to the pre-stylization behaviour.
        if (!scene.accumBuffer) return;
        accumulateWelford(scene.accumBuffer, pixelIndex, wf.radiance[idx], frameIndex);
        return;
    }

    for (int ch = 0; ch < STYLE_CH_COUNT; ++ch) {
        if (!scene.accumChannel[ch] || !wf.channel[ch]) continue;
        accumulateWelford(scene.accumChannel[ch], pixelIndex, wf.channel[ch][idx], frameIndex);
    }

    // Edge coverage is a per-path AOV like albedo and normal, so it is indexed
    // by path slot rather than by pixel.
    if (scene.accumEdge && wf.edgeFactor) {
        const float n = static_cast<float>(frameIndex + 1);
        float prev = scene.accumEdge[idx];
        prev += (wf.edgeFactor[idx] - prev) / n;
        scene.accumEdge[idx] = prev;
    }
}

// ---------------------------------------------------------------------------
//  wfPresent — style operators + gamma encode into the PBO.
//  One thread per path slot / pixel.
// ---------------------------------------------------------------------------
__global__ void wfPresent(
    WavefrontSoA wf,
    WfSceneView  scene,
    uint32_t*    framebuffer)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= wf.maxPaths) return;

    const uint32_t px = wf.pixelIndex[idx];

    // ── Debug views: inspect the AOVs and the channel split directly ─────
    if (scene.debugView != static_cast<int>(DebugView::Off)) {
        Float3 dbg = {0.0f, 0.0f, 0.0f};
        switch (static_cast<DebugView>(scene.debugView)) {
            case DebugView::Albedo:
                if (scene.accumAlbedo) dbg = scene.accumAlbedo[idx];
                break;
            case DebugView::Normal:
                if (scene.accumNormal) {
                    const Float3 n = scene.accumNormal[idx];
                    dbg = {0.5f * n.x + 0.5f, 0.5f * n.y + 0.5f, 0.5f * n.z + 0.5f};
                }
                break;
            case DebugView::ObjectId:
                if (scene.primaryObjectId) {
                    const int id = scene.primaryObjectId[idx];
                    if (id >= 0) {
                        // Cheap hash to a distinguishable colour.
                        const uint32_t h = static_cast<uint32_t>(id) * 2654435761u;
                        dbg = {static_cast<float>((h >> 16) & 0xFF) / 255.0f,
                               static_cast<float>((h >> 8)  & 0xFF) / 255.0f,
                               static_cast<float>( h        & 0xFF) / 255.0f};
                    }
                }
                break;
            case DebugView::Depth:
                if (scene.primaryDepth) {
                    const float d = scene.primaryDepth[idx];
                    // Reciprocal mapping keeps near geometry readable without a
                    // scene-dependent far plane.
                    const float v = (d < 0.0f) ? 0.0f : 1.0f / (1.0f + 0.1f * d);
                    dbg = {v, v, v};
                }
                break;
            case DebugView::Edge:
                if (scene.accumEdge) {
                    const float e = scene.accumEdge[idx];
                    dbg = {e, e, e};
                }
                break;
            case DebugView::DirectDiffuse:
            case DebugView::DirectSpecular:
            case DebugView::IndirectDiffuse:
            case DebugView::Reflected:
            case DebugView::Transmitted:
            case DebugView::Emissive: {
                const int ch = scene.debugView - static_cast<int>(DebugView::DirectDiffuse);
                if (scene.styleModeAnime && scene.readChannel[ch])
                    dbg = scene.readChannel[ch][px];
                else if (scene.readChannel[0])
                    dbg = scene.readChannel[0][px];
                break;
            }
            default:
                break;
        }
        framebuffer[px] = packPixelGamma(dbg);
        return;
    }

    // ── Physical mode: one buffer, no stylization ─────────────────────────
    if (!scene.styleModeAnime) {
        framebuffer[px] = packPixelGamma(scene.readChannel[0][px]);
        return;
    }

    // ── Anime mode: per-channel operators on converged data ──────────────
    const Float3 directDiffuse   = scene.readChannel[STYLE_CH_DIRECT_DIFFUSE][px];
    const Float3 directSpecular  = scene.readChannel[STYLE_CH_DIRECT_SPECULAR][px];
    const Float3 indirectDiffuse = scene.readChannel[STYLE_CH_INDIRECT_DIFFUSE][px];
    const Float3 reflected       = scene.readChannel[STYLE_CH_REFLECTED][px];
    const Float3 transmitted     = scene.readChannel[STYLE_CH_TRANSMITTED][px];
    const Float3 emissive        = scene.readChannel[STYLE_CH_EMISSIVE][px];

    const int styleId = scene.primaryStyleId ? scene.primaryStyleId[idx] : -1;
    const StyleData st = loadStyle(scene, styleId);

    // Body tone: the wall-bounce fill is part of the terminator shape, so
    // direct and gain-scaled indirect are quantized together — once, on the
    // converged sum — rather than leaving noisy GI sitting on a black band.
    const Float3 bodyInput = add3(directDiffuse, mul3(indirectDiffuse, st.indirectGain));
    Float3 out = styleBodyTone(bodyInput, st, scene.rampTex, scene.rampCount);

    // Hard-edged anime highlight rather than a smooth specular falloff.
    out = add3(out, styleHighlight(directSpecular, st));

    // Posterize what the surface actually reflects and refracts.  This is the
    // part raster NPR cannot reach: the bands sit on a real traced reflection.
    out = add3(out, styleMetal(reflected, st));
    out = add3(out, styleJewel(transmitted, st));

    // Rim light, from the primary-hit normal AOV and this pixel's view ray.
    if (st.rimStrength > 0.0f && scene.accumNormal) {
        const int pxX = static_cast<int>(px) % scene.width;
        const int pxY = static_cast<int>(px) / scene.width;
        const Ray viewRay = generatePrimaryRay(pxX, pxY, scene.width, scene.height,
                                               scene.camera, 0.0f, 0.0f);
        out = add3(out, styleRim(scene.accumNormal[idx], viewRay.direction, st));
    }

    // Emitters and the backdrop stay unstyled.
    out = add3(out, emissive);

    // Ink last, after banding, so a thin edge cannot vanish into the same
    // cel level as its neighbour.  Coverage is the max along the path, so a
    // silhouette seen in a reflection still lands on this pixel.
    if (scene.accumEdge)
        out = styleInk(out, st, scene.accumEdge[idx]);

    framebuffer[px] = packPixelGamma(out);
}

// ---------------------------------------------------------------------------
//  Host wrappers called from cudaRenderWavefront()
// ---------------------------------------------------------------------------
void launchWfAccumulate(
    const WavefrontSoA& wf,
    const WfSceneView&  scene)
{
    const int threads = 256;
    const int blocks  = (wf.maxPaths + threads - 1) / threads;
    wfAccumulate<<<blocks, threads>>>(wf, scene);
}

void launchWfPresent(
    const WavefrontSoA& wf,
    const WfSceneView&  scene,
    uint32_t*           framebuffer)
{
    const int threads = 256;
    const int blocks  = (wf.maxPaths + threads - 1) / threads;
    wfPresent<<<blocks, threads>>>(wf, scene, framebuffer);
}
