// ============================================================================
//  wavefront_host.cu — host side of the wavefront path tracer:
//  SoA buffer management, scene view packing, and the per-frame bounce loop.
// ============================================================================
//
//  Per-frame stage sequence (device kernels live in src/wavefront/):
//
//      generate                     (wfGenerate,          plain CUDA)
//      loop while rays remain:
//          extend                   (__raygen__wf_extend, optixLaunch)
//          shade                    (wfShade,             plain CUDA)
//          connect                  (__raygen__wf_shadow, optixLaunch)
//          swap(rayQueue, rayQueueNext)
//      present                      (wfPresent,           plain CUDA)
//
//  Key design points:
//    • Ping-pong ray queues — shade reads rayQueue/rayCount and appends
//      surviving paths to rayQueueNext/rayCountNext; the host swaps the
//      pointers after each bounce.  A single queue would race: shade threads
//      would append into the same buffer other threads are still reading.
//    • Accumulation happens ONCE, in the terminal present stage.  Shade must
//      NOT write to accumBuffer: connect runs after shade, so a path that
//      dies in shade (RR / miss) would snapshot its radiance before its NEE
//      shadow rays from the same bounce were resolved, silently dropping
//      direct lighting.  With one path per pixel per frame, wf.radiance[idx]
//      is final once the loop exits, and present Welford-blends it.
// ============================================================================

#include "renderer_state.h"
#include "cuda_check.h"
#include "rt_constants.cuh"   // shared MAX_BOUNCES

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <cstdint>
#include <cstdio>
#include <utility>            // std::swap

// Forward declarations for host launchers defined in src/wavefront/*.cu
void launchWfGenerate(const WavefrontSoA&, const CameraData&, int, int, uint32_t);
void launchWfShade   (const WavefrontSoA&, const WfSceneView&, int);
void launchWfPresent (const WavefrontSoA&, const WfSceneView&, uint32_t*, int, int);

// ---------------------------------------------------------------------------
//  Wavefront buffer allocation  (called lazily on first wavefront render)
// ---------------------------------------------------------------------------
static void freeWavefrontSoA(WavefrontSoA& w)
{
    cudaFree(w.rayOrigin);      cudaFree(w.rayDir);
    cudaFree(w.throughput);     cudaFree(w.radiance);
    cudaFree(w.eta);            cudaFree(w.brdfPDF);
    cudaFree(w.specularBounce); cudaFree(w.bounceCount);
    cudaFree(w.pixelIndex);     cudaFree(w.rngState);
    cudaFree(w.hitTriIndex);
    cudaFree(w.hitPoint);       cudaFree(w.hitNormal); cudaFree(w.hitUV);
    cudaFree(w.rayQueue);       cudaFree(w.rayQueueNext);
    cudaFree(w.rayCount);       cudaFree(w.rayCountNext);
    cudaFree(w.shadowCount);
    cudaFree(w.shadowOrigin);   cudaFree(w.shadowDir);
    cudaFree(w.shadowTMax);     cudaFree(w.shadowContrib);
    cudaFree(w.shadowPathIdx);
    w = {};
}

static void ensureWavefrontBuffers(int width, int height)
{
    if (s_wf.allocated &&
        s_wf.soa.maxPaths == width * height)
        return;

    // Free old allocation if resolution changed.
    if (s_wf.allocated) {
        freeWavefrontSoA(s_wf.soa);
        s_wf.allocated = false;
    }

    WavefrontSoA& w = s_wf.soa;
    // One path slot per pixel; shadow slots must cover area NEE + all spotlights + env map.
    w.maxPaths   = width * height;
    w.maxShadows = w.maxPaths * (1 + s_spotlightCount + (s_hasEnvMap ? 1 : 0) + 1); // +1 area, +1 env, +1 slack

    const int  N = w.maxPaths;
    const int  M = w.maxShadows;

#define WF_ALLOC(ptr, T, n)  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&(ptr)), (n) * sizeof(T)))
    WF_ALLOC(w.rayOrigin,       Float3,        N);
    WF_ALLOC(w.rayDir,          Float3,        N);
    WF_ALLOC(w.throughput,      Float3,        N);
    WF_ALLOC(w.radiance,        Float3,        N);
    WF_ALLOC(w.eta,             float,         N);
    WF_ALLOC(w.brdfPDF,         float,         N);
    WF_ALLOC(w.specularBounce,  unsigned char, N);
    WF_ALLOC(w.bounceCount,     int,           N);
    WF_ALLOC(w.pixelIndex,      uint32_t,      N);
    WF_ALLOC(w.rngState,        uint32_t,      N);
    WF_ALLOC(w.hitTriIndex,     int,           N);
    WF_ALLOC(w.hitPoint,        Float3,        N);
    WF_ALLOC(w.hitNormal,       Float3,        N);
    WF_ALLOC(w.hitUV,           Float2,        N);
    WF_ALLOC(w.rayQueue,        int,           N);
    WF_ALLOC(w.rayQueueNext,    int,           N);
    WF_ALLOC(w.rayCount,        int,           1);
    WF_ALLOC(w.rayCountNext,    int,           1);
    WF_ALLOC(w.shadowCount,     int,           1);
    WF_ALLOC(w.shadowOrigin,    Float3,        M);
    WF_ALLOC(w.shadowDir,       Float3,        M);
    WF_ALLOC(w.shadowTMax,      float,         M);
    WF_ALLOC(w.shadowContrib,   Float3,        M);
    WF_ALLOC(w.shadowPathIdx,   uint32_t,      M);
#undef WF_ALLOC

    s_wf.allocated = true;
    fprintf(stdout, "[wavefront] Buffers allocated (%dx%d, %d paths, %d shadow slots)\n",
            width, height, N, M);
}

// ---------------------------------------------------------------------------
//  Pack a WfSceneView from the current renderer state.
// ---------------------------------------------------------------------------
static WfSceneView buildSceneView(int frameIndex)
{
    WfSceneView sv = {};
    sv.triangles             = s_triangles_d;
    sv.triangleCount         = s_triangleCount;
    sv.triangleEmitterFlags  = s_triangleEmitterFlags_d;
    sv.triangleEmission      = s_triangleEmission_d;
    sv.bsdfs                 = s_bsdfs_d;
    sv.bsdfCount             = s_bsdfCount;
    sv.triangleBsdfIds       = s_triangleBsdfIds_d;
    sv.triangleMaterialIds   = s_triangleMaterialIds_d;
    sv.emitters              = s_emitters_d;
    sv.emitterCount          = s_emitterCount;
    sv.emitterTriIndices     = s_emitterTriIndices_d;
    sv.emitterTriCdf         = s_emitterTriCdf_d;
    sv.sceneEmitterCdf       = s_sceneEmitterCdf_d;
    sv.spotlights            = s_spotlights_d;
    sv.spotlightCount        = s_spotlightCount;
    sv.texObjects            = s_texObjects_d;
    sv.texObjectsMr          = s_texObjectsMr_d;
    sv.textureCount          = s_textureCount;
    sv.envMap                = s_envMap_h;
    sv.hasEnvMap             = s_hasEnvMap;
    sv.accumBuffer           = s_accumBuffer_d;
    sv.frameIndex            = frameIndex;
    return sv;
}

// ---------------------------------------------------------------------------
//  Wavefront render loop
//  Called instead of the megakernel optixLaunch block when
//  s_renderMode == RenderMode::Wavefront.
// ---------------------------------------------------------------------------
void cudaRenderWavefront(uint32_t* devPtr, int imageWidth, int imageHeight)
{
    ensureWavefrontBuffers(imageWidth, imageHeight);

    WavefrontSoA& wf = s_wf.soa;
    const WfSceneView sv = buildSceneView(static_cast<int>(s_frameIndex));

    const int zero = 0;

    // ── Stage 1: Generate primary rays ──────────────────────────────────
    // Fills every path slot and the current ray queue (rayQueue/rayCount).
    CUDA_CHECK(cudaMemcpy(wf.rayCount, &zero, sizeof(int), cudaMemcpyHostToDevice));
    launchWfGenerate(wf, s_camera_h, imageWidth, imageHeight, s_frameIndex);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Bounce loop (shared cap with the megakernel) ────────────────────
    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {

        int activeCount = 0;
        CUDA_CHECK(cudaMemcpy(&activeCount, wf.rayCount, sizeof(int), cudaMemcpyDeviceToHost));
        if (activeCount <= 0) break;

        // ── Stage 2: Extend — optixLaunch with the wf_extend raygen ─────
        // One thread per active ray; results land in the hit buffer.
        {
            LaunchParams lp = {};
            lp.handle      = s_gasHandle;
            lp.triangles   = s_triangles_d;
            lp.wf          = wf;
            CUDA_CHECK(cudaMemcpy(s_launchParams_d, &lp, sizeof(LaunchParams), cudaMemcpyHostToDevice));

            OPTIX_CHECK(optixLaunch(s_optixPipeline, 0 /*stream*/,
                                    reinterpret_cast<CUdeviceptr>(s_launchParams_d),
                                    sizeof(LaunchParams), &s_sbt_wf_extend,
                                    static_cast<unsigned int>(activeCount), 1, 1));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Reset the NEXT-bounce queue and this bounce's shadow counter;
        // shade appends into both.  The current queue (rayQueue/rayCount)
        // stays intact — shade is still reading it.
        CUDA_CHECK(cudaMemcpy(wf.rayCountNext, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(wf.shadowCount,  &zero, sizeof(int), cudaMemcpyHostToDevice));

        // ── Stage 3: Shade ───────────────────────────────────────────────
        launchWfShade(wf, sv, activeCount);
        CUDA_CHECK(cudaDeviceSynchronize());

        // ── Stage 4: Connect shadow rays ─────────────────────────────────
        int shadowCount = 0;
        CUDA_CHECK(cudaMemcpy(&shadowCount, wf.shadowCount, sizeof(int), cudaMemcpyDeviceToHost));
        if (shadowCount > 0) {
            LaunchParams lp = {};
            lp.handle = s_gasHandle;
            lp.wf     = wf;
            CUDA_CHECK(cudaMemcpy(s_launchParams_d, &lp, sizeof(LaunchParams), cudaMemcpyHostToDevice));

            OPTIX_CHECK(optixLaunch(s_optixPipeline, 0 /*stream*/,
                                    reinterpret_cast<CUdeviceptr>(s_launchParams_d),
                                    sizeof(LaunchParams), &s_sbt_wf_shadow,
                                    static_cast<unsigned int>(shadowCount), 1, 1));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // ── Ping-pong: next bounce reads what shade just wrote ──────────
        std::swap(wf.rayQueue, wf.rayQueueNext);
        std::swap(wf.rayCount, wf.rayCountNext);
    }

    // ── Stage 5: Present ────────────────────────────────────────────────
    // Every path is now terminated, and all its shadow contributions are in
    // wf.radiance.  Welford-blend into the accumulation buffer and tonemap
    // into the PBO (same math as the megakernel tail).
    launchWfPresent(wf, sv, devPtr, imageWidth, imageHeight);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Optional: denoise the accumulated image and re-present, overwriting
    // the raw tonemap above (mirrors the megakernel path in cudaRender).
    if (s_denoiserEnabled) {
        denoiseAndPresent(devPtr, imageWidth, imageHeight);
    }
}

// ---------------------------------------------------------------------------
//  Teardown (called from cudaCleanup)
// ---------------------------------------------------------------------------
void freeWavefrontState()
{
    if (s_wf.allocated) {
        freeWavefrontSoA(s_wf.soa);
        s_wf.allocated = false;
    }
}
