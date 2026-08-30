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
//          outline    [anime only]  (__raygen__wf_outline, optixLaunch)
//          shade                    (wfShade,             plain CUDA)
//          connect                  (__raygen__wf_shadow, optixLaunch)
//          swap(rayQueue, rayQueueNext)
//      accumulate                   (wfAccumulate,        plain CUDA)
//      denoise smooth channels      (OptiX denoiser, host)
//      present                      (wfPresent,           plain CUDA)
//
//  Key design points:
//    • Ping-pong ray queues — shade reads rayQueue/rayCount and appends
//      surviving paths to rayQueueNext/rayCountNext; the host swaps the
//      pointers after each bounce.  A single queue would race: shade threads
//      would append into the same buffer other threads are still reading.
//    • Accumulation happens ONCE, in the terminal accumulate stage.  Shade must
//      NOT write to accumBuffer: connect runs after shade, so a path that
//      dies in shade (RR / miss) would snapshot its radiance before its NEE
//      shadow rays from the same bounce were resolved, silently dropping
//      direct lighting.  With one path per pixel per frame, the style channels
//      are final once the loop exits, and accumulate Welford-blends them.
//    • The outline stage runs BETWEEN extend and shade, because shade consumes
//      wf.edgeFactor as a throughput modulation, and it needs the centre hit
//      that extend just resolved.
//    • Stylization is confined to present, which runs on converged data.  The
//      photorealistic path is unchanged: the outline launch is skipped, the
//      channel pointers all alias the one accumulation buffer, and present
//      takes an early-out branch on a scene-level uniform.
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
void launchWfGenerate  (const WavefrontSoA&, const CameraData&, int, int, uint32_t);
void launchWfShade     (const WavefrontSoA&, const WfSceneView&, int);
void launchWfAccumulate(const WavefrontSoA&, const WfSceneView&);
void launchWfPresent   (const WavefrontSoA&, const WfSceneView&, uint32_t*);

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
    cudaFree(w.shadowContribSpec);
    cudaFree(w.shadowPathIdx);  cudaFree(w.shadowIsPrimary);
    cudaFree(w.styleChannel);
    cudaFree(w.edgeFactor);

    // channel[0] aliases w.radiance and channel[1..] are separate allocations
    // only in anime mode; in physical mode they all alias, so freeing per
    // pointer would double-free.
    for (int c = 1; c < STYLE_CH_COUNT; ++c) {
        if (w.channel[c] && w.channel[c] != w.radiance) cudaFree(w.channel[c]);
    }
    w = {};
}

// Anime mode needs five extra per-path radiance buffers plus the edge buffer.
// They are allocated on the first stylized frame and released when stylization
// is turned off, so the photorealistic path never carries the cost.
static void allocWfStyleChannels(WavefrontSoA& w)
{
    const size_t n = static_cast<size_t>(w.maxPaths);
    for (int c = 1; c < STYLE_CH_COUNT; ++c) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&w.channel[c]), n * sizeof(Float3)));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&w.edgeFactor), n * sizeof(float)));

    fprintf(stdout, "[style] Wavefront style channels allocated (%d paths, "
                    "%.1f MB)\n",
            w.maxPaths,
            static_cast<double>(n * (5 * sizeof(Float3) + sizeof(float))) / (1024.0 * 1024.0));
}

static void freeWfStyleChannels(WavefrontSoA& w)
{
    for (int c = 1; c < STYLE_CH_COUNT; ++c) {
        if (w.channel[c] && w.channel[c] != w.radiance) cudaFree(w.channel[c]);
        w.channel[c] = w.radiance;
    }
    if (w.edgeFactor) { cudaFree(w.edgeFactor); w.edgeFactor = nullptr; }
}

static void ensureWavefrontBuffers(int width, int height)
{
    const bool anime = (s_styleMode == StyleMode::Anime);

    if (s_wf.allocated && s_wf.soa.maxPaths == width * height) {
        // Resolution unchanged; only the style tier may need to change.
        WavefrontSoA& w = s_wf.soa;
        const bool haveStyle = (w.edgeFactor != nullptr);
        if (anime && !haveStyle)      allocWfStyleChannels(w);
        else if (!anime && haveStyle) freeWfStyleChannels(w);
        return;
    }

    // Free old allocation if resolution changed.
    if (s_wf.allocated) {
        freeWavefrontSoA(s_wf.soa);
        s_wf.allocated = false;
    }

    WavefrontSoA& w = s_wf.soa;
    // One path slot per pixel; shadow slots must cover area NEE + all spotlights.
    w.maxPaths   = width * height;
    w.maxShadows = w.maxPaths * (1 + s_spotlightCount + 1); // +1 area, +1 slack

    const int  N = w.maxPaths;
    const int  M = w.maxShadows;

#define WF_ALLOC(ptr, T, n)  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&(ptr)), (n) * sizeof(T)))
    WF_ALLOC(w.rayOrigin,        Float3,        N);
    WF_ALLOC(w.rayDir,           Float3,        N);
    WF_ALLOC(w.throughput,       Float3,        N);
    WF_ALLOC(w.radiance,         Float3,        N);
    WF_ALLOC(w.eta,              float,         N);
    WF_ALLOC(w.brdfPDF,          float,         N);
    WF_ALLOC(w.specularBounce,   unsigned char, N);
    WF_ALLOC(w.styleChannel,     unsigned char, N);
    WF_ALLOC(w.bounceCount,      int,           N);
    WF_ALLOC(w.pixelIndex,       uint32_t,      N);
    WF_ALLOC(w.rngState,         uint32_t,      N);
    WF_ALLOC(w.hitTriIndex,      int,           N);
    WF_ALLOC(w.hitPoint,         Float3,        N);
    WF_ALLOC(w.hitNormal,        Float3,        N);
    WF_ALLOC(w.hitUV,            Float2,        N);
    WF_ALLOC(w.rayQueue,         int,           N);
    WF_ALLOC(w.rayQueueNext,     int,           N);
    WF_ALLOC(w.rayCount,         int,           1);
    WF_ALLOC(w.rayCountNext,     int,           1);
    WF_ALLOC(w.shadowCount,      int,           1);
    WF_ALLOC(w.shadowOrigin,     Float3,        M);
    WF_ALLOC(w.shadowDir,        Float3,        M);
    WF_ALLOC(w.shadowTMax,       float,         M);
    WF_ALLOC(w.shadowContrib,    Float3,        M);
    WF_ALLOC(w.shadowContribSpec, Float3,       M);
    WF_ALLOC(w.shadowPathIdx,    uint32_t,      M);
    WF_ALLOC(w.shadowIsPrimary,  unsigned char, M);
#undef WF_ALLOC

    // Physical mode: all six channels are the same buffer.  Shade and connect
    // then need no style-mode branch — every atomicAdd lands in `radiance` and
    // sums to exactly the physical result.
    for (int c = 0; c < STYLE_CH_COUNT; ++c) w.channel[c] = w.radiance;
    w.edgeFactor = nullptr;

    if (anime) allocWfStyleChannels(w);

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
    sv.triangleObjectIds     = s_triangleObjectIds_d;
    sv.bsdfs                 = s_bsdfs_d;
    sv.bsdfCount             = s_bsdfCount;
    sv.triangleBsdfIds       = s_triangleBsdfIds_d;
    sv.triangleMaterialIds   = s_triangleMaterialIds_d;
    sv.emitters              = s_emitters_d;
    sv.emitterCount          = s_emitterCount;
    sv.emitterTriIndices     = s_emitterTriIndices_d;
    sv.emitterTriCdf         = s_emitterTriCdf_d;
    sv.sceneEmitterCdf       = s_sceneEmitterCdf_d;
    sv.emitterSelectCount    = s_emitterSelectCount;
    sv.envEmitterSlot        = s_envEmitterSlot;
    sv.spotlights            = s_spotlights_d;
    sv.spotlightCount        = s_spotlightCount;
    sv.texObjects            = s_texObjects_d;
    sv.textureCount          = s_textureCount;
    sv.env                   = s_env;

    sv.styles                = s_styles_d;
    sv.styleCount            = s_styleCount;
    sv.rampTex               = s_rampObjects_d;
    sv.rampCount             = s_rampCount;
    sv.styleModeAnime        = (s_styleMode == StyleMode::Anime) ? 1 : 0;

    sv.accumAlbedo           = s_accumAlbedo_d;
    sv.accumNormal           = s_accumNormal_d;
    sv.accumEdge             = s_accumEdge_d;
    sv.primaryDepth          = s_primaryDepth_d;
    sv.primaryStyleId        = s_primaryStyleId_d;
    sv.primaryObjectId       = s_primaryObjectId_d;

    for (int c = 0; c < STYLE_CH_COUNT; ++c) {
        sv.accumChannel[c] = s_accumChannel_d[c];
        // Default: present reads the raw accumulator.  The denoise step below
        // redirects only the smooth channels.
        sv.readChannel[c]  = s_accumChannel_d[c];
    }

    sv.accumBuffer           = s_accumBuffer_d;
    sv.frameIndex            = frameIndex;

    sv.camera                = s_camera_h;
    sv.width                 = s_accumWidth;
    sv.height                = s_accumHeight;
    sv.debugView             = static_cast<int>(s_debugView);
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
    ensureStyleBuffers(imageWidth, imageHeight);

    WavefrontSoA& wf = s_wf.soa;
    WfSceneView   sv = buildSceneView(static_cast<int>(s_frameIndex));

    const bool anime = (s_styleMode == StyleMode::Anime);
    const int  zero  = 0;

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

        // ── Stage 2b: Outline probes (anime only) ────────────────────────
        // Skipped entirely in physical mode — not a device branch, an absent
        // launch — so the photorealistic path pays literally nothing.  Runs
        // after extend because it classifies probes against the centre hit,
        // and before shade because shade folds edgeFactor into throughput.
        if (anime && s_pgWfOutline) {
            LaunchParams lp = {};
            lp.handle            = s_gasHandle;
            lp.triangles         = s_triangles_d;
            lp.triangleBsdfIds   = s_triangleBsdfIds_d;
            lp.triangleObjectIds = s_triangleObjectIds_d;
            lp.bsdfs             = s_bsdfs_d;
            lp.bsdfCount         = s_bsdfCount;
            lp.styles            = s_styles_d;
            lp.styleCount        = s_styleCount;
            lp.wf                = wf;
            CUDA_CHECK(cudaMemcpy(s_launchParams_d, &lp, sizeof(LaunchParams), cudaMemcpyHostToDevice));

            OPTIX_CHECK(optixLaunch(s_optixPipeline, 0 /*stream*/,
                                    reinterpret_cast<CUdeviceptr>(s_launchParams_d),
                                    sizeof(LaunchParams), &s_sbt_wf_outline,
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

    // ── Stage 5: Accumulate ─────────────────────────────────────────────
    // Every path is now terminated, and all its shadow contributions have
    // landed in the style channels.  Welford-blend them into the persistent
    // accumulators.
    launchWfAccumulate(wf, sv);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Stage 6: Denoise, then present ──────────────────────────────────
    if (!anime) {
        // Physical mode is unchanged: raw tonemap, then optionally denoise the
        // whole image and re-present over the top.
        launchWfPresent(wf, sv, devPtr);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (s_denoiserEnabled) {
            denoiseAndPresent(devPtr, imageWidth, imageHeight);
        }
        return;
    }

    // Anime mode denoises SELECTIVELY.  The smooth channels — indirect diffuse,
    // reflected, transmitted — are the noisy ones and are also the ones about to
    // be posterized, and posterizing noise produces crawling band edges.  The
    // direct channels are deliberately left alone: a denoiser trained on natural
    // images rounds off exactly the hard shadow terminator the style depends on.
    if (s_denoiserEnabled) {
        static const int kSmooth[] = { STYLE_CH_INDIRECT_DIFFUSE,
                                       STYLE_CH_REFLECTED,
                                       STYLE_CH_TRANSMITTED };
        for (int ch : kSmooth) {
            if (!s_accumChannel_d[ch] || !s_denoisedChannel_d[ch]) continue;
            denoiseChannel(s_accumChannel_d[ch], s_denoisedChannel_d[ch],
                           imageWidth, imageHeight);
            sv.readChannel[ch] = s_denoisedChannel_d[ch];
        }
    }

    launchWfPresent(wf, sv, devPtr);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
//  Present-only pass — restyle the already-converged buffers without tracing.
// ---------------------------------------------------------------------------
void cudaRepresentWavefront(uint32_t* devPtr, int imageWidth, int imageHeight)
{
    if (!s_wf.allocated) return;

    WfSceneView sv = buildSceneView(static_cast<int>(s_frameIndex));

    if (s_styleMode == StyleMode::Anime && s_denoiserEnabled) {
        static const int kSmooth[] = { STYLE_CH_INDIRECT_DIFFUSE,
                                       STYLE_CH_REFLECTED,
                                       STYLE_CH_TRANSMITTED };
        for (int ch : kSmooth) {
            if (!s_accumChannel_d[ch] || !s_denoisedChannel_d[ch]) continue;
            denoiseChannel(s_accumChannel_d[ch], s_denoisedChannel_d[ch],
                           imageWidth, imageHeight);
            sv.readChannel[ch] = s_denoisedChannel_d[ch];
        }
    }

    launchWfPresent(s_wf.soa, sv, devPtr);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (s_styleMode == StyleMode::Physical && s_denoiserEnabled) {
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
