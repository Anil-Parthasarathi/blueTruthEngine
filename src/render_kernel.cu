// ============================================================================
//  render_kernel.cu  –  public renderer API + per-frame dispatch
// ============================================================================
//  The host-side API declared in render_kernel.h is implemented across a few
//  focused translation units:
//
//    src/host/renderer_state.cu — shared device buffers / handles (s_* state)
//    src/host/optix_setup.cu    — OptiX pipeline, SBTs, GAS build, cudaInitOptix
//    src/host/scene_upload.cu   — cudaInit* scene uploaders
//    src/host/denoiser.cu       — OptiX AI denoiser + linear presentation
//    src/host/wavefront_host.cu — wavefront buffers + per-frame bounce loop
//
//  This file keeps only what glues a frame together: PBO registration,
//  accumulation reset, the megakernel launch, mode dispatch, and cleanup.
// ============================================================================

#include "render_kernel.h"
#include "host/renderer_state.h"
#include "host/cuda_check.h"
#include "optix_launch_params.h"

#include <glad/gl.h>       // Must come before cuda_gl_interop.h (defines GLuint)
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>

#include <cstdio>
#include <cstdint>

void cudaRegisterPBO(uint32_t pbo)
{
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &s_pboResource, pbo,
        cudaGraphicsMapFlagsWriteDiscard));
}

// ---------------------------------------------------------------------------
//  AOV and style-channel buffers.
//
//  Two tiers, deliberately:
//
//    • The AOVs (albedo, normal, depth, style ID, object ID) are ALWAYS
//      allocated.  Albedo and normal are the OptiX denoiser's guide layers, so
//      keeping them on improves the photorealistic path too — that is the one
//      part of this feature worth paying for unconditionally.
//
//    • The five extra channel accumulators, their denoised copies, and the edge
//      buffer are allocated only in anime mode and freed when it turns off.  In
//      physical mode every channel pointer aliases s_accumBuffer_d, so the
//      footprint is exactly what it was before stylization existed.  That
//      matters most at the 4K resolutions this renderer targets, where six
//      Float3 channels would cost roughly 840 MB.
// ---------------------------------------------------------------------------
static int s_styleBufferWidth  = 0;
static int s_styleBufferHeight = 0;

static void freeChannelBuffers()
{
    // [0] aliases s_accumBuffer_d and is not owned here.
    for (int c = 1; c < STYLE_CH_COUNT; ++c) {
        if (s_accumChannel_d[c]) { cudaFree(s_accumChannel_d[c]); }
        s_accumChannel_d[c] = nullptr;
    }
    for (int c = 0; c < STYLE_CH_COUNT; ++c) {
        if (s_denoisedChannel_d[c]) { cudaFree(s_denoisedChannel_d[c]); }
        s_denoisedChannel_d[c] = nullptr;
    }
    if (s_accumEdge_d) { cudaFree(s_accumEdge_d); s_accumEdge_d = nullptr; }
}

void freeStyleBuffers()
{
    freeChannelBuffers();

    if (s_accumAlbedo_d)     { cudaFree(s_accumAlbedo_d);     s_accumAlbedo_d = nullptr; }
    if (s_accumNormal_d)     { cudaFree(s_accumNormal_d);     s_accumNormal_d = nullptr; }
    if (s_primaryDepth_d)    { cudaFree(s_primaryDepth_d);    s_primaryDepth_d = nullptr; }
    if (s_primaryStyleId_d)  { cudaFree(s_primaryStyleId_d);  s_primaryStyleId_d = nullptr; }
    if (s_primaryObjectId_d) { cudaFree(s_primaryObjectId_d); s_primaryObjectId_d = nullptr; }

    s_styleBufferWidth  = 0;
    s_styleBufferHeight = 0;
}

void ensureStyleBuffers(int width, int height)
{
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);

    if (s_styleBufferWidth != width || s_styleBufferHeight != height) {
        freeStyleBuffers();
        s_styleBufferWidth  = width;
        s_styleBufferHeight = height;
    }

    if (!s_accumAlbedo_d) {
        CUDA_CHECK(cudaMalloc(&s_accumAlbedo_d,     sizeof(Float3) * n));
        CUDA_CHECK(cudaMalloc(&s_accumNormal_d,     sizeof(Float3) * n));
        CUDA_CHECK(cudaMalloc(&s_primaryDepth_d,    sizeof(float)  * n));
        CUDA_CHECK(cudaMalloc(&s_primaryStyleId_d,  sizeof(int)    * n));
        CUDA_CHECK(cudaMalloc(&s_primaryObjectId_d, sizeof(int)    * n));
    }

    // Channel [0] always aliases the single accumulation buffer, so anime mode
    // adds five buffers rather than six.
    s_accumChannel_d[0] = s_accumBuffer_d;

    if (s_styleMode == StyleMode::Anime) {
        if (!s_accumChannel_d[1]) {
            for (int c = 1; c < STYLE_CH_COUNT; ++c)
                CUDA_CHECK(cudaMalloc(&s_accumChannel_d[c], sizeof(Float3) * n));

            // Only the smooth channels are ever denoised.
            CUDA_CHECK(cudaMalloc(&s_denoisedChannel_d[STYLE_CH_INDIRECT_DIFFUSE], sizeof(Float3) * n));
            CUDA_CHECK(cudaMalloc(&s_denoisedChannel_d[STYLE_CH_REFLECTED],        sizeof(Float3) * n));
            CUDA_CHECK(cudaMalloc(&s_denoisedChannel_d[STYLE_CH_TRANSMITTED],      sizeof(Float3) * n));

            CUDA_CHECK(cudaMalloc(&s_accumEdge_d, sizeof(float) * n));
        }
    } else if (s_accumChannel_d[1]) {
        freeChannelBuffers();
    }

    // Physical mode: every channel is the same buffer, which is what lets shade
    // and connect stay free of style-mode branches.
    if (s_styleMode == StyleMode::Physical) {
        for (int c = 0; c < STYLE_CH_COUNT; ++c)
            s_accumChannel_d[c] = s_accumBuffer_d;
    }
}

static void clearStyleBuffers(int width, int height)
{
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);

    if (s_accumAlbedo_d)     CUDA_CHECK(cudaMemset(s_accumAlbedo_d,     0, sizeof(Float3) * n));
    if (s_accumNormal_d)     CUDA_CHECK(cudaMemset(s_accumNormal_d,     0, sizeof(Float3) * n));
    if (s_accumEdge_d)       CUDA_CHECK(cudaMemset(s_accumEdge_d,       0, sizeof(float)  * n));
    if (s_primaryDepth_d)    CUDA_CHECK(cudaMemset(s_primaryDepth_d,    0, sizeof(float)  * n));
    if (s_primaryStyleId_d)  CUDA_CHECK(cudaMemset(s_primaryStyleId_d,  0xFF, sizeof(int) * n));
    if (s_primaryObjectId_d) CUDA_CHECK(cudaMemset(s_primaryObjectId_d, 0xFF, sizeof(int) * n));

    // [0] aliases s_accumBuffer_d, which the caller has already cleared.
    for (int c = 1; c < STYLE_CH_COUNT; ++c) {
        if (s_accumChannel_d[c])
            CUDA_CHECK(cudaMemset(s_accumChannel_d[c], 0, sizeof(Float3) * n));
    }
}

void cudaResetAccumulation(int imageWidth, int imageHeight)
{
    // Called before the first frame has sized the buffers (e.g. from a style
    // toggle at startup), in which case there is nothing to reset yet.
    if (imageWidth <= 0 || imageHeight <= 0) {
        s_frameIndex = 0;
        return;
    }

    // Reallocate buffer if the resolution has changed.
    if (s_accumBuffer_d && (s_accumWidth != imageWidth || s_accumHeight != imageHeight)) {
        cudaFree(s_accumBuffer_d);
        s_accumBuffer_d = nullptr;
    }
    if (!s_accumBuffer_d) {
        s_accumWidth  = imageWidth;
        s_accumHeight = imageHeight;
        CUDA_CHECK(cudaMalloc(&s_accumBuffer_d,
                              sizeof(Float3) * static_cast<size_t>(imageWidth * imageHeight)));
    }
    CUDA_CHECK(cudaMemset(s_accumBuffer_d, 0,
                          sizeof(Float3) * static_cast<size_t>(imageWidth * imageHeight)));

    ensureStyleBuffers(imageWidth, imageHeight);
    clearStyleBuffers(imageWidth, imageHeight);

    s_frameIndex = 0;
}

void cudaRender(int imageWidth, int imageHeight)
{
    // Auto-initialise the accumulation buffer on the first call or if the
    // resolution has changed (e.g. window resize).
    if (!s_accumBuffer_d || s_accumWidth != imageWidth || s_accumHeight != imageHeight) {
        cudaResetAccumulation(imageWidth, imageHeight);
    }

    // Map the PBO so CUDA can write into it
    CUDA_CHECK(cudaGraphicsMapResources(1, &s_pboResource, 0));

    uint32_t* devPtr = nullptr;
    size_t    bufSize = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&devPtr), &bufSize, s_pboResource));

    // ── Dispatch to the active render path ─────────────────────────────
    if (s_renderMode == RenderMode::Wavefront) {
        cudaRenderWavefront(devPtr, imageWidth, imageHeight);

        ++s_frameIndex;
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &s_pboResource, 0));
        return;
    }

    // ── Megakernel path (OptiX __raygen__rg) ────────────────────────────
    // Fill the launch parameters and upload to the device.
    LaunchParams lp = {};
    lp.framebuffer          = devPtr;
    lp.accumBuffer          = s_accumBuffer_d;
    lp.width                = imageWidth;
    lp.height               = imageHeight;
    lp.launchOffsetY        = 0;
    lp.frameIndex           = s_frameIndex;
    lp.camera               = s_camera_h;
    lp.handle               = s_gasHandle;
    lp.triangles            = s_triangles_d;
    lp.triangleCount        = s_triangleCount;
    lp.bsdfs                = s_bsdfs_d;
    lp.bsdfCount            = s_bsdfCount;
    lp.triangleBsdfIds      = s_triangleBsdfIds_d;
    lp.triangleMaterialIds  = s_triangleMaterialIds_d;
    lp.triangleEmitterFlags = s_triangleEmitterFlags_d;
    lp.triangleEmission     = s_triangleEmission_d;
    lp.triangleObjectIds    = s_triangleObjectIds_d;
    lp.emitters             = s_emitters_d;
    lp.emitterCount         = s_emitterCount;
    lp.emitterTriIndices    = s_emitterTriIndices_d;
    lp.emitterTriCdf        = s_emitterTriCdf_d;
    lp.sceneEmitterCdf      = s_sceneEmitterCdf_d;
    lp.emitterSelectCount   = s_emitterSelectCount;
    lp.envEmitterSlot       = s_envEmitterSlot;
    lp.spotlights           = s_spotlights_d;
    lp.spotlightCount       = s_spotlightCount;
    lp.texObjects           = s_texObjects_d;
    lp.textureCount         = s_textureCount;
    lp.env                  = s_env;
    lp.styles               = s_styles_d;
    lp.styleCount           = s_styleCount;

    // Render the frame in horizontal row-band tiles. Each optixLaunch covers
    // only TILE_HEIGHT scanlines so that no single GPU command runs long enough
    // to trip the Windows display-driver watchdog (TDR). Output is identical to
    // a single full-frame launch — only the launch granularity changes.
    constexpr int TILE_HEIGHT = 64;
    for (int y0 = 0; y0 < imageHeight; y0 += TILE_HEIGHT) {
        const int rows = (imageHeight - y0 < TILE_HEIGHT) ? (imageHeight - y0) : TILE_HEIGHT;

        lp.launchOffsetY = y0;
        CUDA_CHECK(cudaMemcpy(s_launchParams_d, &lp, sizeof(LaunchParams), cudaMemcpyHostToDevice));

        OPTIX_CHECK(optixLaunch(s_optixPipeline, 0 /*stream*/,
                                reinterpret_cast<CUdeviceptr>(s_launchParams_d),
                                sizeof(LaunchParams), &s_sbt,
                                static_cast<unsigned int>(imageWidth),
                                static_cast<unsigned int>(rows), 1));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ++s_frameIndex;

    // Optional: denoise the accumulated image and tonemap into the PBO,
    // overwriting the raygen's direct write.
    if (s_denoiserEnabled) {
        denoiseAndPresent(devPtr, imageWidth, imageHeight);
    }

    // Unmap so OpenGL can read the PBO
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &s_pboResource, 0));
}

void cudaCleanup()
{
    if (s_pboResource) {
        cudaGraphicsUnregisterResource(s_pboResource);
        s_pboResource = nullptr;
    }

    // Must run before s_accumBuffer_d is released: channel [0] aliases it.
    freeStyleBuffers();

    if (s_accumBuffer_d) {
        cudaFree(s_accumBuffer_d);
        s_accumBuffer_d = nullptr;
        s_accumWidth    = 0;
        s_accumHeight   = 0;
        s_frameIndex    = 0;
    }
    for (int c = 0; c < STYLE_CH_COUNT; ++c) s_accumChannel_d[c] = nullptr;

    freeSceneUploads();
    freeDenoiserState();
    freeWavefrontState();
    freeOptixState();
}

// ---------------------------------------------------------------------------
//  Render mode API
// ---------------------------------------------------------------------------
void cudaSetRenderMode(RenderMode mode)
{
    if (mode == s_renderMode) return;
    s_renderMode = mode;
    cudaResetAccumulation(s_accumWidth, s_accumHeight);
    fprintf(stdout, "[render] Mode switched to %s\n",
            mode == RenderMode::Wavefront ? "Wavefront" : "Megakernel");
}

RenderMode cudaGetRenderMode()
{
    return s_renderMode;
}

// ---------------------------------------------------------------------------
//  Style mode API
//
//  Switching modes changes what the accumulation buffers mean — physical mode
//  collapses all six channels onto one buffer — so accumulation resets, exactly
//  as it does for a render-mode switch.
// ---------------------------------------------------------------------------
void cudaSetStyleMode(StyleMode mode)
{
    if (mode == s_styleMode) return;
    s_styleMode = mode;
    cudaResetAccumulation(s_accumWidth, s_accumHeight);
    fprintf(stdout, "[style] Mode switched to %s\n",
            mode == StyleMode::Anime ? "Anime" : "Physical");

    if (mode == StyleMode::Anime && s_renderMode == RenderMode::Megakernel) {
        fprintf(stdout, "[style] Note: the megakernel is physical-only and stays "
                        "the correctness reference; switch to the wavefront "
                        "renderer to see the stylized result.\n");
    }
}

StyleMode cudaGetStyleMode()
{
    return s_styleMode;
}

bool cudaToggleStyleMode()
{
    cudaSetStyleMode(s_styleMode == StyleMode::Anime ? StyleMode::Physical
                                                     : StyleMode::Anime);
    return s_styleMode == StyleMode::Anime;
}

// Debug views are pure presentation, so cycling one costs a single kernel launch
// and never touches the converged data.
DebugView cudaCycleDebugView()
{
    const int next = (static_cast<int>(s_debugView) + 1) %
                     static_cast<int>(DebugView::Count);
    s_debugView = static_cast<DebugView>(next);

    static const char* kNames[] = {
        "Off", "Albedo", "Normal", "ObjectId", "Depth", "Edge",
        "DirectDiffuse", "DirectSpecular", "IndirectDiffuse",
        "Reflected", "Transmitted", "Emissive"
    };
    fprintf(stdout, "[debug] View: %s\n", kNames[next]);
    return s_debugView;
}

void cudaRepresent(int imageWidth, int imageHeight)
{
    if (!s_pboResource || !s_accumBuffer_d) return;
    if (imageWidth <= 0 || imageHeight <= 0) return;

    CUDA_CHECK(cudaGraphicsMapResources(1, &s_pboResource, 0));

    uint32_t* devPtr = nullptr;
    size_t    bufSize = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&devPtr), &bufSize, s_pboResource));

    if (s_renderMode == RenderMode::Wavefront) {
        cudaRepresentWavefront(devPtr, imageWidth, imageHeight);
    } else if (s_denoiserEnabled) {
        denoiseAndPresent(devPtr, imageWidth, imageHeight);
    }

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &s_pboResource, 0));
    (void)bufSize;
}
