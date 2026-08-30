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

void cudaResetAccumulation(int imageWidth, int imageHeight)
{
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
    lp.emitters             = s_emitters_d;
    lp.emitterCount         = s_emitterCount;
    lp.emitterTriIndices    = s_emitterTriIndices_d;
    lp.emitterTriCdf        = s_emitterTriCdf_d;
    lp.sceneEmitterCdf      = s_sceneEmitterCdf_d;
    lp.spotlights           = s_spotlights_d;
    lp.spotlightCount       = s_spotlightCount;
    lp.texObjects           = s_texObjects_d;
    lp.texObjectsMr         = s_texObjectsMr_d;
    lp.textureCount         = s_textureCount;
    lp.envMap               = s_envMap_h;
    lp.hasEnvMap            = s_hasEnvMap;

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

    if (s_accumBuffer_d) {
        cudaFree(s_accumBuffer_d);
        s_accumBuffer_d = nullptr;
        s_accumWidth    = 0;
        s_accumHeight   = 0;
        s_frameIndex    = 0;
    }

    freeSceneUploads();
    freeEnvMapState();
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
