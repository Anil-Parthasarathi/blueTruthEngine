// ============================================================================
//  denoiser.cu — OptiX AI denoiser (HDR model) + linear-buffer presentation.
// ============================================================================

#include "renderer_state.h"
#include "cuda_check.h"
#include "rt_path_integrator.cuh"   // packPixelGamma

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------------
//  Tonemap a linear-space buffer (e.g. the denoiser output) into the RGBA8 PBO.
//  Uses the identical gamma 2.2 encode as __raygen__rg (packPixelGamma) so
//  denoised and non-denoised frames match tonally.
// ---------------------------------------------------------------------------
__global__ void presentLinearKernel(uint32_t* framebuffer, const Float3* linear,
                                     int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    framebuffer[idx] = packPixelGamma(linear[idx]);
}

// Lazily creates / resizes the denoiser for the given resolution.
static void ensureDenoiser(int width, int height)
{
    if (s_denoiser && s_denoiserWidth == width && s_denoiserHeight == height) return;

    if (s_denoiser)          { optixDenoiserDestroy(s_denoiser); s_denoiser = nullptr; }
    if (s_denoiserState)     { cudaFree(reinterpret_cast<void*>(s_denoiserState));   s_denoiserState = 0; }
    if (s_denoiserScratch)   { cudaFree(reinterpret_cast<void*>(s_denoiserScratch)); s_denoiserScratch = 0; }
    if (s_denoisedBuffer_d)  { cudaFree(s_denoisedBuffer_d); s_denoisedBuffer_d = nullptr; }
    if (!s_denoiserIntensity) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_denoiserIntensity), sizeof(float)));

    OptixDenoiserOptions options = {};
    options.guideAlbedo  = 0;
    options.guideNormal  = 0;
    options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
    OPTIX_CHECK(optixDenoiserCreate(s_optixContext, OPTIX_DENOISER_MODEL_KIND_HDR,
                                    &options, &s_denoiser));

    OptixDenoiserSizes sizes = {};
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(s_denoiser,
                static_cast<unsigned int>(width), static_cast<unsigned int>(height), &sizes));
    s_denoiserStateSize   = sizes.stateSizeInBytes;
    s_denoiserScratchSize = sizes.withoutOverlapScratchSizeInBytes;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_denoiserState),   s_denoiserStateSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_denoiserScratch), s_denoiserScratchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_denoisedBuffer_d),
                          sizeof(Float3) * static_cast<size_t>(width) * static_cast<size_t>(height)));

    OPTIX_CHECK(optixDenoiserSetup(s_denoiser, 0,
                static_cast<unsigned int>(width), static_cast<unsigned int>(height),
                s_denoiserState, s_denoiserStateSize,
                s_denoiserScratch, s_denoiserScratchSize));

    s_denoiserWidth  = width;
    s_denoiserHeight = height;
}

// Runs the denoiser on the accumulated linear image and tonemaps the result
// into the mapped PBO (devPtr).
void denoiseAndPresent(uint32_t* devPtr, int width, int height)
{
    ensureDenoiser(width, height);

    OptixImage2D inputImage = {};
    inputImage.data              = reinterpret_cast<CUdeviceptr>(s_accumBuffer_d);
    inputImage.width             = static_cast<unsigned int>(width);
    inputImage.height            = static_cast<unsigned int>(height);
    inputImage.rowStrideInBytes  = static_cast<unsigned int>(width * sizeof(Float3));
    inputImage.pixelStrideInBytes = static_cast<unsigned int>(sizeof(Float3));
    inputImage.format            = OPTIX_PIXEL_FORMAT_FLOAT3;

    OptixImage2D outputImage = inputImage;
    outputImage.data = reinterpret_cast<CUdeviceptr>(s_denoisedBuffer_d);

    OPTIX_CHECK(optixDenoiserComputeIntensity(s_denoiser, 0, &inputImage,
                s_denoiserIntensity, s_denoiserScratch, s_denoiserScratchSize));

    OptixDenoiserParams params = {};
    params.hdrIntensity = s_denoiserIntensity;
    params.blendFactor  = 0.0f;

    OptixDenoiserGuideLayer guideLayer = {};
    OptixDenoiserLayer layer = {};
    layer.input  = inputImage;
    layer.output = outputImage;

    OPTIX_CHECK(optixDenoiserInvoke(s_denoiser, 0, &params,
                s_denoiserState, s_denoiserStateSize,
                &guideLayer, &layer, 1, 0, 0,
                s_denoiserScratch, s_denoiserScratchSize));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    presentLinearKernel<<<grid, block>>>(devPtr, s_denoisedBuffer_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

bool cudaToggleDenoiser()
{
    s_denoiserEnabled = !s_denoiserEnabled;
    fprintf(stdout, "[denoise] %s\n", s_denoiserEnabled ? "ON" : "OFF");
    return s_denoiserEnabled;
}

// ---------------------------------------------------------------------------
//  Teardown (called from cudaCleanup)
// ---------------------------------------------------------------------------
void freeDenoiserState()
{
    if (s_denoiser)         { optixDenoiserDestroy(s_denoiser); s_denoiser = nullptr; }
    if (s_denoiserState)    { cudaFree(reinterpret_cast<void*>(s_denoiserState));   s_denoiserState = 0; }
    if (s_denoiserScratch)  { cudaFree(reinterpret_cast<void*>(s_denoiserScratch)); s_denoiserScratch = 0; }
    if (s_denoiserIntensity){ cudaFree(reinterpret_cast<void*>(s_denoiserIntensity)); s_denoiserIntensity = 0; }
    if (s_denoisedBuffer_d) { cudaFree(s_denoisedBuffer_d); s_denoisedBuffer_d = nullptr; }
    s_denoiserWidth = 0; s_denoiserHeight = 0;
}
