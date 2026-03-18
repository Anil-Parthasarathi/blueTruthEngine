// ============================================================================
//  render_kernel.cu  –  CUDA-GL interop + simple triangle rasteriser
// ============================================================================
//  This file owns all CUDA resources.  The host-side API declared in
//  render_kernel.h is implemented here.
//
//  Architecture for future path tracing:
//    • cudaInit()       – upload scene data (mesh, materials, BVH, …)
//    • cudaRegisterPBO() – register the GL pixel-buffer for interop
//    • cudaRender()     – launch the render kernel (swap in your tracer here)
//    • cudaCleanup()    – release everything
// ============================================================================

#include "render_kernel.h"

#include <glad/gl.h>       // Must come before cuda_gl_interop.h (defines GLuint)
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
//  Error-checking helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d – %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
//  Device-side constants (uploaded once from host)
// ---------------------------------------------------------------------------
__constant__ TriangleData d_triangle;
__constant__ UniformColor d_color;

// ---------------------------------------------------------------------------
//  CUDA-GL interop state
// ---------------------------------------------------------------------------
static cudaGraphicsResource* s_pboResource = nullptr;

// ---------------------------------------------------------------------------
//  Render kernel
// ---------------------------------------------------------------------------
//  Each thread computes one pixel.  A simple edge-function test determines
//  whether the pixel lies inside the triangle.  If it does, the pixel is
//  coloured with the uniform texture colour; otherwise a dark background
//  is written.
//
//  ► To convert this into a path tracer, replace the body of this kernel
//    with ray generation + tracing + shading.
// ---------------------------------------------------------------------------

__device__ float edgeFunction(float ax, float ay,
                               float bx, float by,
                               float cx, float cy)
{
    // Standard edge function: (B-A) × (C-A)
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

__global__ void renderKernel(uint32_t* framebuffer, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Convert pixel coord to normalised [-1,1] with y-up
    float u = (2.0f * x / width)  - 1.0f;
    float v = (2.0f * y / height) - 1.0f;

    // Fetch triangle vertices from constant memory
    float x0 = d_triangle.v0.x, y0 = d_triangle.v0.y;
    float x1 = d_triangle.v1.x, y1 = d_triangle.v1.y;
    float x2 = d_triangle.v2.x, y2 = d_triangle.v2.y;

    // Edge-function inside test
    float area  = edgeFunction(x0, y0, x1, y1, x2, y2);
    float w0    = edgeFunction(x1, y1, x2, y2, u, v);
    float w1    = edgeFunction(x2, y2, x0, y0, u, v);
    float w2    = edgeFunction(x0, y0, x1, y1, u, v);

    uint8_t r, g, b, a;

    if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
    {
        // Inside triangle → use uniform texture colour
        r = static_cast<uint8_t>(fminf(d_color.r * 255.0f, 255.0f));
        g = static_cast<uint8_t>(fminf(d_color.g * 255.0f, 255.0f));
        b = static_cast<uint8_t>(fminf(d_color.b * 255.0f, 255.0f));
        a = 255;
    }
    else
    {
        // Background: very dark grey
        r = 18; g = 18; b = 24; a = 255;
    }

    // Pack RGBA into a uint32 (ABGR byte order for GL_UNSIGNED_BYTE / RGBA)
    uint32_t pixel = (a << 24) | (b << 16) | (g << 8) | r;
    framebuffer[y * width + x] = pixel;
}

// ---------------------------------------------------------------------------
//  Host API implementation
// ---------------------------------------------------------------------------

void cudaInit(const TriangleData& tri, const UniformColor& color,
              int /*imageWidth*/, int /*imageHeight*/)
{
    CUDA_CHECK(cudaMemcpyToSymbol(d_triangle, &tri,   sizeof(TriangleData)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_color,    &color, sizeof(UniformColor)));
}

void cudaRegisterPBO(uint32_t pbo)
{
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &s_pboResource, pbo,
        cudaGraphicsMapFlagsWriteDiscard));
}

void cudaRender(int imageWidth, int imageHeight)
{
    // Map the PBO so CUDA can write into it
    CUDA_CHECK(cudaGraphicsMapResources(1, &s_pboResource, 0));

    uint32_t* devPtr = nullptr;
    size_t    bufSize = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&devPtr), &bufSize, s_pboResource));

    // Launch kernel – 16×16 threads per block
    dim3 block(16, 16);
    dim3 grid((imageWidth  + block.x - 1) / block.x,
              (imageHeight + block.y - 1) / block.y);

    renderKernel<<<grid, block>>>(devPtr, imageWidth, imageHeight);
    CUDA_CHECK(cudaGetLastError());

    // Unmap so OpenGL can read the PBO
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &s_pboResource, 0));
}

void cudaCleanup()
{
    if (s_pboResource) {
        cudaGraphicsUnregisterResource(s_pboResource);
        s_pboResource = nullptr;
    }
}
