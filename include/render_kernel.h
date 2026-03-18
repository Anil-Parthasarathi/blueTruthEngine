#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// Data types shared between host (C++) and device (CUDA)
// ---------------------------------------------------------------------------

struct Float3 {
    float x, y, z;
};

struct Float4 {
    float x, y, z, w;
};

/// A single triangle defined by three screen-space 2-D positions.
/// Positions are in normalised coordinates: x,y ∈ [-1, 1].
struct TriangleData {
    Float3 v0, v1, v2;
};

/// Uniform color extracted from the loaded texture (average RGB + alpha).
struct UniformColor {
    float r, g, b, a;
};

// ---------------------------------------------------------------------------
// Host-side API  (implemented in render_kernel.cu)
// ---------------------------------------------------------------------------

/// Upload triangle + color data to the GPU and allocate internal buffers.
/// Call once after loading the mesh and texture.
void cudaInit(const TriangleData& tri, const UniformColor& color,
              int imageWidth, int imageHeight);

/// Register an OpenGL PBO with CUDA so the kernel can write into it.
void cudaRegisterPBO(uint32_t pbo);

/// Launch the render kernel.  The kernel writes RGBA8 pixels into the
/// mapped PBO.  Call this every frame.
void cudaRender(int imageWidth, int imageHeight);

/// Unregister the PBO and free device memory.  Call before exit.
void cudaCleanup();
