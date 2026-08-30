// ============================================================================
//  scene_upload.cu — host → device uploads of the flattened scene
//  (triangles, materials, emission, emitter tables, BSDFs, spotlights,
//  textures, camera).
// ============================================================================

#include "render_kernel.h"
#include "renderer_state.h"
#include "cuda_check.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

void cudaInitScene(const TriangleData* triangles, int triangleCount,
                    const Float3* materials, int materialCount,
                    const int* triangleMaterialIds,
                    int /*imageWidth*/, int /*imageHeight*/)
{
    // Free previous scene buffers (if re-initialising)
    if (s_triangles_d) {
        cudaFree(s_triangles_d);
        s_triangles_d = nullptr;
    }
    if (s_materials_d) {
        cudaFree(s_materials_d);
        s_materials_d = nullptr;
    }
    if (s_triangleMaterialIds_d) {
        cudaFree(s_triangleMaterialIds_d);
        s_triangleMaterialIds_d = nullptr;
    }

    s_triangleCount = triangleCount;
    s_materialCount = materialCount;

    if (triangleCount > 0) {
        CUDA_CHECK(cudaMalloc(&s_triangles_d,
                              sizeof(TriangleData) * static_cast<size_t>(triangleCount)));
        CUDA_CHECK(cudaMemcpy(s_triangles_d, triangles,
                               sizeof(TriangleData) * static_cast<size_t>(triangleCount),
                               cudaMemcpyHostToDevice));
    }

    if (materialCount > 0) {
        CUDA_CHECK(cudaMalloc(&s_materials_d,
                              sizeof(Float3) * static_cast<size_t>(materialCount)));
        CUDA_CHECK(cudaMemcpy(s_materials_d, materials,
                               sizeof(Float3) * static_cast<size_t>(materialCount),
                               cudaMemcpyHostToDevice));
    }

    if (triangleMaterialIds && triangleCount > 0) {
        CUDA_CHECK(cudaMalloc(&s_triangleMaterialIds_d,
                              sizeof(int) * static_cast<size_t>(triangleCount)));
        CUDA_CHECK(cudaMemcpy(s_triangleMaterialIds_d, triangleMaterialIds,
                               sizeof(int) * static_cast<size_t>(triangleCount),
                               cudaMemcpyHostToDevice));
    }
}

void cudaInit(const TriangleData& tri, const Float3& color,
              int imageWidth, int imageHeight)
{
    // Wrap the old API into the new flattened scene API.
    cudaInitScene(&tri, 1, &color, 1, nullptr, imageWidth, imageHeight);
}

void cudaInitCamera(const CameraData& camera)
{
    // Stage the camera on the host; it is copied into LaunchParams each frame.
    // (The OptiX module reads the camera from launch params, not a __constant__.)
    s_camera_h = camera;
}

void cudaInitTriangleEmission(const Float3* triangleEmission, int triangleCount)
{
    if (s_triangleEmission_d) {
        cudaFree(s_triangleEmission_d);
        s_triangleEmission_d = nullptr;
    }

    if (!triangleEmission || triangleCount <= 0) return;

    CUDA_CHECK(cudaMalloc(&s_triangleEmission_d,
                          sizeof(Float3) * static_cast<size_t>(triangleCount)));
    CUDA_CHECK(cudaMemcpy(s_triangleEmission_d, triangleEmission,
                          sizeof(Float3) * static_cast<size_t>(triangleCount),
                          cudaMemcpyHostToDevice));
}

void cudaInitTriangleEmitterFlags(const uint8_t* flags, int triangleCount)
{
    if (s_triangleEmitterFlags_d) {
        cudaFree(s_triangleEmitterFlags_d);
        s_triangleEmitterFlags_d = nullptr;
    }

    if (!flags || triangleCount <= 0) return;

    CUDA_CHECK(cudaMalloc(&s_triangleEmitterFlags_d,
                          sizeof(uint8_t) * static_cast<size_t>(triangleCount)));
    CUDA_CHECK(cudaMemcpy(s_triangleEmitterFlags_d, flags,
                          sizeof(uint8_t) * static_cast<size_t>(triangleCount),
                          cudaMemcpyHostToDevice));
}

void cudaInitEmitterTable(const EmitterData* emitters, int emitterCount,
                          const int* emitterTriIndices, int emitterTriIndexCount,
                          const float* emitterTriCdf, int emitterTriCdfCount,
                          const float* sceneEmitterCdf)
{
    // Free previous
    if (s_emitters_d) { cudaFree(s_emitters_d); s_emitters_d = nullptr; }
    if (s_emitterTriIndices_d) { cudaFree(s_emitterTriIndices_d); s_emitterTriIndices_d = nullptr; }
    if (s_emitterTriCdf_d) { cudaFree(s_emitterTriCdf_d); s_emitterTriCdf_d = nullptr; }
    if (s_sceneEmitterCdf_d) { cudaFree(s_sceneEmitterCdf_d); s_sceneEmitterCdf_d = nullptr; }
    s_emitterCount = 0;

    if (!emitters || emitterCount <= 0 ||
        !emitterTriIndices || emitterTriIndexCount <= 0 ||
        !emitterTriCdf || emitterTriCdfCount <= 0 ||
        !sceneEmitterCdf)
        return;

    s_emitterCount = emitterCount;

    // Emitters table
    CUDA_CHECK(cudaMalloc(&s_emitters_d, sizeof(EmitterData) * static_cast<size_t>(emitterCount)));
    CUDA_CHECK(cudaMemcpy(s_emitters_d, emitters,
                          sizeof(EmitterData) * static_cast<size_t>(emitterCount),
                          cudaMemcpyHostToDevice));

    // Concatenated triangle indices
    CUDA_CHECK(cudaMalloc(&s_emitterTriIndices_d,
                          sizeof(int) * static_cast<size_t>(emitterTriIndexCount)));
    CUDA_CHECK(cudaMemcpy(s_emitterTriIndices_d, emitterTriIndices,
                          sizeof(int) * static_cast<size_t>(emitterTriIndexCount),
                          cudaMemcpyHostToDevice));

    // Concatenated CDFs
    CUDA_CHECK(cudaMalloc(&s_emitterTriCdf_d,
                          sizeof(float) * static_cast<size_t>(emitterTriCdfCount)));
    CUDA_CHECK(cudaMemcpy(s_emitterTriCdf_d, emitterTriCdf,
                          sizeof(float) * static_cast<size_t>(emitterTriCdfCount),
                          cudaMemcpyHostToDevice));

    // Scene-level emitter selection CDF
    CUDA_CHECK(cudaMalloc(&s_sceneEmitterCdf_d,
                          sizeof(float) * static_cast<size_t>(emitterCount + 1)));
    CUDA_CHECK(cudaMemcpy(s_sceneEmitterCdf_d, sceneEmitterCdf,
                          sizeof(float) * static_cast<size_t>(emitterCount + 1),
                          cudaMemcpyHostToDevice));
}

void cudaInitBsdfs(const BsdfData* bsdfs, int bsdfCount,
                   const int* triangleBsdfIds, int triangleCount)
{
    if (s_bsdfs_d) { cudaFree(s_bsdfs_d); s_bsdfs_d = nullptr; }
    if (s_triangleBsdfIds_d) { cudaFree(s_triangleBsdfIds_d); s_triangleBsdfIds_d = nullptr; }
    s_bsdfCount = 0;

    if (!bsdfs || bsdfCount <= 0 || !triangleBsdfIds || triangleCount <= 0)
        return;

    s_bsdfCount = bsdfCount;

    CUDA_CHECK(cudaMalloc(&s_bsdfs_d, sizeof(BsdfData) * static_cast<size_t>(bsdfCount)));
    CUDA_CHECK(cudaMemcpy(s_bsdfs_d, bsdfs,
                          sizeof(BsdfData) * static_cast<size_t>(bsdfCount),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&s_triangleBsdfIds_d, sizeof(int) * static_cast<size_t>(triangleCount)));
    CUDA_CHECK(cudaMemcpy(s_triangleBsdfIds_d, triangleBsdfIds,
                          sizeof(int) * static_cast<size_t>(triangleCount),
                          cudaMemcpyHostToDevice));
}

void cudaInitSpotlights(const SpotlightData* spotlights, int spotlightCount)
{
    if (s_spotlights_d) {
        cudaFree(s_spotlights_d);
        s_spotlights_d = nullptr;
    }
    s_spotlightCount = 0;

    if (!spotlights || spotlightCount <= 0) return;

    s_spotlightCount = spotlightCount;
    CUDA_CHECK(cudaMalloc(&s_spotlights_d,
                          sizeof(SpotlightData) * static_cast<size_t>(spotlightCount)));
    CUDA_CHECK(cudaMemcpy(s_spotlights_d, spotlights,
                          sizeof(SpotlightData) * static_cast<size_t>(spotlightCount),
                          cudaMemcpyHostToDevice));
}

void cudaInitTextures(const uint8_t* const* pixels,
                      const int* widths,
                      const int* heights,
                      int textureCount)
{
    // Release any previously uploaded textures.
    for (auto& texObj : s_texObjects_h) if (texObj) cudaDestroyTextureObject(texObj);
    for (auto& arr : s_cuArrays)        if (arr)    cudaFreeArray(arr);
    s_texObjects_h.clear();
    s_cuArrays.clear();
    if (s_texObjects_d) { cudaFree(s_texObjects_d); s_texObjects_d = nullptr; }
    s_textureCount = 0;

    if (!pixels || textureCount <= 0) return;

    s_cuArrays.resize(static_cast<size_t>(textureCount), nullptr);
    s_texObjects_h.resize(static_cast<size_t>(textureCount), 0);

    for (int i = 0; i < textureCount; ++i) {
        // nullptr means this material has no texture — leave the handle as 0.
        // The kernel checks texObj != 0 before sampling, so the BSDF's own
        // base_color / albedo parameters are used unchanged.
        if (!pixels[i]) {
            s_cuArrays[static_cast<size_t>(i)]     = nullptr;
            s_texObjects_h[static_cast<size_t>(i)] = 0;
            continue;
        }

        const uint8_t* srcPixels = pixels[i];
        const int w = widths[i];
        const int h = heights[i];

        // Allocate a 2-D CUDA array (RGBA8 per texel).
        const cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
        CUDA_CHECK(cudaMallocArray(&s_cuArrays[static_cast<size_t>(i)], &desc,
                                   static_cast<size_t>(w), static_cast<size_t>(h)));

        // Copy host pixels into the CUDA array (row-major, 4 bytes per pixel).
        CUDA_CHECK(cudaMemcpy2DToArray(
            s_cuArrays[static_cast<size_t>(i)], 0, 0,
            srcPixels, static_cast<size_t>(w) * 4,
            static_cast<size_t>(w) * 4, static_cast<size_t>(h),
            cudaMemcpyHostToDevice));

        // Create a texture object with bilinear filtering, UV wrap, normalised coords.
        // readMode = NormalizedFloat converts uchar4 → float4 in [0,1] automatically.
        cudaResourceDesc resDesc{};
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = s_cuArrays[static_cast<size_t>(i)];

        cudaTextureDesc texDesc{};
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;

        CUDA_CHECK(cudaCreateTextureObject(
            &s_texObjects_h[static_cast<size_t>(i)], &resDesc, &texDesc, nullptr));

        fprintf(stdout, "[texture] Uploaded material %d: %d×%d\n", i, w, h);
    }

    s_textureCount = textureCount;

    // Copy the array of texture-object handles to the device so the kernel can index them.
    CUDA_CHECK(cudaMalloc(&s_texObjects_d,
                          sizeof(cudaTextureObject_t) * static_cast<size_t>(textureCount)));
    CUDA_CHECK(cudaMemcpy(s_texObjects_d, s_texObjects_h.data(),
                          sizeof(cudaTextureObject_t) * static_cast<size_t>(textureCount),
                          cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
//  Teardown (called from cudaCleanup)
// ---------------------------------------------------------------------------
void freeSceneUploads()
{
    if (s_triangles_d) {
        cudaFree(s_triangles_d);
        s_triangles_d = nullptr;
        s_triangleCount = 0;
    }
    if (s_materials_d) {
        cudaFree(s_materials_d);
        s_materials_d = nullptr;
        s_materialCount = 0;
    }
    if (s_triangleMaterialIds_d) {
        cudaFree(s_triangleMaterialIds_d);
        s_triangleMaterialIds_d = nullptr;
    }

    if (s_triangleEmission_d) {
        cudaFree(s_triangleEmission_d);
        s_triangleEmission_d = nullptr;
    }

    if (s_triangleEmitterFlags_d) {
        cudaFree(s_triangleEmitterFlags_d);
        s_triangleEmitterFlags_d = nullptr;
    }

    if (s_emitters_d) {
        cudaFree(s_emitters_d);
        s_emitters_d = nullptr;
    }
    if (s_emitterTriIndices_d) {
        cudaFree(s_emitterTriIndices_d);
        s_emitterTriIndices_d = nullptr;
    }
    if (s_emitterTriCdf_d) {
        cudaFree(s_emitterTriCdf_d);
        s_emitterTriCdf_d = nullptr;
    }
    if (s_sceneEmitterCdf_d) {
        cudaFree(s_sceneEmitterCdf_d);
        s_sceneEmitterCdf_d = nullptr;
    }
    s_emitterCount = 0;

    if (s_bsdfs_d) {
        cudaFree(s_bsdfs_d);
        s_bsdfs_d = nullptr;
    }
    if (s_triangleBsdfIds_d) {
        cudaFree(s_triangleBsdfIds_d);
        s_triangleBsdfIds_d = nullptr;
    }
    s_bsdfCount = 0;

    if (s_spotlights_d) {
        cudaFree(s_spotlights_d);
        s_spotlights_d = nullptr;
    }
    s_spotlightCount = 0;

    for (auto& texObj : s_texObjects_h) if (texObj) cudaDestroyTextureObject(texObj);
    for (auto& arr : s_cuArrays)        if (arr)    cudaFreeArray(arr);
    s_texObjects_h.clear();
    s_cuArrays.clear();
    if (s_texObjects_d) { cudaFree(s_texObjects_d); s_texObjects_d = nullptr; }
    s_textureCount = 0;
}

// ===========================================================================
//  Environment map (HDRI IBL) upload + marginal–conditional CDF construction
// ===========================================================================

void freeEnvMapState()
{
    if (s_envMapTexObj) {
        cudaDestroyTextureObject(s_envMapTexObj);
        s_envMapTexObj = 0;
    }
    if (s_envMapCuArray) {
        cudaFreeArray(s_envMapCuArray);
        s_envMapCuArray = nullptr;
    }
    if (s_envMapMarginalCdf_d) {
        cudaFree(s_envMapMarginalCdf_d);
        s_envMapMarginalCdf_d = nullptr;
    }
    if (s_envMapConditionalCdf_d) {
        cudaFree(s_envMapConditionalCdf_d);
        s_envMapConditionalCdf_d = nullptr;
    }
    s_envMap_h = {};
    s_hasEnvMap = false;
}

void cudaInitEnvMap(const float* hdriPixelsRGBA, int width, int height,
                    float intensity, float rotationDeg)
{
    // Release any previous env map.
    freeEnvMapState();

    if (!hdriPixelsRGBA || width <= 0 || height <= 0) return;

    // ── 1. Upload HDR float4 data as a CUDA array ────────────────────────
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    CUDA_CHECK(cudaMallocArray(&s_envMapCuArray, &desc,
                               static_cast<size_t>(width), static_cast<size_t>(height)));
    CUDA_CHECK(cudaMemcpy2DToArray(
        s_envMapCuArray, 0, 0,
        hdriPixelsRGBA, static_cast<size_t>(width) * sizeof(float4),
        static_cast<size_t>(width) * sizeof(float4), static_cast<size_t>(height),
        cudaMemcpyHostToDevice));

    // ── 2. Create a texture object (bilinear, wrap, normalised, float) ───
    cudaResourceDesc resDesc{};
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = s_envMapCuArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeClamp; // clamp vertically (poles)
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType; // data is already float
    texDesc.normalizedCoords = 1;

    CUDA_CHECK(cudaCreateTextureObject(&s_envMapTexObj, &resDesc, &texDesc, nullptr));

    // ── 3. Build 2D marginal–conditional CDF on the host ────────────────
    //  For each pixel (ix, iy):
    //    luminance = 0.2126*R + 0.7152*G + 0.0722*B
    //    weight    = luminance * sin(theta)    (solid-angle correction)
    //  Conditional CDF: per-row prefix sum across columns (width+1 entries per row).
    //  Marginal CDF: prefix sum of row integrals (height+1 entries).
    const int W = width;
    const int H = height;
    const size_t condSize = static_cast<size_t>(H) * static_cast<size_t>(W + 1);
    const size_t margSize = static_cast<size_t>(H + 1);

    std::vector<float> condCdf(condSize, 0.0f);
    std::vector<float> margCdf(margSize, 0.0f);

    static constexpr float PI = 3.14159265358979323846f;

    for (int iy = 0; iy < H; ++iy) {
        // θ at the centre of this row
        const float theta = PI * (static_cast<float>(iy) + 0.5f) / static_cast<float>(H);
        const float sinTheta = std::sin(theta);

        float* rowCdf = condCdf.data() + static_cast<size_t>(iy) * static_cast<size_t>(W + 1);
        rowCdf[0] = 0.0f;

        for (int ix = 0; ix < W; ++ix) {
            const size_t pixelBase = (static_cast<size_t>(iy) * static_cast<size_t>(W) + static_cast<size_t>(ix)) * 4;
            const float r = hdriPixelsRGBA[pixelBase + 0];
            const float g = hdriPixelsRGBA[pixelBase + 1];
            const float b = hdriPixelsRGBA[pixelBase + 2];
            const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            rowCdf[ix + 1] = rowCdf[ix] + lum * sinTheta;
        }

        // Marginal CDF: integral of this row
        margCdf[static_cast<size_t>(iy) + 1] = margCdf[static_cast<size_t>(iy)] + rowCdf[W];
    }

    const float totalPower = margCdf[static_cast<size_t>(H)];

    // ── 4. Upload CDFs to device ─────────────────────────────────────────
    CUDA_CHECK(cudaMalloc(&s_envMapMarginalCdf_d,    margSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(s_envMapMarginalCdf_d,     margCdf.data(),
                          margSize * sizeof(float),  cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&s_envMapConditionalCdf_d, condSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(s_envMapConditionalCdf_d,  condCdf.data(),
                          condSize * sizeof(float),  cudaMemcpyHostToDevice));

    // ── 5. Fill the host-side descriptor ──────────────────────────────────
    s_envMap_h.texObj          = static_cast<unsigned long long>(s_envMapTexObj);
    s_envMap_h.width           = W;
    s_envMap_h.height          = H;
    s_envMap_h.intensity       = intensity;
    s_envMap_h.rotationRadians = rotationDeg * (PI / 180.0f);
    s_envMap_h.marginalCdf     = s_envMapMarginalCdf_d;
    s_envMap_h.conditionalCdf  = s_envMapConditionalCdf_d;
    s_envMap_h.totalPower      = totalPower;

    s_hasEnvMap = true;

    fprintf(stdout, "[envmap] Uploaded %dx%d HDR environment map (intensity=%.2f, rotation=%.1f°, totalPower=%.2f)\n",
            W, H, intensity, rotationDeg, totalPower);
}
