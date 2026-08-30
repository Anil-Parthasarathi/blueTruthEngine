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
                          const float* sceneEmitterCdf, int selectCount)
{
    // Free previous
    if (s_emitters_d) { cudaFree(s_emitters_d); s_emitters_d = nullptr; }
    if (s_emitterTriIndices_d) { cudaFree(s_emitterTriIndices_d); s_emitterTriIndices_d = nullptr; }
    if (s_emitterTriCdf_d) { cudaFree(s_emitterTriCdf_d); s_emitterTriCdf_d = nullptr; }
    if (s_sceneEmitterCdf_d) { cudaFree(s_sceneEmitterCdf_d); s_sceneEmitterCdf_d = nullptr; }
    s_emitterCount       = 0;
    s_emitterSelectCount = 0;
    s_envEmitterSlot     = -1;

    // A scene can be lit by the environment alone, so the mesh emitter arrays are
    // optional; only the selection CDF is required.
    if (!sceneEmitterCdf || selectCount <= 0) return;

    s_emitterCount = (emitters && emitterCount > 0) ? emitterCount : 0;

    if (s_emitterCount > 0 &&
        emitterTriIndices && emitterTriIndexCount > 0 &&
        emitterTriCdf && emitterTriCdfCount > 0) {

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
    } else {
        s_emitterCount = 0;
    }

    // Scene-level emitter selection CDF.  When selectCount exceeds the mesh
    // emitter count, the extra trailing slot is the environment light, so it gets
    // picked by the same discrete draw as every other emitter and shares the same
    // MIS machinery.
    s_emitterSelectCount = selectCount;
    s_envEmitterSlot     = (selectCount > s_emitterCount) ? s_emitterCount : -1;

    CUDA_CHECK(cudaMalloc(&s_sceneEmitterCdf_d,
                          sizeof(float) * static_cast<size_t>(selectCount + 1)));
    CUDA_CHECK(cudaMemcpy(s_sceneEmitterCdf_d, sceneEmitterCdf,
                          sizeof(float) * static_cast<size_t>(selectCount + 1),
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

void cudaInitStyles(const StyleData* styles, int styleCount)
{
    if (s_styles_d) { cudaFree(s_styles_d); s_styles_d = nullptr; }
    s_styleCount = 0;

    if (!styles || styleCount <= 0) return;

    s_styleCount = styleCount;
    CUDA_CHECK(cudaMalloc(&s_styles_d, sizeof(StyleData) * static_cast<size_t>(styleCount)));
    CUDA_CHECK(cudaMemcpy(s_styles_d, styles,
                          sizeof(StyleData) * static_cast<size_t>(styleCount),
                          cudaMemcpyHostToDevice));

    fprintf(stdout, "[style] Uploaded %d style record(s)\n", styleCount);
}

void cudaInitTriangleObjectIds(const int* objectIds, int triangleCount)
{
    if (s_triangleObjectIds_d) {
        cudaFree(s_triangleObjectIds_d);
        s_triangleObjectIds_d = nullptr;
    }

    if (!objectIds || triangleCount <= 0) return;

    CUDA_CHECK(cudaMalloc(&s_triangleObjectIds_d,
                          sizeof(int) * static_cast<size_t>(triangleCount)));
    CUDA_CHECK(cudaMemcpy(s_triangleObjectIds_d, objectIds,
                          sizeof(int) * static_cast<size_t>(triangleCount),
                          cudaMemcpyHostToDevice));
}

void cudaInitRampTextures(const uint8_t* const* pixels,
                          const int* widths,
                          int rampCount)
{
    for (auto& t : s_rampObjects_h) if (t) cudaDestroyTextureObject(t);
    for (auto& a : s_rampArrays)    if (a) cudaFreeArray(a);
    s_rampObjects_h.clear();
    s_rampArrays.clear();
    if (s_rampObjects_d) { cudaFree(s_rampObjects_d); s_rampObjects_d = nullptr; }
    s_rampCount = 0;

    if (!pixels || rampCount <= 0) return;

    s_rampArrays.resize(static_cast<size_t>(rampCount), nullptr);
    s_rampObjects_h.resize(static_cast<size_t>(rampCount), 0);

    for (int i = 0; i < rampCount; ++i) {
        if (!pixels[i] || widths[i] <= 0) continue;

        const int w = widths[i];

        // Ramps are Nx1 strips, but a 2-D array with height 1 keeps the sampling
        // code identical to the albedo textures (tex2D with v = 0.5).
        const cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
        CUDA_CHECK(cudaMallocArray(&s_rampArrays[static_cast<size_t>(i)], &desc,
                                   static_cast<size_t>(w), 1));

        CUDA_CHECK(cudaMemcpy2DToArray(
            s_rampArrays[static_cast<size_t>(i)], 0, 0,
            pixels[i], static_cast<size_t>(w) * 4,
            static_cast<size_t>(w) * 4, 1,
            cudaMemcpyHostToDevice));

        cudaResourceDesc resDesc{};
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = s_rampArrays[static_cast<size_t>(i)];

        cudaTextureDesc texDesc{};
        // Clamp: a ramp is a gradient, not a tiling pattern, so the darkest and
        // brightest bands must hold at the ends.
        texDesc.addressMode[0]   = cudaAddressModeClamp;
        texDesc.addressMode[1]   = cudaAddressModeClamp;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;

        CUDA_CHECK(cudaCreateTextureObject(
            &s_rampObjects_h[static_cast<size_t>(i)], &resDesc, &texDesc, nullptr));

        fprintf(stdout, "[style] Uploaded ramp %d: %d texels\n", i, w);
    }

    s_rampCount = rampCount;
    CUDA_CHECK(cudaMalloc(&s_rampObjects_d,
                          sizeof(cudaTextureObject_t) * static_cast<size_t>(rampCount)));
    CUDA_CHECK(cudaMemcpy(s_rampObjects_d, s_rampObjects_h.data(),
                          sizeof(cudaTextureObject_t) * static_cast<size_t>(rampCount),
                          cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
//  Environment light
// ---------------------------------------------------------------------------

// Upload equirect float radiance into a float4 cudaArray with bilinear
// filtering.  Wrap in u because phi is periodic; clamp in v because theta is
// not — sampling past a pole must not wrap around to the other side of the sky.
static cudaTextureObject_t uploadEnvTexture(const float* pixels, int width, int height,
                                             int channels, cudaArray_t* outArray)
{
    *outArray = nullptr;
    if (!pixels || width <= 0 || height <= 0) return 0;

    std::vector<float4> rgba(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (size_t i = 0; i < rgba.size(); ++i) {
        const float* src = pixels + i * static_cast<size_t>(channels);
        rgba[i] = make_float4(src[0],
                              (channels > 1) ? src[1] : src[0],
                              (channels > 2) ? src[2] : src[0],
                              1.0f);
    }

    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
    CUDA_CHECK(cudaMallocArray(outArray, &desc,
                               static_cast<size_t>(width), static_cast<size_t>(height)));
    CUDA_CHECK(cudaMemcpy2DToArray(*outArray, 0, 0,
                                   rgba.data(), static_cast<size_t>(width) * sizeof(float4),
                                   static_cast<size_t>(width) * sizeof(float4),
                                   static_cast<size_t>(height),
                                   cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc{};
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = *outArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0]   = cudaAddressModeWrap;   // phi wraps
    texDesc.addressMode[1]   = cudaAddressModeClamp;  // theta does not
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;  // keep HDR values intact
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
    return tex;
}

static void freeEnvironment()
{
    if (s_env.radianceTex)   { cudaDestroyTextureObject(s_env.radianceTex); }
    if (s_env.backgroundTex) { cudaDestroyTextureObject(s_env.backgroundTex); }
    if (s_envArray)          { cudaFreeArray(s_envArray);   s_envArray = nullptr; }
    if (s_envBgArray)        { cudaFreeArray(s_envBgArray); s_envBgArray = nullptr; }
    if (s_envCondCdf_d)      { cudaFree(s_envCondCdf_d); s_envCondCdf_d = nullptr; }
    if (s_envMargCdf_d)      { cudaFree(s_envMargCdf_d); s_envMargCdf_d = nullptr; }
    s_env = {};
}

float cudaInitEnvironment(const EnvironmentUpload* upload)
{
    freeEnvironment();

    if (!upload) return 0.0f;

    // A flat-colour backdrop with no HDRI is a legitimate configuration: escaped
    // camera rays show the colour, and escaped bounce rays see nothing.  There is
    // no distribution to importance-sample, so it takes no emitter slot and
    // returns a zero selection weight.
    if (!upload->pixels || upload->width <= 0 || upload->height <= 0) {
        if (upload->backgroundMode != ENV_BG_FLAT_COLOR) return 0.0f;

        s_env.enabled         = 1;
        s_env.intensity       = upload->intensity;
        s_env.backgroundMode  = ENV_BG_FLAT_COLOR;
        s_env.backgroundColor = upload->backgroundColor;

        fprintf(stdout, "[env] Flat backdrop (%.3f, %.3f, %.3f), no lighting HDRI\n",
                s_env.backgroundColor.x, s_env.backgroundColor.y, s_env.backgroundColor.z);
        return 0.0f;
    }

    const int width    = upload->width;
    const int height   = upload->height;
    const int channels = (upload->channels > 0) ? upload->channels : 3;

    // ── Hierarchical 2-D CDF for importance sampling ──────────────────────
    //  conditionalCdf[y] : cumulative texel luminance across row y  (width + 1)
    //  marginalCdf       : cumulative row luminance * sin(theta)    (height + 1)
    //
    //  The sin(theta) weighting in the marginal accounts for the solid angle of
    //  a texel shrinking toward the poles, so rows near the horizon are chosen
    //  proportionally more often — which is where a sky HDRI's energy actually is.
    std::vector<float> condCdf(static_cast<size_t>(height) * static_cast<size_t>(width + 1), 0.0f);
    std::vector<float> margCdf(static_cast<size_t>(height) + 1, 0.0f);

    const double pi = 3.14159265358979323846;

    double marginalTotal = 0.0;
    for (int y = 0; y < height; ++y) {
        float* row = condCdf.data() + static_cast<size_t>(y) * static_cast<size_t>(width + 1);
        row[0] = 0.0f;

        double rowSum = 0.0;
        for (int x = 0; x < width; ++x) {
            const float* texel = upload->pixels +
                (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) *
                static_cast<size_t>(channels);

            const float r = texel[0];
            const float g = (channels > 1) ? texel[1] : r;
            const float b = (channels > 2) ? texel[2] : r;

            // Guard against NaN/Inf in the source HDR, which would poison the
            // whole CDF and make every environment sample invalid.
            float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            if (!(lum > 0.0f) || !isfinite(lum)) lum = 0.0f;

            rowSum += static_cast<double>(lum);
            row[x + 1] = static_cast<float>(rowSum);
        }

        // Texel centre theta, so no row gets exactly sin(theta) == 0.
        const double theta   = pi * (static_cast<double>(y) + 0.5) / static_cast<double>(height);
        const double rowWeight = rowSum * sin(theta);

        marginalTotal += rowWeight;
        margCdf[static_cast<size_t>(y) + 1] = static_cast<float>(marginalTotal);
    }

    if (marginalTotal <= 0.0) {
        fprintf(stderr, "[env] HDRI \"%dx%d\" has zero total luminance — "
                        "environment disabled.\n", width, height);
        return 0.0f;
    }

    CUDA_CHECK(cudaMalloc(&s_envCondCdf_d, sizeof(float) * condCdf.size()));
    CUDA_CHECK(cudaMemcpy(s_envCondCdf_d, condCdf.data(),
                          sizeof(float) * condCdf.size(), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&s_envMargCdf_d, sizeof(float) * margCdf.size()));
    CUDA_CHECK(cudaMemcpy(s_envMargCdf_d, margCdf.data(),
                          sizeof(float) * margCdf.size(), cudaMemcpyHostToDevice));

    // ── Device record ─────────────────────────────────────────────────────
    s_env.enabled     = 1;
    s_env.radianceTex = uploadEnvTexture(upload->pixels, width, height, channels, &s_envArray);
    s_env.width       = width;
    s_env.height      = height;
    s_env.conditionalCdf = s_envCondCdf_d;
    s_env.marginalCdf    = s_envMargCdf_d;
    s_env.yawRadians  = upload->yawRadians;
    s_env.intensity   = upload->intensity;

    // p_omega = texelWeight * width * height / (2 pi^2 * marginalTotal).
    // Folding the constant here keeps envPdf to a single multiply, and keeps the
    // one number that must agree between envSample and envPdf in one place.
    s_env.pdfScale = static_cast<float>(
        (static_cast<double>(width) * static_cast<double>(height)) /
        (2.0 * pi * pi * marginalTotal));

    // Power-style selection weight, in the same spirit as a mesh emitter's
    // area * luminance: mean radiance over the full sphere.
    s_env.selectWeight = static_cast<float>(
        (marginalTotal / (static_cast<double>(width) * static_cast<double>(height))) *
        4.0 * pi * static_cast<double>(upload->intensity));

    s_env.backgroundMode  = upload->backgroundMode;
    s_env.backgroundColor = upload->backgroundColor;

    if (upload->backgroundMode == ENV_BG_SEPARATE_TEX && upload->bgPixels) {
        s_env.backgroundTex = uploadEnvTexture(upload->bgPixels,
                                               upload->bgWidth, upload->bgHeight,
                                               (upload->bgChannels > 0) ? upload->bgChannels : 3,
                                               &s_envBgArray);
    }

    fprintf(stdout, "[env] HDRI %dx%d uploaded (intensity %.3f, yaw %.1f deg, "
                    "select weight %.4f)\n",
            width, height, s_env.intensity,
            s_env.yawRadians * 180.0f / 3.14159265f, s_env.selectWeight);

    return s_env.selectWeight;
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

    if (s_triangleObjectIds_d) {
        cudaFree(s_triangleObjectIds_d);
        s_triangleObjectIds_d = nullptr;
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
    s_emitterCount       = 0;
    s_emitterSelectCount = 0;
    s_envEmitterSlot     = -1;

    if (s_bsdfs_d) {
        cudaFree(s_bsdfs_d);
        s_bsdfs_d = nullptr;
    }
    if (s_triangleBsdfIds_d) {
        cudaFree(s_triangleBsdfIds_d);
        s_triangleBsdfIds_d = nullptr;
    }
    s_bsdfCount = 0;

    if (s_styles_d) {
        cudaFree(s_styles_d);
        s_styles_d = nullptr;
    }
    s_styleCount = 0;

    for (auto& t : s_rampObjects_h) if (t) cudaDestroyTextureObject(t);
    for (auto& a : s_rampArrays)    if (a) cudaFreeArray(a);
    s_rampObjects_h.clear();
    s_rampArrays.clear();
    if (s_rampObjects_d) {
        cudaFree(s_rampObjects_d);
        s_rampObjects_d = nullptr;
    }
    s_rampCount = 0;

    freeEnvironment();

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
