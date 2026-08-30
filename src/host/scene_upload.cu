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

void cudaInitObjects(const ObjectDesc* objects, int objectCount)
{
    // Free previous object tables.
    if (s_objectTransforms_d) { cudaFree(s_objectTransforms_d); s_objectTransforms_d = nullptr; }
    if (s_objectTriOffset_d)  { cudaFree(s_objectTriOffset_d);  s_objectTriOffset_d  = nullptr; }
    if (s_triangleObjectId_d) { cudaFree(s_triangleObjectId_d); s_triangleObjectId_d = nullptr; }
    s_objects_h.clear();
    s_objectTransforms_h.clear();
    s_objectCount = 0;

    if (!objects || objectCount <= 0) return;

    s_objectCount = objectCount;
    s_objects_h.assign(objects, objects + objectCount);
    s_objectTransforms_h.resize(static_cast<size_t>(objectCount));
    for (int i = 0; i < objectCount; ++i)
        s_objectTransforms_h[static_cast<size_t>(i)] = objects[i].transform;

    // Prefix-sum style offset array: objectTriOffset[i] = start of object i,
    // objectTriOffset[objectCount] = total triangle count.
    std::vector<int> triOffsets(static_cast<size_t>(objectCount + 1));
    for (int i = 0; i < objectCount; ++i)
        triOffsets[static_cast<size_t>(i)] = objects[i].triOffset;
    triOffsets[static_cast<size_t>(objectCount)] =
        objects[objectCount - 1].triOffset + objects[objectCount - 1].triCount;

    // Per-triangle owning object id.
    std::vector<int> triObjectId(static_cast<size_t>(s_triangleCount), 0);
    for (int i = 0; i < objectCount; ++i) {
        const int begin = objects[i].triOffset;
        const int end   = begin + objects[i].triCount;
        for (int t = begin; t < end && t < s_triangleCount; ++t)
            triObjectId[static_cast<size_t>(t)] = i;
    }

    CUDA_CHECK(cudaMalloc(&s_objectTransforms_d,
                          sizeof(ObjectTransform) * static_cast<size_t>(objectCount)));
    CUDA_CHECK(cudaMemcpy(s_objectTransforms_d, s_objectTransforms_h.data(),
                          sizeof(ObjectTransform) * static_cast<size_t>(objectCount),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&s_objectTriOffset_d,
                          sizeof(int) * static_cast<size_t>(objectCount + 1)));
    CUDA_CHECK(cudaMemcpy(s_objectTriOffset_d, triOffsets.data(),
                          sizeof(int) * static_cast<size_t>(objectCount + 1),
                          cudaMemcpyHostToDevice));

    if (s_triangleCount > 0) {
        CUDA_CHECK(cudaMalloc(&s_triangleObjectId_d,
                              sizeof(int) * static_cast<size_t>(s_triangleCount)));
        CUDA_CHECK(cudaMemcpy(s_triangleObjectId_d, triObjectId.data(),
                              sizeof(int) * static_cast<size_t>(s_triangleCount),
                              cudaMemcpyHostToDevice));
    }

    fprintf(stdout, "[scene] Uploaded %d instanced objects\n", objectCount);
}

void cudaSetObjectTransforms(const ObjectTransform* transforms, int objectCount)
{
    if (!transforms || objectCount != s_objectCount || s_objectCount <= 0) return;

    s_objectTransforms_h.assign(transforms, transforms + objectCount);
    CUDA_CHECK(cudaMemcpy(s_objectTransforms_d, transforms,
                          sizeof(ObjectTransform) * static_cast<size_t>(objectCount),
                          cudaMemcpyHostToDevice));

    // Refit the IAS so the next trace sees the new poses.
    if (s_optixReady)
        buildIAS(/*update=*/true);
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

    if (s_objectTransforms_d) { cudaFree(s_objectTransforms_d); s_objectTransforms_d = nullptr; }
    if (s_objectTriOffset_d)  { cudaFree(s_objectTriOffset_d);  s_objectTriOffset_d  = nullptr; }
    if (s_triangleObjectId_d) { cudaFree(s_triangleObjectId_d); s_triangleObjectId_d = nullptr; }
    s_objects_h.clear();
    s_objectTransforms_h.clear();
    s_objectCount = 0;
}
