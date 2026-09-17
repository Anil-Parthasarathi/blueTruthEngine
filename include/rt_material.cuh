#pragma once

// ---------------------------------------------------------------------------
//  rt_material.cuh — per-hit BSDF lookup with texture override.
//  Shared by the megakernel (traceRay) and the wavefront shade stage.
// ---------------------------------------------------------------------------

#include "render_kernel.h"
#include "rt_shading_context.cuh"

#include <cuda_runtime.h>

// Fetch the BSDF for a triangle (by value, so the texture override can mutate
// the local copy without touching the global table).
__device__ __forceinline__ BsdfData loadBsdfForTriangle(
    const DirectLightingContext& lightingCtx,
    const BsdfData* bsdfs,
    const int* triangleBsdfIds,
    int triangleIndex,
    Float2 uv)
{
    const int bsdfId = triangleBsdfIds[triangleIndex];
    BsdfData bsdf = bsdfs[bsdfId];

    // Texture sampling
    if (lightingCtx.texObjects != nullptr && lightingCtx.triangleMaterialIds != nullptr) {
        const int matId = lightingCtx.triangleMaterialIds[triangleIndex];
        if (matId >= 0 && matId < lightingCtx.textureCount) {
            const cudaTextureObject_t texObj = lightingCtx.texObjects[matId];
            if (texObj != 0) {
                // tex2D returns normalised float4 in [0,1] (readMode = NormalizedFloat).
                // We use the values as-is (no gamma decode) so that textures and
                // XML albedo/base_color parameters live in the same convention.
                const float4 s = tex2D<float4>(texObj, uv.x, uv.y);
                bsdf.p0.x = s.x;
                bsdf.p0.y = s.y;
                bsdf.p0.z = s.z;
                // Microfacet derives ks = 1 - max(kd) from the albedo.
                // Recompute after texture override to stay energy-conserving.
                if (bsdf.type == BSDF_Microfacet) {
                    bsdf.p1.w = 1.0f - fmaxf(bsdf.p0.x, fmaxf(bsdf.p0.y, bsdf.p0.z));
                }
            }

            // glTF metallic-roughness: G = roughness, B = metallic, multiplied by
            // the BSDF factors already stored in p0.w / p1.x.
            if (lightingCtx.texObjectsMr != nullptr) {
                const cudaTextureObject_t mrObj = lightingCtx.texObjectsMr[matId];
                if (mrObj != 0 && bsdf.type == BSDF_Disney) {
                    const float4 mr = tex2D<float4>(mrObj, uv.x, uv.y);
                    bsdf.p0.w *= mr.y;
                    bsdf.p1.x *= mr.z;
                }
            }
        }
    }

    return bsdf;
}
