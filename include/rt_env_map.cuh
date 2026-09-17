#pragma once

// ---------------------------------------------------------------------------
//  rt_env_map.cuh — HDRI environment map sampling for image-based lighting.
//
//  Provides:
//    dirToEquirect    — world direction → equirectangular UV
//    equirectToDir    — equirectangular UV → world direction
//    envMapEval       — evaluate radiance for a given direction
//    envMapPdf        — probability density of a direction under the CDF
//    envMapSample     — importance-sample a direction from the env map
//
//  The importance sampling uses a 2D marginal–conditional CDF built on the
//  host (see cudaInitEnvMap in scene_upload.cu).  Luminance is weighted by
//  sin(θ) to account for the solid-angle distortion of equirectangular maps.
//
//  No OptiX dependency — usable from both PTX and plain CUDA TUs.
// ---------------------------------------------------------------------------

#include "render_kernel.h"
#include "rt_cuda_math.cuh"

#include <cuda_runtime.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ---------------------------------------------------------------------------
//  Direction ↔ equirectangular UV helpers
// ---------------------------------------------------------------------------

/// Convert a normalised world direction to equirectangular UV [0,1]².
/// Applies the env map's Y-axis rotation offset to φ.
__device__ __forceinline__ Float2 dirToEquirect(const Float3& dir, float rotationRadians)
{
    // φ ∈ [−π, π] around Y, θ ∈ [0, π] from +Y pole
    float phi   = atan2f(dir.z, dir.x) + rotationRadians;
    float theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f));

    // Wrap φ into [−π, π]
    phi = fmodf(phi + M_PI, 2.0f * M_PI);
    if (phi < 0.0f) phi += 2.0f * M_PI;
    phi -= M_PI;

    // Map to [0,1]²:  u wraps horizontally, v goes from top (+Y) to bottom (-Y)
    float u = (phi + M_PI) / (2.0f * M_PI);
    float v = theta / M_PI;

    return {u, v};
}

/// Convert equirectangular UV [0,1]² back to a normalised world direction.
/// Applies the env map's Y-axis rotation offset.
__device__ __forceinline__ Float3 equirectToDir(float u, float v, float rotationRadians)
{
    float phi   = u * 2.0f * M_PI - M_PI - rotationRadians;
    float theta = v * M_PI;

    float sinTheta = sinf(theta);
    return normalize3({
        sinTheta * cosf(phi),
        cosf(theta),
        sinTheta * sinf(phi)
    });
}

// ---------------------------------------------------------------------------
//  Evaluate the env map radiance for a given direction.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Float3 envMapEval(const EnvMapData& env, const Float3& dir)
{
    Float2 uv = dirToEquirect(dir, env.rotationRadians);
    float4 texVal = tex2D<float4>(static_cast<cudaTextureObject_t>(env.texObj), uv.x, uv.y);
    return {texVal.x * env.intensity, texVal.y * env.intensity, texVal.z * env.intensity};
}

// ---------------------------------------------------------------------------
//  Binary search helper for CDF inversion (shared with emitter sampling).
// ---------------------------------------------------------------------------
__device__ __forceinline__ int envCdfBinarySearch(const float* cdf, int count, float x)
{
    int lo = 0, hi = count;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (cdf[mid] <= x) lo = mid;
        else hi = mid;
    }
    return lo;
}

// ---------------------------------------------------------------------------
//  Compute the PDF (in solid-angle measure) of sampling a given direction
//  from the env map CDF.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float envMapPdf(const EnvMapData& env, const Float3& dir)
{
    if (env.totalPower <= 0.0f) return 0.0f;

    Float2 uv = dirToEquirect(dir, env.rotationRadians);

    // Pixel coordinates (continuous)
    float fx = uv.x * static_cast<float>(env.width);
    float fy = uv.y * static_cast<float>(env.height);
    int ix = min(max(static_cast<int>(fx), 0), env.width  - 1);
    int iy = min(max(static_cast<int>(fy), 0), env.height - 1);

    // Marginal PDF of row iy
    const float marginalSum = env.marginalCdf[env.height];
    const float rowWeight   = env.marginalCdf[iy + 1] - env.marginalCdf[iy];
    if (marginalSum <= 0.0f || rowWeight <= 0.0f) return 0.0f;
    const float pdfRow = rowWeight / marginalSum;

    // Conditional PDF of column ix within row iy
    const float* rowCdf = env.conditionalCdf + iy * (env.width + 1);
    const float rowSum  = rowCdf[env.width];
    const float colWeight = rowCdf[ix + 1] - rowCdf[ix];
    if (rowSum <= 0.0f || colWeight <= 0.0f) return 0.0f;
    const float pdfCol = colWeight / rowSum;

    // Discrete pixel PDF → solid angle PDF
    // PDF_pixel = pdfRow * pdfCol
    // Each pixel spans (2π/W) × (π/H) in (φ, θ) space
    // Jacobian: dω = sin(θ) dθ dφ, so PDF_solidAngle = PDF_pixel * W * H / (2π² sin(θ))
    const float theta    = uv.y * M_PI;
    const float sinTheta = fmaxf(sinf(theta), 1e-6f);
    const float pdfSolidAngle = (pdfRow * pdfCol * static_cast<float>(env.width) * static_cast<float>(env.height))
                              / (2.0f * M_PI * M_PI * sinTheta);

    return pdfSolidAngle;
}

// ---------------------------------------------------------------------------
//  Importance-sample a direction from the env map using the 2D CDF.
//  Returns the sampled direction, its solid-angle PDF, and the radiance.
// ---------------------------------------------------------------------------
struct EnvMapSampleResult {
    Float3 direction;
    float  pdf;
    Float3 radiance;
};

__device__ __forceinline__ EnvMapSampleResult envMapSample(
    const EnvMapData& env, float u0, float u1)
{
    EnvMapSampleResult result;
    result.direction = {0.0f, 1.0f, 0.0f};
    result.pdf       = 0.0f;
    result.radiance  = {0.0f, 0.0f, 0.0f};

    if (env.totalPower <= 0.0f) return result;

    // 1. Invert the marginal CDF to pick a row
    const float marginalSum = env.marginalCdf[env.height];
    if (marginalSum <= 0.0f) return result;
    const float xi_row = u0 * marginalSum;
    const int row = envCdfBinarySearch(env.marginalCdf, env.height, xi_row);

    const float c0_row = env.marginalCdf[row];
    const float c1_row = env.marginalCdf[row + 1];
    const float dRow = c1_row - c0_row;
    // Reuse: continuous sub-pixel offset within the selected row
    float fracRow = (dRow > 0.0f) ? (xi_row - c0_row) / dRow : 0.0f;

    // 2. Invert the conditional CDF to pick a column within that row
    const float* rowCdf = env.conditionalCdf + row * (env.width + 1);
    const float rowSum = rowCdf[env.width];
    if (rowSum <= 0.0f) return result;
    const float xi_col = u1 * rowSum;
    const int col = envCdfBinarySearch(rowCdf, env.width, xi_col);

    const float c0_col = rowCdf[col];
    const float c1_col = rowCdf[col + 1];
    const float dCol = c1_col - c0_col;
    float fracCol = (dCol > 0.0f) ? (xi_col - c0_col) / dCol : 0.0f;

    // 3. Compute UV with sub-pixel refinement
    float u = (static_cast<float>(col) + fracCol) / static_cast<float>(env.width);
    float v = (static_cast<float>(row) + fracRow) / static_cast<float>(env.height);

    // 4. Convert UV to direction
    result.direction = equirectToDir(u, v, env.rotationRadians);

    // 5. Evaluate radiance
    float4 texVal = tex2D<float4>(static_cast<cudaTextureObject_t>(env.texObj), u, v);
    result.radiance = {texVal.x * env.intensity, texVal.y * env.intensity, texVal.z * env.intensity};

    // 6. Compute PDF
    const float pdfRow = dRow / marginalSum;
    const float pdfCol = dCol / rowSum;

    const float theta    = v * M_PI;
    const float sinTheta = fmaxf(sinf(theta), 1e-6f);
    result.pdf = (pdfRow * pdfCol * static_cast<float>(env.width) * static_cast<float>(env.height))
               / (2.0f * M_PI * M_PI * sinTheta);

    return result;
}
