#pragma once

// ---------------------------------------------------------------------------
//  rt_environment.cuh — HDRI environment lighting.
//
//  This is a universal engine feature, not part of the stylization work: it is
//  active in both render modes and both style modes, and lives in a shared
//  header so the megakernel keeps working as the correctness reference for the
//  environment MIS weights.
//
//  Parameterisation (equirectangular / lat-long):
//      theta = acos(dir.y)          in [0, pi],   v = theta / pi
//      phi   = atan2(dir.z, dir.x)  in [-pi, pi], u = phi / (2 pi) + 0.5
//  v = 0 is +Y (the top row of the loaded image), matching the usual HDRI
//  convention.  A yaw offset rotates the map about +Y so the art direction can
//  aim the sun without re-exporting the image.
//
//  Importance sampling uses a standard hierarchical 2-D distribution built on
//  the host:
//      conditionalCdf[y]  — length width + 1, cumulative texel luminance in row y
//      marginalCdf        — length height + 1, cumulative row luminance * sin(theta)
//  The sin(theta) factor in the marginal accounts for the shrinking solid angle
//  of rows near the poles.
//
//  The solid-angle pdf works out to a pleasantly simple closed form.  With
//  w[y][x] the texel luminance and W the marginal total:
//      p_uv    = (w[y][x] sin0_y / W) * height * width / rowSum_y ... etc.
//      p_omega = p_uv / (2 pi^2 sin0_y)
//              = w[y][x] * width * height / (2 pi^2 W)
//  The sin(theta) cancels exactly, so no pole special-casing is needed and
//  `pdfScale` below folds the whole constant into one host-computed float.
//
//  CRITICAL: envPdf must agree with the pdf that envSample actually divides by,
//  or the MIS weights silently bias every render — including photorealistic
//  ones, since this feature is always on.  Both read the SAME texel weight,
//  recovered from a difference of adjacent conditional-CDF entries, rather than
//  the bilinearly filtered radiance.  Validate with the furnace test.
// ---------------------------------------------------------------------------

#include "render_kernel.h"
#include "rt_cuda_math.cuh"
#include "rt_emitter_sampling.cuh"   // sampleCdfReuse

#include <cuda_runtime.h>            // cudaTextureObject_t

static constexpr float RT_ENV_PI      = 3.14159265358979323846f;
static constexpr float RT_ENV_TWO_PI  = 6.28318530717958647692f;
static constexpr float RT_ENV_INV_PI  = 0.31830988618379067154f;

// EnvBackgroundMode lives in render_kernel.h so main.cpp can select a mode
// without including any CUDA device headers.

struct EnvLightData {
    int enabled;                        // 0 => no environment; escaped rays add nothing

    cudaTextureObject_t radianceTex;    // equirect float4 HDR, 0 when absent
    cudaTextureObject_t backgroundTex;  // optional separate backdrop, 0 when absent

    const float* conditionalCdf;        // height * (width + 1)
    const float* marginalCdf;           // height + 1

    int   width;
    int   height;

    float yawRadians;
    float intensity;

    float pdfScale;      // width * height / (2 pi^2 * marginalTotal)
    float selectWeight;  // power-style weight used in the scene emitter CDF

    int    backgroundMode;
    Float3 backgroundColor;
};

// ---------------------------------------------------------------------------
//  Direction <-> equirect UV
// ---------------------------------------------------------------------------
__device__ __forceinline__ Float2 envDirToUV(const EnvLightData& env, const Float3& dir)
{
    const float theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f));
    float       phi   = atan2f(dir.z, dir.x) - env.yawRadians;

    // Wrap phi into [-pi, pi] so the u coordinate stays in [0,1).
    phi = fmodf(phi + RT_ENV_PI, RT_ENV_TWO_PI);
    if (phi < 0.0f) phi += RT_ENV_TWO_PI;
    phi -= RT_ENV_PI;

    Float2 uv;
    uv.x = phi / RT_ENV_TWO_PI + 0.5f;
    uv.y = theta * RT_ENV_INV_PI;
    return uv;
}

__device__ __forceinline__ Float3 envUVToDir(const EnvLightData& env, float u, float v)
{
    const float theta = v * RT_ENV_PI;
    const float phi   = (u - 0.5f) * RT_ENV_TWO_PI + env.yawRadians;

    const float sinTheta = sinf(theta);
    return { sinTheta * cosf(phi), cosf(theta), sinTheta * sinf(phi) };
}

// ---------------------------------------------------------------------------
//  Radiance lookup along a direction (bilinear, intensity-scaled).
// ---------------------------------------------------------------------------
__device__ __forceinline__ Float3 envEval(const EnvLightData& env, const Float3& dir)
{
    if (!env.enabled || env.radianceTex == 0) return {0.0f, 0.0f, 0.0f};

    const Float2 uv = envDirToUV(env, dir);
    const float4 s  = tex2D<float4>(env.radianceTex, uv.x, uv.y);
    return { s.x * env.intensity, s.y * env.intensity, s.z * env.intensity };
}

// ---------------------------------------------------------------------------
//  Backdrop lookup for escaped CAMERA rays only (bounceCount == 0).  Every
//  other escaped ray uses envEval, so the lighting stays physical no matter
//  what the backdrop shows.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Float3 envEvalBackground(const EnvLightData& env, const Float3& dir)
{
    if (!env.enabled) return {0.0f, 0.0f, 0.0f};

    if (env.backgroundMode == ENV_BG_FLAT_COLOR) {
        return env.backgroundColor;
    }

    if (env.backgroundMode == ENV_BG_SEPARATE_TEX && env.backgroundTex != 0) {
        const Float2 uv = envDirToUV(env, dir);
        const float4 s  = tex2D<float4>(env.backgroundTex, uv.x, uv.y);
        return { s.x * env.intensity, s.y * env.intensity, s.z * env.intensity };
    }

    return envEval(env, dir);
}

// ---------------------------------------------------------------------------
//  Solid-angle pdf of the importance-sampling distribution for `dir`.
//  Reads the same piecewise-constant texel weight envSample draws from.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float envPdf(const EnvLightData& env, const Float3& dir)
{
    if (!env.enabled || !env.conditionalCdf || env.width <= 0 || env.height <= 0)
        return 0.0f;

    const Float2 uv = envDirToUV(env, dir);

    int x = static_cast<int>(uv.x * static_cast<float>(env.width));
    int y = static_cast<int>(uv.y * static_cast<float>(env.height));
    x = min(max(x, 0), env.width  - 1);
    y = min(max(y, 0), env.height - 1);

    const float* row = env.conditionalCdf + static_cast<size_t>(y) * static_cast<size_t>(env.width + 1);
    const float  w   = row[x + 1] - row[x];

    return fmaxf(w, 0.0f) * env.pdfScale;
}

struct EnvSample {
    Float3 direction;
    Float3 radiance;
    float  pdf;        // solid angle
};

// ---------------------------------------------------------------------------
//  Importance-sample the environment.  u0/u1 are consumed (and internally
//  remapped by sampleCdfReuse) so no extra RNG draws are needed relative to the
//  mesh-emitter path.
// ---------------------------------------------------------------------------
__device__ __forceinline__ EnvSample envSample(const EnvLightData& env, float u0, float u1)
{
    EnvSample out{};
    out.direction = {0.0f, 1.0f, 0.0f};
    out.radiance  = {0.0f, 0.0f, 0.0f};
    out.pdf       = 0.0f;

    if (!env.enabled || !env.conditionalCdf || !env.marginalCdf ||
        env.width <= 0 || env.height <= 0)
        return out;

    // Row (theta) from the marginal, then column (phi) from that row's conditional.
    float rowPdf = 0.0f;
    const int y = sampleCdfReuse(env.marginalCdf, env.height, u1, rowPdf);
    if (rowPdf <= 0.0f) return out;

    const float* row = env.conditionalCdf + static_cast<size_t>(y) * static_cast<size_t>(env.width + 1);

    float colPdf = 0.0f;
    const int x = sampleCdfReuse(row, env.width, u0, colPdf);
    if (colPdf <= 0.0f) return out;

    // u0 / u1 now hold the position within the chosen texel.
    const float u = (static_cast<float>(x) + u0) / static_cast<float>(env.width);
    const float v = (static_cast<float>(y) + u1) / static_cast<float>(env.height);

    out.direction = normalize3(envUVToDir(env, u, v));

    // Recover the texel weight the same way envPdf does, so the two agree
    // exactly and the MIS weights stay consistent.
    const float w = fmaxf(row[x + 1] - row[x], 0.0f);
    out.pdf       = w * env.pdfScale;
    if (out.pdf <= 0.0f) return out;

    out.radiance = envEval(env, out.direction);
    return out;
}
