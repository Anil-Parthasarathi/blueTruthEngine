#pragma once

// ---------------------------------------------------------------------------
//  rt_style.cuh — non-photorealistic (anime) stylization records and operators.
//
//  Design rule: the style operators NEVER touch bsdfEval / bsdfPdf.  A
//  quantizing BRDF is non-integrable and would silently break the MIS weights
//  in rt_direct_lighting.cuh.  Instead the path tracer integrates a physically
//  correct estimate, split across StyleChannel buckets, and the operators here
//  run once at present time on the CONVERGED accumulation buffers.
//
//  That ordering is the whole point: posterizing a Monte Carlo sample makes
//  band edges dither and converge to a blur, while posterizing a converged
//  soft penumbra yields a hard anime shadow whose shape is physically exact.
//
//  Style is orthogonal to material: a StyleData record is referenced by
//  BsdfData::styleId and can be attached to any BSDF type.  A "metallic anime"
//  material is the Disney BSDF with metallic = 1 plus a metal style; a jewel is
//  Disney with specular_transmission = 1 plus a gem style.
//
//  No OptiX dependency — includable from plain CUDA TUs and PTX TUs alike.
// ---------------------------------------------------------------------------

#include "render_kernel.h"
#include "rt_cuda_math.cuh"

#include <cuda_runtime.h>   // cudaTextureObject_t

// ---------------------------------------------------------------------------
//  Radiance channels.
//
//  A path's channel is fixed at the primary hit from the BSDF lobe that was
//  sampled there, and persists for the rest of the path.  Everything a metal
//  surface reflects therefore lands in STYLE_CH_REFLECTED and receives the
//  metal operator; everything seen through a gem lands in STYLE_CH_TRANSMITTED.
//
//  In physical mode all six channel pointers alias the single radiance buffer,
//  so the shading kernels need no branch on style mode at all.
// ---------------------------------------------------------------------------
enum StyleChannel : int {
    STYLE_CH_DIRECT_DIFFUSE   = 0,  // NEE at the primary hit, diffuse-ish eval
    STYLE_CH_DIRECT_SPECULAR  = 1,  // NEE at the primary hit, specular-ish eval
    STYLE_CH_INDIRECT_DIFFUSE = 2,  // bounce >= 1 for diffuse-rooted paths
    STYLE_CH_REFLECTED        = 3,  // metal / clearcoat-rooted paths
    STYLE_CH_TRANSMITTED      = 4,  // glass-rooted paths
    STYLE_CH_EMISSIVE         = 5,  // emitter hits + escaped rays (unstyled)
    STYLE_CH_COUNT            = 6
};

// StyleData itself is declared in render_kernel.h, so main.cpp can build the
// upload table without pulling in any CUDA device headers.  Which channel each
// field acts on:
//
//   diffuseRampTex / diffuseBands / bandSoftness / toneScale
//                                     → STYLE_CH_DIRECT_DIFFUSE
//   specThreshold / specSoftness / specIntensity
//                                     → STYLE_CH_DIRECT_SPECULAR
//   reflectBands / reflectGain / reflectTint*
//                                     → STYLE_CH_REFLECTED
//   transmitBands / transmitGain / chromaShift
//                                     → STYLE_CH_TRANSMITTED
//   indirectGain                      → STYLE_CH_INDIRECT_DIFFUSE
//   rim* / line*                      → applied at present / in shade

// A neutral record used when a material has no style attached.  Every operator
// below is an identity (or a no-op) under these values, so the same present
// code path can run for styled and unstyled pixels.
__device__ __forceinline__ StyleData styleDataIdentity()
{
    StyleData s{};
    s.diffuseRampTex         = -1;
    s.diffuseBands           = 1;      // 1 = pass-through, no quantization
    s.bandSoftness           = 0.0f;
    s.toneScale              = 1.0f;
    s.specThreshold          = 0.0f;   // threshold 0 => highlight always passes
    s.specSoftness           = 0.0f;
    s.specIntensity          = 1.0f;
    s.rimStrength            = 0.0f;
    s.rimPower               = 1.0f;
    s.rimColor               = {0.0f, 0.0f, 0.0f};
    s.reflectBands           = 1;
    s.reflectGain            = 1.0f;
    s.reflectTintMix         = 0.0f;
    s.reflectTint            = {1.0f, 1.0f, 1.0f};
    s.transmitBands          = 1;
    s.transmitGain           = 1.0f;
    s.chromaShift            = 0.0f;
    s.indirectGain           = 1.0f;
    s.lineColor              = {0.0f, 0.0f, 0.0f};
    s.lineWidth              = 0.0f;
    s.lineStrength           = 0.0f;
    s.outlineNormalThreshold = 0.5f;
    s.outlineDepthThreshold  = 0.1f;
    return s;
}

// ---------------------------------------------------------------------------
//  Scalar helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float styleLuminance(const Float3& c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ __forceinline__ float styleClamp01(float x)
{
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__device__ __forceinline__ float styleSmoothstep01(float x)
{
    const float t = styleClamp01(x);
    return t * t * (3.0f - 2.0f * t);
}

// Smoothstep between two edges, tolerant of edge0 == edge1 (becomes a hard step).
__device__ __forceinline__ float styleSmoothstep(float edge0, float edge1, float x)
{
    if (edge1 - edge0 <= 1e-6f) return (x >= edge1) ? 1.0f : 0.0f;
    return styleSmoothstep01((x - edge0) / (edge1 - edge0));
}

// ---------------------------------------------------------------------------
//  Cel band quantization.
//
//  Maps a continuous tone in [0,1] onto `bands` discrete levels spanning the
//  full [0,1] range.  With softness > 0 each band gets a soft shoulder at its
//  upper boundary, which is what keeps very high-contrast lighting from
//  aliasing along a band edge while leaving the edge itself crisp.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float styleBandQuantize(float tone, int bands, float softness)
{
    if (bands <= 1) return tone;   // pass-through: no stylization

    const float n = static_cast<float>(bands);
    const float s = styleClamp01(tone);

    const float scaled = s * n;
    float       lower  = floorf(scaled);
    const float frac   = scaled - lower;
    lower = fminf(lower, n - 1.0f);

    const float invTop = 1.0f / (n - 1.0f);
    float level = lower * invTop;

    if (softness > 1e-5f) {
        const float shoulder = fminf(softness, 1.0f);
        if (frac > 1.0f - shoulder) {
            const float w    = styleSmoothstep01((frac - (1.0f - shoulder)) / shoulder);
            const float next = fminf(lower + 1.0f, n - 1.0f) * invTop;
            level += (next - level) * w;
        }
    }

    return level;
}

// ---------------------------------------------------------------------------
//  Tone-preserving quantization of a colour.
//
//  Quantizes luminance and rescales the colour to match, so a coloured light
//  keeps its hue through the band step instead of collapsing toward grey.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Float3 styleQuantizeTone(const Float3& c, int bands, float softness,
                                                     float toneScale)
{
    const float lum = styleLuminance(c);
    if (lum <= 1e-6f) return {0.0f, 0.0f, 0.0f};

    const float target = styleBandQuantize(lum * toneScale, bands, softness);
    return mul3(c, target / lum);
}

// ---------------------------------------------------------------------------
//  Body tone: procedural bands, or a 1-D ramp texture lookup when one is bound.
//  The ramp is indexed by quantized luminance and multiplies the incoming
//  chroma, so an artist-authored ramp can recolour the shadow side.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Float3 styleBodyTone(const Float3& directDiffuse,
                                                 const StyleData& st,
                                                 const cudaTextureObject_t* rampTex,
                                                 int rampCount)
{
    const float lum = styleLuminance(directDiffuse);
    if (lum <= 1e-6f) return {0.0f, 0.0f, 0.0f};

    const float tone = styleBandQuantize(lum * st.toneScale, st.diffuseBands, st.bandSoftness);

    if (rampTex != nullptr && st.diffuseRampTex >= 0 && st.diffuseRampTex < rampCount) {
        const cudaTextureObject_t tex = rampTex[st.diffuseRampTex];
        if (tex != 0) {
            const float4 r = tex2D<float4>(tex, styleClamp01(tone), 0.5f);
            // Ramp supplies the tone (and optionally a tint); chroma comes from
            // the physically integrated direct lighting.
            return mul3(div3(directDiffuse, lum), Float3{r.x, r.y, r.z});
        }
    }

    return mul3(directDiffuse, tone / lum);
}

// ---------------------------------------------------------------------------
//  Anime highlight: a hard-edged specular blob rather than a smooth falloff.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Float3 styleHighlight(const Float3& directSpecular,
                                                  const StyleData& st)
{
    const float lum = styleLuminance(directSpecular);
    if (lum <= 1e-6f) return {0.0f, 0.0f, 0.0f};

    const float gate = styleSmoothstep(st.specThreshold,
                                       st.specThreshold + st.specSoftness,
                                       lum);
    return mul3(directSpecular, gate * st.specIntensity);
}

// ---------------------------------------------------------------------------
//  Anime metal: posterize the physically traced reflection, then optionally
//  pull it toward a tint.  This is the case raster NPR cannot reach, because
//  the banding is applied to a real reflection rather than a cubemap guess.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Float3 styleMetal(const Float3& reflected, const StyleData& st)
{
    Float3 c = styleQuantizeTone(reflected, st.reflectBands, st.bandSoftness, 1.0f);
    c = mul3(c, st.reflectGain);

    if (st.reflectTintMix > 0.0f) {
        const Float3 tinted = mul3(c, st.reflectTint);
        c = add3(mul3(c, 1.0f - st.reflectTintMix), mul3(tinted, st.reflectTintMix));
    }
    return c;
}

// ---------------------------------------------------------------------------
//  Anime jewel: posterize the refraction into facet bands and push saturation
//  so the gem reads as coloured glass rather than grey glass.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Float3 styleJewel(const Float3& transmitted, const StyleData& st)
{
    Float3 c = styleQuantizeTone(transmitted, st.transmitBands, st.bandSoftness, 1.0f);
    c = mul3(c, st.transmitGain);

    if (st.chromaShift > 0.0f) {
        const float lum = styleLuminance(c);
        // Push each channel away from its luminance to increase saturation.
        c.x = lum + (c.x - lum) * (1.0f + st.chromaShift);
        c.y = lum + (c.y - lum) * (1.0f + st.chromaShift);
        c.z = lum + (c.z - lum) * (1.0f + st.chromaShift);
        c.x = fmaxf(c.x, 0.0f);
        c.y = fmaxf(c.y, 0.0f);
        c.z = fmaxf(c.z, 0.0f);
    }
    return c;
}

// ---------------------------------------------------------------------------
//  Rim light from the primary-hit normal AOV and the view direction.
//  Computed at present time, so it costs no rays and carries no noise.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Float3 styleRim(const Float3& normal, const Float3& viewDir,
                                            const StyleData& st)
{
    if (st.rimStrength <= 0.0f) return {0.0f, 0.0f, 0.0f};

    const float n2 = dot3(normal, normal);
    if (n2 <= 1e-8f) return {0.0f, 0.0f, 0.0f};

    // viewDir points from the camera toward the surface, so -viewDir . n is the
    // facing ratio; the rim is the complement of it.
    const float facing = fabsf(dot3(normalize3(normal), normalize3(viewDir)));
    const float rim    = powf(styleClamp01(1.0f - facing), fmaxf(st.rimPower, 1e-3f));

    return mul3(st.rimColor, rim * st.rimStrength);
}
