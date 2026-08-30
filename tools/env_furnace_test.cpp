// Furnace test for the environment importance-sampling distribution.
//
// Build and run (needs nothing but a C++ compiler and stb):
//   c++ -std=c++17 -O2 -Iexternal/stb -o env_furnace_test tools/env_furnace_test.cpp
//   ./env_furnace_test assets/sky_anime.hdr
//
// This mirrors, on the CPU, the CDF construction in cudaInitEnvironment()
// (src/host/scene_upload.cu) and the envSample / envPdf pair in
// include/rt_environment.cuh. It is a mirror rather than the real thing so it can
// run without CUDA, which means it can drift: if you change the environment
// sampling math, change it here too. It exists because environment MIS is the
// easiest part of the lighting code to get subtly wrong, and a bug here degrades
// photorealistic renders as much as stylized ones.
//
// It checks the three things that actually go wrong:
//
//   1. Does the solid-angle pdf integrate to 1 over the sphere? A pdf that does
//      not normalize makes every environment NEE estimate biased.
//   2. Does importance sampling recover the same integral of radiance over the
//      sphere as a brute-force sum over texels? This is the furnace-style check;
//      it catches a mismatched pdfScale, a missing sin(theta), or a flipped v.
//   3. Does envSample's chosen direction, fed back through envPdf, return the
//      pdf that envSample reported? MIS weights are wrong if it does not.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

static const double PI = 3.14159265358979323846;

struct Env {
    int W = 0, H = 0;
    std::vector<float> rgb;          // W*H*3
    std::vector<float> condCdf;      // H * (W+1)
    std::vector<float> margCdf;      // H + 1
    double marginalTotal = 0.0;
    float  pdfScale = 0.0f;

    float lum(int x, int y) const {
        const float* p = &rgb[3 * (static_cast<size_t>(y) * W + x)];
        float l = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
        return (l > 0.0f && std::isfinite(l)) ? l : 0.0f;
    }
};

// --- mirrors cudaInitEnvironment() ------------------------------------------
static void buildCdf(Env& e)
{
    e.condCdf.assign(static_cast<size_t>(e.H) * (e.W + 1), 0.0f);
    e.margCdf.assign(static_cast<size_t>(e.H) + 1, 0.0f);

    double marginalTotal = 0.0;
    for (int y = 0; y < e.H; ++y) {
        float* row = &e.condCdf[static_cast<size_t>(y) * (e.W + 1)];
        double rowSum = 0.0;
        for (int x = 0; x < e.W; ++x) {
            rowSum += e.lum(x, y);
            row[x + 1] = static_cast<float>(rowSum);
        }
        const double theta = PI * (y + 0.5) / e.H;
        marginalTotal += rowSum * std::sin(theta);
        e.margCdf[y + 1] = static_cast<float>(marginalTotal);
    }
    e.marginalTotal = marginalTotal;
    e.pdfScale = static_cast<float>((double(e.W) * e.H) / (2.0 * PI * PI * marginalTotal));
}

// --- mirrors envDirToUV / envUVToDir ---------------------------------------
static void dirToUV(const double d[3], double& u, double& v)
{
    const double theta = std::acos(std::fmin(1.0, std::fmax(-1.0, d[1])));
    double phi = std::atan2(d[2], d[0]);
    u = phi / (2.0 * PI) + 0.5;
    v = theta / PI;
}

static void uvToDir(double u, double v, double d[3])
{
    const double phi = (u - 0.5) * 2.0 * PI;
    const double theta = v * PI;
    const double st = std::sin(theta);
    d[0] = st * std::cos(phi);
    d[1] = std::cos(theta);
    d[2] = st * std::sin(phi);
}

// --- mirrors envPdf --------------------------------------------------------
static double envPdf(const Env& e, const double d[3])
{
    double u, v;
    dirToUV(d, u, v);
    int x = static_cast<int>(u * e.W);
    int y = static_cast<int>(v * e.H);
    x = std::min(std::max(x, 0), e.W - 1);
    y = std::min(std::max(y, 0), e.H - 1);
    const float* row = &e.condCdf[static_cast<size_t>(y) * (e.W + 1)];
    return std::fmax(0.0f, row[x + 1] - row[x]) * double(e.pdfScale);
}

// --- mirrors sampleCdfReuse ------------------------------------------------
static int sampleCdfReuse(const float* cdf, int n, float& u, float& pdf)
{
    const float total = cdf[n];
    if (!(total > 0.0f)) { pdf = 0.0f; return 0; }
    const float target = u * total;

    int lo = 0, hi = n;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) / 2;
        if (cdf[mid] <= target) lo = mid; else hi = mid;
    }
    const float w = cdf[lo + 1] - cdf[lo];
    if (w <= 0.0f) { u = 0.0f; pdf = 0.0f; return lo; }
    pdf = w / total;
    u = (target - cdf[lo]) / w;
    return lo;
}

// --- mirrors envSample ----------------------------------------------------
static bool envSample(const Env& e, float u0, float u1, double dir[3], double& pdf)
{
    float rowPdf = 0.0f;
    const int y = sampleCdfReuse(e.margCdf.data(), e.H, u1, rowPdf);
    if (rowPdf <= 0.0f) return false;

    const float* row = &e.condCdf[static_cast<size_t>(y) * (e.W + 1)];
    float colPdf = 0.0f;
    const int x = sampleCdfReuse(row, e.W, u0, colPdf);
    if (colPdf <= 0.0f) return false;

    const double u = (x + double(u0)) / e.W;
    const double v = (y + double(u1)) / e.H;
    uvToDir(u, v, dir);

    pdf = std::fmax(0.0f, row[x + 1] - row[x]) * double(e.pdfScale);
    return pdf > 0.0;
}

static uint32_t rngState = 0x12345678u;
static float frand()
{
    rngState ^= rngState << 13; rngState ^= rngState >> 17; rngState ^= rngState << 5;
    return (rngState >> 8) * (1.0f / 16777216.0f);
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::fprintf(stderr, "usage: check_env_pdf <file.hdr>\n"); return 2; }

    Env e;
    int n = 0;
    float* px = stbi_loadf(argv[1], &e.W, &e.H, &n, 3);
    if (!px) { std::fprintf(stderr, "load failed: %s\n", stbi_failure_reason()); return 1; }
    e.rgb.assign(px, px + size_t(e.W) * e.H * 3);
    stbi_image_free(px);

    buildCdf(e);
    std::printf("HDRI %dx%d  marginalTotal=%.6g  pdfScale=%.6g\n",
                e.W, e.H, e.marginalTotal, e.pdfScale);

    // (1) Does the pdf integrate to 1? Sum p_omega * dOmega over every texel.
    // dOmega = 2*pi^2*sin(theta)/(W*H).
    double pdfIntegral = 0.0;
    double radianceIntegral[3] = {0, 0, 0};
    for (int y = 0; y < e.H; ++y) {
        const double theta = PI * (y + 0.5) / e.H;
        const double dOmega = 2.0 * PI * PI * std::sin(theta) / (double(e.W) * e.H);
        for (int x = 0; x < e.W; ++x) {
            pdfIntegral += double(e.lum(x, y)) * e.pdfScale * dOmega;
            const float* p = &e.rgb[3 * (size_t(y) * e.W + x)];
            for (int c = 0; c < 3; ++c) radianceIntegral[c] += p[c] * dOmega;
        }
    }
    std::printf("(1) integral of pdf over sphere = %.6f   (want 1.000000)\n", pdfIntegral);

    // (2) Importance-sample the same integral: E[L(w)/p(w)] should match.
    const int N = 4000000;
    double est[3] = {0, 0, 0};
    long   failed = 0, pdfMismatch = 0;
    double worstRelErr = 0.0;
    for (int i = 0; i < N; ++i) {
        double dir[3], pdf;
        if (!envSample(e, frand(), frand(), dir, pdf)) { ++failed; continue; }

        // (3) Round-trip the direction through envPdf.
        const double back = envPdf(e, dir);
        if (back > 0.0) {
            const double rel = std::fabs(back - pdf) / pdf;
            if (rel > 1e-4) { ++pdfMismatch; if (rel > worstRelErr) worstRelErr = rel; }
        } else {
            ++pdfMismatch;
        }

        double u, v;
        dirToUV(dir, u, v);
        int x = std::min(std::max(int(u * e.W), 0), e.W - 1);
        int y = std::min(std::max(int(v * e.H), 0), e.H - 1);
        const float* p = &e.rgb[3 * (size_t(y) * e.W + x)];
        for (int c = 0; c < 3; ++c) est[c] += p[c] / pdf;
    }
    for (int c = 0; c < 3; ++c) est[c] /= N;

    std::printf("(2) integral of radiance over sphere\n");
    std::printf("    brute force  = (%.4f %.4f %.4f)\n",
                radianceIntegral[0], radianceIntegral[1], radianceIntegral[2]);
    std::printf("    importance   = (%.4f %.4f %.4f)  over %d samples\n",
                est[0], est[1], est[2], N);
    for (int c = 0; c < 3; ++c) {
        const double rel = std::fabs(est[c] - radianceIntegral[c]) /
                           std::fmax(1e-9, radianceIntegral[c]);
        std::printf("    channel %d relative error %.4f%%\n", c, rel * 100.0);
    }

    std::printf("(3) envSample/envPdf disagreement: %ld of %d samples", pdfMismatch, N);
    if (pdfMismatch) std::printf(", worst relative error %.3g", worstRelErr);
    std::printf("\n    degenerate samples: %ld\n", failed);
    return 0;
}
