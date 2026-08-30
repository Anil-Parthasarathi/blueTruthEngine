// Inspect an equirectangular HDRI through the same stb path the engine uses.
//
// Build and run:
//   c++ -std=c++17 -O1 -Iexternal/stb -o hdr_inspect tools/hdr_inspect.cpp
//   ./hdr_inspect assets/sky_anime.hdr
//
// Reports dimensions, the zenith-to-ground gradient, and the peak texel's
// recovered direction. The direction readout is the useful part: it confirms the
// file's row order matches envDirToUV()'s convention (v=0 is the +Y zenith), which
// is otherwise only visible as an upside-down sky in a finished render.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cmath>
#include <cstdio>

int main(int argc, char** argv)
{
    if (argc < 2) { std::fprintf(stderr, "usage: check_hdr <file.hdr>\n"); return 2; }

    int w = 0, h = 0, n = 0;
    float* px = stbi_loadf(argv[1], &w, &h, &n, 3);
    if (!px) { std::fprintf(stderr, "stbi_loadf failed: %s\n", stbi_failure_reason()); return 1; }

    std::printf("loaded %dx%d, %d source channels\n", w, h, n);

    auto at = [&](int x, int y) { return px + 3 * (static_cast<size_t>(y) * w + x); };

    auto dump = [&](const char* label, int y) {
        const float* p = at(w / 2, y);
        std::printf("  %-22s v=%.3f  rgb=(%.3f %.3f %.3f)\n",
                    label, (y + 0.5f) / h, p[0], p[1], p[2]);
    };
    dump("zenith (row 0)", 0);
    dump("upper sky", h / 4);
    dump("horizon", h / 2 - 1);
    dump("just below horizon", h / 2 + 1);
    dump("ground (last row)", h - 1);

    // Sun: count texels far brighter than the sky and find the peak.
    double total = 0.0, peak = 0.0;
    long bright = 0;
    int peakX = 0, peakY = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float* p = at(x, y);
            const double lum = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
            total += lum;
            if (lum > 50.0) ++bright;
            if (lum > peak) { peak = lum; peakX = x; peakY = y; }
        }
    }
    const double mean = total / (static_cast<double>(w) * h);
    std::printf("  mean luminance %.4f, peak %.1f at (%d,%d), %ld texels > 50\n",
                mean, peak, peakX, peakY, bright);
    std::printf("  peak/mean ratio %.0f  (high ratio is the point: importance\n"
                "  sampling has to find this or the sun contributes nothing)\n",
                peak / mean);

    // Recover the sun direction from the peak texel and compare with the intent
    // (azimuth 40 deg, elevation 32 deg) to confirm the u/v convention round-trips.
    const double theta = (peakY + 0.5) / h * M_PI;
    const double phi = ((peakX + 0.5) / w - 0.5) * 2.0 * M_PI;
    const double elevation = 90.0 - theta * 180.0 / M_PI;
    double azimuth = phi * 180.0 / M_PI;
    std::printf("  peak direction: azimuth %.1f deg, elevation %.1f deg "
                "(expected 40.0 / 32.0)\n", azimuth, elevation);

    stbi_image_free(px);
    return 0;
}
