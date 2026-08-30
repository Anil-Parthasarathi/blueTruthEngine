#!/usr/bin/env python3
"""Generate assets/sky_anime.hdr, a procedural equirectangular HDRI.

Why a synthetic sky is checked in
---------------------------------
Environment importance sampling plus MIS is the most error-prone part of the
lighting code: envPdf() has to agree with envSample(), and the escaped-ray
weighting has to agree with both. A scene that only ever uses a flat background
colour never touches any of it. This sky is deliberately built to exercise it:

  * A very bright, very small sun disc. Uniform sphere sampling would find it
    almost never, so if the 2D CDF or its pdf is wrong the result is obviously
    wrong (fireflies, or a sun that contributes nothing).
  * A smooth zenith-to-horizon gradient, so the marginal CDF over rows is
    non-degenerate.
  * A dimmer ground hemisphere, so the distribution is not symmetric about the
    horizon and a flipped v coordinate shows up immediately.

Layout must match envDirToUV() in include/rt_environment.cuh: theta =
acos(dir.y) and v = theta/pi, so v=0 is the +Y zenith. Radiance files are
written top row first, so scanline 0 is the zenith.

Values are linear radiance and are free to exceed 1.
"""

import math
import struct
import sys
from pathlib import Path

WIDTH = 1024
HEIGHT = 512

ZENITH = (0.16, 0.28, 0.62)
HORIZON = (0.88, 0.82, 0.74)
GROUND = (0.10, 0.09, 0.08)

SUN_AZIMUTH_DEG = 40.0
SUN_ELEVATION_DEG = 32.0
SUN_RADIUS_DEG = 1.6
SUN_RADIANCE = (620.0, 560.0, 480.0)
# Wide, faint forward-scattering halo so the sun is not a bare disc.
GLOW_RADIUS_DEG = 22.0
GLOW_STRENGTH = 2.2


def sun_direction():
    az = math.radians(SUN_AZIMUTH_DEG)
    el = math.radians(SUN_ELEVATION_DEG)
    # Matches envDirToUV's convention: phi measured from +X toward +Z, +Y up.
    return (math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az))


def sky_radiance(direction, sun):
    y = direction[1]

    if y >= 0.0:
        # Gradient is biased toward the horizon so the band near y=0 stays bright.
        t = y ** 0.45
        base = [HORIZON[c] + (ZENITH[c] - HORIZON[c]) * t for c in range(3)]
    else:
        t = min(1.0, (-y) ** 0.5)
        base = [HORIZON[c] + (GROUND[c] - HORIZON[c]) * t for c in range(3)]

    cos_angle = max(-1.0, min(1.0, sum(direction[c] * sun[c] for c in range(3))))
    angle = math.degrees(math.acos(cos_angle))

    if angle <= SUN_RADIUS_DEG:
        return [base[c] + SUN_RADIANCE[c] for c in range(3)]

    if angle <= GLOW_RADIUS_DEG:
        falloff = 1.0 - (angle - SUN_RADIUS_DEG) / (GLOW_RADIUS_DEG - SUN_RADIUS_DEG)
        glow = GLOW_STRENGTH * (falloff ** 3)
        return [base[c] * (1.0 + glow) for c in range(3)]

    return base


def to_rgbe(rgb):
    """Shared-exponent encoding: mantissas scaled so the largest lands in [128,256)."""
    peak = max(rgb)
    if peak < 1e-32:
        return (0, 0, 0, 0)
    mantissa, exponent = math.frexp(peak)     # peak == mantissa * 2**exponent
    scale = mantissa * 256.0 / peak
    return (min(255, int(rgb[0] * scale)),
            min(255, int(rgb[1] * scale)),
            min(255, int(rgb[2] * scale)),
            exponent + 128)


def run_length_at(values, i, cap):
    n = len(values)
    run = 1
    while i + run < n and run < cap and values[i + run] == values[i]:
        run += 1
    return run


def rle_encode_channel(values):
    """Radiance new-style RLE for one channel of one scanline.

    Matches stb_image's decoder exactly: a length byte above 128 is a run of
    (byte - 128) copies, and a byte of 1..128 is a literal block of that many
    bytes. Both lengths must be non-zero and must not overrun the scanline, so
    runs cap at 127 and literal blocks at 128.
    """
    out = bytearray()
    n = len(values)
    i = 0
    while i < n:
        run = run_length_at(values, i, 127)

        if run >= 4:
            out.append(128 + run)
            out.append(values[i])
            i += run
            continue

        # Literal block. Advance a single byte per step so the 128-byte cap can
        # never be overshot, and stop as soon as a run worth encoding begins.
        start = i
        while i < n and (i - start) < 128:
            if run_length_at(values, i, 4) >= 4:
                break
            i += 1
        block = values[start:i]
        out.append(len(block))
        out.extend(block)
    return out


def main():
    out_dir = Path(__file__).resolve().parent.parent / "assets"
    if not out_dir.is_dir():
        sys.exit(f"assets directory not found at {out_dir}")
    path = out_dir / "sky_anime.hdr"

    sun = sun_direction()
    body = bytearray()

    for row in range(HEIGHT):
        # Pixel centres, so the poles are not sampled exactly.
        theta = (row + 0.5) / HEIGHT * math.pi
        sin_t, cos_t = math.sin(theta), math.cos(theta)

        channels = [bytearray(), bytearray(), bytearray(), bytearray()]
        for col in range(WIDTH):
            # Inverse of envDirToUV: u -> phi, v -> theta.
            phi = ((col + 0.5) / WIDTH - 0.5) * 2.0 * math.pi
            direction = (sin_t * math.cos(phi), cos_t, sin_t * math.sin(phi))

            rgbe = to_rgbe(sky_radiance(direction, sun))
            for c in range(4):
                channels[c].append(rgbe[c])

        body += bytes((2, 2, (WIDTH >> 8) & 0xFF, WIDTH & 0xFF))
        for c in range(4):
            body += rle_encode_channel(channels[c])

    header = (b"#?RADIANCE\n"
              b"# Generated by tools/make_sky_hdr.py\n"
              b"FORMAT=32-bit_rle_rgbe\n\n"
              + f"-Y {HEIGHT} +X {WIDTH}\n".encode())

    path.write_bytes(header + bytes(body))
    print(f"wrote assets/{path.name}  ({WIDTH}x{HEIGHT} RGBE, "
          f"{len(header) + len(body)} bytes)")


if __name__ == "__main__":
    main()
