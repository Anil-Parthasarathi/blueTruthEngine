#!/usr/bin/env python3
"""Generate the 1-D cel ramp strips in assets/.

A ramp is a 256x1 RGB PNG sampled by styleBodyTone() in include/rt_style.cuh.

Lookup coordinate
-----------------
The coordinate is the QUANTIZED tone, not raw luminance. styleBandQuantize()
maps luminance onto `bands` discrete levels spanning [0,1], so with
diffuse_bands="N" the shader reads

    t = 0, 1/(N-1), ..., 1

t=0 is the deepest shadow band and t=1 the fully lit band. band_softness widens
each band's upper edge into a shoulder, during which t sweeps continuously
between two adjacent levels. Stops are therefore interpolated rather than held
flat: with softness=0 only the exact stop positions are ever sampled (crisp
bands), and with softness>0 the shoulder blends through the ramp smoothly.

Colour space
------------
The engine samples 8-bit textures with cudaReadModeNormalizedFloat and no sRGB
decode, i.e. the stored bytes ARE linear radiance, and gamma 2.2 is applied
later in packPixelGamma(). Stops below are written as the colours we want to
SEE, and encoded to linear on the way out. Open one of these PNGs in a viewer
and it will look darker than the stops read here; that is correct.

How the ramp is applied
-----------------------
styleBodyTone() returns (directDiffuse / luminance) * rampColour, so the ramp
supplies tone AND tint while chroma comes from the physically integrated direct
lighting. A ramp meant for surfaces that carry their own albedo (walls, floor,
cloth) needs a near-neutral lit band or it overwrites every material's hue;
only the shadow band should push hard toward a colour.
"""

import struct
import sys
import zlib
from pathlib import Path

WIDTH = 256
GAMMA = 2.2

# name -> (bands, [(t, (r, g, b)), ...]); t in [0,1], stops as 8-bit DISPLAY sRGB.
RAMPS = {
    # Body tone: shifting a shadow toward blue-violet rather than just darkening
    # it is the strongest single anime cue, so the shadow band moves in hue as
    # well as value. The lit band stays warm and near-white to preserve albedo.
    "ramp_skin": (
        3,
        [
            (0.0, (118, 104, 150)),  # shadow: cool violet
            (0.5, (222, 186, 174)),  # mid: warm terminator band
            (1.0, (255, 246, 236)),  # lit: warm near-white
        ],
    ),
    # Cloth / architecture. These surfaces have their own albedo, so the lit band
    # must stay neutral and the shadow band stays fairly unsaturated, letting the
    # red and green walls keep their identity through the band step.
    "ramp_cloth": (
        2,
        [
            (0.0, (150, 158, 196)),  # shadow: muted cool blue
            (1.0, (252, 250, 247)),  # lit: neutral
        ],
    ),
}


def to_linear_byte(display_byte):
    return min(255, max(0, round(((display_byte / 255.0) ** GAMMA) * 255.0)))


def build_scanline(stops):
    """Piecewise-linear interpolation between stops, in display space."""
    row = bytearray()
    for x in range(WIDTH):
        t = x / (WIDTH - 1)

        # Find the segment containing t; clamp outside the first/last stop.
        lo = stops[0]
        hi = stops[-1]
        for i in range(len(stops) - 1):
            if stops[i][0] <= t <= stops[i + 1][0]:
                lo, hi = stops[i], stops[i + 1]
                break

        span = hi[0] - lo[0]
        w = 0.0 if span <= 1e-6 else (t - lo[0]) / span
        for c in range(3):
            display = lo[1][c] + (hi[1][c] - lo[1][c]) * w
            row.append(to_linear_byte(display))
    return row


def write_png(path, scanline):
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", WIDTH, 1, 8, 2, 0, 0, 0)  # 8-bit RGB
    idat = zlib.compress(b"\x00" + bytes(scanline), 9)       # filter type 0
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", ihdr)
                     + chunk(b"IDAT", idat)
                     + chunk(b"IEND", b""))


def main():
    out_dir = Path(__file__).resolve().parent.parent / "assets"
    if not out_dir.is_dir():
        sys.exit(f"assets directory not found at {out_dir}")

    for name, (bands, stops) in RAMPS.items():
        path = out_dir / f"{name}.png"
        write_png(path, build_scanline(stops))
        print(f"wrote {path.relative_to(out_dir.parent)}  "
              f"({WIDTH}x1 RGB8 linear, {bands} bands)")


if __name__ == "__main__":
    main()
