#pragma once

// ---------------------------------------------------------------------------
//  rt_constants.cuh — path-tracing tuning constants shared by the megakernel
//  and wavefront pipelines.
// ---------------------------------------------------------------------------

// Paths traced per pixel per frame by the megakernel (__raygen__rg).
// The wavefront pipeline traces 1 path per pixel per frame, so at equal frame
// counts it accumulates 5x fewer samples (correct, just noisier per frame).
static constexpr int   SAMPLES_PER_PIXEL = 5;

// Hard bounce cap for a single path (Russian roulette usually kills paths
// long before this).  Shared so megakernel and wavefront behave identically.
static constexpr int   MAX_BOUNCES       = 200;

// Self-intersection epsilon: closest-hit tmin, bounce origin offset, and
// shadow-ray tMax shortening all use this value.
static constexpr float RT_EPSILON        = 1e-4f;

// tMax for a shadow ray aimed at the environment light.  There is no finite
// distance to shorten against, so the ray must reach past all geometry; this
// matches the 1e30f tmax traceClosest already uses.
static constexpr float RT_ENV_TMAX       = 1e30f;

// ---------------------------------------------------------------------------
//  Ray-probed outlines (anime style mode only)
// ---------------------------------------------------------------------------
// Outlines are found by shooting probe rays around each path vertex and
// classifying discontinuities, rather than filtering the final image.  Coverage
// is max-accumulated along the path and composited as ink at present time, so a
// line seen inside a mirror still lands on the correct screen pixels.
//
// Cost is OUTLINE_PROBE_COUNT extra rays per active path per bounce, so the
// depth cap keeps it bounded: lines beyond a couple of bounces are not visible
// anyway.
static constexpr int   OUTLINE_MAX_DEPTH   = 2;
static constexpr int   OUTLINE_PROBE_COUNT = 4;
// Floor on the probe angular offset (~3 px at 720p / 40° fov) so authored
// line_width values near 0.004 stay visible instead of collapsing to a hairline.
static constexpr float OUTLINE_MIN_PROBE_ANGLE = 0.003f;
