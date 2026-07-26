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
