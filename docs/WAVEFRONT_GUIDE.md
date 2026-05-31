# Wavefront Path Tracer — Conversion Guide (RedTruthEngine)

This is a step-by-step guide for converting the current **megakernel-on-OptiX**
path tracer into a **wavefront** architecture. It is written for *you* to
implement; it gives the design, the data layout, the kernel breakdown, the host
loop, and the correctness gotchas — but no finished kernel code.

It is tailored to this codebase. Read it alongside:

- `src/optix_programs.cu` — current raygen path loop (`__raygen__rg`, `traceRay`,
  `sampleAreaEmitterNEE`, `sampleSpotlightNEE`, `accumulateEmitterHit`).
- `src/render_kernel.cu` — host: OptiX context/pipeline/SBT/GAS, `cudaRender`.
- `include/optix_launch_params.h` — `LaunchParams`, `RayType`, SBT records.
- `include/rt_intersect.cuh` — `Ray`, `Intersection`.
- `include/rt_rng.cuh` — `RngState`.
- `include/rt_bsdf.cuh` — `bsdfSample`/`bsdfEval`/`bsdfPdf`, `bsdfIsDiffuse`,
  `bsdfIsDelta`, `BsdfType`.
- `include/rt_emitter_sampling.cuh` — emitter sampling helpers.

---

## 0. Why wavefront

The current design runs the **entire path** (generate -> intersect -> shade ->
NEE -> bounce, up to `MAX_BOUNCES = 200`) inside one program invocation per
pixel. Two problems on the GPU:

1. **Divergence.** Paths have wildly different lengths (Russian roulette kills
   most after a few bounces, but some specular/dielectric paths run deep) and
   hit different `BsdfType`s. Within a warp, threads wait on each other and the
   per-material `if (b.type == ...)` chains in `bsdfEval`/`bsdfSample`/`bsdfPdf`
   (`include/rt_bsdf.cuh`) execute divergently.
2. **Register pressure.** One program contains ray-gen + traversal + all five
   BSDFs + NEE + MIS. High register use lowers occupancy, which hides latency
   poorly.

Wavefront fixes both by splitting the work into **stages**, where each stage is
a small kernel that does one job over a large batch of *active* paths held in
**structure-of-arrays (SoA)** global buffers. Each stage is coherent (low
divergence) and small (low register pressure -> high occupancy). It also lets
you **sort paths by material** so each shade launch runs one BSDF coherently.

With Part 1 (OptiX) already done, the "intersect" stage becomes a thin OptiX
launch that only traces and writes hits — this is NVIDIA's standard
"wavefront + OptiX-for-traversal" design.

```
generate -> [ray queue] -> intersect(OptiX) -> [hit buffer] -> logic
   ^                                                              |
   |                                                              v
[ray queue] <----- shade(per material) <--- [material queues] <--+
                        |                                          |
                        v                                          v
                  [shadow queue] -> occlusion(OptiX) -> accumulate radiance
```

---

## 1. Decide the scope of the first version

Keep the first wavefront version **as close to current behavior as possible** so
you can diff images:

- Keep `SAMPLES_PER_PIXEL` semantics by treating each `cudaRender` frame as **one
  path per pixel** and accumulating across frames (the `s_accumBuffer_d` Welford
  blend already does this). The simplest start is **N = width*height paths in
  flight, one bounce-step per stage pass**.
- Keep the same RNG seeding (`makeRng(hashUint(pixelIndex ^ (frameIndex * 0x9e3779b9u)))`).
- Keep MIS exactly as in `traceRay`: `brdfPDF`, `specularBounce`, the
  `convertAreaPDFtoSolidAnglePDF` weighting, and the `bsdfIsDiffuse` vs delta
  routing.

Only after it matches should you add multiple paths-in-flight per pixel and
material sorting.

---

## 2. SoA path-state buffers

Replace the per-thread `PathState` (in `src/optix_programs.cu`) and the local
`Intersection`/`RngState` with parallel device arrays of length
`N = width * height` (one slot per pixel for v1). Allocate once in
`cudaInitScene`/`cudaResetAccumulation`; free in `cudaCleanup`.

Fields to store (split into separate arrays — do **not** keep them interleaved):

Path state (persists across bounces):
- `float3 rayOrigin[N]`, `float3 rayDir[N]`
- `float3 throughput[N]`
- `float3 radiance[N]` (the per-path accumulated color, == `PathState::accumulatedColor`)
- `int    bounceCount[N]`
- `float  eta[N]`
- `float  brdfPDF[N]`          <- MIS state, MUST persist
- `unsigned char specularBounce[N]` (bool)   <- MIS state, MUST persist
- `unsigned int pixelIndex[N]`
- `RngState rng[N]`            <- MUST persist; advancing it in any stage updates the same slot

Intersection results (written by the intersect stage, read by logic/shade):
- `int    hitTriIndex[N]`  (-1 == miss)
- `float  hitT[N]`
- `float2 hitBary[N]` (or precompute `float3 hitPoint`, `float3 hitNormal`,
  `float2 hitUV` if you prefer to do attribute interpolation in the intersect
  closest-hit, mirroring `__closesthit__radiance`).

Queues / counters (device `int` via `atomicAdd`):
- `int rayQueue[N]`     + `int* rayCount`        (paths needing an extend/closest-hit)
- `int shadowQueue[M]`  + `int* shadowCount`     (M sized for worst case; see §6)
- optionally per-material queues (see §8).

Tip: a single big `cudaMalloc` carved into views keeps allocation simple. Keep a
small `struct WavefrontBuffers` of raw pointers and pass it in `LaunchParams`
(extend the struct in `include/optix_launch_params.h`) and/or to CUDA kernels.

---

## 3. Generate stage (CUDA kernel)

One thread per pixel:

1. Compute `pixelIndex`, seed `rng[pixelIndex]` exactly as `__raygen__rg` does.
2. Generate the primary ray (port `generatePrimaryRay`, including the sub-pixel
   jitter draw from the RNG). Write `rayOrigin`, `rayDir`.
3. Initialize `throughput = (1,1,1)`, `radiance = 0`, `bounceCount = 0`,
   `eta = 1`, `brdfPDF = 0`, `specularBounce = 1`.
4. Append the slot index to `rayQueue` via `atomicAdd(rayCount, 1)` (for v1 with
   one path per pixel, the queue is just `0..N-1`, so you can skip the atomic and
   launch the intersect over all N initially).

Reset `rayCount`/`shadowCount` to 0 with `cudaMemset` before the stages that
fill them.

---

## 4. Intersect / Extend stage (OptiX)

Reuse the GAS and pipeline from Part 1, but make a **second, minimal raygen**
that does *only* traversal:

1. Add a launch-params pointer to the ray SoA + the active `rayQueue` + count.
2. The new `__raygen__extend` maps `optixGetLaunchIndex().x` -> queue slot ->
   path index, reads `rayOrigin/rayDir`, calls `optixTrace` (radiance ray type),
   and the closest-hit writes `hitTriIndex/hitT/hitBary` (or the full
   interpolated hit) into the SoA at that path index. The miss program writes
   `hitTriIndex = -1`.
3. Launch with `optixLaunch(width = activeCount, height = 1)` — i.e. a 1-D launch
   over the compacted active set. Keep the **row-band tiling** wrapper from
   `cudaRender` (or cap `activeCount` per launch) so you never trip the Windows
   TDR watchdog.

You can keep your current `__raygen__rg` for the non-wavefront path, and add the
extend raygen + program group + SBT records beside it. The closest-hit logic is
the same attribute interpolation already in `__closesthit__radiance`.

---

## 5. Logic stage (CUDA kernel)

One thread per active path (read from the queue you intersected). This is the
"router" — port the top of `traceRay` plus the loop control from
`__raygen__rg`:

1. If `hitTriIndex < 0` (miss): the path is done. The current code just breaks;
   environment light would be added here. Leave `radiance` as-is, do **not**
   re-enqueue.
2. If hit an emitter (`params.triangleEmitterFlags[tri] != 0`): run
   `accumulateEmitterHit` (the MIS BRDF-side weight). This reads `brdfPDF` and
   `specularBounce` from the path state — that's why they must persist.
3. Russian roulette: port the `bounceCount > 3` block from `__raygen__rg`
   (note it divides `throughput` by the continuation probability). If the path
   dies, finalize it (write its `radiance` into the per-pixel accumulation — see
   §9) and don't re-enqueue.
4. Survivors: append the path index to the **shade queue** (or per-material queue,
   §8). Do not advance the ray here; that happens in shade.

Keep an explicit `bounceCount < MAX_BOUNCES` guard to bound the loop.

> Implementation choice: you can merge "logic" into the front of the shade kernel
> to save a launch, but keeping it separate makes material sorting (§8) cleaner.

---

## 6. Shade stage (CUDA kernel)

One thread per surviving path. Port the **bottom half of `traceRay`**:

1. Load `BsdfData` (`params.bsdfs[triangleBsdfIds[tri]]`), apply the texture
   override exactly as `traceRay` does (the `tex2D<float4>` block).
2. If `bsdfIsDiffuse(bsdf)`: set `specularBounce = 0` and do NEE:
   - **Area emitters:** port `sampleAreaEmitterNEE`, but instead of calling
     `traceOccluded` inline, **write a shadow ray** (origin = hit point,
     direction = `directionToLight`, `tMax = distToLight - RT_EPSILON`) plus the
     precomputed light contribution (`fr * Le * lightWeight * geo / pdf *
     throughput`) into the **shadow queue** with `atomicAdd(shadowCount, 1)`.
     The contribution is only added to `radiance` *after* the occlusion stage
     confirms the ray is unblocked (§7).
   - **Spotlights:** port `sampleSpotlightNEE` the same way — one shadow ray per
     spotlight per path. Size the shadow queue for
     `activePaths * (1 + spotlightCount)` worst case.
   - else set `specularBounce = 1` (delta/specular: no NEE).
3. Sample the BSDF for the next bounce (port the `bsdfSample` block): update
   `rayOrigin = hitPoint + bounceDir * RT_EPSILON`, `rayDir = bounceDir`,
   `throughput *= sampleWeight`, `eta *= bsdfQuery.eta`, set `brdfPDF`,
   `bounceCount++`.
4. Append the path back into the **ray queue** for the next intersect pass.

This is exactly the work `traceRay` does — only the shadow test and the loop are
externalized.

---

## 7. Occlusion stage (OptiX)

A second minimal OptiX raygen (`__raygen__shadow`) over the shadow queue:

1. Read a shadow ray (origin/dir/tMax) and its stored contribution + target path
   index from the shadow SoA.
2. `optixTrace` with the shadow ray type and
   `OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | DISABLE_CLOSESTHIT | DISABLE_ANYHIT`
   (same as `traceOccluded` today). The shadow miss program means "not occluded".
3. If **not** occluded, `atomicAdd` the stored contribution into `radiance[pathIndex]`
   (atomic because multiple shadow rays — area + several spotlights — target the
   same path). If occluded, drop it.

Reuse the existing `RAY_TYPE_SHADOW` SBT entry and miss program from Part 1.

---

## 8. Material sorting (do this after v1 matches)

This is the biggest divergence win for your scene (it has diffuse, dielectric,
mirror, microfacet, and Disney materials).

- After the logic stage, instead of one shade queue, bucket survivors into **per
  `BsdfType` queues** (5 queues; see `enum BsdfType` in `include/render_kernel.h`).
  Either:
  - append to `queue[bsdfType]` with a per-bucket `atomicAdd`, or
  - write a `key = triangleBsdfIds[tri]` per path and radix/`cub::DeviceRadixSort`
    the active index array by key, then launch shade over contiguous ranges.
- Launch the shade kernel **once per bucket** (or one launch reading the sorted
  array). Within a launch all threads run the same BSDF path -> coherent
  `bsdfSample`/`bsdfEval`/`bsdfPdf`, far fewer divergent branches, lower
  registers per kernel.

You can even specialize: a dedicated shade kernel per material that calls only
that material's functions (drop the dispatch in `include/rt_bsdf.cuh`).

---

## 9. Accumulation & present (unchanged math)

When a path terminates (miss, RR death, or `bounceCount == MAX_BOUNCES`), its
`radiance` is the sample value for its pixel. Fold it into `s_accumBuffer_d`
using the **same Welford blend** as `__raygen__rg`:

```
new_avg = prev + (sample - prev) / (frameIndex + 1)
```

Then tonemap to the PBO. You already have a `presentLinearKernel` in
`src/render_kernel.cu` that applies the exact gamma-2.2 encode — reuse it
(and it also feeds the OptiX denoiser path). With one path/pixel/frame, each
pixel terminates exactly once per frame, so the blend stays correct.

---

## 10. Host loop (in `cudaRender`)

Replace the single tiled `optixLaunch` with a bounce loop:

```
reset accumulation-frame state; rayCount = N (all pixels)
generate()                                  // CUDA
for (bounce = 0; bounce < MAX_BOUNCES; ++bounce) {
    if (rayCount == 0) break;
    shadowCount = 0
    intersect(rayCount)                     // OptiX extend raygen (tiled)
    logic(rayCount)                          // CUDA: emitter hit, RR, route, finalize dead
    rayCount = 0                             // will be refilled by shade
    [sort by material]                       // optional
    shade(activePaths)                       // CUDA: NEE -> shadow queue, sample -> ray queue
    occlusion(shadowCount)                   // OptiX shadow raygen (tiled)
}
present()                                    // Welford blend already done as paths die; tonemap
++frameIndex
```

Read `rayCount`/`shadowCount` back with a small `cudaMemcpy` (or keep them in
device memory and launch with a generous fixed grid that early-outs on
`tid >= count`). The early-out approach avoids a sync per bounce.

Keep the **TDR-safe tiling**: cap any single OptiX/CUDA launch so it stays well
under ~1–2 s. Splitting by active count usually does this automatically as paths
die off.

---

## 11. Correctness checklist

- RNG (`rng[N]`) is read-modify-written in generate **and** shade (and NEE). Make
  sure every consumer writes the advanced state back to the same slot, or you'll
  get correlated/repeated samples.
- `brdfPDF` and `specularBounce` are produced in shade and consumed in the *next*
  bounce's logic (`accumulateEmitterHit`). Persist them.
- Preserve `RT_EPSILON` handling: bounce origin offset (`+ bounceDir*RT_EPSILON`),
  shadow `tMax = dist - RT_EPSILON`, closest `tmin = RT_EPSILON`.
- Keep `bsdfIsDiffuse` vs delta routing identical (only non-delta BSDFs do NEE).
- Shadow contributions must be added **atomically** (multiple lights per path).
- Verify against the current renderer: same scene, same `frameIndex` count,
  compare the accumulated image (it should converge to the same result; per-frame
  noise pattern will differ because work ordering changed, which is fine).

---

## 12. Suggested implementation order

1. SoA buffers + generate + a CUDA "shade-all-in-one" that still loops internally
   (just to validate the buffers) — optional scaffolding.
2. Split intersect into the OptiX extend raygen; verify hits match.
3. Split logic + shade; run the host bounce loop with **inline** occlusion first
   (call `traceOccluded` from shade via a CUDA-side trace is not possible — so do
   occlusion as its own OptiX stage from the start).
4. Add the shadow queue + occlusion stage.
5. Match the image to the megakernel.
6. Add per-material sorting (§8) and multiple paths-in-flight; profile with
   Nsight Compute (look at occupancy, warp execution efficiency, and the
   per-stage time split).

---

## 13. Expected payoff (this codebase)

- Wavefront vs the current megakernel-on-OptiX: typically **1.3–2x**, weighted
  toward the high end here because of `MAX_BOUNCES = 200` + mixed/Disney BSDFs
  (severe divergence today). Material sorting is the single largest contributor.
- Combined with the optimizations in the plan (lower `MAX_BOUNCES`, better
  sampler, the denoiser you now have via the `D` key), the practical
  time-to-clean-image improvement is larger than the raw frame-rate number.

Profile before/after each stage — wavefront has real overhead (queue traffic,
extra launches), so the win only shows up once divergence/occupancy actually
dominated, which it does for your deep, mixed-material paths.
