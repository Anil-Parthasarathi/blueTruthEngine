# Wavefront Path Tracer — Conversion Guide (RedTruthEngine)

This is a step-by-step guide for converting the current **megakernel-on-OptiX**
path tracer into a **wavefront** architecture. It is written for *you* to
implement; it gives the design, the data layout, the kernel breakdown, the host
loop, and the correctness gotchas — but no finished kernel code.

Since the megakernel refactor, all the shading math is already extracted into
shared headers that the wavefront kernels can call directly — the stubs are
mostly glue. Read this alongside:

- [WAVEFRONT_IMPLEMENTATION.md](WAVEFRONT_IMPLEMENTATION.md) — the build order, the
  per-step verification, and a signature reference for every function you'll call.
  Work from that one; use this document as the per-stage spec it refers back to.
- [MEGAKERNEL_WALKTHROUGH.md](MEGAKERNEL_WALKTHROUGH.md) — how a frame renders
  today, step by step, and which shared function each wavefront stage should call.
  Start here if you need a refresher on the existing path tracer.
- `src/optix_programs.cu` — the four OptiX entry points (`__raygen__rg`,
  `__closesthit__radiance`, `__miss__radiance`, `__miss__shadow`).
- `include/rt_optix_shading.cuh` — `traceClosest`/`traceOccluded`, the megakernel
  NEE wrappers, and `traceRay` (the per-bounce step).
- `include/rt_direct_lighting.cuh` — `accumulateEmitterHit` and the **pure** NEE
  helpers `prepareAreaEmitterNEE` / `prepareSpotlightNEE` (no tracing inside).
- `include/rt_path_integrator.cuh` — `scatterPath`, `russianRoulette`,
  `accumulateWelford`, `packPixelGamma`.
- `include/rt_material.cuh` — `loadBsdfForTriangle` (BSDF + texture override).
- `include/rt_camera.cuh` — `generatePrimaryRay`.
- `include/rt_constants.cuh` — `RT_EPSILON`, `MAX_BOUNCES`, `SAMPLES_PER_PIXEL`.
- `include/wavefront/wavefront_buffers.h` — `WavefrontSoA`, `WfSceneView`.
- `include/wavefront/wavefront_shading.cuh` — `loadPathState`/`storePathState`,
  `loadIntersection`, `makeDirectLightingContext`, `enqueueShadowRay`.
- `src/host/wavefront_host.cu` — the finished host bounce loop
  (`cudaRenderWavefront`), buffer allocation, `buildSceneView`.
- `include/optix_launch_params.h` — `LaunchParams`, `RayType`, SBT records.
- `include/rt_rng.cuh` — `RngState`, `makeRng`, `hashUint`, `rngNextFloat01`.
- `include/rt_bsdf.cuh` — `bsdfSample`/`bsdfEval`/`bsdfPdf`, `bsdfIsDiffuse`,
  `bsdfIsDelta`, `BsdfType`.

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

The "intersect" stage stays a thin OptiX launch that only traces and writes
hits — this is NVIDIA's standard "wavefront + OptiX-for-traversal" design.

```
generate ─▶ [rayQueue] ─▶ extend(OptiX) ─▶ [hit buffer] ─▶ shade
   ▲                                                        │  │
   │            (host swaps the queues)                     │  ▼
[rayQueue] ◀── [rayQueueNext] ◀─────────────────────────────┘ [shadow SoA]
                                                                 │
                                                                 ▼
                                             connect(OptiX) ─▶ radiance (atomicAdd)
after the loop:  present ─▶ accumBuffer (Welford) ─▶ PBO (gamma)
```

---

## 1. Scope of the first version

Keep the first wavefront version **as close to current behavior as possible** so
you can diff images:

- One `cudaRender` frame traces **one path per pixel** and accumulates across
  frames (the `s_accumBuffer_d` Welford blend already does this). The megakernel
  traces `SAMPLES_PER_PIXEL = 5` paths per pixel per frame, so **at equal frame
  counts the wavefront image is ~5x noisier even when perfectly correct**. It
  converges to the same result; compare at 5x the frame count (or divide by eye).
- Keep the same RNG seeding: `makeRng(hashUint(pixelIndex ^ (frameIndex * 0x9e3779b9u)))`.
- Keep MIS exactly as in the megakernel — you get this for free by calling the
  shared `accumulateEmitterHit` / `prepare*NEE` / `scatterPath` instead of
  re-implementing them.

Only after it matches should you add multiple paths-in-flight per pixel and
material sorting.

---

## 2. SoA path-state buffers

`include/wavefront/wavefront_buffers.h` already defines the layout, and
`src/host/wavefront_host.cu` allocates/frees it. Summary:

Path state (persists across bounces; mirrors the megakernel's `PathState`):
- `Float3 rayOrigin[N]`, `Float3 rayDir[N]`
- `Float3 throughput[N]`
- `Float3 radiance[N]` (== `PathState::accumulatedColor`)
- `int    bounceCount[N]`
- `float  eta[N]`
- `float  brdfPDF[N]`          <- MIS state, MUST persist
- `unsigned char specularBounce[N]`          <- MIS state, MUST persist
- `uint32_t pixelIndex[N]`
- `uint32_t rngState[N]`       <- MUST persist; every stage that draws numbers
  writes the advanced state back to the same slot

Hit buffer (written by extend, read by shade):
- `int    hitTriIndex[N]`  (-1 == miss)
- `Float3 hitPoint[N]`, `Float3 hitNormal[N]`, `Float2 hitUV[N]`

> There are **no barycentric fields**: `Intersection` (in `rt_intersect.cuh`)
> doesn't store them either — `__closesthit__radiance` consumes the barycentrics
> to interpolate `hitNormal` and `uv`, then discards them. The interpolated
> values are everything shade needs.

Queues / counters (device `int` via `atomicAdd`):
- `int rayQueue[N]` + `int* rayCount` — THIS bounce's active paths (read-only
  during shade).
- `int rayQueueNext[N]` + `int* rayCountNext` — filled by shade for the NEXT
  bounce; the host swaps the two pairs after each bounce (**ping-pong**).
  A single queue would race: shade appends into the same buffer other threads
  are still reading from, so one thread can clobber an entry before it is read.
- `int* shadowCount` + the shadow SoA (`shadowOrigin/Dir/TMax/Contrib/PathIdx`,
  each `[M]`, `M = N * (1 + spotlightCount + 1)` worst case).
  There is **no shadow index queue**: `atomicAdd(shadowCount, 1)` already hands
  out contiguous slots, so connect indexes the shadow arrays directly with its
  launch index.

---

## 3. Stage 1 — Generate (`src/wavefront/ray_generate.cu`, CUDA kernel)

One thread per pixel:

1. Seed the RNG exactly as `__raygen__rg` does:
   `RngState rng = makeRng(hashUint(idx ^ (frameIndex * 0x9e3779b9u)));`
2. Draw the sub-pixel jitter (`rngNextFloat01(rng) - 0.5f` twice) and call the
   shared `generatePrimaryRay(px, py, width, height, camera, jx, jy)` from
   `rt_camera.cuh`. Write `rayOrigin`, `rayDir`.
3. Initialize `throughput = (1,1,1)`, `radiance = 0`, `bounceCount = 0`,
   `eta = 1`, `brdfPDF = 0`, `specularBounce = 1`, `pixelIndex = idx`, and
   persist `rng.state`.
4. Append the slot index to the **current** queue:
   `wf.rayQueue[atomicAdd(wf.rayCount, 1)] = idx;`

The host resets `wf.rayCount` to 0 before this launch (already wired in
`cudaRenderWavefront`).

---

## 4. Stage 2 — Extend (`src/wavefront/ray_extend.cu`, OptiX raygen)

`__raygen__wf_extend` does *only* traversal:

1. `queueSlot = optixGetLaunchIndex().x`, `idx = params.wf.rayQueue[queueSlot]`.
2. Build a `Ray` from `rayOrigin[idx]` / `rayDir[idx]` and call the shared
   `traceClosest(params.handle, ray, its)` from `rt_optix_shading.cuh` — the
   same helper the megakernel's `traceRay` uses (payload packing, `RT_EPSILON`
   tmin, `RAY_TYPE_RADIANCE`).
3. Write `its.triangleIndex`, `its.hitPoint`, `its.hitNormal`, `its.uv` into the
   hit buffer at `idx`. Miss ⇒ `hitTriIndex[idx] = -1`.

The existing `__closesthit__radiance` / `__miss__radiance` programs handle the
payload; no new hit/miss programs are needed. The host launches with
`optixLaunch(width = activeCount, height = 1)`.

---

## 5. Stage 3 — Shade (`src/wavefront/ray_shade.cu`, CUDA kernel)

One thread per entry in the current queue. This is glue around the shared
functions — the same sequence as the megakernel's `traceRay`, with two changes:
shadow rays are **enqueued** instead of traced, and terminated paths are simply
**not re-enqueued** (no accumulation here — see §7).

```
idx = wf.rayQueue[queueSlot]
PathState ps = loadPathState(wf, idx);          // wavefront_shading.cuh
RngState  rng; rng.state = wf.rngState[idx];

a) Miss (hitTriIndex < 0): optionally add background × ps.throughput; store
   back; return (no re-enqueue).

b) Hit:
   its  = loadIntersection(wf, idx);
   ctx  = makeDirectLightingContext(scene);
   bsdf = loadBsdfForTriangle(ctx, scene.bsdfs, scene.triangleBsdfIds,
                              its.triangleIndex, its.uv);

   1. if (ctx.triangleEmitterFlags[its.triangleIndex]) accumulateEmitterHit(ps, its, ctx);
   2. if (bsdfIsDiffuse(bsdf)) {
          ps.specularBounce = false;
          ShadowRayRecord sr{};
          if (prepareAreaEmitterNEE(ps, rng, its, bsdf, ctx, sr))
              enqueueShadowRay(wf, sr, idx);
          for (si in 0..ctx.spotlightCount)
              if (prepareSpotlightNEE(ps, its, bsdf, ctx, si, sr))
                  enqueueShadowRay(wf, sr, idx);
      } else ps.specularBounce = true;
   3. scatterPath(ps, rng, its, bsdf);           // BSDF sample + ray advance
   4. if (!russianRoulette(ps, rng)) { store back; return; }
   5. ++ps.bounceCount; storePathState(wf, idx, ps); wf.rngState[idx] = rng.state;
      wf.rayQueueNext[atomicAdd(wf.rayCountNext, 1)] = idx;   // NEXT queue!
```

**Ordering hazard:** `accumulateEmitterHit` reads the *previous* bounce's
`specularBounce`, `brdfPDF`, and ray origin/direction, and `prepare*NEE` reads
the previous ray direction — all of which `scatterPath` overwrites. Work on the
local `PathState` loaded once at the top and only call `scatterPath` after the
emitter-hit and NEE steps, exactly like the megakernel.

**The `bsdfSample` API** (used inside the shared `scatterPath` — you don't call
it yourself, but for reference): the throughput weight is the **return value**
of `Float3 bsdfSample(const BsdfData&, BsdfQueryRecord&, float u1, float u2,
float* outPdf = nullptr)`. There is no `bsdfQ.value`, `.pdf`, or `.sampledType`;
the PDF comes from `bsdfPdf(bsdf, bsdfQ)` and specular-ness from
`bsdfIsDiffuse(bsdf)` / `bsdfIsDelta(bsdf)` on the material type.

---

## 6. Stage 4 — Connect (`src/wavefront/ray_connect.cu`, OptiX raygen)

`__raygen__wf_shadow` over `[0, shadowCount)`; the launch index **is** the
shadow slot (no index queue):

1. Read `shadowOrigin/Dir/TMax/Contrib/PathIdx` at the launch index.
2. Test occlusion with the shared `traceOccluded(params.handle, org, dir, tMax)`
   from `rt_optix_shading.cuh` (terminate-on-first-hit, `RAY_TYPE_SHADOW`;
   `__miss__shadow` clears the payload — identical to the megakernel wrappers).
3. If **not** occluded, `atomicAdd` the stored contribution into
   `radiance[pathIdx]` (atomic because area NEE + several spotlights can target
   the same path). If occluded, drop it.

`shadowContrib` is already MIS-weighted and throughput-multiplied by the
`prepare*NEE` helpers — connect does no shading math.

---

## 7. Stage 5 — Present (`src/wavefront/ray_present.cu`, CUDA kernel)

Runs **once per frame, after the bounce loop** — this is where accumulation
happens, NOT in shade. Why: connect runs *after* shade each bounce, so a path
that writes NEE shadow rays and then dies (miss/RR) in the same shade call would
snapshot its radiance into `accumBuffer` *before* those shadow contributions
land — silently dropping direct lighting on terminating paths. Once the loop
exits, `wf.radiance[idx]` is final (one path per pixel per frame), so a single
terminal pass is both correct and simpler.

One thread per path slot:

1. `blended = accumulateWelford(scene.accumBuffer, wf.pixelIndex[idx],
   wf.radiance[idx], scene.frameIndex)` — shared with `__raygen__rg`.
2. `framebuffer[pixelIndex] = packPixelGamma(blended)` — writes the PBO.

If the denoiser is on, the host then calls `denoiseAndPresent`, overwriting the
PBO with the denoised tonemap (mirrors the megakernel path).

---

## 8. Material sorting (do this after v1 matches)

This is the biggest divergence win for your scene (it has diffuse, dielectric,
mirror, microfacet, and Disney materials).

- After extend, instead of one shade launch, bucket paths into **per
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

## 9. Host loop (`src/host/wavefront_host.cu` — already implemented)

`cudaRenderWavefront` is fully wired; you only implement the kernels:

```
ensureWavefrontBuffers(w, h)
rayCount = 0
generate()                                  // CUDA — fills rayQueue with all N
for (bounce = 0; bounce < MAX_BOUNCES; ++bounce) {   // shared constant (200)
    activeCount = *rayCount;  if (0) break;
    extend(activeCount)                     // OptiX, s_sbt_wf_extend
    rayCountNext = 0; shadowCount = 0       // NEXT queue reset — current stays
    shade(activeCount)                      // CUDA — fills rayQueueNext + shadow SoA
    if (shadowCount > 0) connect(shadowCount)  // OptiX, s_sbt_wf_shadow
    swap(rayQueue, rayQueueNext); swap(rayCount, rayCountNext)   // ping-pong
}
present()                                   // CUDA — Welford + gamma → PBO
if (denoiserEnabled) denoiseAndPresent()
```

Counter readbacks are small `cudaMemcpy`s with a `cudaDeviceSynchronize()`
between stages. As paths die off, `activeCount` shrinks, so per-launch cost
drops automatically (which also keeps launches well under the Windows TDR
watchdog).

---

## 10. Correctness checklist

- RNG (`rngState[N]`) is read-modify-written in generate **and** shade. Make
  sure every consumer writes the advanced state back to the same slot, or you'll
  get correlated/repeated samples.
- `brdfPDF` and `specularBounce` are produced by `scatterPath` in shade and
  consumed by `accumulateEmitterHit` in the *next* bounce's shade. Persist them,
  and call `accumulateEmitterHit` BEFORE `scatterPath` within a bounce.
- Shade appends survivors to `rayQueueNext`/`rayCountNext`, never to the queue
  it is reading. The host swap makes them current for the next bounce.
- Preserve `RT_EPSILON` handling — free if you use the shared helpers: bounce
  origin offset (`+ bounceDir*RT_EPSILON` in `scatterPath`), shadow
  `tMax = dist - RT_EPSILON` (in `prepare*NEE`), closest `tmin = RT_EPSILON`
  (in `traceClosest`).
- Keep `bsdfIsDiffuse` vs delta routing identical (only diffuse BSDFs do NEE).
- Shadow contributions must be added **atomically** (multiple lights per path).
- Accumulate ONLY in present. Shade never touches `accumBuffer`.
- Verify against the megakernel: same scene, and remember the 5x
  samples-per-frame difference (§1) when comparing at equal frame counts.

---

## 11. Suggested implementation order

1. `wfGenerate` + a debug visualisation (e.g. write `rayDir` as color via a
   temporary present pass) to validate the buffers.
2. `__raygen__wf_extend`; visualise `hitNormal` to verify hits match.
3. `wfPresent` (tiny — two shared calls) so terminated paths reach the screen.
4. `wfShade` miss path only → scene silhouette against black.
5. `wfShade` full (emitter hit + scatter + RR + re-enqueue), shadow queue still
   empty → indirect-only image.
6. `__raygen__wf_shadow` + NEE enqueues in shade → direct lighting appears;
   match the image to the megakernel.
7. Add per-material sorting (§8) and multiple paths-in-flight; profile with
   Nsight Compute (occupancy, warp execution efficiency, per-stage time split).

---

## 12. Expected payoff (this codebase)

- Wavefront vs the current megakernel-on-OptiX: typically **1.3–2x**, weighted
  toward the high end here because of `MAX_BOUNCES = 200` + mixed/Disney BSDFs
  (severe divergence today). Material sorting is the single largest contributor.
- Combined with the optimizations in the plan (lower `MAX_BOUNCES`, better
  sampler, the denoiser you now have via the `D` key), the practical
  time-to-clean-image improvement is larger than the raw frame-rate number.

Profile before/after each stage — wavefront has real overhead (queue traffic,
extra launches), so the win only shows up once divergence/occupancy actually
dominated, which it does for your deep, mixed-material paths.
