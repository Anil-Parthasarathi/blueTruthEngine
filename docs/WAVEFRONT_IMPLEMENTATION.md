# Wavefront Implementation Playbook

The actionable checklist for filling in the five stub kernels: what to call, in what
order, and how to prove each step works before starting the next.

This doc deliberately does **not** re-explain the design. For that:

- [MEGAKERNEL_WALKTHROUGH.md](MEGAKERNEL_WALKTHROUGH.md) — how a frame renders today.
- [WAVEFRONT_GUIDE.md](WAVEFRONT_GUIDE.md) — the per-stage spec and correctness rules.
- [WAVEFRONT_DEEP_DIVE.md](WAVEFRONT_DEEP_DIVE.md) — why the design is shaped this way.

---

## Ground rules

**You are writing five kernel bodies and nothing else.** The host loop, the SoA
buffers, the OptiX pipeline, the SBTs, and every shading function already exist and
are wired. If you find yourself editing a shared header, stop and ask which of these
two things is happening: either you found a real bug (fix it — it improves both
renderers at once), or you're about to re-implement something that already exists.

The five bodies, in the files that hold them:

| Stub | File | Kind |
| --- | --- | --- |
| `wfGenerate` | `src/wavefront/ray_generate.cu` | plain CUDA |
| `__raygen__wf_extend` | `src/wavefront/ray_extend.cu` | OptiX raygen (PTX) |
| `wfShade` | `src/wavefront/ray_shade.cu` | plain CUDA |
| `__raygen__wf_shadow` | `src/wavefront/ray_connect.cu` | OptiX raygen (PTX) |
| `wfPresent` | `src/wavefront/ray_present.cu` | plain CUDA |

Each file's header comment already contains the exact call sequence for its stage.
This doc is the ordering and testing layer on top of that.

**Two facts that dictate the order below:**

1. Nothing reaches the screen until `wfPresent` exists — the wavefront path's only
   PBO write is in present. Until then you are debugging blind (host readbacks only).
2. The SoA is `cudaMalloc`'d but never zeroed. Any stage that reads a buffer no
   earlier stage has written gets garbage, not zeros — and garbage in `hitTriIndex`
   or `pixelIndex` means out-of-bounds indexing, not just a wrong colour.

Together those mean: **generate and present go in first, as a pair.** That gets you a
visible feedback loop with all state initialised, and every later step is a
one-line change away from being visible on screen.

**Flags and keys you'll use throughout:**

- `USE_WAVEFRONT` (`src/main.cpp:51`) — `true` for wavefront, `false` for megakernel.
  Plain bool, so switching is just an exe rebuild, no CMake reconfigure.
- `SAMPLES_PER_PIXEL` (`include/rt_constants.cuh:11`) — set to `1` temporarily for the
  A/B comparison in Step 6. Only the megakernel reads it.
- `C` screenshot, `D` toggle denoiser, `R` toggle recording, `Escape` quit.

---

## The functions you will call

Everything here already exists. Full signatures so you don't have to go grepping.
Nothing in this table needs to be written or modified.

### Available to plain CUDA kernels (generate / shade / present)

| Function | Header |
| --- | --- |
| `RngState makeRng(uint32_t seed)` | `rt_rng.cuh` |
| `uint32_t hashUint(uint32_t x)` | `rt_rng.cuh` |
| `float rngNextFloat01(RngState& r)` | `rt_rng.cuh` |
| `Ray generatePrimaryRay(int px, int py, int width, int height, const CameraData&, float jx, float jy)` | `rt_camera.cuh` |
| `PathState loadPathState(const WavefrontSoA&, int idx)` | `wavefront/wavefront_shading.cuh` |
| `void storePathState(const WavefrontSoA&, int idx, const PathState&)` | `wavefront/wavefront_shading.cuh` |
| `Intersection loadIntersection(const WavefrontSoA&, int idx)` | `wavefront/wavefront_shading.cuh` |
| `DirectLightingContext makeDirectLightingContext(const WfSceneView&)` | `wavefront/wavefront_shading.cuh` |
| `void enqueueShadowRay(const WavefrontSoA&, const ShadowRayRecord&, uint32_t pathIdx)` | `wavefront/wavefront_shading.cuh` |
| `BsdfData loadBsdfForTriangle(const DirectLightingContext&, const BsdfData* bsdfs, const int* triangleBsdfIds, int triangleIndex, Float2 uv)` | `rt_material.cuh` |
| `void accumulateEmitterHit(PathState&, const Intersection&, const DirectLightingContext&)` | `rt_direct_lighting.cuh` |
| `bool prepareAreaEmitterNEE(PathState&, RngState&, const Intersection&, const BsdfData&, const DirectLightingContext&, ShadowRayRecord& out)` | `rt_direct_lighting.cuh` |
| `bool prepareSpotlightNEE(const PathState&, const Intersection&, const BsdfData&, const DirectLightingContext&, int spotlightIndex, ShadowRayRecord& out)` | `rt_direct_lighting.cuh` |
| `void scatterPath(PathState&, RngState&, const Intersection&, const BsdfData&)` | `rt_path_integrator.cuh` |
| `bool russianRoulette(PathState&, RngState&)` | `rt_path_integrator.cuh` |
| `Float3 accumulateWelford(Float3* accumBuffer, uint32_t pixelIndex, Float3 sampleColor, uint32_t frameIndex)` | `rt_path_integrator.cuh` |
| `uint32_t packPixelGamma(Float3 linear)` | `rt_path_integrator.cuh` |
| `bool bsdfIsDiffuse(const BsdfData&)` / `bool bsdfIsDelta(const BsdfData&)` | `rt_bsdf.cuh` |

### Available to OptiX raygens (extend / connect)

Everything above, plus — and **only** from a PTX translation unit:

| Function | Header |
| --- | --- |
| `bool traceClosest(OptixTraversableHandle, const Ray&, Intersection& out)` | `rt_optix_shading.cuh` |
| `bool traceOccluded(OptixTraversableHandle, const Float3& origin, const Float3& dir, float tMax)` | `rt_optix_shading.cuh` |

`rt_optix_shading.cuh` calls `optixTrace`, so including it from `ray_generate.cu`,
`ray_shade.cu`, or `ray_present.cu` will not compile. That split is deliberate.

### What you will *not* call

`sampleAreaEmitterNEE` / `sampleSpotlightNEE` (`rt_optix_shading.cuh`) are the
**megakernel's** NEE wrappers — they trace the shadow ray inline. The wavefront calls
the `prepare*NEE` helpers underneath them and defers the trace to connect. Same for
`traceRay`: it's the megakernel's whole per-bounce body, and the wavefront's stages
are that body taken apart.

You also never call `bsdfSample` / `bsdfEval` / `bsdfPdf` directly. `scatterPath`
wraps the sampling, and the `prepare*NEE` helpers wrap the evaluation. (For reference,
since the old docs got this wrong: `bsdfSample` returns the throughput weight as its
`Float3` **return value** — there is no `.value`, `.pdf`, or `.sampledType` field on
`BsdfQueryRecord`.)

---

## Step 1 — Generate + Present

**Write:** `wfGenerate` and `wfPresent` in full. They're the two smallest bodies and
they bracket everything else.

Generate: seed the RNG, draw two jitter values, call `generatePrimaryRay`, write all
ten path-state fields, then `wf.rayQueue[atomicAdd(wf.rayCount, 1)] = idx`.
Present: `accumulateWelford` then `packPixelGamma` into the framebuffer.

**Three easy mistakes here, all silently wrong rather than loud:**

- **Persist `rng.state` *after* the jitter draws**, not before. Store it before and
  shade replays the same two numbers as its first draws every bounce.
- **Use `atomicAdd` to fill the queue.** Writing `wf.rayQueue[idx] = idx` directly
  looks equivalent but leaves `wf.rayCount` at zero, so the host reads
  `activeCount == 0` and breaks out of the bounce loop immediately.
- **Seed with the path slot, not the queue slot.** `makeRng(hashUint(idx ^ (frameIndex
  * 0x9e3779b9u)))`, matching `__raygen__rg` exactly — this is what makes the Step 6
  comparison work.

**Verify:** temporarily replace the value present writes with the ray direction
remapped into `[0,1]` (`0.5f * (dir + 1)`), so present becomes a debug view of the
generate output:

```cpp
// TEMPORARY — Step 1 verification only
const Float3 d = wf.rayDir[idx];
framebuffer[wf.pixelIndex[idx]] = packPixelGamma({0.5f*(d.x+1.f), 0.5f*(d.y+1.f), 0.5f*(d.z+1.f)});
```

**Done when:** you see a smooth two-axis colour gradient across the window, matching
the camera orientation, with no noise or blocky patches. Noise means uninitialised
memory (some slot never got written). A hard discontinuity means the `px`/`py`
decomposition from `idx` is wrong.

Then delete the debug line and confirm you get a black window — that's correct now
(`radiance` is zero until shade exists), and it proves the accumulate/present path
runs without garbage.

---

## Step 2 — Extend

**Write:** `__raygen__wf_extend`. Read `idx` from the queue at the launch index, build
a `Ray` from the SoA, call `traceClosest(params.handle, ray, its)`, write the four hit
fields at `idx`.

**The one trap:** the launch index is a **queue slot, not a path slot**. `atomicAdd`
hands out slots in nondeterministic order, so `queueSlot != idx` in general. Read the
ray from `idx` and write the hit to `idx`; use `queueSlot` only to index `rayQueue`.
Getting this wrong produces a scrambled but plausible-looking image.

**Verify:** switch the debug line in present to visualise the hit normal:

```cpp
// TEMPORARY — Step 2 verification only
const Float3 n = wf.hitNormal[idx];
framebuffer[wf.pixelIndex[idx]] = packPixelGamma({fabsf(n.x), fabsf(n.y), fabsf(n.z)});
```

**Done when:** you see your scene's geometry in false colour, with flat faces reading
as flat colour and smooth surfaces shading smoothly. Background pixels come out black:
`traceClosest` takes an `Intersection its{}` whose members are default-initialised, and
the miss program only sets `triangleIndex = -1`, so a missed ray writes a zero normal.
A background that is *not* black means you're writing the hit fields on a stale or
wrong `idx`.

---

## Step 3 — Shade, miss path only

**Write:** only branch (a) of `wfShade` — load the path state, check
`wf.hitTriIndex[idx] < 0`, and for now add a bright debug colour to
`ps.accumulatedColor` before storing back. Don't re-enqueue. Leave the hit branch
empty so every path dies after one bounce.

**Verify:** remove the present debug line and run normally. You should see your
scene's silhouette as black against the bright debug background.

**Done when:** the silhouette matches the geometry you saw in Step 2. This is the first
step where radiance actually flows through `accumulateWelford`, so it also confirms
the accumulation buffer and frame index are behaving — the background should settle to
a constant colour as frames accumulate, not flicker or drift.

Then remove the debug colour. The megakernel adds nothing on a miss, so the final
version of this branch stores state and returns.

---

## Step 4 — Shade, hit path without NEE

**Write:** the rest of the hit branch except the NEE block — `loadIntersection`,
`makeDirectLightingContext`, `loadBsdfForTriangle`, the emitter-hit check, the
`bsdfIsDiffuse` routing (setting `specularBounce` but skipping the `prepare*NEE`
calls), `scatterPath`, `russianRoulette`, and the re-enqueue into `rayQueueNext`.

**Order matters and the compiler won't catch it.** `accumulateEmitterHit` reads the
*previous* bounce's `specularBounce`, `brdfPDF`, and ray origin/direction; the
`prepare*NEE` helpers read the previous ray direction. `scatterPath` overwrites all of
them. So: load `PathState` once at the top, do emitter-hit and NEE, *then*
`scatterPath`, exactly as `traceRay` does. Reversing this produces an image that looks
almost right, which is the worst kind of wrong.

**Two more:**

- Append to `rayQueueNext` / `rayCountNext`, **never** to `rayQueue` — you're still
  reading it, and other threads are too.
- Don't check `bounceCount` against `MAX_BOUNCES` in the kernel. The host `for` loop
  is the cap. Checking both is harmless but confusing; `russianRoulette`'s internal
  `bounceCount > 3` guard is the only bounce-count test shade needs.

**Verify:** run normally, with the shadow queue still empty every bounce.

**Done when:** you see a recognisable image with emitters visible and indirect
(bounce) lighting present, but no direct lighting from NEE — so it will be dark and
noisy, with light only where paths happened to hit an emitter. Add the counter
`fprintf` described under "The universal debug hook" below and confirm paths die off
over a handful of bounces rather than running to the 200 cap. That validates
throughput, the ping-pong queues, re-enqueueing, and Russian roulette all at once.

---

## Step 5 — Connect + NEE

**Write:** the NEE block in `wfShade` (`prepareAreaEmitterNEE` once, then
`prepareSpotlightNEE` in a loop over `ctx.spotlightCount`, each feeding
`enqueueShadowRay` when it returns `true`), and `__raygen__wf_shadow` in full.

Connect is the smallest stage: the launch index **is** the shadow slot (no index
queue), so read the five shadow arrays at the launch index, call `traceOccluded`, and
on a miss `atomicAdd` the stored contribution into `wf.radiance[pathIdx]`.

**Why the atomics are not optional:** one path can own several in-flight shadow rays
in the same bounce (one area sample plus one per spotlight), and they land in
different launch indices. A non-atomic read-modify-write loses contributions.

**Verify:** run normally. Direct lighting and soft shadows should appear, and the
image should get dramatically brighter and less noisy than Step 4.

**Done when:** it looks like the megakernel's output. Which brings us to actually
measuring that.

---

## Step 6 — Verify against the megakernel

Here's the useful part: **the two renderers draw from identical random number
streams**, so this is a numerical comparison, not an eyeball one.

Both seed `makeRng(hashUint(pixelIndex ^ (frameIndex * 0x9e3779b9u)))` per pixel, then
draw in the same order: two jitter values up front, then per bounce two light samples
(inside `prepareAreaEmitterNEE`, on diffuse hits only), three scatter samples (inside
`scatterPath`), and one roulette sample once past bounce 3. Spotlight NEE draws
nothing. Since both pipelines call the *same functions in the same order*, the
sequences line up bounce for bounce — which is exactly why the shared-header refactor
was worth doing before writing any of this.

So:

1. Set `SAMPLES_PER_PIXEL = 1` (`include/rt_constants.cuh:11`). The megakernel now
   traces one path per pixel per frame, same as the wavefront.
2. Capture both at the same frame count with `C` (`USE_WAVEFRONT = true`, then
   `false`).
3. The images should be **near-identical, pixel for pixel** — not merely similar.

Expect tiny differences only where a pixel's path emitted more than one shadow ray,
because float `atomicAdd` ordering in connect is nondeterministic. Anything larger is
a real bug, and the failure mode tells you where:

| What you see | Where to look |
| --- | --- |
| Same structure, uniformly ~5× noisier | You left `SAMPLES_PER_PIXEL = 5`. Not a bug. |
| Same structure, different noise pattern | RNG draw order diverged — usually a stray `rngNextFloat01` or state stored at the wrong point in generate |
| Direct light missing entirely | Connect's branch polarity — `traceOccluded` returns **true when blocked**, so the contribution is added on the early-return's *other* path |
| Direct light doubled / too bright | Contribution added in both shade and connect, or non-atomic accumulation racing |
| Emitters blown out, indirect fine | `specularBounce` not persisting, so MIS weights fall back to 1 |
| Correct at bounce 1, wrong after | `brdfPDF` not persisting, or `scatterPath` called before emitter-hit/NEE |
| Fireflies the megakernel doesn't have | Accumulating in shade instead of present, so a dying path snapshots before its shadow rays land |

Restore `SAMPLES_PER_PIXEL = 5` when you're done.

---

## The universal debug hook

Present is the only stage that writes the PBO, which makes it the natural place to
inspect anything. Any SoA buffer can be visualised by temporarily swapping what
present writes, remapped into `[0,1]`:

| To inspect | Map as |
| --- | --- |
| `rayDir` / `hitNormal` | `0.5f * (v + 1)` or `fabsf(v)` per component |
| `hitTriIndex` | `< 0` → white, else black (miss mask) |
| `throughput` / `radiance` | directly, or scaled if values exceed 1 |
| `bounceCount` | `count / 8.0f` as greyscale — shows path-depth structure |
| `hitUV` | `{u, v, 0}` — should look like a texture-coordinate chart |

Keep these behind a single `#if 0` block rather than deleting and retyping them; you
will want them again when you add material sorting.

For counters, the host loop already reads `rayCount` and `shadowCount` back every
bounce — a temporary `fprintf` in `cudaRenderWavefront` printing bounce index,
`activeCount`, and `shadowCount` is the cheapest possible progress check. Healthy
output falls off fast: roughly N, then a fraction of N, dropping to zero within
5–15 bounces for a typical scene. If `activeCount` stays flat at N, paths aren't
terminating (RR or the miss branch isn't ending them). If it drops to zero after one
bounce, shade isn't re-enqueueing.

---

## Invariants worth re-reading before you commit

These are the ones that produce plausible-but-wrong images rather than crashes:

1. **Accumulate only in present.** Shade must never touch `accumBuffer`.
2. **`accumulateEmitterHit` and `prepare*NEE` run before `scatterPath`**, on a
   `PathState` loaded once at the top.
3. **Shade reads `rayQueue`, writes `rayQueueNext`.** Never the same buffer.
4. **`rngState`, `brdfPDF`, and `specularBounce` persist across bounces.** They're
   produced in one shade call and consumed by the next.
5. **Shadow contributions are added atomically**, on the unoccluded branch.
6. **Queue slot ≠ path slot** in extend and shade.
7. **Let the shared helpers own `RT_EPSILON`** — bounce-origin offset in
   `scatterPath`, `tMax = dist - RT_EPSILON` in `prepare*NEE`, `tmin` in
   `traceClosest`. Don't add your own offsets on top.

---

## After it matches: optimisation order

Do not start any of this until Step 6 passes. Each of these changes the image's noise
characteristics or its timing, and you want a known-good baseline to diff against.

1. **Material sorting** — the single biggest win, and the reason wavefront exists.
   Bucket paths by `BsdfType` after extend so each shade launch runs one BSDF
   coherently. See `WAVEFRONT_GUIDE.md` §8.
2. **Cut the host round-trips.** The loop currently does a blocking counter readback
   plus a sync per stage, up to `MAX_BOUNCES` times. Check the queue every N bounces,
   or move the loop onto the device with CUDA graphs / persistent kernels.
3. **Multiple paths in flight per pixel**, to close the 5× samples-per-frame gap with
   the megakernel.
4. **Fix `findEmitterIndexForTriangle`** (`rt_emitter_sampling.cuh:132`) — an
   O(emitters × triangles) scan on every emissive hit. A per-triangle emitter-index
   side table makes it O(1) and speeds up both renderers.

Profile between each step with Nsight Compute — occupancy, warp execution efficiency,
and the per-stage time split. Wavefront has real overhead (queue traffic, extra
launches), so a change that should help sometimes doesn't, and the profile is the only
way to know which.
