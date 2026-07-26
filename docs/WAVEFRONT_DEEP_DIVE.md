# Wavefront Path Tracer — Deep Dive Reference
### RedTruthEngine · For someone new to CUDA

This document is the companion to `WAVEFRONT_GUIDE.md`. Where that file tells you
*what* to implement, this one tells you *why it works the way it does*. Read this
first, then use `WAVEFRONT_GUIDE.md` as the task checklist.

For a refresher on the pipeline being converted — how a frame renders today, and
which shared function each wavefront stage should call — see
[MEGAKERNEL_WALKTHROUGH.md](MEGAKERNEL_WALKTHROUGH.md). When you're ready to write
code, [WAVEFRONT_IMPLEMENTATION.md](WAVEFRONT_IMPLEMENTATION.md) is the build order and
verification checklist.

---

## Table of Contents

1. [How the GPU executes your code](#1-how-the-gpu-executes-your-code)
2. [Why the megakernel is slow — divergence and register pressure](#2-why-the-megakernel-is-slow)
3. [Structure-of-Arrays — the memory layout that makes wavefront fast](#3-structure-of-arrays)
4. [The wavefront loop — a birds-eye view](#4-the-wavefront-loop)
5. [Stage 1 — ray_generate.cu](#5-stage-1--ray_generatecu)
6. [Stage 2 — ray_extend.cu (OptiX raygen)](#6-stage-2--ray_extendcu)
7. [Stage 3 — ray_shade.cu](#7-stage-3--ray_shadecu)
8. [Stage 4 — ray_connect.cu (OptiX raygen)](#8-stage-4--ray_connectcu)
9. [Stage 5 — ray_present.cu](#9-stage-5--ray_presentcu)
10. [Queues and atomic counters — why ping-pong](#10-queues-and-atomic-counters)
11. [The RNG — seeding and advancing state](#11-the-rng)
12. [MIS state — brdfPDF and specularBounce](#12-mis-state)
13. [Welford accumulation — how progressive rendering works](#13-welford-accumulation)
14. [OptiX concepts — raygen, closest-hit, miss, payload, SBT](#14-optix-concepts)
15. [PTX and the two compilation paths](#15-ptx-and-the-two-compilation-paths)
16. [What happens in cudaRenderWavefront](#16-what-happens-in-cudarenderwavefront)
17. [Implementation order and how to test each stage](#17-implementation-order-and-testing)
18. [Quick reference — shared functions per stage](#18-quick-reference)

---

## 1. How the GPU executes your code

### Threads, blocks, and the grid

When you write `myKernel<<<blocks, threads>>>()`, you are launching a *grid* of
threads. Every thread runs the same function but gets a unique ID via
`threadIdx.x`, `blockIdx.x` etc. so it can figure out which piece of data to
work on.

For a wavefront kernel we will always use a 1-D grid and compute the index like:
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= activeCount) return;   // guard — grid may be slightly bigger than the work
```

`blockDim.x` is the number of threads per block (we use `256`). `blockIdx.x` is
which block we're in. Multiply and add gives a unique integer per thread across
the whole grid.

### Warps — the real unit of execution

A GPU does not execute threads one at a time or even one block at a time. It
groups 32 consecutive threads into a **warp**. All 32 threads in a warp execute
the *same instruction at the same time* (SIMD). The hardware switches between
hundreds of warps in flight to hide memory latency — while one warp waits for a
memory fetch, another runs.

This is important for us because:
1. **Branching within a warp is expensive** (see §2).
2. **Memory accesses within a warp are coalesced** if the 32 threads access
   consecutive addresses — all 32 bytes land in a single cache-line read. If
   threads scatter across memory, you pay 32 separate reads. This is the
   reason for SoA layout (§3).

### Occupancy

"Occupancy" is the fraction of the maximum possible warps that are actually
resident on a Streaming Multiprocessor (SM) at once. Higher occupancy means
more warps available to hide latency. The main enemy of occupancy is *register
usage* — each thread's local variables live in a register file of fixed size.
A kernel that uses many registers per thread can fit fewer warps simultaneously.
The wavefront approach keeps each kernel small and single-purpose, dramatically
reducing per-thread register pressure compared to the megakernel.

---

## 2. Why the megakernel is slow

The current renderer runs one `__raygen__rg` invocation *per pixel*, and inside
it loops through `SAMPLES_PER_PIXEL` samples, each doing up to `MAX_BOUNCES`
full path steps. Every step includes ray generation, traversal, closest-hit,
BSDF sampling, NEE, and Russian roulette — all in one enormous program.

### Problem 1 — Divergence

Consider 32 threads (one warp) after a few bounces. Some paths hit a diffuse
surface, some hit glass, some have already been killed by Russian roulette,
some are on bounce 3 while one outlier is on bounce 47. In a warp:

```
Thread 0: BSDF_Diffuse  → executes bsdfSample branch A
Thread 1: BSDF_Mirror   → executes bsdfSample branch B  ← must wait for thread 0
Thread 2: dead (RR)     → sits idle                     ← wastes a lane
Thread 3: bounce 47     → still looping while others finished long ago
...
```

Whenever threads in a warp take different branches, the GPU serialises them: it
runs the "true" side with the "false" threads masked off, then the "false" side
with "true" threads masked off. The warp takes as long as the *longest* path in
any branch. This is **warp divergence** and it is the biggest performance killer
in a deep path tracer.

### Problem 2 — Register pressure

The megakernel has to keep alive at the same time:
- The current ray
- The entire `PathState` (throughput, accumulated radiance, MIS state, RNG state)
- The `Intersection` result
- Local variables for BSDF sampling, NEE, texture lookups…

All of these live in registers. With many local variables you can easily end up
needing 128+ registers per thread. On an RTX 4080 each SM has 65,536 registers
total. At 128 registers per thread that is only 512 threads per SM — roughly
25% of the theoretical maximum. The other 75% of hardware is sitting idle.

### The wavefront solution

Split the work into small, single-purpose stages. Each stage runs over a *batch*
of paths, all doing the same narrow job:

- The shade kernel only shades — it does not trace rays.
- The extend kernel only traces — it does not shade.
- Within the shade kernel, if you add material sorting (§8 of WAVEFRONT_GUIDE),
  every thread in a warp hits the same `if (bsdf.type == ...)` branch.

Small kernels → low register pressure → high occupancy → better latency hiding.

---

## 3. Structure-of-Arrays

In the megakernel, per-path state is stored as a struct on the *stack*
(registers/local memory), one copy per thread. There is no persistent layout.

In wavefront, state must persist across multiple kernel launches (the path lives
from generate through to termination, possibly 50+ bounces later). We store it
in **global device memory** as **Structure-of-Arrays (SoA)**.

### SoA vs AoS

**Array-of-Structures (AoS)** — what you'd naturally write in C++:
```cpp
struct Path { float3 origin; float3 dir; float3 throughput; ... };
Path paths[N];
// Thread 0 reads paths[0].origin
// Thread 1 reads paths[1].origin
// These are NOT contiguous in memory!
```

**Structure-of-Arrays (SoA)** — what we use:
```cpp
float3 rayOrigin[N];   // ALL origins together
float3 rayDir[N];      // ALL directions together
float3 throughput[N];  // ALL throughputs together
// Thread 0 reads rayOrigin[0]
// Thread 1 reads rayOrigin[1]  ← adjacent in memory!
```

When thread 0 and thread 1 (in the same warp) both read `rayOrigin[idx]`, the
hardware sees two adjacent 12-byte reads and can satisfy both with a single
cache-line fetch instead of two separate ones. With 32 threads in a warp this
is a 32x reduction in memory transactions — the difference between a kernel that
runs at 80% of memory bandwidth and one that runs at 5%.

This is why `WavefrontSoA` in `include/wavefront/wavefront_buffers.h` has
separate arrays for every field instead of one array of structs.

### The bridge back to the megakernel's structs

You do not shade against raw arrays. `include/wavefront/wavefront_shading.cuh`
provides `loadPathState(wf, idx)` / `storePathState(wf, idx, ps)` and
`loadIntersection(wf, idx)`, which reassemble the exact `PathState` and
`Intersection` structs the megakernel's shared shading functions expect. Load
once at the top of the kernel, mutate the local copy, store once at the end.

---

## 4. The wavefront loop

Here is the full render loop for one frame, explained in English:

```
cudaRenderWavefront() in src/host/wavefront_host.cu:

  1. rayCount = 0
  2. Launch wfGenerate  → fills rayQueue with ALL N path indices
                        → initialises per-path state
  for bounce in 0..MAX_BOUNCES:                       (shared constant, 200)
      3. Read rayCount from GPU.  If 0, all paths are done → break.
      4. Launch __raygen__wf_extend (OptiX, width=rayCount)
             → reads rayQueue[0..rayCount-1]
             → for each path: traceClosest → RT cores → writes hit buffer
      5. Reset rayCountNext = 0, shadowCount = 0      (NOT rayCount!)
      6. Launch wfShade (CUDA, threads=rayCount)
             → reads rayQueue[0..rayCount-1] and the hit buffer
             → for each path: shared shading calls, shadow rays → shadow SoA,
               survivors → rayQueueNext
      7. Read shadowCount from GPU.  If > 0:
      8. Launch __raygen__wf_shadow (OptiX, width=shadowCount)
             → for each shadow ray: traceOccluded → if clear, atomicAdd
               contribution into radiance[pathIdx]
      9. swap(rayQueue, rayQueueNext); swap(rayCount, rayCountNext)
  10. Launch wfPresent → Welford-blend radiance[] into accumBuffer, gamma-encode
      into the PBO.  (Then denoiseAndPresent if the denoiser is on.)
```

Each numbered step corresponds to one GPU kernel launch. Between steps we do a
`cudaDeviceSynchronize()` — this blocks the CPU until the GPU finishes the
previous step. We must sync before reading `rayCount` because the GPU writes it
and the CPU reads it.

Two things worth noticing:

- **Shade never resets or writes the queue it reads.** It reads
  `rayQueue`/`rayCount` and appends to `rayQueueNext`/`rayCountNext`; the host
  swap in step 9 makes the "next" queue current. §10 explains why.
- **Nothing accumulates until step 10.** Radiance lives in `wf.radiance[idx]`
  until every bounce (and every shadow ray) has resolved. §9 explains why.

---

## 5. Stage 1 — ray_generate.cu

**File:** `src/wavefront/ray_generate.cu`  
**Kernel:** `wfGenerate<<<(N+255)/256, 256>>>(wf, camera, width, height, frameIndex)`  
**One thread per pixel.**

### What it does

1. Computes `px = idx % width`, `py = idx / width` — the 2-D pixel coordinate
   for this thread.
2. Seeds the RNG with a hash of the pixel index and frame index (same formula
   as the megakernel so noise patterns are comparable).
3. Draws two jitter samples from the RNG and calls the shared
   `generatePrimaryRay` (from `include/rt_camera.cuh` — the same function the
   megakernel calls) to get a world-space ray through this pixel.
4. Writes the initial path state: throughput = white, radiance = black,
   bounceCount = 0, specularBounce = 1 (treat camera ray as specular so a
   direct emitter hit gets full radiance).
5. Atomically appends `idx` to the ray queue (`wf.rayQueue` — generate fills
   the *current* queue; only shade uses `rayQueueNext`).

### The RNG

Look at `include/rt_rng.cuh`. `RngState` is just a `uint32_t` wrapping a
xorshift32 generator. Seeding uses the same expression as `__raygen__rg`:

```cpp
RngState rng = makeRng(hashUint(static_cast<uint32_t>(idx) ^ (frameIndex * 0x9e3779b9u)));
```

(there is no `rngInit()` helper — `makeRng` + `hashUint` is the whole API).
`rngNextFloat01(rng)` advances the state and returns a float in [0,1).

The key rule: **every time you call `rngNextFloat01`, you must write `rng.state`
back to `wf.rngState[idx]`** before the kernel returns. If you forget, the next
stage that reads `rngState[idx]` will restart from the same seed and produce
correlated samples (visible as structured noise).

### generatePrimaryRay

Lives in `include/rt_camera.cuh` and takes the camera explicitly:
`generatePrimaryRay(px, py, width, height, camera, jx, jy)`. The camera has:

- `camera.origin` — the eye position
- `camera.forward`, `.right`, `.up` — the camera's orientation vectors
- `camera.fovYRadians` — vertical field of view angle
- `camera.aspect` — width/height ratio

The idea: map pixel `(px, py)` to NDC space `[-1,1]`, scale by the half-tangent
of the FOV, then build a direction vector `forward + right*sx + up*sy`. Nothing
to port — just call it.

### Initialising specularBounce = 1

We mark the first bounce as "specular" even though a camera ray is not a BSDF
sample. The reason is MIS (Multiple Importance Sampling) in the shade stage. On
diffuse surfaces, the shade stage computes NEE (explicit light sampling) to
directly estimate the direct lighting contribution. But it must not *also* add
that contribution when the next extend step happens to hit the light directly —
that would be double-counting. The `specularBounce` flag prevents this:
- If `specularBounce = 1`, `accumulateEmitterHit` adds the full emitter radiance
  when the ray hits a light (BRDF MIS weight = 1 because there was no NEE for
  this bounce).
- If `specularBounce = 0`, it applies a fractional MIS weight because NEE
  already contributed the other fraction.

Setting it to 1 at the start means camera rays that directly hit a light see it
correctly.

---

## 6. Stage 2 — ray_extend.cu

**File:** `src/wavefront/ray_extend.cu`  
**Program:** `__raygen__wf_extend` — an OptiX *raygen* program, compiled to PTX  
**Launched with:** `optixLaunch(pipeline, ..., &s_sbt_wf_extend, activeCount, 1, 1)`

### What OptiX raygen programs are

An OptiX *raygen* program is a GPU function that OptiX launches in a grid, one
thread per "pixel" in the launch (here we abuse "pixel" to mean "one active
ray"). Inside it you call `optixTrace(...)` which triggers the hardware BVH
traversal (RT cores) and calls your closest-hit or miss program when done.

`optixGetLaunchIndex()` returns the 3-D index of the current thread within the
OptiX launch — equivalent to `blockIdx.x * blockDim.x + threadIdx.x` but for
OptiX. We launch 1-D (width = `activeCount`, height = 1, depth = 1) so we only
use `.x`.

### Reading from the queue

```cpp
uint32_t queueSlot = optixGetLaunchIndex().x;   // which active ray are we?
int idx = params.wf.rayQueue[queueSlot];         // which path slot is that ray?
```

`queueSlot` runs from 0 to `activeCount-1`. It is an index *into the queue*, not
into the path SoA directly. The queue holds path slot indices — that is how we
handle paths terminating (dead paths are never re-enqueued so they disappear from
future queues).

### Tracing via the shared helper

The megakernel already wraps the payload-pointer mechanics in
`traceClosest(handle, ray, its)` (`include/rt_optix_shading.cuh`), and this
raygen just calls it:

```cpp
Ray ray;
ray.origin    = params.wf.rayOrigin[idx];
ray.direction = params.wf.rayDir[idx];

Intersection its{};
traceClosest(params.handle, ray, its);
// its.triangleIndex == -1  ⇒ miss (set by __miss__radiance)
```

Under the hood: `optixTrace` can only pass small integers as payload registers,
so `traceClosest` splits the 64-bit `&its` pointer into two 32-bit payload
values (`packPointer`). `__closesthit__radiance` (in `optix_programs.cu`)
reconstructs the pointer, then interpolates the hit attributes from the triangle
barycentrics and writes `triangleIndex`, `hitPoint`, `hitNormal`, and `uv` into
the struct. You do not write any new hit/miss programs.

### Writing the hit buffer

After `traceClosest` returns, copy the results from the stack `Intersection`
into the SoA hit buffer arrays at slot `idx`:

```cpp
params.wf.hitTriIndex[idx] = its.triangleIndex;
params.wf.hitPoint[idx]    = its.hitPoint;
params.wf.hitNormal[idx]   = its.hitNormal;
params.wf.hitUV[idx]       = its.uv;
```

(No barycentrics are stored — `__closesthit__radiance` consumed them to produce
the interpolated normal and UV, which is all shade needs.)

The shade kernel reads these in the next stage. Why store to global memory
instead of keeping them in registers? Because the shade kernel is a *different*
kernel launch — registers don't survive across kernel boundaries. Global memory
(device DRAM, the big GPU memory) is the only way to pass data between kernels.

### RT_EPSILON

`traceClosest` uses `tmin = RT_EPSILON = 1e-4f` (from `rt_constants.cuh`). This
tiny offset prevents self-intersection: when a ray is shot from a surface point,
the origin is slightly *outside* the surface so the ray doesn't immediately
re-hit the same triangle at t≈0 due to floating-point error.

---

## 7. Stage 3 — ray_shade.cu

**File:** `src/wavefront/ray_shade.cu`  
**Kernel:** `wfShade<<<(activeCount+255)/256, 256>>>(wf, scene, activeCount)`  
**One thread per active path in the queue.**

This stage runs everything the megakernel's per-bounce step (`traceRay` in
`rt_optix_shading.cuh`) does, *except* the actual ray tracing. Because all the
shading math lives in shared headers, the kernel is mostly glue.

### Thread indexing

```cpp
int queueSlot = blockIdx.x * blockDim.x + threadIdx.x;
if (queueSlot >= activeCount) return;
int idx = wf.rayQueue[queueSlot];   // actual path slot
```

We index the *queue*, not the full pixel array, so terminated paths (not in the
queue) are simply not processed. This is how wavefront gets faster as bounces
progress and paths die — later bounces have smaller `activeCount`, so the shade
kernel launches fewer threads.

### Load once, store once

```cpp
PathState ps = loadPathState(wf, idx);     // wavefront_shading.cuh
RngState  rng; rng.state = wf.rngState[idx];
```

This matters for correctness, not just style: `accumulateEmitterHit` reads the
**previous** bounce's `specularBounce`, `brdfPDF`, and ray origin/direction, and
the NEE helpers read the previous ray direction. `scatterPath` overwrites all of
them. By working on a local `PathState` in megakernel order — emitter hit, NEE,
*then* scatter — the reads always see the right values. Store back with
`storePathState(wf, idx, ps)` (and `wf.rngState[idx] = rng.state`) exactly once,
at every exit point.

### Step a — Miss

If `wf.hitTriIndex[idx] == -1`, the ray escaped the scene. The megakernel adds
nothing on a miss (no environment light); optionally add a sky colour ×
`ps.throughput` to `ps.accumulatedColor` here. Store back and return — **do not
re-enqueue, do not accumulate** (present handles that after the loop).

### Step b — Emitter hit

```cpp
Intersection its              = loadIntersection(wf, idx);
DirectLightingContext ctx     = makeDirectLightingContext(scene);
BsdfData bsdf                 = loadBsdfForTriangle(ctx, scene.bsdfs,
                                    scene.triangleBsdfIds, its.triangleIndex, its.uv);

if (ctx.triangleEmitterFlags[its.triangleIndex] != 0)
    accumulateEmitterHit(ps, its, ctx);
```

`accumulateEmitterHit` (shared, `rt_direct_lighting.cuh`) internally handles
both cases: specular previous bounce ⇒ weight 1, diffuse previous bounce ⇒
MIS-weighted by `ps.brdfPDF`. You do not branch on `specularBounce` yourself.

### Step c — NEE (shadow ray writing)

The megakernel wrappers `sampleAreaEmitterNEE`/`sampleSpotlightNEE` trace
inline; a CUDA kernel cannot call `optixTrace`, so you use the **pure** halves
they are built on — `prepareAreaEmitterNEE` / `prepareSpotlightNEE` — which do
all the sampling/MIS math and fill a `ShadowRayRecord` without tracing:

```cpp
if (bsdfIsDiffuse(bsdf)) {
    ps.specularBounce = false;
    ShadowRayRecord sr{};
    if (prepareAreaEmitterNEE(ps, rng, its, bsdf, ctx, sr))
        enqueueShadowRay(wf, sr, static_cast<uint32_t>(idx));
    for (int si = 0; si < ctx.spotlightCount; ++si)
        if (prepareSpotlightNEE(ps, its, bsdf, ctx, si, sr))
            enqueueShadowRay(wf, sr, static_cast<uint32_t>(idx));
} else {
    ps.specularBounce = true;
}
```

`enqueueShadowRay` (`wavefront_shading.cuh`) does the `atomicAdd(wf.shadowCount)`
and writes origin/dir/tMax/contribution/pathIdx into the shadow SoA. The
contribution is already MIS-weighted and throughput-multiplied; connect adds it
to `radiance[pathIdx]` only if the ray turns out unoccluded.

### Step d — BSDF sampling

One shared call:

```cpp
scatterPath(ps, rng, its, bsdf);   // rt_path_integrator.cuh
```

Inside it (for reference — you don't write this): `bsdfSample(bsdf, bsdfQ, u0, u1)`
**returns** the throughput weight as a `Float3` (there is no `bsdfQ.value` or
`bsdfQ.pdf` field — the PDF is queried separately with `bsdfPdf` and stored into
`ps.brdfPDF` for the next bounce's MIS). It then sets
`ps.ray.origin = hitPoint + bounceDir * RT_EPSILON` (offset along the **bounce
direction** — there is no `faceForward` helper in this codebase),
`ps.ray.direction = bounceDir`, multiplies `ps.throughput`, and accumulates
`ps.eta`.

`toLocalFromNormal` / `toWorldFromNormal` rotate directions into/out of the
local frame where `hitNormal` is +Z; all BSDF evaluation happens in that frame.

### Step e — Russian roulette

```cpp
if (!russianRoulette(ps, rng)) {          // rt_path_integrator.cuh
    storePathState(wf, idx, ps);
    wf.rngState[idx] = rng.state;
    return;                                // terminated — no re-enqueue
}
```

`russianRoulette` only acts after `ps.bounceCount > 3` (matching the
megakernel), uses `fminf(maxCoeff3(ps.throughput) * ps.eta * ps.eta, 0.99f)`
as the continuation probability (note the 0.99 cap, and `maxCoeff3` from
`rt_cuda_math.cuh` — there is no `maxComponent`), and divides survivors'
throughput to stay unbiased.

### Step f — Re-enqueue for next extend

```cpp
++ps.bounceCount;
storePathState(wf, idx, ps);
wf.rngState[idx] = rng.state;

int slot = atomicAdd(wf.rayCountNext, 1);
wf.rayQueueNext[slot] = idx;               // the NEXT queue, never rayQueue!
```

Appending to `rayQueueNext` instead of `rayQueue` is what makes the queue safe —
see §10. The host swaps the queues after connect.

### WfSceneView

`wfShade` receives a `WfSceneView` struct (defined in `wavefront_buffers.h`)
instead of `params` because it is a regular CUDA kernel (not OptiX) and cannot
read `extern "C" __constant__ LaunchParams params`. The host function
`buildSceneView()` in `src/host/wavefront_host.cu` packages all the scene device
pointers into this struct and passes it by value.
`makeDirectLightingContext(scene)` (`wavefront_shading.cuh`) then converts it to
the same `DirectLightingContext` the megakernel builds from `LaunchParams`, so
the shared shading functions see identical data.

---

## 8. Stage 4 — ray_connect.cu

**File:** `src/wavefront/ray_connect.cu`  
**Program:** `__raygen__wf_shadow` — an OptiX raygen, compiled to PTX  
**Launched with:** `optixLaunch(pipeline, ..., &s_sbt_wf_shadow, shadowCount, 1, 1)`

### What it does

There is **no shadow index queue**: `atomicAdd(wf.shadowCount, 1)` in shade
hands out contiguous slots, so the launch index *is* the shadow slot:

```cpp
uint32_t sIdx = optixGetLaunchIndex().x;

Float3   org     = params.wf.shadowOrigin[sIdx];
Float3   dir     = params.wf.shadowDir[sIdx];
float    tMax    = params.wf.shadowTMax[sIdx];
Float3   contrib = params.wf.shadowContrib[sIdx];
uint32_t pathIdx = params.wf.shadowPathIdx[sIdx];
```

The occlusion test is the shared `traceOccluded` from `rt_optix_shading.cuh` —
the very same helper the megakernel's NEE wrappers call:

```cpp
if (traceOccluded(params.handle, org, dir, tMax))
    return;                                   // blocked — drop the contribution
```

Internally it traces with `OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
DISABLE_CLOSESTHIT | DISABLE_ANYHIT` and `RAY_TYPE_SHADOW`; the
`__miss__shadow` program clears the payload to signal "not occluded".

If unoccluded, accumulate the pre-computed contribution into the path's radiance:

```cpp
atomicAdd(&params.wf.radiance[pathIdx].x, contrib.x);
atomicAdd(&params.wf.radiance[pathIdx].y, contrib.y);
atomicAdd(&params.wf.radiance[pathIdx].z, contrib.z);
```

### Why atomicAdd for radiance?

A single path can have multiple shadow rays in flight (one area emitter + one
per spotlight). All of them have the same `shadowPathIdx`. Multiple threads in
the connect kernel will therefore try to add to the same `radiance` element at
the same time. Without `atomicAdd`, two threads could read the same old value,
each add their contribution, and one write would overwrite the other — you would
lose a light contribution. `atomicAdd` serialises these writes so no update is
lost. It is slightly slower than a plain write but necessary for correctness.

---

## 9. Stage 5 — ray_present.cu

**File:** `src/wavefront/ray_present.cu`  
**Kernel:** `wfPresent<<<(N+255)/256, 256>>>(wf, scene, framebuffer, width, height)`  
**One thread per path slot. Launched ONCE per frame, after the bounce loop.**

### Why accumulation is NOT done in shade

Within one bounce, connect runs *after* shade. Suppose a path enqueues NEE
shadow rays and then dies to Russian roulette in the same shade invocation. If
shade accumulated on termination, it would snapshot `wf.radiance[idx]` into the
accumulation buffer **before** connect added those shadow contributions —
silently dropping the direct lighting of every terminating bounce. Paths that
survive to the bounce cap would never accumulate at all.

Instead, radiance simply stays in `wf.radiance[idx]` until the loop fully
drains. With one path per pixel per frame, that value is final once the loop
exits, so a single terminal pass is both correct and simpler.

### What it does

```cpp
Float3 blended = accumulateWelford(scene.accumBuffer, wf.pixelIndex[idx],
                                   wf.radiance[idx],
                                   static_cast<uint32_t>(scene.frameIndex));
framebuffer[wf.pixelIndex[idx]] = packPixelGamma(blended);
```

Both helpers come from `rt_path_integrator.cuh` and are byte-identical to what
`__raygen__rg` runs at the end of its pixel — so wavefront and megakernel
converge to the same displayed image. If the denoiser is enabled, the host then
runs `denoiseAndPresent` (in `src/host/denoiser.cu`), which overwrites the PBO
with a tonemap of the denoised accumulation buffer — the same flow as the
megakernel path.

---

## 10. Queues and atomic counters

### How the queue works

The ray queue `wf.rayQueue` is a flat array of integers. Each integer is a
*path slot index* — an index into the SoA arrays. A path is "active" if and
only if its slot index appears in the queue.

`wf.rayCount` is a single integer in device memory that acts as the
next-free-slot counter. To append a path index `idx`:

```cpp
int slot = atomicAdd(wf.rayCountNext, 1);  // atomically increment, returns old value
wf.rayQueueNext[slot] = idx;
```

`atomicAdd` is a GPU atomic operation: it reads the counter, adds 1, stores the
new value, and returns the *original* value — all as one indivisible operation
even if thousands of threads are doing this simultaneously. The returned value
is the unique slot number for this thread. No two threads get the same slot,
and no slot is ever overwritten.

After the generate kernel runs, `*wf.rayCount == N` (all pixels are queued).
After one full bounce, some paths will have died (missed, RR-terminated) and
not been re-enqueued, so the next bounce's count is smaller.

### Why ping-pong (two queues)?

Suppose shade read from and appended to the *same* queue. Thread A (processing
queue slot 500) might still be waiting to read `rayQueue[500]` while thread B
(processing slot 3) finishes early, calls `atomicAdd`, gets slot 500, and
overwrites it with its own path index. Block execution order is undefined, so
this is a genuine data race — paths get duplicated and lost nondeterministically.

The fix is the classic **ping-pong buffer**: shade reads
`rayQueue`/`rayCount` (which nothing mutates during the launch) and appends to
`rayQueueNext`/`rayCountNext`. After connect, the host swaps the pointer pairs
(`std::swap` on the struct members — no data is copied). The host resets only
`rayCountNext` (and `shadowCount`) before shade; the current `rayCount` must
stay intact because shade is still using it.

### Why not use CUDA streams or dynamic parallelism?

For a first implementation, host-side `cudaDeviceSynchronize()` + explicit
counter copies is the clearest and least error-prone approach. The overhead of a
sync per bounce is small compared to the GPU work itself. Advanced optimisations
(persistent kernels, cooperative groups, device-side loop) can come later.

---

## 11. The RNG

`RngState` in `include/rt_rng.cuh` is a xorshift32 generator — just a single
`uint32_t`. It is fast and has decent statistical properties for path tracing.

### Seeding

```cpp
RngState rng = makeRng(hashUint(pixelIndex ^ (frameIndex * 0x9e3779b9u)));
```

`hashUint` scrambles the combined pixel/frame value so every pixel starts with
a different state on every frame. Without this you would see the same noise
pattern every frame (the accumulation would not converge). This is the exact
seeding `__raygen__rg` uses.

### Advancing state

Every call to `rngNextFloat01(rng)` changes `rng.state` and returns a float in
[0, 1). You must write `rng.state` back:

```cpp
float u0 = rngNextFloat01(rng);  // changes rng.state internally
float u1 = rngNextFloat01(rng);  // changes rng.state again
wf.rngState[idx] = rng.state;   // persist the advanced state for next stage
```

Shade reads the state written by generate, uses it to sample the BSDF and NEE,
then writes the advanced state back for the next bounce's shade step. Each path
slot's `rngState[idx]` is effectively an independent RNG stream.

---

## 12. MIS state

MIS (Multiple Importance Sampling) combines two ways of estimating direct
lighting:
1. **BRDF sampling** — sample a new direction from the BSDF and see if it hits a
   light.
2. **Light sampling (NEE)** — explicitly pick a point on a light and check
   visibility.

Combining both estimators reduces variance. But you must not double-count. The
MIS weight for each estimator is computed based on its probability density
function (PDF) relative to the other's. The two fields that persist for this
are:

**`brdfPDF`**: the PDF of the BSDF sample direction that was chosen in the
*previous* shade step (recorded by `scatterPath` via `bsdfPdf`). When the extend
step hits an emitter, `accumulateEmitterHit` needs this to compute
`weight = brdfPDF / (brdfPDF + lightPDF)`.

**`specularBounce`**: whether the *previous* BSDF sample was from a delta
distribution (perfect mirror or refraction). Delta BSDFs have zero probability
of randomly hitting a specific light direction, so no NEE is performed for
them. When `specularBounce = 1`, the weight for the BRDF path is 1.0 — use the
full emitted radiance. When 0, apply the fractional MIS weight.

The rule: **these two values are written in shade (by `scatterPath` / the
diffuse-routing branch) and read in the *next* shade step (by
`accumulateEmitterHit`).** They must survive in the SoA across the extend
between them — and within one shade invocation, `accumulateEmitterHit` must run
BEFORE `scatterPath` overwrites them.

---

## 13. Welford accumulation

The renderer accumulates samples across frames. Rather than storing all samples
and averaging at the end (wastes memory), it maintains a running mean using
Welford's online algorithm:

```
new_mean = old_mean + (new_sample - old_mean) / frame_count
```

This is implemented once, in `accumulateWelford` (`rt_path_integrator.cuh`),
shared by `__raygen__rg` and `wfPresent`. `accumBuffer` stores the *current
mean* in linear colour space; `packPixelGamma` (same header) applies the
gamma-2.2 encode for display.

In the wavefront pipeline the update happens exactly once per pixel per frame,
in the present stage (§9) — identical statistics to the megakernel, except the
megakernel averages `SAMPLES_PER_PIXEL = 5` paths into each per-frame sample
while the wavefront sample is a single path. Same converged image; ~5x more
per-frame variance at equal frame counts.

---

## 14. OptiX concepts

### The pipeline

When we call `ensureOptixPipeline()` in `src/host/optix_setup.cu`, we:
1. **Compile** the PTX (pre-compiled from `optix_programs.cu`, `ray_extend.cu`,
   `ray_connect.cu`) into an OptiX **Module**.
2. Create **Program Groups** — wrappers around individual GPU functions. Each
   raygen, closest-hit, any-hit, and miss function gets its own program group.
3. **Link** all program groups into a **Pipeline** — this is the equivalent of
   linking object files into an executable. All programs in one pipeline share
   the same traversal hardware and can call each other via the SBT.
4. Create the **SBT** (Shader Binding Table).

### The SBT

The SBT is a table that maps ray types and geometry to programs. When
`optixTrace` fires a ray and hits a triangle, OptiX looks up which
closest-hit program to call by indexing into the SBT using the ray type and
geometry SBT index. Our SBT layout:

| Slot | Content |
|------|---------|
| `raygenRecord` | Pointer to the raygen to run when `optixLaunch` is called |
| `missRecords[0]` | `__miss__radiance` — called for RAY_TYPE_RADIANCE misses |
| `missRecords[1]` | `__miss__shadow` — called for RAY_TYPE_SHADOW misses |
| `hitgroupRecords[0]` | `__closesthit__radiance` — called for RAY_TYPE_RADIANCE hits |
| `hitgroupRecords[1]` | (reuses __closesthit__radiance) — shadow rays disable CH |

We have *three* SBTs:
- `s_sbt` — megakernel, raygen = `__raygen__rg`
- `s_sbt_wf_extend` — wavefront extend, raygen = `__raygen__wf_extend`
- `s_sbt_wf_shadow` — wavefront shadow, raygen = `__raygen__wf_shadow`

The miss and hitgroup records are *shared* between all three SBTs (same device
pointers). Only the raygen record differs. Swapping which SBT is passed to
`optixLaunch` selects which raygen runs.

### LaunchParams

`LaunchParams` (in `include/optix_launch_params.h`) is the struct that the host
fills every frame and passes to `optixLaunch`. OptiX copies it into a CUDA
constant memory variable named `params` (declared
`extern "C" __constant__ LaunchParams params;` in the device files). Every
thread in the raygen reads from `params` to access scene data, the GAS handle,
and the wavefront SoA pointers.

Constant memory is cached and broadcast — all 32 threads in a warp reading the
same constant memory address costs the same as one read. This is why scene-wide
data (triangle pointer, emitter count, etc.) goes in `params` rather than being
passed via registers.

---

## 15. PTX and the two compilation paths

### Two kinds of CUDA files in this project

| File | Compiled to | How it runs |
|------|------------|-------------|
| `ray_generate.cu`, `ray_shade.cu`, `ray_present.cu` | Regular CUDA object → linked into `.exe` | Called by host via `<<<>>>` syntax |
| `ray_extend.cu`, `ray_connect.cu`, `optix_programs.cu` | PTX text → embedded as a C array → loaded by OptiX at runtime | Called by OptiX via `optixLaunch` |

**PTX** (Parallel Thread eXecution) is NVIDIA's intermediate assembly language.
When these files are compiled with `CUDA_PTX_COMPILATION ON`, the compiler
outputs human-readable text assembly instead of binary object code. OptiX then
JIT-compiles this PTX for the exact GPU at runtime using the display driver.
This is why we use `bin2c` to embed the PTX as a byte array inside the exe:
the exe carries the PTX with it so it does not need external `.ptx` files.

### Why raygens must be in PTX

Regular CUDA kernels (launched via `<<<>>>`) go through CUDA's runtime and
cannot call `optixTrace`, `optixGetLaunchIndex`, or any other OptiX built-in.
These built-ins only work inside an OptiX execution context, which is set up
when OptiX launches the raygen. Hence `ray_extend.cu` and `ray_connect.cu` are
compiled as OptiX device programs. The same split applies to headers:
`rt_optix_shading.cuh` (contains `optixTrace` calls) is PTX-only; everything
else (`rt_direct_lighting.cuh`, `rt_path_integrator.cuh`, …) works in both.

### `extern "C" __constant__ LaunchParams params`

All three device files that read `params` declare it as `extern "C"` —
"external C linkage, defined elsewhere". Since they all compile into the *same
PTX module* (the OBJECT library concatenates their PTX), there is only one
definition. OptiX sees `pipelineLaunchParamsVariableName = "params"` and populates
this constant with the pointer you give to `optixLaunch` before each launch.

---

## 16. What happens in cudaRenderWavefront

The real function lives in `src/host/wavefront_host.cu` and is already
implemented. Annotated:

```cpp
void cudaRenderWavefront(uint32_t* devPtr, int imageWidth, int imageHeight)
{
    // Lazy-allocate SoA buffers (only if resolution changed or first call).
    ensureWavefrontBuffers(imageWidth, imageHeight);

    WavefrontSoA& wf = s_wf.soa;
    const WfSceneView sv = buildSceneView(s_frameIndex);   // scene pointers, by value

    // Stage 1: fill the CURRENT ray queue with all N paths.
    // rayCount lives in device memory so the kernel can atomicAdd on it.
    rayCount = 0;
    launchWfGenerate(wf, s_camera_h, imageWidth, imageHeight, s_frameIndex);
    cudaDeviceSynchronize();   // ← CPU must wait before reading rayCount

    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {   // shared constant

        // How many paths still need a ray cast?
        int activeCount = *wf.rayCount;   // small D2H memcpy
        if (activeCount <= 0) break;      // all paths terminated

        // Stage 2: trace rays via OptiX / RT cores.
        // LaunchParams.wf carries the SoA pointers to the raygen via `params`.
        LaunchParams lp = {};
        lp.handle = s_gasHandle;  lp.triangles = s_triangles_d;  lp.wf = wf;
        upload lp;  optixLaunch(..., &s_sbt_wf_extend, activeCount, 1, 1);  sync;

        // Reset the NEXT queue + shadow counter.  NOT wf.rayCount —
        // shade is about to read the current queue!
        rayCountNext = 0;  shadowCount = 0;

        // Stage 3: shade — shared shading calls, shadow enqueue, rayQueueNext.
        launchWfShade(wf, sv, activeCount);  sync;

        // Stage 4: fire shadow rays (if shade wrote any).
        int shadowCount = *wf.shadowCount;
        if (shadowCount > 0) {
            lp.wf = wf;  upload lp;
            optixLaunch(..., &s_sbt_wf_shadow, shadowCount, 1, 1);  sync;
        }

        // Ping-pong: what shade wrote becomes the next bounce's input.
        std::swap(wf.rayQueue, wf.rayQueueNext);
        std::swap(wf.rayCount, wf.rayCountNext);
    }

    // Stage 5: every path is final — Welford-blend wf.radiance into
    // s_accumBuffer_d and gamma-encode into the PBO.
    launchWfPresent(wf, sv, devPtr, imageWidth, imageHeight);  sync;

    if (s_denoiserEnabled)
        denoiseAndPresent(devPtr, imageWidth, imageHeight);
}
```

`cudaRender` (in `src/render_kernel.cu`) maps the PBO, dispatches here when
`s_renderMode == RenderMode::Wavefront`, increments `s_frameIndex`, and unmaps.

---

## 17. Implementation order and testing

> [WAVEFRONT_IMPLEMENTATION.md](WAVEFRONT_IMPLEMENTATION.md) is the canonical version
> of this section, with per-step done-criteria, a symptom → cause table, and the
> numerical megakernel comparison. The summary below is kept for context.

Work in this order to always have something runnable after each step:

### Step A — Generate + Present

Implement `wfGenerate` and `wfPresent` together. Present is two shared calls
(`accumulateWelford`, `packPixelGamma`) and is the wavefront's only PBO write, so
until it exists nothing is visible; generate is what initialises the SoA that present
reads, and the SoA is never zeroed, so present alone would read garbage.

Together they give you a feedback loop: temporarily have present write
`wf.rayDir[idx]` remapped into `[0,1]` and you should see a smooth gradient matching
the camera frustum. Then remove the debug write — a black window is correct at this
point, since `radiance` stays zero until shade exists.

### Step B — Extend

Implement `__raygen__wf_extend`. After one generate + one extend, every path
should have a valid hit (or miss). Verify by reading `wf.hitTriIndex` back:
most should be ≥ 0 (hit geometry), border pixels should be -1 (missed). You
can visualise hit normals through present as RGB — you should see a false-colour
version of your scene, with a black background (a miss leaves a zero normal).

### Step C — Shade (misses only, no BSDF)

Implement just the miss path in `wfShade` with a temporary bright background
colour added to `ps.accumulatedColor`. Run generate → extend → shade → present:
you should see the scene silhouette against the background. (Remove the debug
colour afterwards — the megakernel adds nothing on a miss.)

### Step D — Shade (BSDF + re-enqueue, no NEE)

Add the emitter-hit / scatter / RR / re-enqueue logic (steps b, d–f from §7).
Shadow queue stays empty. Run the full bounce loop. You should see emitters and
indirect light but soft-shadowless direct lighting — paths bounce until RR or
the cap kills them. This validates throughput, the ping-pong queues, and
re-enqueueing.

### Step E — Shadow queue + connect

Add the NEE enqueues to shade (step c from §7) and implement
`__raygen__wf_shadow`. Now full direct lighting appears. Compare visually with
the megakernel (set `USE_WAVEFRONT = false` to switch back).

### Switching back to verify

Because `USE_WAVEFRONT` in `main.cpp` is a simple boolean, you can flip it,
recompile (no CMake reconfigure needed, just the exe build), and run both
renderers side by side in separate windows to compare. Use the `C` key
(existing screenshot hotkey) to capture both at the same frame count.
**Remember the sample-count difference:** the megakernel traces 5 paths per
pixel per frame, the wavefront 1 — at equal frame counts the wavefront image is
correct but ~5x noisier. Let it run ~5x longer for an apples-to-apples visual.

Better still, set `SAMPLES_PER_PIXEL = 1` and compare numerically: both pipelines
seed and draw from the RNG in the same order, so at 1 sample per pixel per frame they
should agree pixel-for-pixel. See the Step 6 comparison in
[WAVEFRONT_IMPLEMENTATION.md](WAVEFRONT_IMPLEMENTATION.md).

---

## 18. Quick reference — shared functions per stage

Nothing is ported anymore — the megakernel's logic lives in shared headers and
each stage just calls it:

| Stage | Calls | From |
|------|-------|------|
| `wfGenerate` | `makeRng` + `hashUint`, `generatePrimaryRay` | `rt_rng.cuh`, `rt_camera.cuh` |
| `__raygen__wf_extend` | `traceClosest` | `rt_optix_shading.cuh` |
| `wfShade` | `loadPathState` / `storePathState`, `loadIntersection`, `makeDirectLightingContext`, `enqueueShadowRay` | `wavefront/wavefront_shading.cuh` |
| `wfShade` | `loadBsdfForTriangle` | `rt_material.cuh` |
| `wfShade` | `accumulateEmitterHit`, `prepareAreaEmitterNEE`, `prepareSpotlightNEE` | `rt_direct_lighting.cuh` |
| `wfShade` | `scatterPath`, `russianRoulette` | `rt_path_integrator.cuh` |
| `wfShade` | `bsdfIsDiffuse` (routing only) | `rt_bsdf.cuh` |
| `__raygen__wf_shadow` | `traceOccluded` | `rt_optix_shading.cuh` |
| `wfPresent` | `accumulateWelford`, `packPixelGamma` | `rt_path_integrator.cuh` |

The megakernel calls the *same* functions (its NEE wrappers
`sampleAreaEmitterNEE`/`sampleSpotlightNEE` in `rt_optix_shading.cuh` are thin
shells around `prepare*NEE` + `traceOccluded`), so any fix or feature you add in
a shared header improves both renderers at once.
