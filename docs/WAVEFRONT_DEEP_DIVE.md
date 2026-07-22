# Wavefront Path Tracer — Deep Dive Reference
### RedTruthEngine · For someone new to CUDA

This document is the companion to `WAVEFRONT_GUIDE.md`. Where that file tells you
*what* to implement, this one tells you *why it works the way it does*. Read this
first, then use `WAVEFRONT_GUIDE.md` as the task checklist.

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
9. [Queues and atomic counters](#9-queues-and-atomic-counters)
10. [The RNG — seeding and advancing state](#10-the-rng)
11. [MIS state — brdfPDF and specularBounce](#11-mis-state)
12. [Welford accumulation — how progressive rendering works](#12-welford-accumulation)
13. [OptiX concepts — raygen, closest-hit, miss, payload, SBT](#13-optix-concepts)
14. [PTX and the two compilation paths](#14-ptx-and-the-two-compilation-paths)
15. [What happens in cudaRenderWavefront](#15-what-happens-in-cudarenderwavefront)
16. [Implementation order and how to test each stage](#16-implementation-order-and-testing)
17. [Quick reference — functions to port](#17-quick-reference)

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

---

## 4. The wavefront loop

Here is the full render loop for one frame, explained in English:

```
cudaRenderWavefront() in render_kernel.cu:

  1. Reset queue counters to 0  (rayCount = 0, shadowCount = 0)
  2. Launch wfGenerate  → fills rayQueue with ALL N path indices
                        → initialises per-path state
  for bounce in 0..MAX_BOUNCES:
      3. Read rayCount from GPU.  If 0, all paths are done → break.
      4. Launch __raygen__wf_extend (OptiX, width=rayCount)
             → reads rayQueue[0..rayCount-1]
             → for each path: optixTrace → RT cores → writes hitTriIndex etc.
      5. Reset rayCount = 0, shadowCount = 0
      6. Launch wfShade (CUDA, threads=rayCount)
             → reads rayQueue[0..rayCount-1] and hit buffer
             → for each alive path: compute BSDF, write shadow rays to shadowQueue,
               write new ray to rayQueue (if continuing)
      7. Read shadowCount from GPU.  If > 0:
      8. Launch __raygen__wf_shadow (OptiX, width=shadowCount)
             → for each shadow ray: optixTrace → if unoccluded add contrib to radiance
  Present: radiance[] was already blended into accumBuffer by shade on termination.
```

Each numbered step corresponds to one GPU kernel launch. Between steps we do a
`cudaDeviceSynchronize()` — this blocks the CPU until the GPU finishes the
previous step. We must sync before reading `rayCount` because the GPU writes it
and the CPU reads it.

Why do we reset `rayCount` and `shadowCount` to 0 before shade? Because shade
itself *refills* them with the next bounce's rays. If we did not reset, the old
values would accumulate.

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
3. Draws two jitter samples from the RNG and calls `generatePrimaryRay` to get
   a world-space ray through this pixel (with sub-pixel AA offset).
4. Writes the initial path state: throughput = white, radiance = black,
   bounceCount = 0, specularBounce = 1 (treat camera ray as specular so we
   don't add NEE on the first bounce from the camera).
5. Atomically appends `idx` to the ray queue.

### The RNG

Look at `include/rt_rng.cuh`. `RngState` is just a `uint32_t` wrapping a
xorshift32 generator. `rngInit(pixelIndex, frameIndex)` hashes the two values
together to give a different starting state for every pixel every frame.
`rngNextFloat01(rng)` advances the state and returns a float in [0,1).

The key rule: **every time you call `rngNextFloat01`, you must write `rng.state`
back to `wf.rngState[idx]`** before the kernel returns. If you forget, the next
stage that reads `rngState[idx]` will restart from the same seed and produce
correlated samples (visible as structured noise).

### generatePrimaryRay

This function already exists in `optix_programs.cu` (and an older version in
`render_kernel.cu`). The camera has:

- `camera.origin` — the eye position
- `camera.forward`, `.right`, `.up` — the camera's orientation vectors
- `camera.fovYRadians` — vertical field of view angle
- `camera.aspect` — width/height ratio

The idea: map pixel `(px, py)` to NDC space `[-1,1]`, scale by the half-tangent
of the FOV, then build a direction vector `forward + right*sx + up*sy`. Copy the
version from `optix_programs.cu` and adapt it to use the `CameraData` struct
passed by value.

### Initialising specularBounce = 1

We mark the first bounce as "specular" even though a camera ray is not a BSDF
sample. The reason is MIS (Multiple Importance Sampling) in the shade stage. On
diffuse surfaces, the shade stage computes NEE (explicit light sampling) to
directly estimate the direct lighting contribution. But it must not *also* add
that contribution when the next extend step happens to hit the light directly —
that would be double-counting. The `specularBounce` flag prevents this:
- If `specularBounce = 1`, the shade stage will add the full emitter radiance
  when the ray hits a light (BRDF MIS weight = 1 because there was no NEE for
  this bounce).
- If `specularBounce = 0`, the shade stage applies a fractional MIS weight
  because NEE already contributed the other fraction.

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

### Payload pointer packing

`optixTrace` can only pass small integers as payload registers between the raygen
and the closest-hit/miss programs. We want to pass a *pointer* to an
`Intersection` struct, but pointers are 64 bits on a 64-bit GPU. Solution: split
the 64-bit pointer into two 32-bit values with the `packPointer` helper:

```cpp
Intersection its{};            // allocate on the stack (stays alive during trace)
unsigned int p0, p1;
packPointer(&its, p0, p1);     // split address into two ints

optixTrace(params.handle,      // the BVH (GAS)
           toF3(org), toF3(dir),
           RT_EPSILON, 1e16f, 0.f,          // tmin, tmax, rayTime
           255u,                             // visibility mask
           OPTIX_RAY_FLAG_NONE,
           RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE,
           p0, p1);            // payload: our split pointer

// After optixTrace returns, its has been filled by __closesthit__radiance
// (or left at its.triangleIndex = -1 by __miss__radiance).
```

The `__closesthit__radiance` program in `optix_programs.cu` reconstructs the
pointer with `unpackPointer(optixGetPayload_0(), optixGetPayload_1())`, casts it
to `Intersection*`, and fills in `triangleIndex`, `hitPoint`, `hitNormal`,
`hitUV`, etc. You do not need to write a new closest-hit program — the existing
one already does exactly this.

### Writing the hit buffer

After `optixTrace` returns, copy the results from the stack `Intersection` into
the SoA hit buffer arrays at slot `idx`:

```cpp
params.wf.hitTriIndex[idx] = its.triangleIndex;
params.wf.hitBaryU[idx]    = its.baryU;
params.wf.hitBaryV[idx]    = its.baryV;
params.wf.hitPoint[idx]    = its.hitPoint;
params.wf.hitNormal[idx]   = its.hitNormal;
params.wf.hitUV[idx]       = its.hitUV;
```

The shade kernel reads these in the next stage. Why store to global memory
instead of keeping them in registers? Because the shade kernel is a *different*
kernel launch — registers don't survive across kernel boundaries. Global memory
(device DRAM, the big GPU memory) is the only way to pass data between kernels.

### RT_EPSILON

We use `tmin = RT_EPSILON = 1e-4f`. This tiny offset prevents self-intersection:
when a ray is shot from a surface point, the origin is slightly *outside* the
surface so the ray doesn't immediately re-hit the same triangle at t≈0 due to
floating-point error.

---

## 7. Stage 3 — ray_shade.cu

**File:** `src/wavefront/ray_shade.cu`  
**Kernel:** `wfShade<<<(activeCount+255)/256, 256>>>(wf, scene, activeCount)`  
**One thread per active path in the queue.**

This is the largest and most complex stage. It does everything that the inner
loop body of `traceRay()` in `optix_programs.cu` does, *except* the actual ray
tracing (which happens in extend and connect).

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

### Step a — Miss

If `wf.hitTriIndex[idx] == -1`, the ray escaped the scene without hitting
anything. Actions:
- Optionally add an environment map / sky colour contribution:
  `wf.radiance[idx] += envLight * wf.throughput[idx]`
- Accumulate the final path colour into `accumBuffer` (Welford update, §12).
- **Do not re-enqueue.** Return.

### Step b — Emitter hit

Check `scene.triangleEmitterFlags[triIdx]`. If non-zero, the surface is emissive.
You need to port `accumulateEmitterHit` from `optix_programs.cu`:

```cpp
// The emitted radiance of this surface
Float3 Le = scene.triangleEmission[triIdx];

// MIS: if the previous bounce was diffuse (specularBounce==0), we may have
// already counted this emitter via NEE. Apply the BRDF-side MIS weight.
// If specularBounce==1, the weight is 1 (no NEE was done before this bounce).
float brdfWeight = computeMisWeight(...);

wf.radiance[idx] = add3(wf.radiance[idx], mul3(mul3(Le, wf.throughput[idx]), brdfWeight));
```

After this, if the surface also has a material (not a pure light), continue to
BSDF sampling. If it is a pure emitter with no BSDF, terminate the path here
(same as a miss — call `accumulateWelford` and return).

### Step c — NEE (shadow ray writing)

This replaces `sampleAreaEmitterNEE` and `sampleSpotlightNEE` from
`optix_programs.cu`. Instead of calling `traceOccluded()` inline (you cannot —
you are in a CUDA kernel, not an OptiX raygen), you write the shadow ray into
the `shadowQueue` for the connect stage to process:

```cpp
// (inside NEE for one area emitter sample)
int sSlot = atomicAdd(wf.shadowCount, 1);   // reserve a slot
wf.shadowOrigin[sSlot]  = hitPoint + RT_EPSILON * hitNormal;
wf.shadowDir[sSlot]     = normalize(lightPoint - hitPoint);
wf.shadowTMax[sSlot]    = distance(lightPoint, hitPoint) - RT_EPSILON;
wf.shadowContrib[sSlot] = misWeightedContrib;   // radiance × MIS × throughput
wf.shadowPathIdx[sSlot] = (uint32_t)idx;        // which path this belongs to
```

The connect stage fires an OptiX shadow ray for each slot and adds
`shadowContrib` to `radiance[shadowPathIdx]` only if the ray is unoccluded.

For spotlights, loop over all `scene.spotlights` and write one shadow slot per
spotlight per path.

### Step d — BSDF sampling

Port the `bsdfSample` block from `traceRay()`. The key function is in
`include/rt_bsdf.cuh`:

```cpp
BsdfQueryRecord bsdfQ{};
bsdfQ.wi = toLocalFromNormal(hitNormal, neg(wf.rayDir[idx]));  // incident dir in local space
bsdfSample(bsdf, bsdfQ, rng);  // fills bsdfQ.wo (outgoing local dir), pdf, value, eta
Float3 newWorldDir = toWorldFromNormal(hitNormal, bsdfQ.wo);
```

`toLocalFromNormal` rotates the world-space direction into a coordinate system
where `hitNormal` is the +Z axis. BSDF evaluation always works in this local
frame. `toWorldFromNormal` is the inverse.

Update path state:
```cpp
wf.throughput[idx] = mul3(wf.throughput[idx], bsdfQ.value / bsdfQ.pdf);
wf.eta[idx]       *= bsdfQ.eta;        // track IOR accumulation for RR
wf.brdfPDF[idx]    = bsdfQ.pdf;        // save for next bounce's MIS weight
wf.specularBounce[idx] = bsdfIsDelta(bsdf) ? 1 : 0;  // was this a delta scatter?
```

### Step e — Russian roulette

After `bounceCount > 3` (matches the megakernel), try to terminate the path
early. The survival probability is proportional to the current throughput
magnitude (paths that have lost most of their energy can be terminated without
bias if we boost the survivors' throughput to compensate):

```cpp
if (wf.bounceCount[idx] > 3) {
    float rrProb = fminf(1.f, maxComponent(wf.throughput[idx]) * wf.eta[idx] * wf.eta[idx]);
    if (rngNextFloat01(rng) >= rrProb) {
        // Path terminates — accumulate its colour
        accumulateWelford(scene.accumBuffer, wf.pixelIndex[idx],
                          wf.radiance[idx], scene.frameIndex);
        wf.rngState[idx] = rng.state;
        return;  // do NOT re-enqueue
    }
    // Path survives — unbias by dividing throughput
    wf.throughput[idx] = div3(wf.throughput[idx], {rrProb, rrProb, rrProb});
}
```

### Step f — Re-enqueue for next extend

If the path is still alive:
```cpp
wf.rayOrigin[idx] = hitPoint + RT_EPSILON_WF * faceForward(hitNormal, neg(newWorldDir));
wf.rayDir[idx]    = newWorldDir;
wf.bounceCount[idx]++;
wf.rngState[idx]  = rng.state;

int slot = atomicAdd(wf.rayCount, 1);
wf.rayQueue[slot] = idx;
```

The `faceForward` nudge pushes the origin to the correct side of the surface so
the next extend does not immediately re-intersect the same triangle.

### WfSceneView

`wfShade` receives a `WfSceneView` struct (defined in `wavefront_buffers.h`)
instead of `params` because it is a regular CUDA kernel (not OptiX) and cannot
read `extern "C" __constant__ LaunchParams params`. The host function
`buildSceneView()` in `render_kernel.cu` packages all the static device pointers
into this struct and passes it by value. All fields are read-only pointers — the
kernel does not allocate or free anything.

---

## 8. Stage 4 — ray_connect.cu

**File:** `src/wavefront/ray_connect.cu`  
**Program:** `__raygen__wf_shadow` — an OptiX raygen, compiled to PTX  
**Launched with:** `optixLaunch(pipeline, ..., &s_sbt_wf_shadow, shadowCount, 1, 1)`

### What it does

Each thread handles one shadow ray from the shadow queue:

```cpp
uint32_t queueSlot = optixGetLaunchIndex().x;
int sIdx = params.wf.shadowQueue[queueSlot];

Float3   org     = params.wf.shadowOrigin[sIdx];
Float3   dir     = params.wf.shadowDir[sIdx];
float    tMax    = params.wf.shadowTMax[sIdx];
```

Trace as an *occlusion test* (any-hit / terminate-on-first-hit):

```cpp
unsigned int occluded = 1u;   // assume occluded before the trace
optixTrace(
    params.handle,
    toF3(org), toF3(dir),
    RT_EPSILON, tMax, 0.f,
    255u,
    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |   // stop at first geometry hit
    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT     |   // skip closest-hit shading
    OPTIX_RAY_FLAG_DISABLE_ANYHIT,            // skip any-hit (no alpha)
    RAY_TYPE_SHADOW, RAY_TYPE_COUNT, RAY_TYPE_SHADOW,
    occluded);
// After trace: occluded is still 1 if something was hit,
//              0 if __miss__shadow ran (clear line of sight)
```

The `__miss__shadow` program in `optix_programs.cu` calls `optixSetPayload_0(0u)`
to signal "not occluded". This is the same mechanism the megakernel uses in
`traceOccluded()` — you are just moving it into a separate raygen.

If unoccluded, accumulate the pre-computed contribution into the path's radiance:

```cpp
if (occluded == 0u) {
    uint32_t pathIdx = params.wf.shadowPathIdx[sIdx];
    Float3   contrib = params.wf.shadowContrib[sIdx];
    atomicAdd(&params.wf.radiance[pathIdx].x, contrib.x);
    atomicAdd(&params.wf.radiance[pathIdx].y, contrib.y);
    atomicAdd(&params.wf.radiance[pathIdx].z, contrib.z);
}
```

### Why atomicAdd for radiance?

A single path can have multiple shadow rays in the queue (one area emitter + one
per spotlight). All of them have the same `shadowPathIdx`. Multiple threads in
the connect kernel will therefore try to add to the same `radiance` element at
the same time. Without `atomicAdd`, two threads could read the same old value,
each add their contribution, and one write would overwrite the other — you would
lose a light contribution. `atomicAdd` serialises these writes so no update is
lost. It is slightly slower than a plain write but necessary for correctness.

---

## 9. Queues and atomic counters

### How the queue works

The ray queue `wf.rayQueue` is a flat array of integers. Each integer is a
*path slot index* — an index into the SoA arrays. A path is "active" if and
only if its slot index appears in the queue.

`wf.rayCount` is a single integer in device memory that acts as the
next-free-slot counter. To append a path index `idx`:

```cpp
int slot = atomicAdd(wf.rayCount, 1);  // atomically increment, returns old value
wf.rayQueue[slot] = idx;
```

`atomicAdd` is a GPU atomic operation: it reads `*wf.rayCount`, adds 1 to it,
stores the new value, and returns the *original* value — all as one indivisible
operation even if thousands of threads are doing this simultaneously. The
returned value is the unique slot number for this thread. No two threads get the
same slot, and no slot is ever overwritten.

After the generate kernel runs, `*wf.rayCount == N` (all pixels are queued).
After one full bounce, some paths will have died (missed, RR-terminated) and
not been re-enqueued, so `*wf.rayCount < N`.

### Resetting between stages

Before the shade kernel runs, the host resets:
```cpp
cudaMemcpy(wf.rayCount,    &zero, sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(wf.shadowCount, &zero, sizeof(int), cudaMemcpyHostToDevice);
```

This allows shade to *refill* these counters from scratch. The old queue values
in `wf.rayQueue[0..*prevCount-1]` are not erased — they just get overwritten as
shade fills in new values. This is fine because the extend kernel only reads
`[0..*rayCount-1]` based on the current count.

### Why not use CUDA streams or dynamic parallelism?

For a first implementation, host-side `cudaDeviceSynchronize()` + explicit
counter copies is the clearest and least error-prone approach. The overhead of a
sync per bounce is small compared to the GPU work itself. Advanced optimisations
(persistent kernels, cooperative groups, device-side loop) can come later.

---

## 10. The RNG

`RngState` in `include/rt_rng.cuh` is a xorshift32 generator — just a single
`uint32_t`. It is fast and has decent statistical properties for path tracing.

### Seeding

```cpp
RngState rng = rngInit(pixelIndex, frameIndex);
```

`rngInit` hashes `pixelIndex` and `frameIndex` together so every pixel starts
with a different state on every frame. Without this you would see the same noise
pattern every frame (the accumulation would not converge).

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

## 11. MIS state

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
*previous* shade step. When the extend step hits an emitter, the shade step
needs this to compute `weight = brdfPDF / (brdfPDF + lightPDF)`. If `brdfPDF`
was not persisted, you would have to recompute it, which is only possible if you
save the sampled direction — more complex.

**`specularBounce`**: whether the *previous* BSDF sample was from a delta
distribution (perfect mirror or refraction). Delta BSDFs have zero probability
of randomly hitting a specific light direction, so no NEE is performed for
them. When `specularBounce = 1`, the weight for the BRDF path is 1.0 — use the
full emitted radiance. When 0, apply the fractional MIS weight.

The rule: **these two values are written in shade (when sampling the BSDF) and
read in the *next* shade step (when the ray hits an emitter).** They must survive
in the SoA across the extend between them.

---

## 12. Welford accumulation

The renderer accumulates samples across frames. Rather than storing all samples
and averaging at the end (wastes memory), it maintains a running mean using
Welford's online algorithm:

```
new_mean = old_mean + (new_sample - old_mean) / frame_count
```

In code (from `__raygen__rg` in `optix_programs.cu`):
```cpp
int    n     = frameIndex + 1;                 // number of samples after this one
Float3 prev  = accumBuffer[pixelIndex];        // running mean so far
Float3 delta = sub3(pathColor, prev);          // difference from mean
accumBuffer[pixelIndex] = add3(prev, div3(delta, {(float)n, (float)n, (float)n}));
```

`accumBuffer` stores the *current mean* in linear colour space. The
`presentLinearKernel` in `render_kernel.cu` reads it and applies gamma-2.2
encode to produce the display image.

When a path terminates in `wfShade` (on miss, Russian roulette, or max bounces),
call this update once for that path's pixel. With one path per pixel per frame,
each pixel is updated exactly once per `cudaRenderWavefront` call — identical
to the megakernel.

---

## 13. OptiX concepts

### The pipeline

When we call `ensureOptixPipeline()` in `render_kernel.cu`, we:
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

## 14. PTX and the two compilation paths

### Two kinds of CUDA files in this project

| File | Compiled to | How it runs |
|------|------------|-------------|
| `ray_generate.cu`, `ray_shade.cu` | Regular CUDA object → linked into `.exe` | Called by host via `<<<>>>` syntax |
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
compiled as OptiX device programs.

### `extern "C" __constant__ LaunchParams params`

All three device files that read `params` declare it as `extern "C"` —
"external C linkage, defined elsewhere". Since they all compile into the *same
PTX module* (the OBJECT library concatenates their PTX), there is only one
definition. OptiX sees `pipelineLaunchParamsVariableName = "params"` and populates
this constant with the pointer you give to `optixLaunch` before each launch.

---

## 15. What happens in cudaRenderWavefront

Here is the full annotated version of the bounce loop in `render_kernel.cu`:

```cpp
static void cudaRenderWavefront(uint32_t* devPtr, int imageWidth, int imageHeight)
{
    // Lazy-allocate SoA buffers (only if resolution changed or first call).
    ensureWavefrontBuffers(imageWidth, imageHeight);

    WavefrontSoA& wf = s_wf.soa;

    // Reset both queue counters.  These live in device memory so the GPU
    // kernels can use atomicAdd on them.
    int zero = 0;
    cudaMemcpy(wf.rayCount,    &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(wf.shadowCount, &zero, sizeof(int), cudaMemcpyHostToDevice);

    // Stage 1: fill the ray queue with all N paths.
    launchWfGenerate(wf, s_camera_h, imageWidth, imageHeight, s_frameIndex);
    cudaDeviceSynchronize();   // ← CPU must wait until generate is done before
                               //   reading rayCount (which it fills with N)

    for (int bounce = 0; bounce < MAX_WF_BOUNCES; ++bounce) {

        // How many paths still need a ray cast?
        int activeCount = 0;
        cudaMemcpy(&activeCount, wf.rayCount, sizeof(int), cudaMemcpyDeviceToHost);
        if (activeCount <= 0) break;   // all paths terminated

        // Stage 2: trace rays via OptiX / RT cores.
        // Fill LaunchParams.wf with the current SoA pointers so the raygen
        // can access them through `params`.
        LaunchParams lp = {};
        lp.handle    = s_gasHandle;     // the BVH accelerated structure
        lp.triangles = s_triangles_d;
        lp.wf        = wf;             // copy the SoA pointer struct
        cudaMemcpy(s_launchParams_d, &lp, sizeof(LaunchParams), cudaMemcpyHostToDevice);

        optixLaunch(s_optixPipeline, 0, ..., &s_sbt_wf_extend,
                    activeCount, 1, 1);  // 1-D launch: one raygen thread per active ray
        cudaDeviceSynchronize();

        // Reset counters — shade will refill them.
        cudaMemcpy(wf.rayCount,    &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(wf.shadowCount, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // Stage 3: shade, NEE, BSDF sampling.
        WfSceneView sv = buildSceneView(s_frameIndex);
        launchWfShade(wf, sv, activeCount);
        cudaDeviceSynchronize();

        // Stage 4: fire shadow rays (if any were written by shade).
        int shadowCount = 0;
        cudaMemcpy(&shadowCount, wf.shadowCount, sizeof(int), cudaMemcpyDeviceToHost);
        if (shadowCount > 0) {
            // Update params.wf and re-upload.
            lp.wf = wf;
            cudaMemcpy(s_launchParams_d, &lp, sizeof(LaunchParams), ...);
            optixLaunch(..., &s_sbt_wf_shadow, shadowCount, 1, 1);
            cudaDeviceSynchronize();
        }
        // loop back to bounce+1
    }
    // After the loop, dead paths have written their final colour into
    // s_accumBuffer_d via accumulateWelford inside wfShade.
}
```

---

## 16. Implementation order and testing

Work in this order to always have something runnable after each step:

### Step A — Generate + a debug view

Implement `wfGenerate` fully. After the generate launch, the ray queue should
have `width * height` entries and all path state should be initialised. To
verify: copy `wf.rayOrigin` back to the host and check that the rays point
roughly in the expected directions. Or write a small debug kernel that reads
`wf.rayDir[idx]` and maps it to a colour in the PBO — you should see a
smooth gradient matching the camera frustum.

### Step B — Extend

Implement `__raygen__wf_extend`. After one generate + one extend, every path
should have a valid hit (or miss). Verify by reading `wf.hitTriIndex` back:
most should be ≥ 0 (hit geometry), border pixels should be -1 (missed). You
can visualise hit normals: copy `wf.hitNormal` to the PBO as RGB — you should
see a shaded version of your scene.

### Step C — Shade (misses only, no BSDF)

Implement just the miss path in `wfShade`. When `hitTriIndex == -1`, write
a background colour (e.g. dark blue) to `accumBuffer`. Do NOT re-enqueue
anything. Run generate → extend → shade and you should see the scene outline
against the background.

### Step D — Shade (BSDF + re-enqueue, no NEE)

Add the BSDF sampling and re-enqueue logic (steps d–f from §7). Shadow queue
can stay empty. Run the full bounce loop. You should see ambient occlusion /
unlit diffuse — paths bouncing multiple times before dying to RR or max bounces,
no direct lighting contribution. This validates that the path's throughput and
re-enqueue logic are correct.

### Step E — Shadow queue + connect

Add NEE to shade (step c from §7), implement `__raygen__wf_shadow`, and add
the connect launch to the host loop. Now direct lighting should appear. Compare
visually with the megakernel (set `USE_WAVEFRONT = false` to switch back).

### Step F — Emitter hit MIS

Implement `accumulateEmitterHit` (step b from §7). Without this, bright spots
from paths that hit emitters via BRDF sampling are missing. After this step the
image should match the megakernel output closely.

### Switching back to verify

Because `USE_WAVEFRONT` in `main.cpp` is a simple boolean, you can flip it,
recompile (no CMake reconfigure needed, just the exe build), and run both
renderers side by side in separate windows to compare. Use the `C` key
(existing screenshot hotkey) to capture both at the same frame count.

---

## 17. Quick reference — functions to port

| Function in `optix_programs.cu` / `render_kernel.cu` | Port into |
|------|---------|
| `generatePrimaryRay()` | `wfGenerate` |
| `rngInit()` | `wfGenerate` |
| `accumulateEmitterHit()` | `wfShade` step b |
| `sampleAreaEmitterNEE()` | `wfShade` step c (shadow write variant) |
| `sampleSpotlightNEE()` | `wfShade` step c (shadow write variant) |
| `bsdfSample()` | `wfShade` step d |
| Russian roulette block | `wfShade` step e |
| `traceClosest()` | `__raygen__wf_extend` |
| `traceOccluded()` | `__raygen__wf_shadow` |
| Welford update in `__raygen__rg` | `wfShade` on path termination |

All the BSDF, emitter sampling, and math helpers are unchanged — they live in
`include/rt_bsdf.cuh`, `include/rt_emitter_sampling.cuh`,
`include/rt_cuda_math.cuh`, and `include/rt_rng.cuh`. You are only rearranging
*when and where* they are called, not rewriting them.
