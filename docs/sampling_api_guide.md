# Sampling API guide (RNG, emitters, BSDF)

This is a quick reference for the CUDA-side sampling helpers in `include/rt_rng.cuh`, `include/rt_emitter_sampling.cuh`, and `include/rt_bsdf.cuh`.

All functions below are `__device__` and intended to be called from inside your render/integrator kernels.

## RNG (`include/rt_rng.cuh`)

### Types
- **`RngState`**: 32-bit RNG state (xorshift32).

### Functions
- **`uint32_t hashUint(uint32_t x)`**
  - Hash/mix helper for creating seeds (good diffusion).
- **`RngState makeRng(uint32_t seed)`**
  - Creates a RNG state. If `seed==0`, it will use `1`.
- **`uint32_t rngNextU32(RngState& r)`**
  - Advances state and returns a random 32-bit integer.
- **`float rngNextFloat01(RngState& r)`**
  - Returns a uniform float in **\([0,1)\)** (24-bit mantissa).
- **`void rngNextFloat2(RngState& r, float& u0, float& u1)`**
  - Convenience for two uniforms.

### Typical usage (per pixel / per path)

```cpp
const uint32_t pixelIndex = (uint32_t)y * (uint32_t)width + (uint32_t)x;
RngState rng = makeRng(hashUint(pixelIndex ^ (frameIndex * 0x9e3779b9u)));

float u0 = rngNextFloat01(rng);
float u1 = rngNextFloat01(rng);
```

## Emitters (`include/rt_emitter_sampling.cuh`)

These helpers sample mesh area lights defined by:
- **Per-emitter triangle lists** (`emitterTriIndices`)
- **Per-emitter area CDFs** (`emitterTriCdf`, length `triCount+1`)
- **Scene-level emitter selection CDF** (`sceneEmitterCdf`, length `emitterCount+1`)

### Types
- **`MeshSample`**
  - `position`: sampled point on the emitter (world space)
  - `normal`: triangle geometric normal (world space)
  - `pdf`: **area PDF** (currently `1 / totalEmitterArea`)
  - `triangleIndex`: index into the flattened scene triangle buffer

- **`RandomEmitterSample`**
  - `emitterIndex`: chosen emitter table index
  - `emitterPdf`: **discrete** probability of selecting that emitter from `sceneEmitterCdf`
  - `mesh`: the `MeshSample` on that emitter

### Functions
- **`int sampleCdfReuse(const float* cdf, int count, float& u, float& outPdf)`**
  - Discrete sample from a CDF (stored with `cdf[0]=0`, `cdf[count]=sum`).
  - **Mutates** `u` into the “residual” uniform in the selected bin.
  - Returns selected index in `[0, count-1]`.
  - Outputs **discrete pdf** of selecting that bin (`outPdf`).

- **`MeshSample sampleMeshEmitter(...)`**
  - Samples a triangle proportional to its area using the per-emitter triangle CDF.
  - Then samples barycentrics using the standard “sqrt” method (Nori-style).
  - Returns `MeshSample` with **area pdf**.

- **`RandomEmitterSample sampleRandomEmitter(...)`**
  - Picks an emitter from `sceneEmitterCdf` (power-weighted).
  - Samples a point on that emitter using `sampleMeshEmitter`.

### Typical usage (direct lighting: one light sample)

```cpp
float uE0 = rngNextFloat01(rng);
float uE1 = rngNextFloat01(rng);

RandomEmitterSample ls = sampleRandomEmitter(
    triangles,
    emitters, emitterCount,
    emitterTriIndices,
    emitterTriCdf,
    sceneEmitterCdf,
    uE0, uE1);

if (ls.emitterIndex >= 0 && ls.emitterPdf > 0.0f && ls.mesh.pdf > 0.0f) {
    // Light sample
    Float3 xL = ls.mesh.position;
    Float3 nL = ls.mesh.normal;

    // Radiance for the selected emitter
    Float3 Le = emitters[ls.emitterIndex].radiance;

    // PDFs:
    // - ls.emitterPdf  : P(select emitter)
    // - ls.mesh.pdf    : p_A(xL) on that emitter (area density)
}
```

### PDF notes (area vs solid angle)

The emitter sampler returns an **area** density `p_A(x)` (stored in `MeshSample::pdf`).
If you want a **solid-angle** density \(p_\omega\) at shading point `x`:

\[
p_\omega(\omega) = p_A(x_L)\ \frac{d^2}{|\langle n_L, -\omega\rangle|}
\]

where \(d^2 = \|x_L - x\|^2\), \(\omega = \frac{x_L-x}{\|x_L-x\|}\), and \(n_L\) is the light normal.

Then the total emitter-sampling pdf becomes:

\[
p(\omega) = P(\text{select emitter}) \cdot p_\omega(\omega)
\]

## BSDF (`include/rt_bsdf.cuh`)

This is minimal “Nori-like” plumbing. Currently, **diffuse** is implemented; **dielectric** is stubbed (returns 0).

### Coordinate convention (important)
All directions passed to BSDF functions are assumed to be in a **local shading frame** where:
- **+Z is the surface normal**
- `cosTheta(w) == w.z`

Use `makeFrameFromNormal(normal)` plus `toLocal(frame, vWorld)` / `toWorld(frame, vLocal)`, or the convenience wrappers `toLocalFromNormal(normal, vWorld)` / `toWorldFromNormal(normal, vLocal)`, from `include/rt_cuda_math.cuh`.

### Types
- **`BsdfQueryRecord`**
  - `wi`: incident direction (**local**)
  - `wo`: outgoing direction (**local**, set by sampling)
  - `measure`: currently only `BSDF_ESolidAngle`
  - `eta`: relative IOR (used by specular transmission/refraction; set by `sample()`)

### Functions (dispatch)
- **`Float3 bsdfEval(const BsdfData& b, const BsdfQueryRecord& rec)`**
  - Returns \(f_r(\omega_i,\omega_o)\).
  - Diffuse returns `albedo / pi` if both directions are above the surface.
- **`float bsdfPdf(const BsdfData& b, const BsdfQueryRecord& rec)`**
  - Returns `pdf(wo)` for the measure in `rec` (diffuse: `cos(theta)/pi`).

### Functions (diffuse)
- **`Float3 bsdfEvalDiffuse(...)`**, **`float bsdfPdfDiffuse(...)`**
- **`Float3 bsdfSampleDiffuse(...)`**
  - Cosine-hemisphere sampling.
  - Returns the sample “weight” in a Nori-like way (currently returns `albedo`).
  - Optionally outputs the pdf via `outPdf`.

### Typical usage (evaluate diffuse for direct lighting)

```cpp
const BsdfData b = bsdfs[bsdfId];

BsdfQueryRecord bRec{};
bRec.measure = BSDF_ESolidAngle;
bRec.wi = wi_local; // incident (local)
bRec.wo = wo_local; // outgoing (local)

Float3 fr = bsdfEval(b, bRec);
```

### Typical usage (sample next direction)

```cpp
BsdfQueryRecord bRec{};
bRec.wi = wi_local;

float u0 = rngNextFloat01(rng);
float u1 = rngNextFloat01(rng);

float pdf = 0.0f;
Float3 weight = bsdfSampleDiffuse(bsdf, bRec, u0, u1, &pdf);
// bRec.wo is now set (local). Convert to world with your frame.
```

## Quick “wiring checklist”

- **RNG**: make one `RngState` per pixel/path and pass it around by reference.
- **Emitters**:
  - For direct lighting, call `sampleRandomEmitter(...)` using two uniforms.
  - Use `ls.emitterPdf` (discrete) and `ls.mesh.pdf` (area) correctly (convert if needed).
- **BSDF**:
  - Use `triangleBsdfIds[triIndex]` to look up `BsdfData`.
  - Provide **local-frame** directions using the helpers above.

