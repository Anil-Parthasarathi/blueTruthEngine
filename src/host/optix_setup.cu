// ============================================================================
//  optix_setup.cu — OptiX context, module, program groups, pipeline, SBTs,
//  per-object GAS builds, and the top-level IAS (with update/refit).
// ============================================================================

#include "renderer_state.h"
#include "cuda_check.h"

#include <cuda.h>          // CUdeviceptr / CUcontext typedefs used by the OptiX API
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>   // MUST appear in exactly one TU
#include <optix_stack_size.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Embedded PTX arrays (each OptiX .cu is its own module — see CMake).
extern "C" const char* getOptixPtx();
extern "C" size_t      getOptixPtxSize();
extern "C" const char* getWfExtendPtx();
extern "C" size_t      getWfExtendPtxSize();
extern "C" const char* getWfShadowPtx();
extern "C" size_t      getWfShadowPtxSize();

// Rebuild the IAS from scratch every N refits so AABB quality does not drift.
static constexpr int kIasRebuildInterval = 64;

// ---------------------------------------------------------------------------
//  OptiX setup helpers
// ---------------------------------------------------------------------------
static void optixLogCallback(unsigned int level, const char* tag, const char* message, void*)
{
    fprintf(stderr, "[optix][%u][%s] %s\n", level, tag ? tag : "", message ? message : "");
}

// Lazily creates the OptiX context, module, program groups, pipeline and SBT.
// Scene-independent: called once before the first GAS/IAS build.
void ensureOptixPipeline()
{
    if (s_optixReady) return;

    // Make sure a CUDA context exists for OptiX to attach to.
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions ctxOptions = {};
    ctxOptions.logCallbackFunction = &optixLogCallback;
    ctxOptions.logCallbackLevel    = 4;
    OPTIX_CHECK(optixDeviceContextCreate(0 /*current CUDA context*/, &ctxOptions, &s_optixContext));

    // ── Module ──────────────────────────────────────────────────────
    OptixModuleCompileOptions moduleOptions = {};
    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOptions.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOptions.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur                   = 0;
    pipelineCompileOptions.traversableGraphFlags            =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOptions.numPayloadValues                 = 2;
    pipelineCompileOptions.numAttributeValues               = 2;
    pipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    char   log[8192];
    size_t logSize = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(s_optixContext, &moduleOptions, &pipelineCompileOptions,
                                  getOptixPtx(), getOptixPtxSize(),
                                  log, &logSize, &s_optixModule));

    // Wavefront modules (separate PTX to avoid duplicate-symbol issues).
    logSize = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(s_optixContext, &moduleOptions, &pipelineCompileOptions,
                                  getWfExtendPtx(), getWfExtendPtxSize(),
                                  log, &logSize, &s_optixModuleWfExtend));
    logSize = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(s_optixContext, &moduleOptions, &pipelineCompileOptions,
                                  getWfShadowPtx(), getWfShadowPtxSize(),
                                  log, &logSize, &s_optixModuleWfShadow));

    // ── Program groups ──────────────────────────────────────────────
    OptixProgramGroupOptions pgOptions = {};

    OptixProgramGroupDesc rgDesc = {};
    rgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgDesc.raygen.module            = s_optixModule;
    rgDesc.raygen.entryFunctionName = "__raygen__rg";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(s_optixContext, &rgDesc, 1, &pgOptions, log, &logSize, &s_pgRaygen));

    OptixProgramGroupDesc msDesc = {};
    msDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msDesc.miss.module            = s_optixModule;
    msDesc.miss.entryFunctionName = "__miss__radiance";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(s_optixContext, &msDesc, 1, &pgOptions, log, &logSize, &s_pgMissRadiance));

    OptixProgramGroupDesc msShadowDesc = {};
    msShadowDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msShadowDesc.miss.module            = s_optixModule;
    msShadowDesc.miss.entryFunctionName = "__miss__shadow";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(s_optixContext, &msShadowDesc, 1, &pgOptions, log, &logSize, &s_pgMissShadow));

    OptixProgramGroupDesc hgDesc = {};
    hgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hgDesc.hitgroup.moduleCH            = s_optixModule;
    hgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(s_optixContext, &hgDesc, 1, &pgOptions, log, &logSize, &s_pgHitRadiance));

    // ── Wavefront raygen program groups ─────────────────────────────
    OptixProgramGroupDesc wfExtendDesc = {};
    wfExtendDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    wfExtendDesc.raygen.module            = s_optixModuleWfExtend;
    wfExtendDesc.raygen.entryFunctionName = "__raygen__wf_extend";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(s_optixContext, &wfExtendDesc, 1, &pgOptions, log, &logSize, &s_pgWfExtend));

    OptixProgramGroupDesc wfShadowDesc = {};
    wfShadowDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    wfShadowDesc.raygen.module            = s_optixModuleWfShadow;
    wfShadowDesc.raygen.entryFunctionName = "__raygen__wf_shadow";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(s_optixContext, &wfShadowDesc, 1, &pgOptions, log, &logSize, &s_pgWfShadow));

    // ── Pipeline ────────────────────────────────────────────────────
    OptixProgramGroup groups[] = { s_pgRaygen, s_pgWfExtend, s_pgWfShadow,
                                    s_pgMissRadiance, s_pgMissShadow, s_pgHitRadiance };

    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 2;
    logSize = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(s_optixContext, &pipelineCompileOptions, &linkOptions,
                                    groups, static_cast<unsigned int>(sizeof(groups) / sizeof(groups[0])),
                                    log, &logSize, &s_optixPipeline));

    // ── Stack sizes ─────────────────────────────────────────────────
    OptixStackSizes stackSizes = {};
    for (OptixProgramGroup pg : groups)
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stackSizes, s_optixPipeline));

    unsigned int dcFromTraversal = 0, dcFromState = 0, contStack = 0;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes,
                                           2 /*maxTraceDepth*/, 0 /*maxCCDepth*/, 0 /*maxDCDepth*/,
                                           &dcFromTraversal, &dcFromState, &contStack));
    // maxTraversableGraphDepth = 2 for IAS → GAS
    OPTIX_CHECK(optixPipelineSetStackSize(s_optixPipeline,
                                          dcFromTraversal, dcFromState, contStack,
                                          2 /*maxTraversableGraphDepth*/));

    // ── Shader binding table ────────────────────────────────────────
    RayGenSbtRecord rgRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(s_pgRaygen, &rgRecord));
    CUdeviceptr d_raygen = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen), sizeof(RayGenSbtRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen), &rgRecord,
                          sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

    MissSbtRecord missRecords[RAY_TYPE_COUNT];
    OPTIX_CHECK(optixSbtRecordPackHeader(s_pgMissRadiance, &missRecords[RAY_TYPE_RADIANCE]));
    OPTIX_CHECK(optixSbtRecordPackHeader(s_pgMissShadow,   &missRecords[RAY_TYPE_SHADOW]));
    CUdeviceptr d_miss = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss), sizeof(missRecords)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss), missRecords,
                          sizeof(missRecords), cudaMemcpyHostToDevice));

    HitGroupSbtRecord hitRecords[RAY_TYPE_COUNT];
    OPTIX_CHECK(optixSbtRecordPackHeader(s_pgHitRadiance, &hitRecords[RAY_TYPE_RADIANCE]));
    // Shadow rays disable CH/AH, but still index a hitgroup record — reuse the
    // radiance hitgroup header so the SBT index is always valid.
    OPTIX_CHECK(optixSbtRecordPackHeader(s_pgHitRadiance, &hitRecords[RAY_TYPE_SHADOW]));
    CUdeviceptr d_hit = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit), sizeof(hitRecords)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hit), hitRecords,
                          sizeof(hitRecords), cudaMemcpyHostToDevice));

    s_sbt.raygenRecord                = d_raygen;
    s_sbt.missRecordBase              = d_miss;
    s_sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
    s_sbt.missRecordCount             = RAY_TYPE_COUNT;
    s_sbt.hitgroupRecordBase          = d_hit;
    s_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    s_sbt.hitgroupRecordCount         = RAY_TYPE_COUNT;

    // ── Wavefront SBTs (share miss/hit records; only the raygen differs) ─
    RayGenSbtRecord wfExtendRgRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(s_pgWfExtend, &wfExtendRgRecord));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_d_raygen_wf_extend), sizeof(RayGenSbtRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(s_d_raygen_wf_extend), &wfExtendRgRecord,
                          sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

    RayGenSbtRecord wfShadowRgRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(s_pgWfShadow, &wfShadowRgRecord));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_d_raygen_wf_shadow), sizeof(RayGenSbtRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(s_d_raygen_wf_shadow), &wfShadowRgRecord,
                          sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

    s_sbt_wf_extend = s_sbt;
    s_sbt_wf_extend.raygenRecord = s_d_raygen_wf_extend;

    s_sbt_wf_shadow = s_sbt;
    s_sbt_wf_shadow.raygenRecord = s_d_raygen_wf_shadow;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_launchParams_d), sizeof(LaunchParams)));

    s_optixReady = true;
    fprintf(stdout, "[optix] Pipeline ready (IAS + per-object GAS)\n");
}

// ---------------------------------------------------------------------------
//  Build one GAS over a contiguous triangle range in s_gasVertices_d.
// ---------------------------------------------------------------------------
static void buildOneGAS(int triOffset, int triCount,
                        OptixTraversableHandle& outHandle, CUdeviceptr& outBuffer)
{
    outHandle = 0;
    outBuffer = 0;
    if (triCount <= 0) return;

    const unsigned int numVertices = static_cast<unsigned int>(triCount) * 3u;
    CUdeviceptr d_vertices = reinterpret_cast<CUdeviceptr>(s_gasVertices_d + triOffset * 3);

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(Float3);
    buildInput.triangleArray.numVertices         = numVertices;
    buildInput.triangleArray.vertexBuffers       = &d_vertices;

    const unsigned int triangleFlags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    buildInput.triangleArray.flags        = triangleFlags;
    buildInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufferSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(s_optixContext, &accelOptions, &buildInput, 1, &bufferSizes));

    CUdeviceptr d_temp = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp), bufferSizes.tempSizeInBytes));

    CUdeviceptr d_outputUncompacted = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_outputUncompacted), bufferSizes.outputSizeInBytes));

    CUdeviceptr d_compactedSize = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compactedSize), sizeof(uint64_t)));
    OptixAccelEmitDesc emitDesc = {};
    emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = d_compactedSize;

    OptixTraversableHandle uncompactedHandle = 0;
    OPTIX_CHECK(optixAccelBuild(s_optixContext, 0 /*stream*/, &accelOptions, &buildInput, 1,
                                d_temp, bufferSizes.tempSizeInBytes,
                                d_outputUncompacted, bufferSizes.outputSizeInBytes,
                                &uncompactedHandle, &emitDesc, 1));
    CUDA_CHECK(cudaDeviceSynchronize());

    uint64_t compactedSize = 0;
    CUDA_CHECK(cudaMemcpy(&compactedSize, reinterpret_cast<void*>(d_compactedSize),
                          sizeof(uint64_t), cudaMemcpyDeviceToHost));

    if (compactedSize > 0 && compactedSize < bufferSizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&outBuffer), compactedSize));
        OPTIX_CHECK(optixAccelCompact(s_optixContext, 0, uncompactedHandle,
                                      outBuffer, compactedSize, &outHandle));
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(reinterpret_cast<void*>(d_outputUncompacted));
    } else {
        outBuffer = d_outputUncompacted;
        outHandle = uncompactedHandle;
    }

    cudaFree(reinterpret_cast<void*>(d_temp));
    cudaFree(reinterpret_cast<void*>(d_compactedSize));
}

// ---------------------------------------------------------------------------
//  Build / update the top-level IAS from the current object transforms.
// ---------------------------------------------------------------------------
void buildIAS(bool update)
{
    if (s_objectCount <= 0 || s_gasHandles.empty()) return;

    // Force a full rebuild periodically — repeated UPDATEs leave the AABBs loose.
    if (update && s_iasFramesSinceRebuild >= kIasRebuildInterval)
        update = false;

    std::vector<OptixInstance> instances(static_cast<size_t>(s_objectCount));
    for (int i = 0; i < s_objectCount; ++i) {
        OptixInstance& inst = instances[static_cast<size_t>(i)];
        inst = {};
        // ObjectTransform is laid out identically to OptixInstance::transform.
        memcpy(inst.transform, s_objectTransforms_h[static_cast<size_t>(i)].m, sizeof(inst.transform));
        inst.instanceId        = static_cast<unsigned int>(i);
        inst.sbtOffset         = 0;
        inst.visibilityMask    = 255;
        inst.flags             = OPTIX_INSTANCE_FLAG_NONE;
        inst.traversableHandle = s_gasHandles[static_cast<size_t>(i)];
    }

    const size_t instancesBytes = sizeof(OptixInstance) * static_cast<size_t>(s_objectCount);
    if (!s_instances_d) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_instances_d), instancesBytes));
    }
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(s_instances_d), instances.data(),
                          instancesBytes, cudaMemcpyHostToDevice));

    OptixBuildInput buildInput = {};
    buildInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    buildInput.instanceArray.instances    = s_instances_d;
    buildInput.instanceArray.numInstances = static_cast<unsigned int>(s_objectCount);

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.operation  = update ? OPTIX_BUILD_OPERATION_UPDATE
                                     : OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufferSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(s_optixContext, &accelOptions, &buildInput, 1, &bufferSizes));

    if (!update) {
        // Full rebuild — (re)allocate output / temp for the new size.
        if (s_iasOutputBuffer) { cudaFree(reinterpret_cast<void*>(s_iasOutputBuffer)); s_iasOutputBuffer = 0; }
        if (s_iasTempBuffer)   { cudaFree(reinterpret_cast<void*>(s_iasTempBuffer));   s_iasTempBuffer = 0; }

        s_iasOutputSize = bufferSizes.outputSizeInBytes;
        s_iasTempSize   = bufferSizes.tempSizeInBytes;
        // UPDATE needs tempUpdateSizeInBytes; keep the larger of the two.
        if (bufferSizes.tempUpdateSizeInBytes > s_iasTempSize)
            s_iasTempSize = bufferSizes.tempUpdateSizeInBytes;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_iasOutputBuffer), s_iasOutputSize));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_iasTempBuffer),   s_iasTempSize));

        OPTIX_CHECK(optixAccelBuild(s_optixContext, 0 /*stream*/, &accelOptions, &buildInput, 1,
                                    s_iasTempBuffer, s_iasTempSize,
                                    s_iasOutputBuffer, s_iasOutputSize,
                                    &s_iasHandle, nullptr, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        s_iasFramesSinceRebuild = 0;
        fprintf(stdout, "[optix] IAS built (%d instances, %.2f KB)\n",
                s_objectCount, static_cast<double>(s_iasOutputSize) / 1024.0);
    } else {
        // Refit — reuse existing buffers. tempUpdateSizeInBytes can exceed the
        // original tempSizeInBytes, so grow if needed.
        if (bufferSizes.tempUpdateSizeInBytes > s_iasTempSize) {
            cudaFree(reinterpret_cast<void*>(s_iasTempBuffer));
            s_iasTempSize = bufferSizes.tempUpdateSizeInBytes;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_iasTempBuffer), s_iasTempSize));
        }

        OPTIX_CHECK(optixAccelBuild(s_optixContext, 0 /*stream*/, &accelOptions, &buildInput, 1,
                                    s_iasTempBuffer, s_iasTempSize,
                                    s_iasOutputBuffer, s_iasOutputSize,
                                    &s_iasHandle, nullptr, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        ++s_iasFramesSinceRebuild;
    }
}

// ---------------------------------------------------------------------------
//  Build per-object GAS from s_triangles_d, then the top-level IAS.
//  Requires cudaInitObjects() (or synthesises a single identity object).
// ---------------------------------------------------------------------------
void buildGAS()
{
    if (s_triangleCount <= 0 || !s_triangles_d) return;

    // If the host never called cudaInitObjects, synthesise one identity object
    // covering the entire triangle array so a static scene still works.
    if (s_objectCount <= 0) {
        ObjectDesc obj{};
        obj.triOffset = 0;
        obj.triCount  = s_triangleCount;
        obj.transform = objectTransformIdentity();
        cudaInitObjects(&obj, 1);
    }

    // Pack a contiguous vertex buffer (v0,v1,v2 per triangle) from the
    // interleaved TriangleData (v0,v1,v2 sit at offset 0 of each struct).
    const size_t vertexCount = static_cast<size_t>(s_triangleCount) * 3;
    if (s_gasVertices_d) { cudaFree(s_gasVertices_d); s_gasVertices_d = nullptr; }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s_gasVertices_d), vertexCount * sizeof(Float3)));
    CUDA_CHECK(cudaMemcpy2D(s_gasVertices_d, 3 * sizeof(Float3),
                            s_triangles_d, sizeof(TriangleData),
                            3 * sizeof(Float3), static_cast<size_t>(s_triangleCount),
                            cudaMemcpyDeviceToDevice));

    // Tear down any previous per-object GAS.
    for (CUdeviceptr buf : s_gasBuffers)
        if (buf) cudaFree(reinterpret_cast<void*>(buf));
    s_gasBuffers.clear();
    s_gasHandles.clear();
    s_gasHandles.resize(static_cast<size_t>(s_objectCount), 0);
    s_gasBuffers.resize(static_cast<size_t>(s_objectCount), 0);

    for (int i = 0; i < s_objectCount; ++i) {
        const ObjectDesc& obj = s_objects_h[static_cast<size_t>(i)];
        buildOneGAS(obj.triOffset, obj.triCount,
                    s_gasHandles[static_cast<size_t>(i)],
                    s_gasBuffers[static_cast<size_t>(i)]);
    }

    fprintf(stdout, "[optix] Built %d per-object GAS from %d triangles\n",
            s_objectCount, s_triangleCount);

    // Fresh IAS (full build).
    if (s_iasOutputBuffer) { cudaFree(reinterpret_cast<void*>(s_iasOutputBuffer)); s_iasOutputBuffer = 0; }
    if (s_iasTempBuffer)   { cudaFree(reinterpret_cast<void*>(s_iasTempBuffer));   s_iasTempBuffer = 0; }
    s_iasHandle = 0;
    s_iasFramesSinceRebuild = kIasRebuildInterval; // force rebuild path
    buildIAS(false);
}

// ---------------------------------------------------------------------------
//  Public API: initialise the OptiX pipeline (once) and build GAS + IAS.
// ---------------------------------------------------------------------------
void cudaInitOptix()
{
    ensureOptixPipeline();
    buildGAS();
}

// ---------------------------------------------------------------------------
//  Teardown (called from cudaCleanup)
// ---------------------------------------------------------------------------
void freeOptixState()
{
    if (s_sbt.raygenRecord)      { cudaFree(reinterpret_cast<void*>(s_sbt.raygenRecord));      s_sbt.raygenRecord = 0; }
    if (s_sbt.missRecordBase)    { cudaFree(reinterpret_cast<void*>(s_sbt.missRecordBase));    s_sbt.missRecordBase = 0; }
    if (s_sbt.hitgroupRecordBase){ cudaFree(reinterpret_cast<void*>(s_sbt.hitgroupRecordBase)); s_sbt.hitgroupRecordBase = 0; }
    if (s_launchParams_d)        { cudaFree(s_launchParams_d); s_launchParams_d = nullptr; }

    if (s_iasOutputBuffer) { cudaFree(reinterpret_cast<void*>(s_iasOutputBuffer)); s_iasOutputBuffer = 0; }
    if (s_iasTempBuffer)   { cudaFree(reinterpret_cast<void*>(s_iasTempBuffer));   s_iasTempBuffer = 0; }
    if (s_instances_d)     { cudaFree(reinterpret_cast<void*>(s_instances_d));     s_instances_d = 0; }
    s_iasHandle = 0;
    s_iasTempSize = 0;
    s_iasOutputSize = 0;
    s_iasFramesSinceRebuild = 0;

    for (CUdeviceptr buf : s_gasBuffers)
        if (buf) cudaFree(reinterpret_cast<void*>(buf));
    s_gasBuffers.clear();
    s_gasHandles.clear();
    if (s_gasVertices_d) { cudaFree(s_gasVertices_d); s_gasVertices_d = nullptr; }

    // ── Wavefront SBT records and program groups ────────────────────
    if (s_d_raygen_wf_extend) { cudaFree(reinterpret_cast<void*>(s_d_raygen_wf_extend)); s_d_raygen_wf_extend = 0; }
    if (s_d_raygen_wf_shadow) { cudaFree(reinterpret_cast<void*>(s_d_raygen_wf_shadow)); s_d_raygen_wf_shadow = 0; }
    if (s_pgWfExtend)  { optixProgramGroupDestroy(s_pgWfExtend); s_pgWfExtend = nullptr; }
    if (s_pgWfShadow)  { optixProgramGroupDestroy(s_pgWfShadow); s_pgWfShadow = nullptr; }

    if (s_optixPipeline)  { optixPipelineDestroy(s_optixPipeline);   s_optixPipeline = nullptr; }
    if (s_pgRaygen)       { optixProgramGroupDestroy(s_pgRaygen);    s_pgRaygen = nullptr; }
    if (s_pgMissRadiance) { optixProgramGroupDestroy(s_pgMissRadiance); s_pgMissRadiance = nullptr; }
    if (s_pgMissShadow)   { optixProgramGroupDestroy(s_pgMissShadow);   s_pgMissShadow = nullptr; }
    if (s_pgHitRadiance)  { optixProgramGroupDestroy(s_pgHitRadiance);  s_pgHitRadiance = nullptr; }
    if (s_optixModuleWfExtend) { optixModuleDestroy(s_optixModuleWfExtend); s_optixModuleWfExtend = nullptr; }
    if (s_optixModuleWfShadow) { optixModuleDestroy(s_optixModuleWfShadow); s_optixModuleWfShadow = nullptr; }
    if (s_optixModule)    { optixModuleDestroy(s_optixModule);       s_optixModule = nullptr; }
    if (s_optixContext)   { optixDeviceContextDestroy(s_optixContext); s_optixContext = nullptr; }
    s_optixReady = false;
}
