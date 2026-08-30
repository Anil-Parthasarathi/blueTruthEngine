cmake_minimum_required(VERSION 3.24)
project(RedTruthEngine LANGUAGES CXX CUDA)

# ── C++ / CUDA standards ────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target RTX 4080 SUPER  (SM 8.9)
set(CMAKE_CUDA_ARCHITECTURES 89)

# ── OptiX SDK (RT-core acceleration) ────────────────────────────────
# Header-only API: no library to link, the implementation is loaded from the
# display driver at runtime via optixInit().
set(OPTIX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0"
    CACHE PATH "Path to the NVIDIA OptiX SDK install directory")
set(OPTIX_INCLUDE_DIR "${OPTIX_INSTALL_DIR}/include")
if(NOT EXISTS "${OPTIX_INCLUDE_DIR}/optix.h")
    message(FATAL_ERROR "OptiX SDK not found. Set OPTIX_INSTALL_DIR (looked in ${OPTIX_INSTALL_DIR}).")
endif()

# Locate bin2c (ships next to nvcc) for embedding the compiled OptiX PTX.
get_filename_component(CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
find_program(BIN2C bin2c HINTS "${CUDA_BIN_DIR}")
if(NOT BIN2C)
    message(FATAL_ERROR "bin2c not found next to nvcc (${CUDA_BIN_DIR}).")
endif()

# ── Third-party: GLFW (windowing) ───────────────────────────────────
include(FetchContent)

set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glfw)

# ── Third-party: GLM (math) ─────────────────────────────────────────
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glm)

# ── Third-party: GLAD (OpenGL loader) ───────────────────────────────
add_library(glad STATIC
    ${CMAKE_SOURCE_DIR}/external/glad/src/gl.c
)
target_include_directories(glad PUBLIC
    ${CMAKE_SOURCE_DIR}/external/glad/include
)

# ── Third-party: stb + tinyobjloader (header-only) ──────────────────
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE
    ${CMAKE_SOURCE_DIR}/external/stb
)

add_library(tinyobjloader INTERFACE)
target_include_directories(tinyobjloader INTERFACE
    ${CMAKE_SOURCE_DIR}/external/tinyobjloader
)

add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE
    ${CMAKE_SOURCE_DIR}/external/tinygltf
)

# ── OptiX device programs → PTX → embedded byte arrays ───────────────
# Each OptiX .cu file is compiled to PTX in its own OBJECT library, then
# embedded into a separate C array via bin2c.  OptiX loads each array as
# a distinct module — concatenating PTX files is NOT valid (duplicate
# .version / global declarations).

# Helper macro: define an OBJECT library that compiles a single .cu to PTX.
macro(add_optix_ptx_lib TARGET_NAME SOURCE_FILE)
    add_library(${TARGET_NAME} OBJECT ${SOURCE_FILE})
    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_PTX_COMPILATION ON
        CUDA_ARCHITECTURES 89
    )
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${OPTIX_INCLUDE_DIR}
    )
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endmacro()

# 1. Megakernel raygens + CH/miss (optix_programs.cu)
add_optix_ptx_lib(optix_programs     src/optix_programs.cu)
# 2. Wavefront extend raygen (ray_extend.cu)
add_optix_ptx_lib(optix_wf_extend    src/wavefront/ray_extend.cu)
# 3. Wavefront shadow raygen (ray_connect.cu)
add_optix_ptx_lib(optix_wf_shadow    src/wavefront/ray_connect.cu)

# Helper macro: embed one PTX file via bin2c.
macro(add_ptx_embed EMBED_TARGET OBJECT_TARGET VAR_NAME OUTPUT_FILE)
    add_custom_command(
        OUTPUT  "${OUTPUT_FILE}"
        COMMAND ${CMAKE_COMMAND}
                -DBIN2C=${BIN2C}
                "-DPTX_FILE=$<TARGET_OBJECTS:${OBJECT_TARGET}>"
                -DOUTPUT=${OUTPUT_FILE}
                -DVAR_NAME=${VAR_NAME}
                -P "${CMAKE_SOURCE_DIR}/cmake/bin2c_wrapper.cmake"
        DEPENDS ${OBJECT_TARGET} "$<TARGET_OBJECTS:${OBJECT_TARGET}>"
        VERBATIM
        COMMENT "Embedding ${OBJECT_TARGET} PTX into ${OUTPUT_FILE}"
    )
    add_custom_target(${EMBED_TARGET} DEPENDS "${OUTPUT_FILE}")
endmacro()

set(EMBEDDED_PTX_C          "${CMAKE_BINARY_DIR}/embedded_optix_ptx.c")
set(EMBEDDED_WF_EXTEND_C    "${CMAKE_BINARY_DIR}/embedded_wf_extend_ptx.c")
set(EMBEDDED_WF_SHADOW_C    "${CMAKE_BINARY_DIR}/embedded_wf_shadow_ptx.c")

add_ptx_embed(embed_optix_ptx       optix_programs   optixProgramsPtx     "${EMBEDDED_PTX_C}")
add_ptx_embed(embed_wf_extend_ptx   optix_wf_extend  optixWfExtendPtx     "${EMBEDDED_WF_EXTEND_C}")
add_ptx_embed(embed_wf_shadow_ptx   optix_wf_shadow  optixWfShadowPtx     "${EMBEDDED_WF_SHADOW_C}")

# ── RedTruthEngine executable ────────────────────────────────────────
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/render_kernel.cu           # public API + per-frame dispatch
    src/scene.cpp
    src/gltf_loader.cpp
    src/optix_ptx_embed.cpp
    src/host/renderer_state.cu     # shared s_* renderer state (single definition)
    src/host/optix_setup.cu        # OptiX pipeline / SBTs / GAS build
    src/host/scene_upload.cu       # cudaInit* scene uploaders
    src/host/denoiser.cu           # OptiX AI denoiser + linear presentation
    src/host/wavefront_host.cu     # wavefront buffers + per-frame bounce loop
    src/wavefront/ray_generate.cu  # wfGenerate kernel  (regular CUDA, linked to exe)
    src/wavefront/ray_shade.cu     # wfShade kernel     (regular CUDA, linked to exe)
    src/wavefront/ray_present.cu   # wfPresent kernel   (regular CUDA, linked to exe)
)

add_dependencies(${PROJECT_NAME} embed_optix_ptx embed_wf_extend_ptx embed_wf_shadow_ptx)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OPTIX_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}          # for the generated embedded_*_ptx.c files
)

# Enable separable compilation so future .cu files can call device funcs
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    glad
    glfw
    glm::glm
    stb
    tinyobjloader
    tinygltf
    opengl32          # Windows OpenGL
)

# Copy assets next to the binary so the exe can find them at runtime
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/assets"
    COMMENT "Copying assets to output directory"
)
