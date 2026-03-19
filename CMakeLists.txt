cmake_minimum_required(VERSION 3.24)
project(RedTruthEngine LANGUAGES CXX CUDA)

# ── C++ / CUDA standards ────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target RTX 4080 SUPER  (SM 8.9)
set(CMAKE_CUDA_ARCHITECTURES 89)

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

# ── RedTruthEngine executable ────────────────────────────────────────
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/render_kernel.cu
    src/scene.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
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
    opengl32          # Windows OpenGL
)

# Copy assets next to the binary so the exe can find them at runtime
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/assets"
    COMMENT "Copying assets to output directory"
)
