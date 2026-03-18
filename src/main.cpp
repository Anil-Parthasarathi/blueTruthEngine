// ============================================================================
//  main.cpp  –  RedTruthEngine entry point
// ============================================================================
//  Sets up GLFW window, OpenGL context, loads mesh (tinyobjloader) and
//  texture (stb_image), creates a Pixel Buffer Object for CUDA-GL interop,
//  and enters the render loop.
// ============================================================================

// ── OpenGL (GLAD must come before GLFW) ─────────────────────────────
#include <glad/gl.h>
#include <GLFW/glfw3.h>

// ── GLM ─────────────────────────────────────────────────────────────
#include <glm/glm.hpp>

// ── stb_image ───────────────────────────────────────────────────────
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ── tinyobjloader ───────────────────────────────────────────────────
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

// ── Project headers ─────────────────────────────────────────────────
#include "render_kernel.h"

// ── Standard library ────────────────────────────────────────────────
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

// ── Constants ───────────────────────────────────────────────────────
static constexpr int WINDOW_WIDTH  = 1280;
static constexpr int WINDOW_HEIGHT = 720;

// ── OpenGL objects ──────────────────────────────────────────────────
static GLuint g_pbo     = 0;   // Pixel Buffer Object (CUDA writes here)
static GLuint g_texture = 0;   // Screen-sized texture (PBO → texture)
static GLuint g_vao     = 0;   // Fullscreen-quad VAO
static GLuint g_vbo     = 0;   // Fullscreen-quad VBO
static GLuint g_shader  = 0;   // Minimal shader program

// =====================================================================
//  Shader sources (fullscreen textured quad)
// =====================================================================

static const char* vertSrc = R"(
#version 460 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
void main() {
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

static const char* fragSrc = R"(
#version 460 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D screenTex;
void main() {
    fragColor = texture(screenTex, vUV);
}
)";

// =====================================================================
//  Helpers
// =====================================================================

static GLuint compileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::cerr << "Shader compile error:\n" << log << '\n';
        std::exit(EXIT_FAILURE);
    }
    return s;
}

static GLuint createShaderProgram()
{
    GLuint vs  = compileShader(GL_VERTEX_SHADER,   vertSrc);
    GLuint fs  = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    GLuint prg = glCreateProgram();
    glAttachShader(prg, vs);
    glAttachShader(prg, fs);
    glLinkProgram(prg);
    GLint ok = 0;
    glGetProgramiv(prg, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(prg, sizeof(log), nullptr, log);
        std::cerr << "Shader link error:\n" << log << '\n';
        std::exit(EXIT_FAILURE);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prg;
}

// =====================================================================
//  Initialise OpenGL resources
// =====================================================================

static void initGL()
{
    // --- Fullscreen quad (two triangles) ---------------------------------
    // Positions + UVs
    float quad[] = {
        // x     y     u     v
        -1.f, -1.f,  0.f, 0.f,
         1.f, -1.f,  1.f, 0.f,
         1.f,  1.f,  1.f, 1.f,

        -1.f, -1.f,  0.f, 0.f,
         1.f,  1.f,  1.f, 1.f,
        -1.f,  1.f,  0.f, 1.f,
    };

    glGenVertexArrays(1, &g_vao);
    glGenBuffers(1, &g_vbo);
    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    // pos
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    // uv
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          reinterpret_cast<void*>(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // --- Screen-sized texture (updated from PBO each frame) ---------------
    glGenTextures(1, &g_texture);
    glBindTexture(GL_TEXTURE_2D, g_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                 WINDOW_WIDTH, WINDOW_HEIGHT, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // --- Pixel Buffer Object (CUDA-GL interop target) ---------------------
    glGenBuffers(1, &g_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 WINDOW_WIDTH * WINDOW_HEIGHT * 4,
                 nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // --- Shader -----------------------------------------------------------
    g_shader = createShaderProgram();
}

// =====================================================================
//  Load triangle mesh via tinyobjloader
// =====================================================================

static TriangleData loadTriangleMesh(const std::string& path)
{
    tinyobj::attrib_t                attrib;
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials;
    std::string                      warn, err;

    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials,
                               &warn, &err, path.c_str());
    if (!warn.empty()) std::cerr << "[tinyobj warn] " << warn << '\n';
    if (!err.empty())  std::cerr << "[tinyobj err]  " << err  << '\n';
    if (!ok) {
        std::cerr << "Failed to load OBJ: " << path << '\n';
        std::exit(EXIT_FAILURE);
    }

    // Extract the first 3 vertices from the first face
    TriangleData tri{};
    if (shapes.empty() || shapes[0].mesh.indices.size() < 3) {
        std::cerr << "OBJ has no triangle face\n";
        std::exit(EXIT_FAILURE);
    }

    auto idx0 = shapes[0].mesh.indices[0];
    auto idx1 = shapes[0].mesh.indices[1];
    auto idx2 = shapes[0].mesh.indices[2];

    auto v = [&](int vi) -> Float3 {
        return { attrib.vertices[3 * vi],
                 attrib.vertices[3 * vi + 1],
                 attrib.vertices[3 * vi + 2] };
    };

    tri.v0 = v(idx0.vertex_index);
    tri.v1 = v(idx1.vertex_index);
    tri.v2 = v(idx2.vertex_index);

    std::cout << "[mesh] Loaded triangle from \"" << path << "\"\n";
    std::cout << "       v0=(" << tri.v0.x << ", " << tri.v0.y << ", " << tri.v0.z << ")\n";
    std::cout << "       v1=(" << tri.v1.x << ", " << tri.v1.y << ", " << tri.v1.z << ")\n";
    std::cout << "       v2=(" << tri.v2.x << ", " << tri.v2.y << ", " << tri.v2.z << ")\n";

    return tri;
}

// =====================================================================
//  Load texture via stb_image and compute average colour
// =====================================================================

static UniformColor loadTextureAverage(const std::string& path)
{
    int w, h, channels;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!data) {
        std::cerr << "Failed to load texture: " << path << '\n';
        std::exit(EXIT_FAILURE);
    }

    // Compute the average colour across all pixels (using GLM for accumulation)
    glm::dvec4 acc(0.0);
    int totalPixels = w * h;
    for (int i = 0; i < totalPixels; ++i) {
        acc.r += data[4 * i + 0] / 255.0;
        acc.g += data[4 * i + 1] / 255.0;
        acc.b += data[4 * i + 2] / 255.0;
        acc.a += data[4 * i + 3] / 255.0;
    }
    acc /= static_cast<double>(totalPixels);

    stbi_image_free(data);

    UniformColor color{};
    color.r = static_cast<float>(acc.r);
    color.g = static_cast<float>(acc.g);
    color.b = static_cast<float>(acc.b);
    color.a = static_cast<float>(acc.a);

    std::cout << "[tex]  Average colour from \"" << path << "\": ("
              << color.r << ", " << color.g << ", " << color.b
              << ", " << color.a << ")\n";

    return color;
}

// =====================================================================
//  GLFW callbacks
// =====================================================================

static void keyCallback(GLFWwindow* window, int key, int /*scancode*/,
                         int action, int /*mods*/)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

static void framebufferSizeCallback(GLFWwindow* /*window*/, int w, int h)
{
    glViewport(0, 0, w, h);
}

// =====================================================================
//  Main
// =====================================================================

int main()
{
    // ── GLFW init ────────────────────────────────────────────────────
    if (!glfwInit()) {
        std::cerr << "Failed to initialise GLFW\n";
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(
        WINDOW_WIDTH, WINDOW_HEIGHT, "RedTruthEngine", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    // ── Load OpenGL (GLAD 2) ─────────────────────────────────────────
    int gladVersion = gladLoadGL(glfwGetProcAddress);
    if (!gladVersion) {
        std::cerr << "Failed to initialise GLAD\n";
        return EXIT_FAILURE;
    }
    std::cout << "[gl]   OpenGL " << GLAD_VERSION_MAJOR(gladVersion) << "."
              << GLAD_VERSION_MINOR(gladVersion) << " loaded\n";

    // ── GL resources ─────────────────────────────────────────────────
    initGL();

    // ── Load assets ──────────────────────────────────────────────────
    TriangleData tri   = loadTriangleMesh("assets/triangle.obj");
    UniformColor color = loadTextureAverage("assets/checkerboard.png");

    // ── CUDA init ────────────────────────────────────────────────────
    cudaInit(tri, color, WINDOW_WIDTH, WINDOW_HEIGHT);
    cudaRegisterPBO(g_pbo);

    std::cout << "[cuda] Ready – entering render loop\n";

    // ── Render loop ──────────────────────────────────────────────────
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // 1. CUDA renders into the PBO
        cudaRender(WINDOW_WIDTH, WINDOW_HEIGHT);

        // 2. Copy PBO → texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
        glBindTexture(GL_TEXTURE_2D, g_texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        WINDOW_WIDTH, WINDOW_HEIGHT,
                        GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // 3. Draw fullscreen quad
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(g_shader);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_texture);
        glBindVertexArray(g_vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window);
    }

    // ── Cleanup ──────────────────────────────────────────────────────
    cudaCleanup();
    glDeleteProgram(g_shader);
    glDeleteBuffers(1, &g_pbo);
    glDeleteTextures(1, &g_texture);
    glDeleteBuffers(1, &g_vbo);
    glDeleteVertexArrays(1, &g_vao);
    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "[exit] Clean shutdown\n";
    return EXIT_SUCCESS;
}
