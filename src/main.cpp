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
#include "scene.h"

// ── Standard library ────────────────────────────────────────────────
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <cmath>

static int g_windowWidth  = 1280;
static int g_windowHeight = 720;

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
                 g_windowWidth, g_windowHeight, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // --- Pixel Buffer Object (CUDA-GL interop target) ---------------------
    glGenBuffers(1, &g_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 g_windowWidth * g_windowHeight * 4,
                 nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // --- Shader -----------------------------------------------------------
    g_shader = createShaderProgram();
}

// =====================================================================
//  Load all triangles from an OBJ via tinyobjloader
// =====================================================================

static Float3 transformPoint(const Float3& p,
                               const MeshDesc::TransformDesc& t)
{
    // Apply Scale -> Rx -> Ry -> Rz -> Translate (Euler, degrees).
    float x = p.x * t.scaleX;
    float y = p.y * t.scaleY;
    float z = p.z * t.scaleZ;

    const float degToRad = 3.14159265358979323846f / 180.0f;
    float rx = t.rotXDegrees * degToRad;
    float ry = t.rotYDegrees * degToRad;
    float rz = t.rotZDegrees * degToRad;

    // Rx
    {
        float cx = std::cos(rx);
        float sx = std::sin(rx);
        float y2 = y * cx - z * sx;
        float z2 = y * sx + z * cx;
        y = y2;
        z = z2;
    }

    // Ry
    {
        float cy = std::cos(ry);
        float sy = std::sin(ry);
        float x2 = x * cy + z * sy;
        float z2 = -x * sy + z * cy;
        x = x2;
        z = z2;
    }

    // Rz
    {
        float cz = std::cos(rz);
        float sz = std::sin(rz);
        float x2 = x * cz - y * sz;
        float y2 = x * sz + y * cz;
        x = x2;
        y = y2;
    }

    // Translate
    x += t.posX;
    y += t.posY;
    z += t.posZ;

    return {x, y, z};
}

static std::vector<TriangleData> loadTrianglesFromObj(
    const std::string& path,
    const MeshDesc::TransformDesc& transform)
{
    tinyobj::attrib_t                attrib;
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials; // not used; materials live in scene.xml
    std::string                      warn, err;

    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials,
                               &warn, &err, path.c_str());
    if (!warn.empty()) std::cerr << "[tinyobj warn] " << warn << '\n';
    if (!err.empty())  std::cerr << "[tinyobj err]  " << err  << '\n';
    if (!ok) {
        std::cerr << "Failed to load OBJ: " << path << '\n';
        std::exit(EXIT_FAILURE);
    }

    std::vector<TriangleData> tris;

    auto getV = [&](int vertexIndex) -> Float3 {
        // tinyobj uses 0-based indices; negative index indicates missing data.
        if (vertexIndex < 0) {
            std::cerr << "[mesh] Invalid vertex index in \"" << path << "\"\n";
            std::exit(EXIT_FAILURE);
        }
        return { attrib.vertices[3 * vertexIndex + 0],
                 attrib.vertices[3 * vertexIndex + 1],
                 attrib.vertices[3 * vertexIndex + 2] };
    };

    for (const auto& shape : shapes) {
        const auto& idx = shape.mesh.indices;
        if (idx.size() < 3) continue;

        if (idx.size() % 3 != 0) {
            std::cerr << "[mesh] Warning: \"" << path
                      << "\" has non-triangle faces. Ignoring trailing indices.\n";
        }

        for (size_t i = 0; i + 2 < idx.size(); i += 3) {
            const auto i0 = idx[i + 0];
            const auto i1 = idx[i + 1];
            const auto i2 = idx[i + 2];

            TriangleData t{};
            t.v0 = transformPoint(getV(i0.vertex_index), transform);
            t.v1 = transformPoint(getV(i1.vertex_index), transform);
            t.v2 = transformPoint(getV(i2.vertex_index), transform);
            tris.push_back(t);
        }
    }

    if (tris.empty()) {
        std::cerr << "[mesh] No triangles found in OBJ: " << path << '\n';
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[mesh] Loaded " << tris.size() << " triangles from \""
              << path << "\"\n";
    return tris;
}

// =====================================================================
//  Load texture via stb_image and compute average colour
// =====================================================================

static Float3 loadTextureAverage(const std::string& path)
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

    Float3 color{};
    color.x = static_cast<float>(acc.r);
    color.y = static_cast<float>(acc.g);
    color.z = static_cast<float>(acc.b);

    std::cout << "[tex]  Average colour from \"" << path << "\": ("
              << color.x << ", " << color.y << ", " << color.z << ")\n";

    return color;
}

static float triangleAreaHost(const TriangleData& t)
{
    const float ax = t.v1.x - t.v0.x;
    const float ay = t.v1.y - t.v0.y;
    const float az = t.v1.z - t.v0.z;
    const float bx = t.v2.x - t.v0.x;
    const float by = t.v2.y - t.v0.y;
    const float bz = t.v2.z - t.v0.z;

    const float cx = ay * bz - az * by;
    const float cy = az * bx - ax * bz;
    const float cz = ax * by - ay * bx;
    const float len = std::sqrt(cx * cx + cy * cy + cz * cz);
    return 0.5f * len;
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

int main(int argc, char** argv)
{
    // ── Scene description (Nori-style) ─────────────────────────────────
    const std::string scenePath =
        (argc > 1) ? argv[1] : std::string("assets/scene.xml");
    SceneDescription scene = loadSceneDescription(scenePath);
    g_windowWidth  = scene.windowWidth;
    g_windowHeight = scene.windowHeight;

    // ── Camera (convert look-at/up to camera basis) ───────────────────
    {
        glm::vec3 eye(scene.camera.eyeX, scene.camera.eyeY, scene.camera.eyeZ);
        glm::vec3 target(scene.camera.lookAtX,
                         scene.camera.lookAtY,
                         scene.camera.lookAtZ);
        glm::vec3 up(scene.camera.upX, scene.camera.upY, scene.camera.upZ);

        glm::vec3 forward = glm::normalize(target - eye);
        glm::vec3 right   = glm::normalize(glm::cross(forward, up));
        glm::vec3 trueUp  = glm::cross(right, forward);

        if (glm::length(right) < 1e-6f) {
            std::cerr << "[camera] Invalid camera basis. `up` must not be "
                         "colinear with forward.\n";
            return EXIT_FAILURE;
        }
        if (g_windowHeight <= 0) {
            std::cerr << "[camera] Invalid window height.\n";
            return EXIT_FAILURE;
        }

        CameraData cam{};
        cam.origin = { eye.x, eye.y, eye.z };
        cam.forward = { forward.x, forward.y, forward.z };
        cam.right = { right.x, right.y, right.z };
        cam.up = { trueUp.x, trueUp.y, trueUp.z };
        cam.fovYRadians = glm::radians(scene.camera.fovYDegrees);
        cam.aspect = static_cast<float>(g_windowWidth) /
                      static_cast<float>(g_windowHeight);

        cudaInitCamera(cam);
    }

    // ── GLFW init ────────────────────────────────────────────────────
    if (!glfwInit()) {
        std::cerr << "Failed to initialise GLFW\n";
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(
        g_windowWidth, g_windowHeight, "RedTruthEngine", nullptr, nullptr);
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

    // ── Load assets (from scene description) ─────────────────────────
    std::vector<Float3> materials;
    materials.reserve(scene.materials.size());
    for (const auto& m : scene.materials) {
        materials.push_back(loadTextureAverage(m.albedoTexture));
    }

    auto materialIndexOf = [&](const std::string& name) -> int {
        for (int i = 0; i < static_cast<int>(scene.materials.size()); ++i) {
            if (scene.materials[static_cast<size_t>(i)].name == name) return i;
        }
        return -1;
    };

    auto bsdfIndexOf = [&](const std::string& name) -> int {
        for (int i = 0; i < static_cast<int>(scene.bsdfs.size()); ++i) {
            if (scene.bsdfs[static_cast<size_t>(i)].name == name) return i;
        }
        return -1;
    };

    // Build BSDF table (templates only; behavior implemented by you later)
    std::vector<BsdfData> bsdfs;
    bsdfs.reserve(scene.bsdfs.size());
    for (const auto& b : scene.bsdfs) {
        BsdfData d{};
        d.p0 = {0, 0, 0, 0};
        d.p1 = {0, 0, 0, 0};

        if (b.type == "diffuse") {
            d.type = BSDF_Diffuse;
            d.p0 = { b.albedoR, b.albedoG, b.albedoB, 0.0f };
        } else if (b.type == "dielectric") {
            d.type = BSDF_Dielectric;
            d.p1 = { b.intIOR, b.extIOR, 0.0f, 0.0f };
        } else {
            std::cerr << "[bsdf] Unsupported bsdf type: " << b.type << "\n";
            std::exit(EXIT_FAILURE);
        }

        bsdfs.push_back(d);
    }

    std::vector<TriangleData> triangles;
    std::vector<int> triangleMaterialIds;
    std::vector<int> triangleBsdfIds;
    std::vector<Float3> triangleEmission;
    triangles.reserve(1024);
    triangleMaterialIds.reserve(1024);
    triangleBsdfIds.reserve(1024);
    triangleEmission.reserve(1024);

    // Global mesh-emitter sampling (union of all emissive triangles for now)
    std::vector<int> emissiveTriangleIndices;
    std::vector<float> emissiveTriangleCdf; // length N+1, cdf[0]=0
    emissiveTriangleCdf.push_back(0.0f);
    float emissiveAreaSum = 0.0f;

    // Nori-like emitter list (one emitter per emissive mesh)
    std::vector<EmitterData> emitters;
    std::vector<int> emitterTriIndices;   // concatenated per-emitter triangle indices
    std::vector<float> emitterTriCdf;     // concatenated per-emitter area CDFs
    std::vector<float> sceneEmitterCdf;   // power-weighted CDF over emitters
    sceneEmitterCdf.push_back(0.0f);
    float sceneEmitterWeightSum = 0.0f;

    for (const auto& mesh : scene.meshes) {
        const int matId = materialIndexOf(mesh.materialName);
        if (matId < 0) {
            std::cerr << "[scene] mesh references unknown material: "
                      << mesh.materialName << '\n';
            std::exit(EXIT_FAILURE);
        }

        std::vector<TriangleData> meshTris =
            loadTrianglesFromObj(mesh.filename, mesh.transform);

        const int bsdfId = [&]() -> int {
            const auto& mat = scene.materials[static_cast<size_t>(matId)];
            int id = bsdfIndexOf(mat.bsdfName);
            if (id < 0) {
                std::cerr << "[scene] material references unknown bsdf: "
                          << mat.bsdfName << '\n';
                std::exit(EXIT_FAILURE);
            }
            return id;
        }();

        const Float3 emitColor = { mesh.radianceR, mesh.radianceG, mesh.radianceB };
        // If this mesh is emissive, create an emitter entry backed by its triangles.
        int emitterIndex = -1;
        int triIndexOffset = 0;
        int cdfOffset = 0;
        float emitterAreaSum = 0.0f;
        if (mesh.isEmitter) {
            emitterIndex = static_cast<int>(emitters.size());
            triIndexOffset = static_cast<int>(emitterTriIndices.size());
            cdfOffset = static_cast<int>(emitterTriCdf.size());
            emitterTriCdf.push_back(0.0f);
        }

        for (const auto& t : meshTris) {
            triangles.push_back(t);
            triangleMaterialIds.push_back(matId);
            triangleBsdfIds.push_back(bsdfId);

            if (mesh.isEmitter) {
                triangleEmission.push_back(emitColor);

                const int triIdx = static_cast<int>(triangles.size()) - 1;
                emissiveTriangleIndices.push_back(triIdx);
                const float a = triangleAreaHost(t);
                emissiveAreaSum += a;
                emissiveTriangleCdf.push_back(emissiveAreaSum);

                // Per-emitter lists
                emitterTriIndices.push_back(triIdx);
                emitterAreaSum += a;
                emitterTriCdf.push_back(emitterAreaSum);
            } else {
                triangleEmission.push_back({0.0f, 0.0f, 0.0f});
            }
        }

        if (mesh.isEmitter) {
            // Power-weight: areaSum * luminance(radiance) (simple)
            const float lum = 0.2126f * mesh.radianceR + 0.7152f * mesh.radianceG + 0.0722f * mesh.radianceB;
            const float weight = emitterAreaSum * lum;
            sceneEmitterWeightSum += weight;
            sceneEmitterCdf.push_back(sceneEmitterWeightSum);

            EmitterData e{};
            e.triIndexOffset = triIndexOffset;
            e.triCount = static_cast<int>(emitterTriIndices.size()) - triIndexOffset;
            e.cdfOffset = cdfOffset;
            e.radiance = { mesh.radianceR, mesh.radianceG, mesh.radianceB };
            e.areaSum = emitterAreaSum;
            e.powerWeight = weight;
            emitters.push_back(e);
        }
    }

    if (triangles.empty()) {
        std::cerr << "[scene] Scene has no triangles after loading meshes.\n";
        std::exit(EXIT_FAILURE);
    }

    // ── CUDA init ────────────────────────────────────────────────────
    cudaInitScene(triangles.data(), static_cast<int>(triangles.size()),
                   materials.data(), static_cast<int>(materials.size()),
                   triangleMaterialIds.data(),
                   g_windowWidth, g_windowHeight);

    cudaInitTriangleEmission(triangleEmission.data(),
                             static_cast<int>(triangleEmission.size()));

    if (!emissiveTriangleIndices.empty()) {
        cudaInitEmitters(emissiveTriangleIndices.data(),
                         emissiveTriangleCdf.data(),
                         static_cast<int>(emissiveTriangleIndices.size()));
    } else {
        cudaInitEmitters(nullptr, nullptr, 0);
    }

    if (!emitters.empty()) {
        cudaInitEmitterTable(emitters.data(), static_cast<int>(emitters.size()),
                             emitterTriIndices.data(), static_cast<int>(emitterTriIndices.size()),
                             emitterTriCdf.data(), static_cast<int>(emitterTriCdf.size()),
                             sceneEmitterCdf.data());
    } else {
        cudaInitEmitterTable(nullptr, 0, nullptr, 0, nullptr, 0, nullptr);
    }

    if (!bsdfs.empty() && !triangleBsdfIds.empty()) {
        cudaInitBsdfs(bsdfs.data(), static_cast<int>(bsdfs.size()),
                      triangleBsdfIds.data(), static_cast<int>(triangleBsdfIds.size()));
    } else {
        cudaInitBsdfs(nullptr, 0, nullptr, 0);
    }
    cudaRegisterPBO(g_pbo);

    std::cout << "[cuda] Ready – entering render loop\n";

    // ── Render loop ──────────────────────────────────────────────────
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // 1. CUDA renders into the PBO
        cudaRender(g_windowWidth, g_windowHeight);

        // 2. Copy PBO → texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_pbo);
        glBindTexture(GL_TEXTURE_2D, g_texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        g_windowWidth, g_windowHeight,
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
