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

// ── stb_image / stb_image_write ─────────────────────────────────────
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <cmath>
#include <algorithm>

static int  g_windowWidth   = 1280;
static int  g_windowHeight  = 720;

// ---------------------------------------------------------------------------
//  Render mode selector — edit this to switch between implementations.
//
//    true  → wavefront pipeline  (wfGenerate → extend → shade → connect loop)
//    false → megakernel pipeline (single OptiX __raygen__rg per frame)
//
//  This is read once at startup; changing it requires a recompile.
// ---------------------------------------------------------------------------
static constexpr bool USE_WAVEFRONT = true;

// Set to true whenever the camera, geometry, or lighting changes so the
// accumulation buffer is cleared before the next frame.  Wire this up to
// any animation / camera-movement code you add in the future.
static bool g_sceneChanged  = false;
static int  g_accumSamples  = 0;   // samples accumulated since last reset
static bool g_representOnly = false; // restyle without tracing another sample

// Host-side copy of the uploaded StyleData table so live knobs can mutate it
// and push the change back to the GPU without rebuilding the scene.
static std::vector<StyleData> g_styleTable;

inline void markSceneChanged() { g_sceneChanged = true; }

// ── OpenGL objects ──────────────────────────────────────────────────
static GLuint g_pbo     = 0;   // Pixel Buffer Object (CUDA writes here)
static GLuint g_texture = 0;   // Screen-sized texture (PBO → texture)
static GLuint g_vao     = 0;   // Fullscreen-quad VAO
static GLuint g_vbo     = 0;   // Fullscreen-quad VBO
static GLuint g_shader  = 0;   // Minimal shader program

// ── Video recording ─────────────────────────────────────────────────
// Press R to start/stop.  Requires FFmpeg on PATH.
// Output playback FPS is independent of render FPS — slow renders still
// produce a smooth video.
static FILE*                 g_recordPipe  = nullptr;
static bool                  g_recording   = false;
static std::vector<uint8_t>  g_pixelBuf;
static constexpr int         RECORD_FPS    = 30;   // playback FPS of output video

static void startRecording()
{
    char cmd[512];
    // -f rawvideo + -pixel_format rgba: we send raw RGBA bytes from glReadPixels
    // -vf vflip:                        OpenGL origin is bottom-left; flip to top-left
    // -crf 18:                          high quality H.264
    snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -f rawvideo -pixel_format rgba -video_size %dx%d -r %d -i pipe:0 "
        "-vf vflip -c:v libx264 -crf 18 -pix_fmt yuv420p output.mp4",
        g_windowWidth, g_windowHeight, RECORD_FPS);

#ifdef _WIN32
    g_recordPipe = _popen(cmd, "wb");
#else
    g_recordPipe = popen(cmd, "w");
#endif

    if (g_recordPipe) {
        g_recording = true;
        g_pixelBuf.resize(static_cast<size_t>(g_windowWidth * g_windowHeight * 4));
        std::cout << "[record] Recording started → output.mp4  (playback " << RECORD_FPS << " fps)\n";
    } else {
        std::cerr << "[record] Could not open FFmpeg pipe. Is ffmpeg on PATH?\n";
    }
}

static void stopRecording()
{
    if (!g_recordPipe) return;
    g_recording = false;
#ifdef _WIN32
    _pclose(g_recordPipe);
#else
    pclose(g_recordPipe);
#endif
    g_recordPipe = nullptr;
    std::cout << "[record] Recording stopped → output.mp4\n";
}

static void captureScreenshot()
{
    std::vector<uint8_t> pixels(static_cast<size_t>(g_windowWidth * g_windowHeight * 4));
    glReadPixels(0, 0, g_windowWidth, g_windowHeight,
                 GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    // OpenGL pixel origin is bottom-left; flip rows so the PNG is top-left.
    const int stride = g_windowWidth * 4;
    for (int y = 0; y < g_windowHeight / 2; ++y) {
        uint8_t* rowA = pixels.data() + y * stride;
        uint8_t* rowB = pixels.data() + (g_windowHeight - 1 - y) * stride;
        for (int x = 0; x < stride; ++x)
            std::swap(rowA[x], rowB[x]);
    }

    // Build a filename with a frame counter so shots don't overwrite each other.
    static int shotIndex = 0;
    char filename[64];
    snprintf(filename, sizeof(filename), "screenshot_%04d.png", shotIndex++);

    if (stbi_write_png(filename, g_windowWidth, g_windowHeight, 4, pixels.data(), stride))
        std::cout << "[screenshot] Saved " << filename << "\n";
    else
        std::cerr << "[screenshot] Failed to write " << filename << "\n";
}

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

// Like transformPoint but for directions: apply rotations only (no scale, no translation).
static Float3 transformNormal(const Float3& n,
                               const MeshDesc::TransformDesc& t)
{
    float x = n.x, y = n.y, z = n.z;

    const float degToRad = 3.14159265358979323846f / 180.0f;
    float rx = t.rotXDegrees * degToRad;
    float ry = t.rotYDegrees * degToRad;
    float rz = t.rotZDegrees * degToRad;

    {
        float cx = std::cos(rx), sx = std::sin(rx);
        float y2 = y * cx - z * sx;
        float z2 = y * sx + z * cx;
        y = y2; z = z2;
    }
    {
        float cy = std::cos(ry), sy = std::sin(ry);
        float x2 = x * cy + z * sy;
        float z2 = -x * sy + z * cy;
        x = x2; z = z2;
    }
    {
        float cz = std::cos(rz), sz = std::sin(rz);
        float x2 = x * cz - y * sz;
        float y2 = x * sz + y * cz;
        x = x2; y = y2;
    }

    // Normalize (handles any uniform scale implicitly).
    float len = std::sqrt(x * x + y * y + z * z);
    if (len > 1e-8f) { x /= len; y /= len; z /= len; }
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
        if (vertexIndex < 0) {
            std::cerr << "[mesh] Invalid vertex index in \"" << path << "\"\n";
            std::exit(EXIT_FAILURE);
        }
        return { attrib.vertices[3 * vertexIndex + 0],
                 attrib.vertices[3 * vertexIndex + 1],
                 attrib.vertices[3 * vertexIndex + 2] };
    };

    // Returns the vertex normal for the given tinyobj normal_index, or a
    // zero vector when the OBJ has no normals (caller falls back to geometric).
    auto getN = [&](int normalIndex) -> Float3 {
        if (normalIndex < 0 || attrib.normals.empty()) return {0.0f, 0.0f, 0.0f};
        return { attrib.normals[3 * normalIndex + 0],
                 attrib.normals[3 * normalIndex + 1],
                 attrib.normals[3 * normalIndex + 2] };
    };

    // Returns UV texture coordinates from the OBJ texcoord table.
    auto getUV = [&](int texcoordIndex) -> Float2 {
        if (texcoordIndex < 0 || attrib.texcoords.empty()) return {0.0f, 0.0f};
        return { attrib.texcoords[2 * texcoordIndex + 0],
                 attrib.texcoords[2 * texcoordIndex + 1] };
    };

    // Detect degenerate UV layout: if the OBJ defines ≤ 1 unique UV vertex
    // (a common artefact of simple OBJ exports), every face gets the same
    // texcoord and the texture tiles nowhere.  In that case we fall back to
    // world-space planar projection so textures actually appear.
    // Scale controls how many times the texture tiles per world unit.
    const bool hasRealUVs = (attrib.texcoords.size() > 2); // > 1 unique UV
    static constexpr float kUVTileScale = 1.0f;

    // World-space planar UV: project the vertex onto the two axes that are
    // most perpendicular to the dominant normal component.
    auto planarUV = [](const Float3& pos, const Float3& n) -> Float2 {
        const float ax = std::fabs(n.x);
        const float ay = std::fabs(n.y);
        const float az = std::fabs(n.z);
        if (ay >= ax && ay >= az) return { pos.x * kUVTileScale, pos.z * kUVTileScale };
        if (ax >= ay && ax >= az) return { pos.y * kUVTileScale, pos.z * kUVTileScale };
        return { pos.x * kUVTileScale, pos.y * kUVTileScale };
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

            Float3 rawN0 = getN(i0.normal_index);
            Float3 rawN1 = getN(i1.normal_index);
            Float3 rawN2 = getN(i2.normal_index);

            // If any vertex is missing a normal, fall back to the geometric normal
            // for all three vertices so the triangle is consistently flat-shaded.
            const bool hasNormals = (i0.normal_index >= 0) &&
                                    (i1.normal_index >= 0) &&
                                    (i2.normal_index >= 0) &&
                                    !attrib.normals.empty();
            if (hasNormals) {
                t.n0 = transformNormal(rawN0, transform);
                t.n1 = transformNormal(rawN1, transform);
                t.n2 = transformNormal(rawN2, transform);
            } else {
                // Compute geometric normal and use it for all three vertices.
                const Float3 e1 = { t.v1.x - t.v0.x, t.v1.y - t.v0.y, t.v1.z - t.v0.z };
                const Float3 e2 = { t.v2.x - t.v0.x, t.v2.y - t.v0.y, t.v2.z - t.v0.z };
                Float3 gn = { e1.y * e2.z - e1.z * e2.y,
                              e1.z * e2.x - e1.x * e2.z,
                              e1.x * e2.y - e1.y * e2.x };
                float len = std::sqrt(gn.x*gn.x + gn.y*gn.y + gn.z*gn.z);
                if (len > 1e-8f) { gn.x /= len; gn.y /= len; gn.z /= len; }
                t.n0 = t.n1 = t.n2 = gn;
            }

            if (hasRealUVs) {
                t.uv0 = getUV(i0.texcoord_index);
                t.uv1 = getUV(i1.texcoord_index);
                t.uv2 = getUV(i2.texcoord_index);
            } else {
                // Use the geometric normal (already computed in t.n0) to pick
                // the projection plane, then project each vertex into UV space.
                t.uv0 = planarUV(t.v0, t.n0);
                t.uv1 = planarUV(t.v1, t.n0);
                t.uv2 = planarUV(t.v2, t.n0);
            }

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
    if (path.empty()) return {1.0f, 1.0f, 1.0f};

    int w, h, channels;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!data) {
        std::cerr << "[tex]  Failed to load \"" << path << "\" — using white fallback\n";
        return {1.0f, 1.0f, 1.0f};
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

/// Load a texture image as raw RGBA8 pixels (4 bytes per pixel, row-major).
/// Returns a 1×1 white fallback if path is empty or loading fails.
/// The caller owns the returned data and should free it with stbi_image_free()
/// only if `ownedByStb` is true; for the fallback the data lives in the
/// returned vector.
struct ImageRGBA {
    std::vector<uint8_t> pixels;   // RGBA8, width*height*4 bytes
    int width  = 1;
    int height = 1;
};

static ImageRGBA loadImageRGBA(const std::string& path)
{
    ImageRGBA img;
    // Empty path → no texture.  Caller passes nullptr to cudaInitTextures so
    // the engine leaves the material's BSDF base_color untouched.
    if (path.empty()) return img;

    int channels;
    unsigned char* data = stbi_load(path.c_str(), &img.width, &img.height, &channels, 4);
    if (!data) {
        std::cerr << "[texture] Failed to load \"" << path << "\" — material will use BSDF colour\n";
        return img;  // pixels remains empty → treated as no texture
    }

    img.pixels.assign(data, data + img.width * img.height * 4);
    stbi_image_free(data);

    std::cout << "[texture] Loaded \"" << path << "\" ("
              << img.width << "×" << img.height << ")\n";
    return img;
}

/// Load a float HDR image (.hdr) for use as an environment map.
///
/// stbi_loadf keeps the values linear and unclamped, which is the whole point:
/// an HDRI's sun can be thousands of times brighter than its sky, and that ratio
/// is exactly what makes environment importance sampling worth doing.
struct ImageHDR {
    std::vector<float> pixels;   // width*height*channels, row-major, top row = +Y
    int width    = 0;
    int height   = 0;
    int channels = 3;
};

static ImageHDR loadImageHDR(const std::string& path)
{
    ImageHDR img;
    if (path.empty()) return img;

    int w = 0, h = 0, c = 0;
    float* data = stbi_loadf(path.c_str(), &w, &h, &c, 3);
    if (!data) {
        std::cerr << "[env] Failed to load HDRI \"" << path << "\": "
                  << stbi_failure_reason() << "\n";
        return img;
    }

    img.width    = w;
    img.height   = h;
    img.channels = 3;
    img.pixels.assign(data, data + static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
    stbi_image_free(data);

    std::cout << "[env] Loaded HDRI \"" << path << "\" ("
              << w << "×" << h << ")\n";
    return img;
}

/// Build a 1-D ramp strip as RGBA8 texels.
///
/// A ramp file is normally a wide, 1-pixel-tall PNG authored in a paint program;
/// only its top row is read, so a taller image still works.
static ImageRGBA loadRampStrip(const std::string& path)
{
    ImageRGBA img;
    img.width  = 0;
    img.height = 1;
    if (path.empty()) return img;

    int w = 0, h = 0, channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!data) {
        std::cerr << "[style] Failed to load ramp \"" << path
                  << "\" — falling back to procedural bands\n";
        return img;
    }

    img.width  = w;
    img.height = 1;
    img.pixels.assign(data, data + static_cast<size_t>(w) * 4);   // top row only
    stbi_image_free(data);

    std::cout << "[style] Loaded ramp \"" << path << "\" (" << w << " texels)\n";
    return img;
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

    if (key == GLFW_KEY_R && action == GLFW_PRESS) {
        if (!g_recording) startRecording();
        else              stopRecording();
    }

    if (key == GLFW_KEY_C && action == GLFW_PRESS)
        captureScreenshot();

    // Toggle the OptiX AI denoiser (off by default).
    if (key == GLFW_KEY_D && action == GLFW_PRESS)
        cudaToggleDenoiser();

    // A/B photorealistic vs stylized.  Switching modes changes the channel
    // layout, so accumulation restarts — that is expected.  Photoreal stays a
    // first-class mode: N off restores the original path and footprint.
    if (key == GLFW_KEY_N && action == GLFW_PRESS) {
        cudaToggleStyleMode();
        markSceneChanged();
    }

    // Cycle the AOV / channel debug views.  Purely presentational, so it costs
    // one kernel launch and never disturbs the accumulated data.
    if (key == GLFW_KEY_V && action == GLFW_PRESS) {
        cudaCycleDebugView();
        g_representOnly = true;
    }

    // Live style knobs — mutate the host table, re-upload, and re-present the
    // already-converged image.  No re-accumulation.
    auto restyle = [&]() {
        if (g_styleTable.empty()) return;
        cudaInitStyles(g_styleTable.data(), static_cast<int>(g_styleTable.size()));
        g_representOnly = true;
    };

    if (key == GLFW_KEY_LEFT_BRACKET && action == GLFW_PRESS) {
        for (auto& s : g_styleTable)
            s.diffuseBands = std::max(1, s.diffuseBands - 1);
        if (!g_styleTable.empty())
            std::cout << "[style] diffuseBands = " << g_styleTable[0].diffuseBands << "\n";
        restyle();
    }
    if (key == GLFW_KEY_RIGHT_BRACKET && action == GLFW_PRESS) {
        for (auto& s : g_styleTable)
            s.diffuseBands = std::min(8, s.diffuseBands + 1);
        if (!g_styleTable.empty())
            std::cout << "[style] diffuseBands = " << g_styleTable[0].diffuseBands << "\n";
        restyle();
    }
    if (key == GLFW_KEY_MINUS && action == GLFW_PRESS) {
        for (auto& s : g_styleTable)
            s.indirectGain = std::max(0.0f, s.indirectGain - 0.1f);
        if (!g_styleTable.empty())
            std::cout << "[style] indirectGain = " << g_styleTable[0].indirectGain << "\n";
        restyle();
    }
    if (key == GLFW_KEY_EQUAL && action == GLFW_PRESS) {
        for (auto& s : g_styleTable)
            s.indirectGain = std::min(4.0f, s.indirectGain + 0.1f);
        if (!g_styleTable.empty())
            std::cout << "[style] indirectGain = " << g_styleTable[0].indirectGain << "\n";
        restyle();
    }
    if (key == GLFW_KEY_COMMA && action == GLFW_PRESS) {
        for (auto& s : g_styleTable)
            s.lineStrength = std::max(0.0f, s.lineStrength - 0.1f);
        if (!g_styleTable.empty())
            std::cout << "[style] lineStrength = " << g_styleTable[0].lineStrength
                      << " (re-accumulating — outlines modulate throughput)\n";
        restyle();
        markSceneChanged();   // baked into the path, not a present-time operator
    }
    if (key == GLFW_KEY_PERIOD && action == GLFW_PRESS) {
        for (auto& s : g_styleTable)
            s.lineStrength = std::min(2.0f, s.lineStrength + 0.1f);
        if (!g_styleTable.empty())
            std::cout << "[style] lineStrength = " << g_styleTable[0].lineStrength
                      << " (re-accumulating — outlines modulate throughput)\n";
        restyle();
        markSceneChanged();
    }
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

    // Load per-material textures as full RGBA images and upload to the GPU.
    // Each entry aligns with scene.materials[i]; materials with no albedoTexture
    // get an implicit 1×1 white fallback (texture modulation has no visible effect).
    {
        std::vector<ImageRGBA> texImages;
        texImages.reserve(scene.materials.size());
        for (const auto& m : scene.materials)
            texImages.push_back(loadImageRGBA(m.albedoTexture));

        std::vector<const uint8_t*> ptrs;
        std::vector<int>            widths, heights;
        ptrs.reserve(texImages.size());
        widths.reserve(texImages.size());
        heights.reserve(texImages.size());
        for (const auto& img : texImages) {
            // Empty pixels means no texture for this material — pass nullptr
            // so cudaInitTextures leaves the texture object handle as 0.
            ptrs.push_back(img.pixels.empty() ? nullptr : img.pixels.data());
            widths.push_back(img.width);
            heights.push_back(img.height);
        }
        cudaInitTextures(ptrs.data(), widths.data(), heights.data(),
                         static_cast<int>(texImages.size()));
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

    // ── Style table + ramp textures ──────────────────────────────────
    //  Styles are uploaded as a flat table indexed by BsdfData::styleId.  Ramps
    //  get their own texture table so several styles can share one strip.
    std::vector<StyleData> styleTable;
    {
        std::vector<ImageRGBA> rampImages;
        std::vector<std::string> rampPaths;

        auto rampIndexOf = [&](const std::string& path) -> int {
            if (path.empty()) return -1;
            for (int i = 0; i < static_cast<int>(rampPaths.size()); ++i) {
                if (rampPaths[static_cast<size_t>(i)] == path) return i;
            }
            ImageRGBA img = loadRampStrip(path);
            if (img.pixels.empty()) return -1;
            rampPaths.push_back(path);
            rampImages.push_back(std::move(img));
            return static_cast<int>(rampPaths.size()) - 1;
        };

        styleTable.reserve(scene.styles.size());
        for (const auto& s : scene.styles) {
            StyleData d{};
            d.diffuseRampTex = rampIndexOf(s.diffuseRamp);
            d.diffuseBands   = s.diffuseBands;
            d.bandSoftness   = s.bandSoftness;
            d.toneScale      = s.toneScale;

            d.specThreshold = s.specThreshold;
            d.specSoftness  = s.specSoftness;
            d.specIntensity = s.specIntensity;

            d.rimStrength = s.rimStrength;
            d.rimPower    = s.rimPower;
            d.rimColor    = { s.rimColorR, s.rimColorG, s.rimColorB };

            d.reflectBands   = s.reflectBands;
            d.reflectGain    = s.reflectGain;
            d.reflectTintMix = s.reflectTintMix;
            d.reflectTint    = { s.reflectTintR, s.reflectTintG, s.reflectTintB };

            d.transmitBands = s.transmitBands;
            d.transmitGain  = s.transmitGain;
            d.chromaShift   = s.chromaShift;

            d.indirectGain = s.indirectGain;

            d.lineColor              = { s.lineColorR, s.lineColorG, s.lineColorB };
            d.lineWidth              = s.lineWidth;
            d.lineStrength           = s.lineStrength;
            d.outlineNormalThreshold = s.outlineNormalThreshold;
            d.outlineDepthThreshold  = s.outlineDepthThreshold;

            styleTable.push_back(d);
        }
        g_styleTable = styleTable;

        if (!rampImages.empty()) {
            std::vector<const uint8_t*> ptrs;
            std::vector<int>            widths;
            ptrs.reserve(rampImages.size());
            widths.reserve(rampImages.size());
            for (const auto& img : rampImages) {
                ptrs.push_back(img.pixels.empty() ? nullptr : img.pixels.data());
                widths.push_back(img.width);
            }
            cudaInitRampTextures(ptrs.data(), widths.data(),
                                 static_cast<int>(rampImages.size()));
        }
    }

    auto styleIndexOf = [&](const std::string& name) -> int {
        if (name.empty()) return -1;   // no style is the normal photoreal case
        for (int i = 0; i < static_cast<int>(scene.styles.size()); ++i) {
            if (scene.styles[static_cast<size_t>(i)].name == name) return i;
        }
        std::cerr << "[scene] material references unknown style: " << name << '\n';
        std::exit(EXIT_FAILURE);
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
        } else if (b.type == "mirror") {
            d.type = BSDF_Mirror;
            d.p0 = { b.albedoR > 0.0f ? b.albedoR : 1.0f,
                     b.albedoG > 0.0f ? b.albedoG : 1.0f,
                     b.albedoB > 0.0f ? b.albedoB : 1.0f, 0.0f };
        } else if (b.type == "microfacet") {
            d.type = BSDF_Microfacet;
            const float ks = 1.0f - std::fmax(b.albedoR, std::fmax(b.albedoG, b.albedoB));
            d.p0 = { b.albedoR, b.albedoG, b.albedoB, 0.0f };
            d.p1 = { b.alpha, b.intIOR, b.extIOR, ks };
        } else if (b.type == "disney") {
            d.type = BSDF_Disney;
            d.p0 = { b.baseColorR, b.baseColorG, b.baseColorB, b.roughness };
            d.p1 = { b.metallic, b.specular, b.specularTransmission, b.specularTint };
            d.p2 = { b.sheen, b.sheenTint, b.subsurface, b.anisotropic };
            d.p3 = { b.clearcoat, b.clearcoatGloss, b.eta, 0.0f };
        } else {
            std::cerr << "[bsdf] Unsupported bsdf type: " << b.type << "\n";
            std::exit(EXIT_FAILURE);
        }

        d.styleId = -1;   // resolved per material below
        bsdfs.push_back(d);
    }

    // A style is authored on the MATERIAL, but the device reads it through
    // BsdfData::styleId — one indirection instead of a second per-triangle table.
    // When two materials share a BSDF but want different styles, the BSDF record
    // is duplicated with the other styleId.  Scenes have a handful of BSDFs, so
    // duplicating a few is cheaper than an extra per-triangle lookup in every
    // shading kernel.
    struct BsdfVariantKey { int baseBsdfId; int styleId; int resolvedId; };
    std::vector<BsdfVariantKey> bsdfVariants;

    auto bsdfVariantFor = [&](int baseBsdfId, int styleId) -> int {
        if (styleId < 0) return baseBsdfId;

        for (const auto& v : bsdfVariants) {
            if (v.baseBsdfId == baseBsdfId && v.styleId == styleId) return v.resolvedId;
        }

        int resolvedId;
        if (bsdfs[static_cast<size_t>(baseBsdfId)].styleId < 0) {
            // First style to claim this BSDF can use the record in place.
            bsdfs[static_cast<size_t>(baseBsdfId)].styleId = styleId;
            resolvedId = baseBsdfId;
        } else {
            BsdfData copy = bsdfs[static_cast<size_t>(baseBsdfId)];
            copy.styleId = styleId;
            bsdfs.push_back(copy);
            resolvedId = static_cast<int>(bsdfs.size()) - 1;
        }

        bsdfVariants.push_back({ baseBsdfId, styleId, resolvedId });
        return resolvedId;
    };

    std::vector<TriangleData> triangles;
    std::vector<int> triangleMaterialIds;
    std::vector<int> triangleBsdfIds;
    std::vector<Float3> triangleEmission;
    std::vector<uint8_t> triangleEmitterFlags;
    // Per-triangle mesh index.  The outline probes compare object IDs to tell a
    // silhouette from a crease, which is what lets an outline be found by a ray
    // rather than by a screen-space filter.
    std::vector<int> triangleObjectIds;
    triangles.reserve(1024);
    triangleMaterialIds.reserve(1024);
    triangleBsdfIds.reserve(1024);
    triangleEmission.reserve(1024);
    triangleEmitterFlags.reserve(1024);
    triangleObjectIds.reserve(1024);

    // Nori-like emitter list (one emitter per emissive mesh)
    std::vector<EmitterData> emitters;
    std::vector<int> emitterTriIndices;   // concatenated per-emitter triangle indices
    std::vector<float> emitterTriCdf;     // concatenated per-emitter area CDFs
    std::vector<float> sceneEmitterCdf;   // power-weighted CDF over emitters
    sceneEmitterCdf.push_back(0.0f);
    float sceneEmitterWeightSum = 0.0f;

    // Per-triangle emitter flag: set from scene mesh isEmitter (Nori-style).
    int meshObjectId = 0;
    for (const auto& mesh : scene.meshes) {
        const int objectId = meshObjectId++;
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
            return bsdfVariantFor(id, styleIndexOf(mat.styleName));
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
            triangleObjectIds.push_back(objectId);
            triangleEmitterFlags.push_back(mesh.isEmitter ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0));

            if (mesh.isEmitter) {
                triangleEmission.push_back(emitColor);

                const int triIdx = static_cast<int>(triangles.size()) - 1;
                const float a = triangleAreaHost(t);

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

    // Intersection runs on the RT cores via an OptiX GAS, which cudaInitOptix
    // builds on-GPU directly from the triangle data uploaded above.
    cudaInitOptix();

    cudaInitTriangleEmission(triangleEmission.data(),
                             static_cast<int>(triangleEmission.size()));
    cudaInitTriangleEmitterFlags(triangleEmitterFlags.data(),
                                 static_cast<int>(triangleEmitterFlags.size()));
    cudaInitTriangleObjectIds(triangleObjectIds.data(),
                              static_cast<int>(triangleObjectIds.size()));

    // ── Environment light ────────────────────────────────────────────
    //  Uploaded BEFORE the emitter table, because the environment participates
    //  in the same discrete emitter selection as the mesh lights: it gets a
    //  trailing slot in sceneEmitterCdf weighted by the same power-style metric.
    //  That is what keeps NEE and BSDF-sampling MIS weights consistent — the
    //  alternative, treating the environment as a special case outside the CDF,
    //  is the classic source of subtly wrong environment lighting.
    {
        ImageHDR envImage;
        ImageHDR envBgImage;
        float    envSelectWeight = 0.0f;

        if (scene.environment.enabled) {
            envImage = loadImageHDR(scene.environment.filename);

            EnvironmentUpload up{};
            up.pixels   = envImage.pixels.empty() ? nullptr : envImage.pixels.data();
            up.width    = envImage.width;
            up.height   = envImage.height;
            up.channels = envImage.channels;

            up.intensity  = scene.environment.intensity;
            up.yawRadians = glm::radians(scene.environment.yawDegrees);

            if (scene.environment.backgroundMode == "color") {
                up.backgroundMode = ENV_BG_FLAT_COLOR;
            } else if (scene.environment.backgroundMode == "texture") {
                up.backgroundMode = ENV_BG_SEPARATE_TEX;
                envBgImage = loadImageHDR(scene.environment.backgroundFile);
                up.bgPixels   = envBgImage.pixels.empty() ? nullptr : envBgImage.pixels.data();
                up.bgWidth    = envBgImage.width;
                up.bgHeight   = envBgImage.height;
                up.bgChannels = envBgImage.channels;
            } else {
                up.backgroundMode = ENV_BG_USE_ENV;
            }
            up.backgroundColor = { scene.environment.backgroundR,
                                   scene.environment.backgroundG,
                                   scene.environment.backgroundB };

            envSelectWeight = cudaInitEnvironment(&up);
        } else {
            cudaInitEnvironment(nullptr);
        }

        int selectCount = static_cast<int>(emitters.size());
        if (envSelectWeight > 0.0f) {
            sceneEmitterWeightSum += envSelectWeight;
            sceneEmitterCdf.push_back(sceneEmitterWeightSum);
            ++selectCount;
        }

        if (selectCount > 0) {
            cudaInitEmitterTable(emitters.empty() ? nullptr : emitters.data(),
                                 static_cast<int>(emitters.size()),
                                 emitterTriIndices.empty() ? nullptr : emitterTriIndices.data(),
                                 static_cast<int>(emitterTriIndices.size()),
                                 emitterTriCdf.empty() ? nullptr : emitterTriCdf.data(),
                                 static_cast<int>(emitterTriCdf.size()),
                                 sceneEmitterCdf.data(), selectCount);
        } else {
            cudaInitEmitterTable(nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0);
        }
    }

    if (!bsdfs.empty() && !triangleBsdfIds.empty()) {
        cudaInitBsdfs(bsdfs.data(), static_cast<int>(bsdfs.size()),
                      triangleBsdfIds.data(), static_cast<int>(triangleBsdfIds.size()));
    } else {
        cudaInitBsdfs(nullptr, 0, nullptr, 0);
    }

    cudaInitStyles(styleTable.empty() ? nullptr : styleTable.data(),
                   static_cast<int>(styleTable.size()));

    // Build and upload spotlight table.
    // caching the cosine so it doesnt have to be found every time the emitter is sampled
    // converting to radians since degrees are easier to work with for scene setup but cos function takes radians
    {
        std::vector<SpotlightData> spotlightsGpu;
        spotlightsGpu.reserve(scene.spotlights.size());
        for (const auto& sd : scene.spotlights) {
            SpotlightData sp{};
            sp.position  = { sd.posX, sd.posY, sd.posZ };

            // Normalize the direction vector
            float dx = sd.dirX, dy = sd.dirY, dz = sd.dirZ;
            float dlen = std::sqrtf(dx*dx + dy*dy + dz*dz);
            if (dlen > 1e-6f) { dx /= dlen; dy /= dlen; dz /= dlen; }
            sp.direction = { dx, dy, dz };

            sp.radiance  = { sd.radianceR, sd.radianceG, sd.radianceB };
            sp.intensity = sd.intensity;
            static constexpr float kPi = 3.14159265358979323846f;
            sp.innerConeCosine = std::cosf(sd.innerConeAngle * kPi / 180.0f);
            sp.outerConeCosine = std::cosf(sd.outerConeAngle * kPi / 180.0f);
            spotlightsGpu.push_back(sp);
        }
        if (!spotlightsGpu.empty()) {
            cudaInitSpotlights(spotlightsGpu.data(), static_cast<int>(spotlightsGpu.size()));
        } else {
            cudaInitSpotlights(nullptr, 0);
        }
    }

    cudaRegisterPBO(g_pbo);

    // Apply the render mode chosen at the top of this file.
    cudaSetRenderMode(USE_WAVEFRONT ? RenderMode::Wavefront : RenderMode::Megakernel);

    // Scene default is physical unless the XML asked for anime.  Existing
    // scenes have no `style` attribute, so they keep rendering photorealistically.
    if (scene.styleMode == "anime") {
        cudaSetStyleMode(StyleMode::Anime);
        std::cout << "[style] Scene requested anime mode (N toggles back to physical)\n";
    }

    std::cout << "[cuda] Ready – entering render loop\n";
    std::cout << "        keys: N style A/B  V debug view  [ ] bands  - = indirect  , . lines\n";

    // ── Render loop ──────────────────────────────────────────────────
    double fpsLastTime  = glfwGetTime();
    int    fpsFrameCount = 0;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Reset accumulation if the scene/camera changed since the last frame.
        // To trigger this from animation code, call markSceneChanged() anywhere.
        if (g_sceneChanged) {
            cudaResetAccumulation(g_windowWidth, g_windowHeight);
            g_sceneChanged = false;
            g_accumSamples = 0;
            g_representOnly = false;   // a reset invalidates present-only
        }

        // 1. CUDA renders into the PBO — or just re-presents after a style knob.
        if (g_representOnly) {
            cudaRepresent(g_windowWidth, g_windowHeight);
            g_representOnly = false;
        } else {
            cudaRender(g_windowWidth, g_windowHeight);
            ++g_accumSamples;
        }

        // FPS counter — update window title once per second
        ++fpsFrameCount;
        double now     = glfwGetTime();
        double elapsed = now - fpsLastTime;
        if (elapsed >= 1.0) {
            double fps = fpsFrameCount / elapsed;
            char title[128];
            snprintf(title, sizeof(title),
                     "BlueTruthEngine  |  %s  |  %.1f fps  |  %.2f ms  |  %d spp",
                     cudaGetStyleMode() == StyleMode::Anime ? "Anime" : "Physical",
                     fps, 1000.0 / fps, g_accumSamples);
            glfwSetWindowTitle(window, title);
            fpsLastTime   = now;
            fpsFrameCount = 0;
        }

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

        // Capture frame for video recording if active.
        // glReadPixels syncs the GPU before reading, so no extra fence needed.
        if (g_recording && g_recordPipe) {
            glReadPixels(0, 0, g_windowWidth, g_windowHeight,
                         GL_RGBA, GL_UNSIGNED_BYTE, g_pixelBuf.data());
            fwrite(g_pixelBuf.data(), 1, g_pixelBuf.size(), g_recordPipe);
        }

        glfwSwapBuffers(window);
    }

    // ── Cleanup ──────────────────────────────────────────────────────
    if (g_recording) stopRecording();
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
