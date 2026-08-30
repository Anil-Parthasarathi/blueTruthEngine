#pragma once

#include "render_kernel.h"
#include "scene.h"

#include <cstdint>
#include <string>
#include <vector>

struct GltfImageRGBA {
    std::vector<uint8_t> pixels; // empty = no texture
    int width  = 0;
    int height = 0;
};

struct GltfMaterialSlot {
    BsdfData     bsdf;
    GltfImageRGBA albedo;
    GltfImageRGBA metallicRoughness;
    Float3       emission  = {0.0f, 0.0f, 0.0f};
    bool         isEmitter = false;
};

struct GltfLoadResult {
    std::vector<TriangleData>     triangles;
    std::vector<int>              materialIds; // per triangle, index into materials
    std::vector<GltfMaterialSlot> materials;
};

// Flatten a glTF 2.0 / GLB file into triangles + Disney BSDFs.
// `xmlTransform` is applied after the glTF node graph (same convention as <mesh>).
GltfLoadResult loadGltfModel(const std::string& path, const TransformDesc& xmlTransform);
