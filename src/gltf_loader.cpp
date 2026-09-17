// ============================================================================
//  gltf_loader.cpp — flatten glTF 2.0 / GLB into triangles + Disney BSDFs
// ============================================================================

#include "gltf_loader.h"

#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

#include "stb_image.h"

#include <cstdlib>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#include "tiny_gltf.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

glm::mat4 xmlTransformToMat4(const TransformDesc& t)
{
    const float d2r = 3.14159265358979323846f / 180.0f;
    glm::mat4 M(1.0f);
    M = glm::translate(M, glm::vec3(t.posX, t.posY, t.posZ));
    M = glm::rotate(M, t.rotZDegrees * d2r, glm::vec3(0.0f, 0.0f, 1.0f));
    M = glm::rotate(M, t.rotYDegrees * d2r, glm::vec3(0.0f, 1.0f, 0.0f));
    M = glm::rotate(M, t.rotXDegrees * d2r, glm::vec3(1.0f, 0.0f, 0.0f));
    M = glm::scale(M, glm::vec3(t.scaleX, t.scaleY, t.scaleZ));
    return M;
}

glm::mat4 nodeLocalMatrix(const tinygltf::Node& node)
{
    if (node.matrix.size() == 16) {
        float m[16];
        for (int i = 0; i < 16; ++i)
            m[i] = static_cast<float>(node.matrix[static_cast<size_t>(i)]);
        return glm::make_mat4(m);
    }

    glm::mat4 T(1.0f), R(1.0f), S(1.0f);
    if (node.translation.size() == 3) {
        T = glm::translate(glm::mat4(1.0f), glm::vec3(
            static_cast<float>(node.translation[0]),
            static_cast<float>(node.translation[1]),
            static_cast<float>(node.translation[2])));
    }
    if (node.rotation.size() == 4) {
        // glTF quaternion is (x, y, z, w); glm::quat ctor is (w, x, y, z).
        glm::quat q(
            static_cast<float>(node.rotation[3]),
            static_cast<float>(node.rotation[0]),
            static_cast<float>(node.rotation[1]),
            static_cast<float>(node.rotation[2]));
        R = glm::mat4_cast(q);
    }
    if (node.scale.size() == 3) {
        S = glm::scale(glm::mat4(1.0f), glm::vec3(
            static_cast<float>(node.scale[0]),
            static_cast<float>(node.scale[1]),
            static_cast<float>(node.scale[2])));
    }
    return T * R * S;
}

int typeComponentCount(int type)
{
    switch (type) {
        case TINYGLTF_TYPE_SCALAR: return 1;
        case TINYGLTF_TYPE_VEC2:   return 2;
        case TINYGLTF_TYPE_VEC3:   return 3;
        case TINYGLTF_TYPE_VEC4:   return 4;
        default: return 0;
    }
}

int componentByteSize(int componentType)
{
    switch (componentType) {
        case TINYGLTF_COMPONENT_TYPE_BYTE:
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  return 1;
        case TINYGLTF_COMPONENT_TYPE_SHORT:
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: return 2;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        case TINYGLTF_COMPONENT_TYPE_FLOAT:          return 4;
        default: return 0;
    }
}

float readFloatComponent(const uint8_t* p, int componentType, bool normalized)
{
    switch (componentType) {
        case TINYGLTF_COMPONENT_TYPE_BYTE: {
            const int8_t v = static_cast<int8_t>(*p);
            return normalized ? std::max(static_cast<float>(v) / 127.0f, -1.0f)
                              : static_cast<float>(v);
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return normalized ? static_cast<float>(*p) / 255.0f
                              : static_cast<float>(*p);
        case TINYGLTF_COMPONENT_TYPE_SHORT: {
            int16_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return normalized ? std::max(static_cast<float>(v) / 32767.0f, -1.0f)
                              : static_cast<float>(v);
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
            uint16_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return normalized ? static_cast<float>(v) / 65535.0f
                              : static_cast<float>(v);
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
            uint32_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return static_cast<float>(v);
        }
        case TINYGLTF_COMPONENT_TYPE_FLOAT: {
            float v = 0.0f;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        default:
            return 0.0f;
    }
}

uint32_t readIndexComponent(const uint8_t* p, int componentType)
{
    switch (componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            return *p;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
            uint16_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
            uint32_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        default:
            return 0;
    }
}

bool accessorPointer(const tinygltf::Model& model,
                     int accessorIndex,
                     const uint8_t*& outPtr,
                     size_t& outStride,
                     size_t& outCount,
                     const tinygltf::Accessor** outAcc)
{
    if (accessorIndex < 0 || accessorIndex >= static_cast<int>(model.accessors.size()))
        return false;
    const tinygltf::Accessor& acc = model.accessors[static_cast<size_t>(accessorIndex)];
    if (acc.bufferView < 0 || acc.bufferView >= static_cast<int>(model.bufferViews.size()))
        return false;
    const tinygltf::BufferView& view = model.bufferViews[static_cast<size_t>(acc.bufferView)];
    if (view.buffer < 0 || view.buffer >= static_cast<int>(model.buffers.size()))
        return false;
    const tinygltf::Buffer& buf = model.buffers[static_cast<size_t>(view.buffer)];

    const int nComp = typeComponentCount(acc.type);
    const int cSize = componentByteSize(acc.componentType);
    if (nComp <= 0 || cSize <= 0)
        return false;

    const size_t elemSize = static_cast<size_t>(nComp * cSize);
    const size_t stride = view.byteStride > 0 ? static_cast<size_t>(view.byteStride) : elemSize;
    const size_t offset = static_cast<size_t>(view.byteOffset) + static_cast<size_t>(acc.byteOffset);
    if (offset + (acc.count ? (acc.count - 1) * stride + elemSize : 0) > buf.data.size())
        return false;

    outPtr    = buf.data.data() + offset;
    outStride = stride;
    outCount  = static_cast<size_t>(acc.count);
    if (outAcc) *outAcc = &acc;
    return true;
}

std::vector<glm::vec3> readVec3Attr(const tinygltf::Model& model, int accessorIndex)
{
    std::vector<glm::vec3> out;
    const uint8_t* ptr = nullptr;
    size_t stride = 0, count = 0;
    const tinygltf::Accessor* acc = nullptr;
    if (!accessorPointer(model, accessorIndex, ptr, stride, count, &acc))
        return out;
    if (typeComponentCount(acc->type) < 3)
        return out;

    out.resize(count);
    const int cSize = componentByteSize(acc->componentType);
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* e = ptr + i * stride;
        out[i] = glm::vec3(
            readFloatComponent(e, acc->componentType, acc->normalized),
            readFloatComponent(e + cSize, acc->componentType, acc->normalized),
            readFloatComponent(e + 2 * cSize, acc->componentType, acc->normalized));
    }
    return out;
}

std::vector<glm::vec2> readVec2Attr(const tinygltf::Model& model, int accessorIndex)
{
    std::vector<glm::vec2> out;
    const uint8_t* ptr = nullptr;
    size_t stride = 0, count = 0;
    const tinygltf::Accessor* acc = nullptr;
    if (!accessorPointer(model, accessorIndex, ptr, stride, count, &acc))
        return out;
    if (typeComponentCount(acc->type) < 2)
        return out;

    out.resize(count);
    const int cSize = componentByteSize(acc->componentType);
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* e = ptr + i * stride;
        out[i] = glm::vec2(
            readFloatComponent(e, acc->componentType, acc->normalized),
            readFloatComponent(e + cSize, acc->componentType, acc->normalized));
    }
    return out;
}

std::vector<uint32_t> readIndices(const tinygltf::Model& model, int accessorIndex, size_t vertexCount)
{
    std::vector<uint32_t> out;
    if (accessorIndex < 0) {
        out.resize(vertexCount);
        for (size_t i = 0; i < vertexCount; ++i)
            out[i] = static_cast<uint32_t>(i);
        return out;
    }

    const uint8_t* ptr = nullptr;
    size_t stride = 0, count = 0;
    const tinygltf::Accessor* acc = nullptr;
    if (!accessorPointer(model, accessorIndex, ptr, stride, count, &acc))
        return out;

    out.resize(count);
    for (size_t i = 0; i < count; ++i)
        out[i] = readIndexComponent(ptr + i * stride, acc->componentType);
    return out;
}

std::vector<uint32_t> triangulate(const std::vector<uint32_t>& indices, int mode)
{
    std::vector<uint32_t> tris;
    if (mode == TINYGLTF_MODE_TRIANGLES) {
        const size_t n = indices.size() - indices.size() % 3;
        tris.assign(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(n));
        return tris;
    }
    if (mode == TINYGLTF_MODE_TRIANGLE_FAN && indices.size() >= 3) {
        for (size_t i = 1; i + 1 < indices.size(); ++i) {
            tris.push_back(indices[0]);
            tris.push_back(indices[i]);
            tris.push_back(indices[i + 1]);
        }
        return tris;
    }
    if (mode == TINYGLTF_MODE_TRIANGLE_STRIP && indices.size() >= 3) {
        for (size_t i = 0; i + 2 < indices.size(); ++i) {
            if (i % 2 == 0) {
                tris.push_back(indices[i]);
                tris.push_back(indices[i + 1]);
                tris.push_back(indices[i + 2]);
            } else {
                tris.push_back(indices[i + 1]);
                tris.push_back(indices[i]);
                tris.push_back(indices[i + 2]);
            }
        }
        return tris;
    }
    return tris;
}

double extNumber(const tinygltf::Value& ext, const char* key, double fallback)
{
    if (!ext.IsObject() || !ext.Has(key))
        return fallback;
    const tinygltf::Value& v = ext.Get(key);
    if (v.IsNumber())
        return v.GetNumberAsDouble();
    return fallback;
}

GltfImageRGBA imageToRGBA(const tinygltf::Image& img)
{
    GltfImageRGBA out;
    if (img.width <= 0 || img.height <= 0 || img.image.empty())
        return out;

    out.width  = img.width;
    out.height = img.height;
    out.pixels.resize(static_cast<size_t>(img.width) * static_cast<size_t>(img.height) * 4);

    const int srcComp = img.component > 0 ? img.component : 4;
    const size_t nPix = static_cast<size_t>(img.width) * static_cast<size_t>(img.height);
    for (size_t i = 0; i < nPix; ++i) {
        const uint8_t* s = img.image.data() + i * static_cast<size_t>(srcComp);
        uint8_t r = s[0];
        uint8_t g = srcComp > 1 ? s[1] : r;
        uint8_t b = srcComp > 2 ? s[2] : r;
        uint8_t a = srcComp > 3 ? s[3] : 255;
        out.pixels[i * 4 + 0] = r;
        out.pixels[i * 4 + 1] = g;
        out.pixels[i * 4 + 2] = b;
        out.pixels[i * 4 + 3] = a;
    }
    return out;
}

GltfImageRGBA textureImage(const tinygltf::Model& model,
                           const std::vector<GltfImageRGBA>& images,
                           int textureIndex)
{
    if (textureIndex < 0 || textureIndex >= static_cast<int>(model.textures.size()))
        return {};
    const int src = model.textures[static_cast<size_t>(textureIndex)].source;
    if (src < 0 || src >= static_cast<int>(images.size()))
        return {};
    return images[static_cast<size_t>(src)];
}

void multiplyAlbedoByFactor(GltfImageRGBA& img, const glm::vec4& factor)
{
    if (img.pixels.empty())
        return;
    const size_t nPix = static_cast<size_t>(img.width) * static_cast<size_t>(img.height);
    for (size_t i = 0; i < nPix; ++i) {
        img.pixels[i * 4 + 0] = static_cast<uint8_t>(std::clamp(
            (img.pixels[i * 4 + 0] / 255.0f) * factor.r * 255.0f + 0.5f, 0.0f, 255.0f));
        img.pixels[i * 4 + 1] = static_cast<uint8_t>(std::clamp(
            (img.pixels[i * 4 + 1] / 255.0f) * factor.g * 255.0f + 0.5f, 0.0f, 255.0f));
        img.pixels[i * 4 + 2] = static_cast<uint8_t>(std::clamp(
            (img.pixels[i * 4 + 2] / 255.0f) * factor.b * 255.0f + 0.5f, 0.0f, 255.0f));
        img.pixels[i * 4 + 3] = static_cast<uint8_t>(std::clamp(
            (img.pixels[i * 4 + 3] / 255.0f) * factor.a * 255.0f + 0.5f, 0.0f, 255.0f));
    }
}

GltfMaterialSlot disneyFromGltf(const tinygltf::Material& mat,
                                const tinygltf::Model& model,
                                const std::vector<GltfImageRGBA>& images)
{
    GltfMaterialSlot slot{};
    const auto& pbr = mat.pbrMetallicRoughness;

    glm::vec4 baseColor(1.0f);
    if (pbr.baseColorFactor.size() >= 3) {
        baseColor.r = static_cast<float>(pbr.baseColorFactor[0]);
        baseColor.g = static_cast<float>(pbr.baseColorFactor[1]);
        baseColor.b = static_cast<float>(pbr.baseColorFactor[2]);
        baseColor.a = pbr.baseColorFactor.size() >= 4
                          ? static_cast<float>(pbr.baseColorFactor[3])
                          : 1.0f;
    }
    const float metallic  = static_cast<float>(pbr.metallicFactor);
    const float roughness = static_cast<float>(pbr.roughnessFactor);

    slot.albedo = textureImage(model, images, pbr.baseColorTexture.index);
    if (!slot.albedo.pixels.empty())
        multiplyAlbedoByFactor(slot.albedo, baseColor);

    slot.metallicRoughness = textureImage(model, images, pbr.metallicRoughnessTexture.index);

    float specular             = 0.5f;
    float specularTransmission = 0.0f;
    float specularTint         = 0.0f;
    float sheen                = 0.0f;
    float sheenTint            = 0.5f;
    float subsurface           = 0.0f;
    float anisotropic          = 0.0f;
    float clearcoat            = 0.0f;
    float clearcoatGloss       = 1.0f;
    float eta                  = 1.5f;

    auto extIt = mat.extensions.find("KHR_materials_ior");
    if (extIt != mat.extensions.end())
        eta = static_cast<float>(extNumber(extIt->second, "ior", eta));

    extIt = mat.extensions.find("KHR_materials_transmission");
    if (extIt != mat.extensions.end())
        specularTransmission = static_cast<float>(
            extNumber(extIt->second, "transmissionFactor", 0.0));

    extIt = mat.extensions.find("KHR_materials_clearcoat");
    if (extIt != mat.extensions.end()) {
        clearcoat = static_cast<float>(extNumber(extIt->second, "clearcoatFactor", 0.0));
        const float ccRough = static_cast<float>(
            extNumber(extIt->second, "clearcoatRoughnessFactor", 0.0));
        clearcoatGloss = 1.0f - ccRough;
    }

    extIt = mat.extensions.find("KHR_materials_sheen");
    if (extIt != mat.extensions.end() && extIt->second.IsObject() &&
        extIt->second.Has("sheenColorFactor")) {
        const tinygltf::Value& sc = extIt->second.Get("sheenColorFactor");
        if (sc.IsArray() && sc.ArrayLen() >= 3) {
            const float r = static_cast<float>(sc.Get(0).GetNumberAsDouble());
            const float g = static_cast<float>(sc.Get(1).GetNumberAsDouble());
            const float b = static_cast<float>(sc.Get(2).GetNumberAsDouble());
            sheen     = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            sheenTint = 1.0f;
        }
    }

    glm::vec3 emissive(0.0f);
    if (mat.emissiveFactor.size() >= 3) {
        emissive.r = static_cast<float>(mat.emissiveFactor[0]);
        emissive.g = static_cast<float>(mat.emissiveFactor[1]);
        emissive.b = static_cast<float>(mat.emissiveFactor[2]);
    }
    extIt = mat.extensions.find("KHR_materials_emissive_strength");
    if (extIt != mat.extensions.end()) {
        const float strength = static_cast<float>(
            extNumber(extIt->second, "emissiveStrength", 1.0));
        emissive *= strength;
    }
    slot.emission  = {emissive.r, emissive.g, emissive.b};
    slot.isEmitter = (emissive.r + emissive.g + emissive.b) > 1e-4f;

    // Unlit → Disney with metallic 0 and high roughness so it reads as flat paint.
    const bool unlit = mat.extensions.find("KHR_materials_unlit") != mat.extensions.end();

    const float outMetallic  = unlit ? 0.0f : metallic;
    const float outRoughness = unlit ? 1.0f : roughness;
    const glm::vec3 outColor = slot.albedo.pixels.empty()
                                   ? glm::vec3(baseColor)
                                   : glm::vec3(1.0f); // factor baked into the texture

    BsdfData d{};
    d.type = BSDF_Disney;
    d.p0   = {outColor.r, outColor.g, outColor.b, outRoughness};
    d.p1   = {outMetallic, specular, specularTransmission, specularTint};
    d.p2   = {sheen, sheenTint, subsurface, anisotropic};
    d.p3   = {clearcoat, clearcoatGloss, eta, 0.0f};
    slot.bsdf = d;
    return slot;
}

GltfMaterialSlot defaultMaterial()
{
    GltfMaterialSlot slot{};
    BsdfData d{};
    d.type = BSDF_Disney;
    d.p0   = {0.8f, 0.8f, 0.8f, 0.5f};
    d.p1   = {0.0f, 0.5f, 0.0f, 0.0f};
    d.p2   = {0.0f, 0.5f, 0.0f, 0.0f};
    d.p3   = {0.0f, 1.0f, 1.5f, 0.0f};
    slot.bsdf = d;
    return slot;
}

Float3 toFloat3(const glm::vec3& v)
{
    return {v.x, v.y, v.z};
}

Float3 geometricNormal(const Float3& v0, const Float3& v1, const Float3& v2)
{
    const float e1x = v1.x - v0.x, e1y = v1.y - v0.y, e1z = v1.z - v0.z;
    const float e2x = v2.x - v0.x, e2y = v2.y - v0.y, e2z = v2.z - v0.z;
    Float3 n = {e1y * e2z - e1z * e2y,
                e1z * e2x - e1x * e2z,
                e1x * e2y - e1y * e2x};
    const float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len > 1e-20f) { n.x /= len; n.y /= len; n.z /= len; }
    return n;
}

void emitPrimitive(const tinygltf::Model& model,
                   const tinygltf::Primitive& prim,
                   const glm::mat4& world,
                   int materialIndex,
                   GltfLoadResult& result)
{
    auto posIt = prim.attributes.find("POSITION");
    if (posIt == prim.attributes.end())
        return;

    std::vector<glm::vec3> positions = readVec3Attr(model, posIt->second);
    if (positions.empty())
        return;

    std::vector<glm::vec3> normals;
    auto nIt = prim.attributes.find("NORMAL");
    if (nIt != prim.attributes.end())
        normals = readVec3Attr(model, nIt->second);

    std::vector<glm::vec2> uvs;
    auto uvIt = prim.attributes.find("TEXCOORD_0");
    if (uvIt != prim.attributes.end())
        uvs = readVec2Attr(model, uvIt->second);

    const int texCoord = (prim.material >= 0 &&
                          prim.material < static_cast<int>(model.materials.size()))
                             ? model.materials[static_cast<size_t>(prim.material)]
                                   .pbrMetallicRoughness.baseColorTexture.texCoord
                             : 0;
    if (texCoord != 0)
        std::cerr << "[gltf] TEXCOORD_" << texCoord
                  << " requested; using TEXCOORD_0\n";

    std::vector<uint32_t> indices = readIndices(model, prim.indices, positions.size());
    const int mode = prim.mode < 0 ? TINYGLTF_MODE_TRIANGLES : prim.mode;
    if (mode != TINYGLTF_MODE_TRIANGLES &&
        mode != TINYGLTF_MODE_TRIANGLE_STRIP &&
        mode != TINYGLTF_MODE_TRIANGLE_FAN) {
        std::cerr << "[gltf] Skipping non-triangle primitive (mode " << mode << ")\n";
        return;
    }
    indices = triangulate(indices, mode);

    const glm::mat3 normalMat = glm::transpose(glm::inverse(glm::mat3(world)));

    auto xformP = [&](const glm::vec3& p) -> Float3 {
        const glm::vec4 w = world * glm::vec4(p, 1.0f);
        return {w.x, w.y, w.z};
    };
    auto xformN = [&](const glm::vec3& n) -> Float3 {
        const glm::vec3 wn = glm::normalize(normalMat * n);
        return {wn.x, wn.y, wn.z};
    };

    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        const uint32_t i0 = indices[i + 0];
        const uint32_t i1 = indices[i + 1];
        const uint32_t i2 = indices[i + 2];
        if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size())
            continue;

        TriangleData t{};
        t.v0 = xformP(positions[i0]);
        t.v1 = xformP(positions[i1]);
        t.v2 = xformP(positions[i2]);

        const bool hasN = i0 < normals.size() && i1 < normals.size() && i2 < normals.size();
        if (hasN) {
            t.n0 = xformN(normals[i0]);
            t.n1 = xformN(normals[i1]);
            t.n2 = xformN(normals[i2]);
        } else {
            const Float3 gn = geometricNormal(t.v0, t.v1, t.v2);
            t.n0 = t.n1 = t.n2 = gn;
        }

        auto uvAt = [&](uint32_t idx) -> Float2 {
            if (idx < uvs.size())
                return {uvs[idx].x, uvs[idx].y};
            return {0.0f, 0.0f};
        };
        t.uv0 = uvAt(i0);
        t.uv1 = uvAt(i1);
        t.uv2 = uvAt(i2);

        result.triangles.push_back(t);
        result.materialIds.push_back(materialIndex);
    }
}

std::string asciiLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool skipBackdropNode(const tinygltf::Node& node)
{
    const std::string lowerName = asciiLower(node.name);
    return lowerName == "background" ||
           lowerName.find("backgroundgradient") != std::string::npos;
}

bool skipToonOrBackdropPrim(const tinygltf::Model& model, const tinygltf::Primitive& prim)
{
    if (prim.material < 0 || prim.material >= static_cast<int>(model.materials.size()))
        return false;
    const std::string matName = asciiLower(model.materials[static_cast<size_t>(prim.material)].name);
    return matName.find("outline") != std::string::npos ||
           matName == "backgroundgradient";
}

uint64_t primKey(int nodeIndex, int primIndex)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(nodeIndex)) << 32) |
           static_cast<uint32_t>(primIndex);
}

bool primWorldBounds(const tinygltf::Model& model,
                     const tinygltf::Primitive& prim,
                     const glm::mat4& world,
                     glm::vec3& center,
                     float& extent)
{
    auto posIt = prim.attributes.find("POSITION");
    if (posIt == prim.attributes.end())
        return false;
    const int accIndex = posIt->second;
    if (accIndex < 0 || accIndex >= static_cast<int>(model.accessors.size()))
        return false;
    const tinygltf::Accessor& acc = model.accessors[static_cast<size_t>(accIndex)];
    if (acc.minValues.size() < 3 || acc.maxValues.size() < 3)
        return false;

    const glm::vec3 localMin(static_cast<float>(acc.minValues[0]),
                             static_cast<float>(acc.minValues[1]),
                             static_cast<float>(acc.minValues[2]));
    const glm::vec3 localMax(static_cast<float>(acc.maxValues[0]),
                             static_cast<float>(acc.maxValues[1]),
                             static_cast<float>(acc.maxValues[2]));

    glm::vec3 wmin(1e30f), wmax(-1e30f);
    for (int i = 0; i < 8; ++i) {
        const glm::vec3 p((i & 1) ? localMax.x : localMin.x,
                          (i & 2) ? localMax.y : localMin.y,
                          (i & 4) ? localMax.z : localMin.z);
        const glm::vec3 w = glm::vec3(world * glm::vec4(p, 1.0f));
        wmin = glm::min(wmin, w);
        wmax = glm::max(wmax, w);
    }
    center = 0.5f * (wmin + wmax);
    const glm::vec3 size = wmax - wmin;
    extent = std::max(size.x, std::max(size.y, size.z));
    return extent > 0.0f;
}

struct PrimStat {
    uint64_t  key    = 0;
    glm::vec3 center = {0.0f, 0.0f, 0.0f};
    float     extent = 0.0f;
};

void gatherPrimStats(const tinygltf::Model& model,
                     int nodeIndex,
                     const glm::mat4& parent,
                     std::vector<PrimStat>& stats)
{
    if (nodeIndex < 0 || nodeIndex >= static_cast<int>(model.nodes.size()))
        return;
    const tinygltf::Node& node = model.nodes[static_cast<size_t>(nodeIndex)];
    if (skipBackdropNode(node))
        return;

    const glm::mat4 world = parent * nodeLocalMatrix(node);
    if (node.mesh >= 0 && node.mesh < static_cast<int>(model.meshes.size())) {
        const tinygltf::Mesh& mesh = model.meshes[static_cast<size_t>(node.mesh)];
        for (int pi = 0; pi < static_cast<int>(mesh.primitives.size()); ++pi) {
            const auto& prim = mesh.primitives[static_cast<size_t>(pi)];
            if (skipToonOrBackdropPrim(model, prim))
                continue;
            PrimStat s;
            s.key = primKey(nodeIndex, pi);
            if (primWorldBounds(model, prim, world, s.center, s.extent))
                stats.push_back(s);
        }
    }
    for (int child : node.children)
        gatherPrimStats(model, child, world, stats);
}

// VRChat / toon avatars often include a detached cape or collider orders of
// magnitude larger than the body. Drop those so scale-to-fit stays on the character.
std::unordered_set<uint64_t> outlierPrims(const std::vector<PrimStat>& stats)
{
    std::unordered_set<uint64_t> skip;
    if (stats.size() < 4)
        return skip;

    std::vector<float> extents;
    extents.reserve(stats.size());
    std::vector<float> cx, cy, cz;
    cx.reserve(stats.size());
    cy.reserve(stats.size());
    cz.reserve(stats.size());
    for (const auto& s : stats) {
        extents.push_back(s.extent);
        cx.push_back(s.center.x);
        cy.push_back(s.center.y);
        cz.push_back(s.center.z);
    }
    auto median = [](std::vector<float> v) -> float {
        const size_t n = v.size();
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(n / 2), v.end());
        return v[n / 2];
    };
    const float medExtent = median(std::move(extents));
    const glm::vec3 medCenter(median(std::move(cx)), median(std::move(cy)), median(std::move(cz)));
    if (medExtent <= 1e-8f)
        return skip;

    const float maxExtent = 12.0f * medExtent;
    const float maxDist   = 10.0f * medExtent;
    for (const auto& s : stats) {
        if (s.extent > maxExtent || glm::length(s.center - medCenter) > maxDist)
            skip.insert(s.key);
    }
    return skip;
}

void walkNode(const tinygltf::Model& model,
              int nodeIndex,
              const glm::mat4& parent,
              const std::vector<int>& gltfMatToSlot,
              int defaultSlot,
              const std::unordered_set<uint64_t>& skipPrims,
              GltfLoadResult& result)
{
    if (nodeIndex < 0 || nodeIndex >= static_cast<int>(model.nodes.size()))
        return;
    const tinygltf::Node& node = model.nodes[static_cast<size_t>(nodeIndex)];

    // Sketchfab often ships a studio backdrop and inverted-hull toon outline.
    // Both break a path-traced scene (huge gradient cube / solid black shell).
    if (skipBackdropNode(node))
        return;

    const glm::mat4 world = parent * nodeLocalMatrix(node);

    if (node.mesh >= 0 && node.mesh < static_cast<int>(model.meshes.size())) {
        const tinygltf::Mesh& mesh = model.meshes[static_cast<size_t>(node.mesh)];
        for (int pi = 0; pi < static_cast<int>(mesh.primitives.size()); ++pi) {
            const auto& prim = mesh.primitives[static_cast<size_t>(pi)];
            if (skipToonOrBackdropPrim(model, prim))
                continue;
            if (skipPrims.count(primKey(nodeIndex, pi)) != 0)
                continue;
            int slot = defaultSlot;
            if (prim.material >= 0 &&
                prim.material < static_cast<int>(gltfMatToSlot.size()))
                slot = gltfMatToSlot[static_cast<size_t>(prim.material)];
            emitPrimitive(model, prim, world, slot, result);
        }
    }

    for (int child : node.children)
        walkNode(model, child, world, gltfMatToSlot, defaultSlot, skipPrims, result);
}

} // namespace

GltfLoadResult loadGltfModel(const std::string& path, const TransformDesc& xmlTransform)
{
    tinygltf::Model    model;
    tinygltf::TinyGLTF loader;
    std::string        err, warn;

    loader.SetStoreOriginalJSONForExtrasAndExtensions(false);

    const bool isGlb = path.size() >= 4 &&
                       (path.compare(path.size() - 4, 4, ".glb") == 0 ||
                        path.compare(path.size() - 4, 4, ".GLB") == 0);

    const bool ok = isGlb ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
                          : loader.LoadASCIIFromFile(&model, &err, &warn, path);

    if (!warn.empty())
        std::cerr << "[gltf warn] " << warn << "\n";
    if (!ok) {
        std::cerr << "[gltf] Failed to load \"" << path << "\"\n";
        if (!err.empty())
            std::cerr << "        " << err << "\n";
        std::exit(EXIT_FAILURE);
    }

    GltfLoadResult result;

    std::vector<GltfImageRGBA> images(model.images.size());
    for (size_t i = 0; i < model.images.size(); ++i)
        images[i] = imageToRGBA(model.images[i]);

    std::vector<int> gltfMatToSlot(model.materials.size(), 0);
    for (size_t i = 0; i < model.materials.size(); ++i) {
        gltfMatToSlot[i] = static_cast<int>(result.materials.size());
        result.materials.push_back(disneyFromGltf(model.materials[i], model, images));
    }
    const int defaultSlot = static_cast<int>(result.materials.size());
    result.materials.push_back(defaultMaterial());

    const glm::mat4 xmlMat = xmlTransformToMat4(xmlTransform);

    std::vector<int> roots;
    int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (!model.scenes.empty() && sceneIndex < static_cast<int>(model.scenes.size())) {
        roots = model.scenes[static_cast<size_t>(sceneIndex)].nodes;
    } else {
        for (int i = 0; i < static_cast<int>(model.nodes.size()); ++i)
            roots.push_back(i);
    }

    std::vector<PrimStat> stats;
    for (int root : roots)
        gatherPrimStats(model, root, xmlMat, stats);
    const std::unordered_set<uint64_t> skipPrims = outlierPrims(stats);
    if (!skipPrims.empty())
        std::cout << "[gltf] Skipping " << skipPrims.size()
                  << " detached/outlier mesh(es) in \"" << path << "\"\n";

    for (int root : roots)
        walkNode(model, root, xmlMat, gltfMatToSlot, defaultSlot, skipPrims, result);

    // Drop the unused default material if every primitive had a real one.
    if (!result.materialIds.empty()) {
        const bool usedDefault = std::find(result.materialIds.begin(),
                                           result.materialIds.end(),
                                           defaultSlot) != result.materialIds.end();
        if (!usedDefault && defaultSlot == static_cast<int>(result.materials.size()) - 1)
            result.materials.pop_back();
    }

    if (result.triangles.empty()) {
        std::cerr << "[gltf] No triangles in \"" << path << "\"\n";
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[gltf] Loaded \"" << path << "\" — "
              << result.triangles.size() << " triangles, "
              << result.materials.size() << " materials" << std::endl;
    return result;
}
