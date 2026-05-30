#pragma once

#include <string>
#include <vector>

// High-level scene description, inspired by Nori but kept minimal and
// engine-agnostic so it can evolve (meshes, materials, emitters, BRDFs, …).

struct MeshDesc {
    std::string filename;
    std::string materialName; // name of a <material> entry

    // Optional: mark this mesh as an emissive area light.
    bool  isEmitter = false;
    float radianceR = 0.0f;
    float radianceG = 0.0f;
    float radianceB = 0.0f;

    // Optional transform for placing this mesh in world/scene space.
    // Rotation is Euler angles in degrees, applied as Scale -> Rx -> Ry -> Rz -> Translate.
    struct TransformDesc {
        float posX = 0.0f;
        float posY = 0.0f;
        float posZ = 0.0f;

        float rotXDegrees = 0.0f;
        float rotYDegrees = 0.0f;
        float rotZDegrees = 0.0f;

        float scaleX = 1.0f;
        float scaleY = 1.0f;
        float scaleZ = 1.0f;
    } transform;
};

struct MaterialDesc {
    std::string name;          // symbolic name, e.g. "white_diffuse"
    std::string albedoTexture; // path to texture, may be empty
    std::string bsdfName;      // name of a <bsdf> entry
};

struct BsdfDesc {
    std::string name;
    std::string type; // "diffuse" | "dielectric" | "mirror" | "microfacet" | "disney"

    // Diffuse params
    float albedoR = 1.0f;
    float albedoG = 1.0f;
    float albedoB = 1.0f;

    // Dielectric params
    float intIOR = 1.5f;
    float extIOR = 1.0f;

    // Microfacet params
    float alpha = 0.1f;

    // Disney params (defaults mirror the Nori Disney constructor)
    float baseColorR = 0.5f;
    float baseColorG = 0.5f;
    float baseColorB = 0.5f;
    float roughness = 0.1f;
    float metallic = 0.0f;
    float specular = 0.5f;
    float specularTransmission = 0.0f;
    float specularTint = 0.0f;
    float sheen = 0.0f;
    float sheenTint = 0.5f;
    float subsurface = 0.0f;
    float anisotropic = 0.0f;
    float clearcoat = 0.0f;
    float clearcoatGloss = 1.0f;
    float eta = 1.5f;
};

struct EmitterDesc {
    std::string name;
    std::string type;          // e.g. "area", "env", "point"
    std::string targetMesh;    // for area lights: mesh/material to bind to
};

struct CameraDesc {
    // Eye position
    float eyeX = 0.0f;
    float eyeY = 0.0f;
    float eyeZ = 0.0f;

    // Look-at target point
    float lookAtX = 0.0f;
    float lookAtY = 0.0f;
    float lookAtZ = 0.0f;

    // Up direction
    float upX = 0.0f;
    float upY = 1.0f;
    float upZ = 0.0f;

    // Vertical field-of-view in degrees
    float fovYDegrees = 45.0f;
};

struct SceneDescription {
    // Core
    std::vector<MeshDesc>     meshes;
    std::vector<MaterialDesc> materials;
    std::vector<BsdfDesc>     bsdfs;
    std::vector<EmitterDesc>  emitters;

    // Camera
    CameraDesc camera;

    // Simple window / film settings
    int windowWidth  = 1280;
    int windowHeight = 720;
};

/// Load a scene description from an XML-like file.
/// The parser is intentionally simple and forgiving; unknown tags are ignored.
SceneDescription loadSceneDescription(const std::string& path);

