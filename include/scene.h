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
    std::string styleName;     // name of a <style> entry, may be empty
};

/// Non-photorealistic style record, referenced by name from a <material>.
///
/// Style is deliberately ORTHOGONAL to BSDF type: the same record can be
/// attached to a diffuse surface, a metallic Disney surface, or a glass one, and
/// the operators that apply it are chosen by which radiance channel the path
/// landed in rather than by the material's type.  That is what makes "metallic
/// anime" and "jewel anime" fall out of the existing Disney BSDF instead of
/// needing new BSDFs.
struct StyleDesc {
    std::string name;
    std::string diffuseRamp;   // path to a 1-D ramp strip, may be empty

    // Body tone
    int   diffuseBands  = 3;
    float bandSoftness  = 0.05f;
    float toneScale     = 1.0f;

    // Anime highlight
    float specThreshold = 0.5f;
    float specSoftness  = 0.05f;
    float specIntensity = 1.0f;

    // Rim light
    float rimStrength = 0.0f;
    float rimPower    = 3.0f;
    float rimColorR   = 1.0f;
    float rimColorG   = 1.0f;
    float rimColorB   = 1.0f;

    // Anime metal.  Bands default to 1 (pass-through) rather than to a banded
    // value: diffuse cel shading is what a <style> is normally for, so banding
    // a surface's reflections and refractions is opt-in.  Otherwise attaching a
    // skin style to a glossy floor would posterize its reflection unasked.
    int   reflectBands   = 1;
    float reflectGain    = 1.0f;
    float reflectTintMix = 0.0f;
    float reflectTintR   = 1.0f;
    float reflectTintG   = 1.0f;
    float reflectTintB   = 1.0f;

    // Anime jewel — pass-through by default, as above.
    int   transmitBands = 1;
    float transmitGain  = 1.0f;
    float chromaShift   = 0.0f;

    // Smooth indirect
    float indirectGain = 1.0f;

    // Outline
    float lineColorR   = 0.0f;
    float lineColorG   = 0.0f;
    float lineColorB   = 0.0f;
    float lineWidth    = 0.0f;   // 0 disables the probe stage for this material
    float lineStrength = 0.0f;
    float outlineNormalThreshold = 0.5f;
    float outlineDepthThreshold  = 0.1f;
};

/// HDRI environment light.  A universal engine feature, not part of the
/// stylization work: it is active in both renderers and both style modes.
struct EnvironmentDesc {
    bool        enabled = false;
    std::string filename;         // .hdr / .exr-ish float image (stbi_loadf)
    float       intensity  = 1.0f;
    float       yawDegrees = 0.0f;

    // "env" | "color" | "texture" — decoupling the visible backdrop from the
    // lighting environment is standard anime art direction (a painted sky over
    // physically sensible light) and costs nothing.
    std::string backgroundMode = "env";
    std::string backgroundFile;
    float       backgroundR = 0.0f;
    float       backgroundG = 0.0f;
    float       backgroundB = 0.0f;
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

/// Spotlight point light with smooth cone falloff.
/// Angles are in degrees (converted to cosines on load, matching the Nori constructor).
struct SpotlightDesc {
    float posX = 0.0f, posY = 0.0f, posZ = 0.0f;
    float dirX = 0.0f, dirY = -1.0f, dirZ = 0.0f; // default points down
    float radianceR = 1.0f, radianceG = 1.0f, radianceB = 1.0f;
    float intensity = 1.0f;
    float innerConeAngle = 0.0f;    // full intensity inside this angle
    float outerConeAngle = 180.0f;  // zero outside this angle
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
    std::vector<MeshDesc>      meshes;
    std::vector<MaterialDesc>  materials;
    std::vector<BsdfDesc>      bsdfs;
    std::vector<StyleDesc>     styles;
    std::vector<EmitterDesc>   emitters;
    std::vector<SpotlightDesc> spotlights;

    // Environment lighting
    EnvironmentDesc environment;

    // Optional scene-level style mode.  "physical" (default) keeps photoreal
    // rendering; "anime" starts the wavefront path in stylized mode.  Either
    // can still be toggled at runtime with N.
    std::string styleMode = "physical";

    // Camera
    CameraDesc camera;

    // Simple window / film settings
    int windowWidth  = 1280;
    int windowHeight = 720;
};

/// Load a scene description from an XML-like file.
/// The parser is intentionally simple and forgiving; unknown tags are ignored.
SceneDescription loadSceneDescription(const std::string& path);

