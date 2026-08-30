#include "scene.h"

#include <fstream>
#include <iostream>
#include <cstdlib>

namespace {

std::string readFileToString(const std::string& path)
{
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        std::cerr << "[scene] Could not open scene file \"" << path
                  << "\" – using defaults.\n";
        return {};
    }
    std::string contents;
    ifs.seekg(0, std::ios::end);
    contents.resize(static_cast<size_t>(ifs.tellg()));
    ifs.seekg(0, std::ios::beg);
    ifs.read(&contents[0], static_cast<std::streamsize>(contents.size()));
    return contents;
}

std::string extractAttribute(const std::string& src,
                             const std::string& tagName,
                             const std::string& attrName,
                             size_t searchFrom = 0,
                             size_t* outTagEnd = nullptr)
{
    const std::string tagOpen = "<" + tagName;
    auto tagPos = src.find(tagOpen, searchFrom);
    if (tagPos == std::string::npos) return {};

    auto tagEnd = src.find('>', tagPos);
    if (tagEnd == std::string::npos) return {};
    if (outTagEnd) *outTagEnd = tagEnd;

    std::string tagContent = src.substr(tagPos, tagEnd - tagPos);

    const std::string key = attrName + "=\"";
    auto attrPos = tagContent.find(key);
    if (attrPos == std::string::npos) return {};
    attrPos += key.size();

    auto endQuote = tagContent.find('"', attrPos);
    if (endQuote == std::string::npos) return {};

    return tagContent.substr(attrPos, endQuote - attrPos);
}

std::string extractAttributeFromTag(const std::string& tagContent,
                                      const std::string& attrName)
{
    const std::string key = attrName + "=\"";
    auto attrPos = tagContent.find(key);
    if (attrPos == std::string::npos) return {};
    attrPos += key.size();
    auto endQuote = tagContent.find('"', attrPos);
    if (endQuote == std::string::npos) return {};
    return tagContent.substr(attrPos, endQuote - attrPos);
}

bool extractNextTag(const std::string& src,
                    const std::string& tagName,
                    size_t& searchFrom,
                    std::string& outTagContent)
{
    const std::string tagOpen = "<" + tagName;
    auto tagPos = src.find(tagOpen, searchFrom);
    if (tagPos == std::string::npos) return false;

    auto tagEnd = src.find('>', tagPos);
    if (tagEnd == std::string::npos) return false;

    outTagContent = src.substr(tagPos, tagEnd - tagPos);
    searchFrom = tagEnd + 1;
    return true;
}

} // namespace

SceneDescription loadSceneDescription(const std::string& path)
{
    SceneDescription desc;
    std::string xml = readFileToString(path);
    if (xml.empty()) {
        std::cerr << "[scene] Scene file \"" << path
                  << "\" is empty or unreadable. Please provide a valid scene.\n";
        std::exit(EXIT_FAILURE);
    }

    {
        std::string mode = extractAttribute(xml, "scene", "style");
        if (!mode.empty()) desc.styleMode = mode;
        if (desc.styleMode != "physical" && desc.styleMode != "anime") {
            std::cerr << "[scene] <scene style=\"" << desc.styleMode
                      << "\"> is not \"physical\" or \"anime\" — using physical.\n";
            desc.styleMode = "physical";
        }
    }

    // --- BSDFs --------------------------------------------------------------
    {
        size_t searchFrom = 0;
        while (true) {
            std::string tagContent;
            if (!extractNextTag(xml, "bsdf", searchFrom, tagContent))
                break;

            BsdfDesc b;
            b.name = extractAttributeFromTag(tagContent, "name");
            b.type = extractAttributeFromTag(tagContent, "type");

            if (b.name.empty() || b.type.empty()) {
                std::cerr << "[scene] <bsdf> missing required `name` and/or `type`.\n";
                std::exit(EXIT_FAILURE);
            }

            auto readFOpt = [&](const char* attrName, float& out) {
                std::string s = extractAttributeFromTag(tagContent, attrName);
                if (s.empty()) return;
                try { out = std::stof(s); } catch (...) {
                    std::cerr << "[scene] <bsdf name=\"" << b.name
                              << "\"> invalid float attribute `" << attrName
                              << "`: \"" << s << "\"\n";
                    std::exit(EXIT_FAILURE);
                }
            };

            if (b.type == "diffuse") {
                readFOpt("albedoR", b.albedoR);
                readFOpt("albedoG", b.albedoG);
                readFOpt("albedoB", b.albedoB);
            } else if (b.type == "dielectric") {
                readFOpt("intIOR", b.intIOR);
                readFOpt("extIOR", b.extIOR);
            } else if (b.type == "mirror") {
                readFOpt("albedoR", b.albedoR);
                readFOpt("albedoG", b.albedoG);
                readFOpt("albedoB", b.albedoB);
            } else if (b.type == "microfacet") {
                readFOpt("albedoR", b.albedoR);
                readFOpt("albedoG", b.albedoG);
                readFOpt("albedoB", b.albedoB);
                readFOpt("alpha", b.alpha);
                readFOpt("intIOR", b.intIOR);
                readFOpt("extIOR", b.extIOR);
            } else if (b.type == "disney") {
                readFOpt("base_colorR", b.baseColorR);
                readFOpt("base_colorG", b.baseColorG);
                readFOpt("base_colorB", b.baseColorB);
                readFOpt("roughness", b.roughness);
                readFOpt("metallic", b.metallic);
                readFOpt("specular", b.specular);
                readFOpt("specular_transmission", b.specularTransmission);
                readFOpt("specular_tint", b.specularTint);
                readFOpt("sheen", b.sheen);
                readFOpt("sheen_tint", b.sheenTint);
                readFOpt("subsurface", b.subsurface);
                readFOpt("anisotropic", b.anisotropic);
                readFOpt("clearcoat", b.clearcoat);
                readFOpt("clearcoat_gloss", b.clearcoatGloss);
                readFOpt("eta", b.eta);
            } else {
                std::cerr << "[scene] <bsdf name=\"" << b.name
                          << "\"> unsupported type: \"" << b.type
                          << "\" (expected diffuse|dielectric|mirror|microfacet|disney)\n";
                std::exit(EXIT_FAILURE);
            }

            desc.bsdfs.push_back(b);
        }
    }

    // --- Styles -------------------------------------------------------------
    //  A <style> is independent of BSDF type; a material points at one by name.
    {
        size_t searchFrom = 0;
        while (true) {
            std::string tagContent;
            if (!extractNextTag(xml, "style", searchFrom, tagContent))
                break;

            StyleDesc s;
            s.name        = extractAttributeFromTag(tagContent, "name");
            s.diffuseRamp = extractAttributeFromTag(tagContent, "diffuse_ramp");

            if (s.name.empty()) {
                std::cerr << "[scene] <style> missing required attribute `name`.\n";
                std::exit(EXIT_FAILURE);
            }

            auto readFOpt = [&](const char* attrName, float& out) {
                std::string v = extractAttributeFromTag(tagContent, attrName);
                if (v.empty()) return;
                try { out = std::stof(v); } catch (...) {
                    std::cerr << "[scene] <style name=\"" << s.name
                              << "\"> invalid float attribute `" << attrName
                              << "`: \"" << v << "\"\n";
                    std::exit(EXIT_FAILURE);
                }
            };
            auto readIOpt = [&](const char* attrName, int& out) {
                std::string v = extractAttributeFromTag(tagContent, attrName);
                if (v.empty()) return;
                try { out = std::stoi(v); } catch (...) {
                    std::cerr << "[scene] <style name=\"" << s.name
                              << "\"> invalid int attribute `" << attrName
                              << "`: \"" << v << "\"\n";
                    std::exit(EXIT_FAILURE);
                }
            };

            readIOpt("diffuse_bands",  s.diffuseBands);
            readFOpt("band_softness",  s.bandSoftness);
            readFOpt("tone_scale",     s.toneScale);

            readFOpt("spec_threshold", s.specThreshold);
            readFOpt("spec_softness",  s.specSoftness);
            readFOpt("spec_intensity", s.specIntensity);

            readFOpt("rim_strength", s.rimStrength);
            readFOpt("rim_power",    s.rimPower);
            readFOpt("rim_colorR",   s.rimColorR);
            readFOpt("rim_colorG",   s.rimColorG);
            readFOpt("rim_colorB",   s.rimColorB);

            readIOpt("reflect_bands",    s.reflectBands);
            readFOpt("reflect_gain",     s.reflectGain);
            readFOpt("reflect_tint_mix", s.reflectTintMix);
            readFOpt("reflect_tintR",    s.reflectTintR);
            readFOpt("reflect_tintG",    s.reflectTintG);
            readFOpt("reflect_tintB",    s.reflectTintB);

            readIOpt("transmit_bands", s.transmitBands);
            readFOpt("transmit_gain",  s.transmitGain);
            readFOpt("chroma_shift",   s.chromaShift);

            readFOpt("indirect_gain", s.indirectGain);

            readFOpt("line_colorR",   s.lineColorR);
            readFOpt("line_colorG",   s.lineColorG);
            readFOpt("line_colorB",   s.lineColorB);
            readFOpt("line_width",    s.lineWidth);
            readFOpt("line_strength", s.lineStrength);
            readFOpt("outline_normal_threshold", s.outlineNormalThreshold);
            readFOpt("outline_depth_threshold",  s.outlineDepthThreshold);

            desc.styles.push_back(s);
        }
    }

    // --- Environment light --------------------------------------------------
    {
        size_t searchFrom = 0;
        std::string tagContent;
        if (extractNextTag(xml, "environment", searchFrom, tagContent)) {
            EnvironmentDesc& e = desc.environment;
            e.filename       = extractAttributeFromTag(tagContent, "filename");
            e.backgroundFile = extractAttributeFromTag(tagContent, "background_file");

            std::string mode = extractAttributeFromTag(tagContent, "background");
            if (!mode.empty()) e.backgroundMode = mode;

            auto readFOpt = [&](const char* attrName, float& out) {
                std::string v = extractAttributeFromTag(tagContent, attrName);
                if (v.empty()) return;
                try { out = std::stof(v); } catch (...) {
                    std::cerr << "[scene] <environment> invalid float attribute `"
                              << attrName << "`: \"" << v << "\"\n";
                    std::exit(EXIT_FAILURE);
                }
            };
            readFOpt("intensity",  e.intensity);
            readFOpt("yaw",        e.yawDegrees);
            readFOpt("backgroundR", e.backgroundR);
            readFOpt("backgroundG", e.backgroundG);
            readFOpt("backgroundB", e.backgroundB);

            // A flat-colour backdrop is a legitimate environment with no HDRI at
            // all, so `enabled` does not require a filename.
            e.enabled = !e.filename.empty() || e.backgroundMode == "color";

            if (e.filename.empty() && e.backgroundMode != "color") {
                std::cerr << "[scene] <environment> has no `filename` and "
                             "background is not \"color\" — ignoring.\n";
                e.enabled = false;
            }
        }
    }

    // --- Materials ----------------------------------------------------------
    {
        size_t searchFrom = 0;
        while (true) {
            std::string tagContent;
            if (!extractNextTag(xml, "material", searchFrom, tagContent))
                break;

            MaterialDesc mat;
            mat.name          = extractAttributeFromTag(tagContent, "name");
            mat.albedoTexture = extractAttributeFromTag(tagContent, "albedoTexture");
            mat.bsdfName      = extractAttributeFromTag(tagContent, "bsdf");
            mat.styleName     = extractAttributeFromTag(tagContent, "style");

            if (mat.name.empty()) {
                std::cerr << "[scene] <material> missing required attribute "
                             "`name` in \"" << path << "\".\n";
                std::exit(EXIT_FAILURE);
            }
            // albedoTexture is optional — empty means no texture; the BSDF
            // base_color / albedo parameters drive the colour instead.
            if (mat.bsdfName.empty()) {
                std::cerr << "[scene] <material> \"" << mat.name
                          << "\" missing required attribute `bsdf`.\n";
                std::exit(EXIT_FAILURE);
            }

            desc.materials.push_back(mat);
        }
    }

    // --- Meshes -------------------------------------------------------------
    {
        size_t searchFrom = 0;
        while (true) {
            std::string tagContent;
            if (!extractNextTag(xml, "mesh", searchFrom, tagContent))
                break;

            MeshDesc m;
            m.filename    = extractAttributeFromTag(tagContent, "filename");
            m.materialName = extractAttributeFromTag(tagContent, "material");

            // Optional transform attributes.
            // If not provided, MeshDesc keeps default identity transform.
            auto readF = [&](const char* attrName, float& out) {
                std::string s = extractAttributeFromTag(tagContent, attrName);
                if (s.empty()) return;
                try {
                    out = std::stof(s);
                } catch (...) {
                    std::cerr << "[scene] <mesh filename=\"" << m.filename
                              << "\"> invalid float attribute `" << attrName
                              << "`: \"" << s << "\"\n";
                    std::exit(EXIT_FAILURE);
                }
            };

            // Optional emitter attributes
            {
                std::string isEm = extractAttributeFromTag(tagContent, "isEmitter");
                if (!isEm.empty()) {
                    if (isEm == "1" || isEm == "true" || isEm == "True") m.isEmitter = true;
                    else if (isEm == "0" || isEm == "false" || isEm == "False") m.isEmitter = false;
                    else {
                        std::cerr << "[scene] <mesh filename=\"" << m.filename
                                  << "\"> invalid bool attribute `isEmitter`: \""
                                  << isEm << "\" (use true/false or 1/0)\n";
                        std::exit(EXIT_FAILURE);
                    }
                }

                readF("radianceR", m.radianceR);
                readF("radianceG", m.radianceG);
                readF("radianceB", m.radianceB);

                if (m.isEmitter) {
                    // If radiance isn't specified, default to something visible.
                    // (Still optional, but avoids "black light" confusion.)
                    if (m.radianceR == 0.0f && m.radianceG == 0.0f && m.radianceB == 0.0f) {
                        m.radianceR = 10.0f;
                        m.radianceG = 10.0f;
                        m.radianceB = 10.0f;
                    }
                }
            }

            readF("posX", m.transform.posX);
            readF("posY", m.transform.posY);
            readF("posZ", m.transform.posZ);
            readF("rotX", m.transform.rotXDegrees);
            readF("rotY", m.transform.rotYDegrees);
            readF("rotZ", m.transform.rotZDegrees);
            readF("scaleX", m.transform.scaleX);
            readF("scaleY", m.transform.scaleY);
            readF("scaleZ", m.transform.scaleZ);

            if (m.filename.empty()) {
                std::cerr << "[scene] <mesh> missing required attribute "
                             "`filename` in \"" << path << "\".\n";
                std::exit(EXIT_FAILURE);
            }
            if (m.materialName.empty()) {
                std::cerr << "[scene] <mesh filename=\"" << m.filename
                          << "\"> missing required attribute `material`.\n";
                std::exit(EXIT_FAILURE);
            }

            desc.meshes.push_back(m);
        }
    }

    // Validate mesh material references
    {
        auto materialExists = [&](const std::string& name) -> bool {
            for (const auto& m : desc.materials) {
                if (m.name == name) return true;
            }
            return false;
        };

        for (const auto& mesh : desc.meshes) {
            if (!materialExists(mesh.materialName)) {
                std::cerr << "[scene] mesh material reference \""
                          << mesh.materialName
                          << "\" not found in <material> entries.\n";
                std::exit(EXIT_FAILURE);
            }
        }
    }

    // Validate material -> bsdf references
    {
        auto bsdfExists = [&](const std::string& name) -> bool {
            for (const auto& b : desc.bsdfs) {
                if (b.name == name) return true;
            }
            return false;
        };

        for (const auto& mat : desc.materials) {
            if (!bsdfExists(mat.bsdfName)) {
                std::cerr << "[scene] material bsdf reference \""
                          << mat.bsdfName
                          << "\" not found in <bsdf> entries.\n";
                std::exit(EXIT_FAILURE);
            }
        }
    }

    // Validate material -> style references (style is optional)
    {
        auto styleExists = [&](const std::string& name) -> bool {
            for (const auto& s : desc.styles) {
                if (s.name == name) return true;
            }
            return false;
        };

        for (const auto& mat : desc.materials) {
            if (!mat.styleName.empty() && !styleExists(mat.styleName)) {
                std::cerr << "[scene] material style reference \""
                          << mat.styleName
                          << "\" not found in <style> entries.\n";
                std::exit(EXIT_FAILURE);
            }
        }
    }

    // --- Emitters -----------------------------------------------------------
    {
        size_t searchFrom = 0;
        while (true) {
            size_t tagEnd = 0;
            auto name = extractAttribute(xml, "emitter", "name",
                                         searchFrom, &tagEnd);
            auto type = extractAttribute(xml, "emitter", "type",
                                         searchFrom, &tagEnd);
            auto target = extractAttribute(xml, "emitter", "targetMesh",
                                           searchFrom, &tagEnd);
            if (name.empty() && type.empty() && target.empty())
                break;
            EmitterDesc e;
            e.name       = name;
            e.type       = type;
            e.targetMesh = target;
            desc.emitters.push_back(e);
            searchFrom = tagEnd;
        }
    }

    // --- Spotlights ---------------------------------------------------------
    {
        size_t searchFrom = 0;
        while (true) {
            std::string tagContent;
            if (!extractNextTag(xml, "spotlight", searchFrom, tagContent))
                break;

            SpotlightDesc s;

            auto readF = [&](const char* attrName, float& out) {
                std::string v = extractAttributeFromTag(tagContent, attrName);
                if (v.empty()) return;
                try { out = std::stof(v); } catch (...) {
                    std::cerr << "[scene] <spotlight> invalid float attribute `"
                              << attrName << "`: \"" << v << "\"\n";
                    std::exit(EXIT_FAILURE);
                }
            };

            readF("posX",          s.posX);
            readF("posY",          s.posY);
            readF("posZ",          s.posZ);
            readF("dirX",          s.dirX);
            readF("dirY",          s.dirY);
            readF("dirZ",          s.dirZ);
            readF("radianceR",     s.radianceR);
            readF("radianceG",     s.radianceG);
            readF("radianceB",     s.radianceB);
            readF("intensity",     s.intensity);
            readF("innerConeAngle", s.innerConeAngle);
            readF("outerConeAngle", s.outerConeAngle);

            desc.spotlights.push_back(s);
        }
    }

    // --- Window / film ------------------------------------------------------
    {
        auto w = extractAttribute(xml, "window", "width");
        auto h = extractAttribute(xml, "window", "height");
        if (w.empty() || h.empty()) {
            std::cerr << "[scene] <window> missing required `width` and/or `height`.\n";
            std::exit(EXIT_FAILURE);
        }
        try { desc.windowWidth = std::stoi(w); } catch (...) {}
        try { desc.windowHeight = std::stoi(h); } catch (...) {}
    }

    // --- Camera -------------------------------------------------------------
    {
        size_t searchFrom = 0;
        std::string tagContent;
        if (!extractNextTag(xml, "camera", searchFrom, tagContent)) {
            std::cerr << "[scene] No <camera ...> tag found in \"" << path
                      << "\". Please add camera parameters.\n";
            std::exit(EXIT_FAILURE);
        }

        auto readF = [&](const char* attrName) -> float {
            std::string s = extractAttributeFromTag(tagContent, attrName);
            if (s.empty()) {
                std::cerr << "[scene] <camera> missing required attribute `"
                          << attrName << "` in \"" << path << "\".\n";
                std::exit(EXIT_FAILURE);
            }
            try {
                return std::stof(s);
            } catch (...) {
                std::cerr << "[scene] <camera> invalid float value for `"
                          << attrName << "`: \"" << s << "\".\n";
                std::exit(EXIT_FAILURE);
            }
            return 0.0f;
        };

        desc.camera.eyeX       = readF("eyeX");
        desc.camera.eyeY       = readF("eyeY");
        desc.camera.eyeZ       = readF("eyeZ");
        desc.camera.lookAtX   = readF("lookAtX");
        desc.camera.lookAtY   = readF("lookAtY");
        desc.camera.lookAtZ   = readF("lookAtZ");
        desc.camera.upX        = readF("upX");
        desc.camera.upY        = readF("upY");
        desc.camera.upZ        = readF("upZ");
        desc.camera.fovYDegrees = readF("fovY");
    }

    if (desc.meshes.empty()) {
        std::cerr << "[scene] No <mesh ...> elements found in \"" << path
                  << "\". Please add at least one mesh.\n";
        std::exit(EXIT_FAILURE);
    }
    if (desc.materials.empty()) {
        std::cerr << "[scene] No <material ...> elements found in \"" << path
                  << "\". Please add at least one material.\n";
        std::exit(EXIT_FAILURE);
    }
    if (desc.bsdfs.empty()) {
        std::cerr << "[scene] No <bsdf ...> elements found in \"" << path
                  << "\". Please add at least one bsdf.\n";
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[scene] Loaded scene from \"" << path << "\"\n";
    std::cout << "        meshes    = " << desc.meshes.size()    << "\n";
    std::cout << "        materials = " << desc.materials.size() << "\n";
    std::cout << "        styles    = " << desc.styles.size()    << "\n";
    std::cout << "        styleMode = " << desc.styleMode        << "\n";
    std::cout << "        emitters  = " << desc.emitters.size()  << "\n";
    if (desc.environment.enabled) {
        std::cout << "        environment = \"" << desc.environment.filename
                  << "\" (intensity " << desc.environment.intensity
                  << ", background " << desc.environment.backgroundMode << ")\n";
    }
    std::cout << "        window    = " << desc.windowWidth << " x "
              << desc.windowHeight << "\n";

    return desc;
}

