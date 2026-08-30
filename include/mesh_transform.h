#pragma once

#include "render_kernel.h"
#include "scene.h"

#include <cmath>

// Scale -> Rx -> Ry -> Rz -> Translate (Euler, degrees). Matches <mesh> / <model>.
inline Float3 transformPoint(const Float3& p, const TransformDesc& t)
{
    float x = p.x * t.scaleX;
    float y = p.y * t.scaleY;
    float z = p.z * t.scaleZ;

    const float degToRad = 3.14159265358979323846f / 180.0f;
    float rx = t.rotXDegrees * degToRad;
    float ry = t.rotYDegrees * degToRad;
    float rz = t.rotZDegrees * degToRad;

    {
        float cx = std::cos(rx);
        float sx = std::sin(rx);
        float y2 = y * cx - z * sx;
        float z2 = y * sx + z * cx;
        y = y2;
        z = z2;
    }
    {
        float cy = std::cos(ry);
        float sy = std::sin(ry);
        float x2 = x * cy + z * sy;
        float z2 = -x * sy + z * cy;
        x = x2;
        z = z2;
    }
    {
        float cz = std::cos(rz);
        float sz = std::sin(rz);
        float x2 = x * cz - y * sz;
        float y2 = x * sz + y * cz;
        x = x2;
        y = y2;
    }

    return {x + t.posX, y + t.posY, z + t.posZ};
}

// Rotations only, then re-normalize. Does not apply non-uniform scale.
inline Float3 transformNormal(const Float3& n, const TransformDesc& t)
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

    float len = std::sqrt(x * x + y * y + z * z);
    if (len > 1e-8f) { x /= len; y /= len; z /= len; }
    return {x, y, z};
}
