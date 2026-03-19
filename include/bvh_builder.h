#pragma once

#include "render_kernel.h"

#include <vector>

/// Builds a BVH from a flat triangle array using FastBVH and converts the
/// result into GPU-ready flat arrays.
///
/// Returns two parallel outputs:
///   - nodes:       flat LinearBVHNode array for GPU traversal
///   - primIndices: reordered triangle indices referenced by leaf nodes
struct BVHBuildResult {
    std::vector<LinearBVHNode> nodes;
    std::vector<int>           primIndices;
};

BVHBuildResult buildBVH(const TriangleData* triangles, int triangleCount);
