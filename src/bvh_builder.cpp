// bvh_builder.cpp
// Recursive median-split BVH builder.  Produces a flat LinearBVHNode array
// compatible with the iterative GPU traversal in rt_intersect.cuh.
//
// Node layout (mirrors FastBVH convention):
//   - left  child: nodes[nodeIdx + 1]
//   - right child: nodes[nodeIdx + rightOffset]
//   - leaf:        rightOffset == 0,  primCount > 0

#include "bvh_builder.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>

static constexpr int LEAF_SIZE = 4; // max triangles per leaf

// Returns the centroid of a triangle along one axis (0=X, 1=Y, 2=Z).
static float centroid(const TriangleData& t, int axis)
{
    if (axis == 0) return (t.v0.x + t.v1.x + t.v2.x) * (1.0f / 3.0f);
    if (axis == 1) return (t.v0.y + t.v1.y + t.v2.y) * (1.0f / 3.0f);
                   return (t.v0.z + t.v1.z + t.v2.z) * (1.0f / 3.0f);
}

// Expand bmin/bmax to include all three vertices of a triangle.
static void expandAABB(float bmin[3], float bmax[3], const TriangleData& t)
{
    const float* verts[3][3] = {
        {&t.v0.x, &t.v0.y, &t.v0.z},
        {&t.v1.x, &t.v1.y, &t.v1.z},
        {&t.v2.x, &t.v2.y, &t.v2.z},
    };
    for (int v = 0; v < 3; v++)
        for (int a = 0; a < 3; a++) {
            if (*verts[v][a] < bmin[a]) bmin[a] = *verts[v][a];
            if (*verts[v][a] > bmax[a]) bmax[a] = *verts[v][a];
        }
}

// Recursive build.  Writes nodes into `nodes` and appends triangle indices
// into `orderedPrims`.  Uses indexed access (nodes[nodeIdx]) so that vector
// reallocation during recursion does not invalidate anything.
static void buildNode(
    std::vector<LinearBVHNode>& nodes,
    std::vector<int>&           orderedPrims,
    std::vector<int>&           work,        // mutable working index array
    int start, int end,
    const TriangleData*         tris)
{
    const int nodeIdx = (int)nodes.size();
    nodes.emplace_back();   // reserve slot; fill below via nodes[nodeIdx]

    // Compute AABB for [start, end).
    float bmin[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float bmax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (int i = start; i < end; i++)
        expandAABB(bmin, bmax, tris[work[i]]);

    const int count = end - start;

    if (count <= LEAF_SIZE) {
        // Leaf node
        nodes[nodeIdx].primStart   = (uint32_t)orderedPrims.size();
        nodes[nodeIdx].primCount   = (uint32_t)count;
        nodes[nodeIdx].rightOffset = 0u;
        for (int i = start; i < end; i++)
            orderedPrims.push_back(work[i]);
    } else {
        // Interior node — split along the longest AABB axis at the median.
        nodes[nodeIdx].primCount = 0u;

        int axis = 0;
        if ((bmax[1] - bmin[1]) > (bmax[axis] - bmin[axis])) axis = 1;
        if ((bmax[2] - bmin[2]) > (bmax[axis] - bmin[axis])) axis = 2;

        const int mid = (start + end) / 2;
        std::nth_element(
            work.begin() + start,
            work.begin() + mid,
            work.begin() + end,
            [&](int a, int b) {
                return centroid(tris[a], axis) < centroid(tris[b], axis);
            });

        // Left subtree immediately follows this node (nodeIdx + 1).
        buildNode(nodes, orderedPrims, work, start, mid, tris);

        // Right subtree starts wherever the left finished.
        const int rightOffset = (int)nodes.size() - nodeIdx;
        buildNode(nodes, orderedPrims, work, mid, end, tris);

        nodes[nodeIdx].rightOffset = (uint32_t)rightOffset;
    }

    // Write bounding box (after recursion; vector may have reallocated).
    for (int a = 0; a < 3; a++) {
        nodes[nodeIdx].bmin[a] = bmin[a];
        nodes[nodeIdx].bmax[a] = bmax[a];
    }
    nodes[nodeIdx].pad[0] = nodes[nodeIdx].pad[1] = nodes[nodeIdx].pad[2] = 0u;
}

BVHBuildResult buildBVH(const TriangleData* triangles, int triangleCount)
{
    BVHBuildResult result;
    if (!triangles || triangleCount <= 0)
        return result;

    std::vector<int> work(triangleCount);
    for (int i = 0; i < triangleCount; i++) work[i] = i;

    result.nodes.reserve(triangleCount * 2);
    result.primIndices.reserve(triangleCount);

    buildNode(result.nodes, result.primIndices, work, 0, triangleCount, triangles);

    printf("[bvh] %d nodes, %d primitives\n",
           (int)result.nodes.size(), triangleCount);
    return result;
}
