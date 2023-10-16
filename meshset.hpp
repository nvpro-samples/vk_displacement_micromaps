/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include <nvmath/nvmath.h>
#include <vector>
#include <string>

struct MeshSetID
{
  static const inline uint32_t INVALID = ~0u;
};

struct MeshTexture
{
  std::string filename;
};

struct MeshMaterial
{
  std::string name;

  nvmath::vec3f diffuse{1.0f};

  uint32_t normalMapTextureID = MeshSetID::INVALID;
};

struct MeshAttributes
{
  std::vector<nvmath::vec3f> positions;
  std::vector<nvmath::vec3f> normals;

  // optional
  std::vector<nvmath::vec2f> texcoords0;
  std::vector<nvmath::vec3f> tangents;
  std::vector<nvmath::vec3f> bitangents;

  // displacement (lo poly only)
  std::vector<nvmath::vec3f> directions;
  std::vector<nvmath::vec2f> directionBounds;
};

struct MeshBBox
{
  nvmath::vec3f mins{FLT_MAX, FLT_MAX, FLT_MAX};
  nvmath::vec3f maxs{-FLT_MAX, -FLT_MAX, -FLT_MAX};

  inline void merge(const nvmath::vec3f& point)
  {
    mins = nvmath::nv_min(mins, point);
    maxs = nvmath::nv_max(maxs, point);
  }

  inline void merge(const MeshBBox& bbox)
  {
    mins = nvmath::nv_min(mins, bbox.mins);
    maxs = nvmath::nv_max(maxs, bbox.maxs);
  }

  inline nvmath::vec3f diagonal() const { return maxs - mins; }

  inline MeshBBox transformed(const nvmath::mat4f& matrix) const
  {
    int           i;
    nvmath::vec4f box[16];
    // create box corners
    box[0] = nvmath::vec4f(mins.x, mins.y, mins.z, 1.0f);
    box[1] = nvmath::vec4f(maxs.x, mins.y, mins.z, 1.0f);
    box[2] = nvmath::vec4f(mins.x, maxs.y, mins.z, 1.0f);
    box[3] = nvmath::vec4f(maxs.x, maxs.y, mins.z, 1.0f);
    box[4] = nvmath::vec4f(mins.x, mins.y, maxs.z, 1.0f);
    box[5] = nvmath::vec4f(maxs.x, mins.y, maxs.z, 1.0f);
    box[6] = nvmath::vec4f(mins.x, maxs.y, maxs.z, 1.0f);
    box[7] = nvmath::vec4f(maxs.x, maxs.y, maxs.z, 1.0f);

    // transform box corners
    // and find new mins,maxs
    MeshBBox bbox;

    for(i = 0; i < 8; i++)
    {
      nvmath::vec4f point = matrix * box[i];
      bbox.merge({point.x, point.y, point.z});
    }

    return bbox;
  }
};


// Descriptor for unique mesh data
struct MeshInfo
{
  // Vertex attributes are accessed as follows:
  // attribute[firstVertex + indices[firstIndex + i]], where i = 0...numIndices-1.
  uint32_t firstVertex    = 0;
  uint32_t numVertices    = 0;
  uint32_t firstIndex     = 0;
  uint32_t numIndices     = 0;
  uint32_t firstPrimitive = 0;
  uint32_t numPrimitives  = 0;

  uint32_t largestInstanceID = MeshSetID::INVALID;

  float longestEdge = 0;
  float averageEdge = 0;

  MeshBBox bbox;

  // indicates if baker should bake this mesh
  bool doBake = true;

  // Indicates that this mesh's directions array came from the mesh normals,
  // and more specifically, that we should keep the directions the same as the
  // mesh normals.
  bool directionsUseMeshNormals = true;
  // indicates if direction bounds are uniform or varying per-vertex
  bool directionBoundsAreUniform = true;

  // This is either MeshSetID::INVALID, or the index of one of the scene's
  // pairs of uncompressed and compressed displacement to use as microdisplacement.
  uint32_t displacementID = MeshSetID::INVALID;
  // Each displacement source has a list of groups, which define offsets into
  // arrays as well as scale and bias (for uncompressed displacement). This
  // gives the index into that list. In other words, things are indexed like so:
  // baryMeshes[displacementId].groups[displacementGroup]
  uint32_t displacementGroup = 0;
  // This corresponds to the map_offset property from glTF.
  uint32_t displacementMapOffset = 0;

  uint32_t baryNormalID = MeshSetID::INVALID;

  // written by baker, used for instance lod
  uint32_t displacementMaxSubdiv = 0;

  // used for normal map
  uint32_t materialID = MeshSetID::INVALID;
};


// Descriptor for instances of Mesh
struct MeshInstance
{
  uint32_t meshID = MeshSetID::INVALID;

  MeshBBox bbox;

  nvmath::mat4f xform;
};

// MeshSet aggregates a number of meshes instanced from a set
// with a variety of materials attached (see 'prefab')
struct MeshSet
{
  MeshAttributes attributes;

  std::vector<uint32_t>     indices;
  std::vector<MeshInfo>     meshInfos;
  std::vector<MeshInstance> meshInstances;
  std::vector<MeshTexture>  textures;
  std::vector<MeshMaterial> materials;

  // optional
  // indices with firstVertex applied
  std::vector<uint32_t> globalIndices;

  // map from global tri idx to mesh idx
  std::vector<uint32_t> globalTriangleToMeshID;

  // mapping table with identical positions
  std::vector<uint32_t> globalUniquePosMap;
  // indices with firstVertex applied
  std::vector<uint32_t> globalUniquePosIndices;

  // mapping table with identical positions & directions
  std::vector<uint32_t> globalUniquePosDirMap;
  // indices with firstVertex applied
  std::vector<uint32_t> globalUniquePosDirIndices;


  // adaptive displacement generates these
  // per-primitive
  std::vector<uint8_t> decimateEdgeFlags;
  // optionally provided at load time
  // per-primitive
  std::vector<uint16_t> subdivisionLevels;

  // Indicates that all Meshs' directions come from the mesh normals,
  // and more specifically, that we should keep the directions the same as the
  // mesh normals. The directions attribute will still exist.
  bool directionsUseMeshNormals = true;
  // indicates if direction bounds are uniform for alles meshes or varying per-vertex for
  // some/all
  bool directionBoundsAreUniform = true;

  MeshBBox bbox;

  void setupProcessingGlobals(uint32_t numThreads = 0);
  void setupDirectionBoundsGlobals(uint32_t numThreads = 0);
  void clearDirectionBoundsGlobals();

  void setupInstanceGrid(size_t numOrig, size_t copies, uint32_t axis, nvmath::vec3f refShift);

  // setup largestInstance and longestEdge
  void setupLargestInstance();
  void setupEdgeLengths(uint32_t numThreads = 0);

  // returns true if all mesh.firstIndex/numIndices are multiple of 3
  bool hasContiguousIndices() const;

  std::vector<uint32_t> getDisplacementGroupOrderedMeshs(uint32_t displacementID) const;
  uint32_t              findDisplacementGroupMesh(uint32_t displacementID, uint32_t groupID) const;
};
