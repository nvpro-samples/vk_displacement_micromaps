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


#include <glm/glm.hpp>
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

  glm::vec3 diffuse{1.0f};

  uint32_t normalMapTextureID = MeshSetID::INVALID;
};

struct MeshAttributes
{
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;

  // optional
  std::vector<glm::vec2> texcoords0;
  std::vector<glm::vec3> tangents;
  std::vector<glm::vec3> bitangents;

  // displacement (lo poly only)
  std::vector<glm::vec3> directions;
  std::vector<glm::vec2> directionBounds;
};

struct MeshBBox
{
  glm::vec3 mins{FLT_MAX, FLT_MAX, FLT_MAX};
  glm::vec3 maxs{-FLT_MAX, -FLT_MAX, -FLT_MAX};

  inline void merge(const glm::vec3& point)
  {
    mins = glm::min(mins, point);
    maxs = glm::max(maxs, point);
  }

  inline void merge(const MeshBBox& bbox)
  {
    mins = glm::min(mins, bbox.mins);
    maxs = glm::max(maxs, bbox.maxs);
  }

  inline glm::vec3 diagonal() const { return maxs - mins; }

  inline MeshBBox transformed(const glm::mat4& matrix) const
  {
    int           i;
    glm::vec4 box[16];
    // create box corners
    box[0] = glm::vec4(mins.x, mins.y, mins.z, 1.0f);
    box[1] = glm::vec4(maxs.x, mins.y, mins.z, 1.0f);
    box[2] = glm::vec4(mins.x, maxs.y, mins.z, 1.0f);
    box[3] = glm::vec4(maxs.x, maxs.y, mins.z, 1.0f);
    box[4] = glm::vec4(mins.x, mins.y, maxs.z, 1.0f);
    box[5] = glm::vec4(maxs.x, mins.y, maxs.z, 1.0f);
    box[6] = glm::vec4(mins.x, maxs.y, maxs.z, 1.0f);
    box[7] = glm::vec4(maxs.x, maxs.y, maxs.z, 1.0f);

    // transform box corners
    // and find new mins,maxs
    MeshBBox bbox;

    for(i = 0; i < 8; i++)
    {
      glm::vec4 point = matrix * box[i];
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

  glm::mat4 xform;
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

  void setupInstanceGrid(size_t numOrig, size_t copies, uint32_t axis, glm::vec3 refShift);

  // setup largestInstance and longestEdge
  void setupLargestInstance();
  void setupEdgeLengths(uint32_t numThreads = 0);

  // returns true if all mesh.firstIndex/numIndices are multiple of 3
  bool hasContiguousIndices() const;

  std::vector<uint32_t> getDisplacementGroupOrderedMeshs(uint32_t displacementID) const;
  uint32_t              findDisplacementGroupMesh(uint32_t displacementID, uint32_t groupID) const;
};
