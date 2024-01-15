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

#include "micromesh_decoder_utils_vk.hpp"

namespace microdisp {

glm::vec4 computeSphere(const MeshSet& meshSet, size_t baseTriangleIdx, const glm::vec3 barys[3], float minDisp, float maxDisp, bool directionBoundsAreUniform)
{
  const glm::uvec3* triIndices = reinterpret_cast<const glm::uvec3*>(meshSet.globalIndices.data());

  glm::vec4 sphere;

  glm::uvec3 indices = triIndices[baseTriangleIdx];
  glm::vec3  verts[3];
  glm::vec3  dirs[3];
  glm::vec2  bounds[3];

  verts[0] = meshSet.attributes.positions[indices.x];
  verts[1] = meshSet.attributes.positions[indices.y];
  verts[2] = meshSet.attributes.positions[indices.z];
  dirs[0]  = meshSet.attributes.directions[indices.x];
  dirs[1]  = meshSet.attributes.directions[indices.y];
  dirs[2]  = meshSet.attributes.directions[indices.z];

  if(!directionBoundsAreUniform && !meshSet.attributes.directionBounds.empty())
  {
    bounds[0] = meshSet.attributes.directionBounds[indices.x];
    bounds[1] = meshSet.attributes.directionBounds[indices.y];
    bounds[2] = meshSet.attributes.directionBounds[indices.z];

    verts[0] = verts[0] + dirs[0] * bounds[0].x;
    verts[1] = verts[1] + dirs[1] * bounds[1].x;
    verts[2] = verts[2] + dirs[2] * bounds[2].x;

    dirs[0] = dirs[0] * bounds[0].y;
    dirs[1] = dirs[1] * bounds[1].y;
    dirs[2] = dirs[2] * bounds[2].y;
  }

  glm::vec3 vertExtents[6];
  vertExtents[0] = getInterpolated(verts, barys[0]) + getInterpolated(dirs, barys[0]) * minDisp;
  vertExtents[1] = getInterpolated(verts, barys[1]) + getInterpolated(dirs, barys[1]) * minDisp;
  vertExtents[2] = getInterpolated(verts, barys[2]) + getInterpolated(dirs, barys[2]) * minDisp;
  vertExtents[3] = getInterpolated(verts, barys[0]) + getInterpolated(dirs, barys[0]) * maxDisp;
  vertExtents[4] = getInterpolated(verts, barys[1]) + getInterpolated(dirs, barys[1]) * maxDisp;
  vertExtents[5] = getInterpolated(verts, barys[2]) + getInterpolated(dirs, barys[2]) * maxDisp;

  glm::vec3 center = vertExtents[0];
  center += vertExtents[1];
  center += vertExtents[2];
  center += vertExtents[3];
  center += vertExtents[4];
  center += vertExtents[5];
  center /= 6.0f;

  float radius = 0;
  for(uint32_t i = 0; i < 6; i++)
  {
    radius = std::max(radius, glm::length(vertExtents[i] - center));
  }

  sphere.x = center.x;
  sphere.y = center.y;
  sphere.z = center.z;
  sphere.w = radius;

  return sphere;
}

void initBmapIndices(RBuffer&                    tableBuffer,
                     uint64_t                    tableBinding[MAX_BARYMAP_LEVELS],
                     ResourcesVK&                res,
                     nvvk::StagingMemoryManager* staging,
                     VkCommandBuffer             cmd,
                     const BaryAttributesSet&    barySet)
{
  // uncompressed map used here for accessing micro-vertex shading attributes
  baryutils::BaryLevelsMap bmap = barySet.makeBaryLevelsMapShading();
  uint32_t bmapLevelsCount = std::min(bmap.getNumLevels(), std::min(uint32_t(MAX_BARYMAP_LEVELS), uint32_t(MICRO_MAX_LEVELS)));

  // for micro-vertex attributes
  uint32_t level2bmapOffsets[MAX_BARYMAP_LEVELS];
  {
    uint32_t bary2bmapTotal = 0;
    for(uint32_t lvl = 0; lvl < bmapLevelsCount; lvl++)
    {
      level2bmapOffsets[lvl] = bary2bmapTotal;
      bary2bmapTotal += uint32_t(bmap.getLevel(lvl).coordinates.size());
    }
    tableBuffer = res.createBuffer(sizeof(uint32_t) * bary2bmapTotal, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  }

  for(uint32_t lvl = 0; lvl < bmapLevelsCount; lvl++)
  {
    tableBinding[lvl] = tableBuffer.addr + sizeof(uint32_t) * level2bmapOffsets[lvl];
  }

  uint32_t* bindicesAll =
      staging->cmdToBufferT<uint32_t>(cmd, tableBuffer.buffer, tableBuffer.info.offset, tableBuffer.info.range);

  for(uint32_t lvl = 0; lvl < bmapLevelsCount; lvl++)
  {
    const baryutils::BaryLevelsMap::Level& level    = bmap.getLevel(lvl);
    uint32_t*                              bindices = bindicesAll + level2bmapOffsets[lvl];
    uint32_t                               msize    = (1 << lvl) + 1;
    for(size_t i = 0; i < level.coordinates.size(); i++)
    {
      uint32_t idx  = umajorUV_toLinear(msize, glm::ivec2(level.coordinates[i].u, level.coordinates[i].v));
      bindices[idx] = uint32_t(i);
    }
  }
}

void initAttributes(MicromeshSetCompressedVK& micro, ResourcesVK& res, const MeshSet& meshSet, const BaryAttributesSet& barySet, uint32_t numThreads)
{
  micro.initAttributeNormals(res, meshSet, barySet, numThreads);

  {
    nvvk::StagingMemoryManager* staging = res.m_allocator.getStaging();
    VkCommandBuffer             cmd     = res.createTempCmdBuffer();

    uint64_t tableBinding[MAX_BARYMAP_LEVELS] = {0};

    initBmapIndices(micro.umajor2bmap, tableBinding, res, staging, cmd, barySet);

    for(auto& meshData : micro.meshDatas)
    {
      for(uint32_t lvl = 0; lvl < MAX_BARYMAP_LEVELS; lvl++)
      {
        meshData.combinedData->bindingData.umajor2bmap[lvl] = tableBinding[lvl];
      }
    }

    res.tempSyncSubmit(cmd);
  }
}


void MicroSplitParts::initMergeIndices()
{
  memset(partVertexMergeIndices, 0, sizeof(partVertexMergeIndices));

  // setup merge indices info
  {
    // original, preserve
    partVertexMergeIndices[0].a = 0;
    partVertexMergeIndices[0].b = 0;
    partVertexMergeIndices[1].a = 1;
    partVertexMergeIndices[1].b = 1;
    partVertexMergeIndices[2].a = 2;
    partVertexMergeIndices[2].b = 2;

    // Hierarchical decoder loop
    for(uint32_t lvl = 0; lvl < MICRO_PART_MAX_SUBDIV; lvl++)
    {
      uint32_t numSegments   = (1 << (lvl + 1));
      uint32_t edgeVerts     = numSegments + 1;
      uint32_t edgeVertsPrev = (numSegments / 2) + 1;

      // Compute number of vertices
      uint numVerts     = ((edgeVerts) * ((edgeVerts) + 1)) / 2;
      uint numVertsPrev = ((edgeVertsPrev) * ((edgeVertsPrev) + 1)) / 2;

      for(uint i = 0; i < numVerts; i++)
      {
        // these stay where they are
        if(i < numVertsPrev)
        {
        }
        else
        {
          // get triplet base coord
          baryutils::BaryWUV_uint16 coordBase = map.getLevel(lvl + 1).coordinates[i];
          coordBase.u /= 2;
          coordBase.v /= 2;
          coordBase.w = uint16_t((1u << (lvl)) - coordBase.u - coordBase.v);

          baryutils::BaryWUV_uint16 coordL = coordBase;
          baryutils::BaryWUV_uint16 coordR = coordBase;

          // edge 0 = AC split
          // edge 1 = CB split
          // edge 2 = BA split

          uint32_t tripletEdge = (i - numVertsPrev) % 3;
          switch(tripletEdge)
          {
            case 0:
              coordR.w -= 1;
              coordR.v += 1;
              break;
            case 1:
              coordL.w -= 1;
              coordL.v += 1;
              coordR.w -= 1;
              coordR.u += 1;
              break;
            case 2:
              coordR.w -= 1;
              coordR.u += 1;
              break;
          }

          partVertexMergeIndices[i].a = map.getLevel(lvl).getCoordIndex(coordL);
          partVertexMergeIndices[i].b = map.getLevel(lvl).getCoordIndex(coordR);
        }
      }
    }
  }
}

void MicroSplitParts::uploadTriangleIndices(nvvk::StagingMemoryManager* staging, VkCommandBuffer cmd, const RBuffer& triangleIndices, bool doPartFlip)
{
  glm::u8vec4* trianglesAll =
      staging->cmdToBufferT<glm::u8vec4>(cmd, triangleIndices.buffer, triangleIndices.info.offset, triangleIndices.info.range);

  // iterate over all edge decimate permutations
  for(uint32_t decimateEdgeBits = 0; decimateEdgeBits < MICRO_MESHLET_TOPOS; decimateEdgeBits++)
  {
    // level 0,1,2,3
    // these levels fit in a single meshlet part
    // can just take indices/vertices as is
    for(uint32_t lvl = 0; lvl <= MICRO_PART_MAX_SUBDIV; lvl++)
    {
      const baryutils::BaryLevelsMap::Level& birdLevel = map.getLevel(lvl);

      std::vector<baryutils::BaryLevelsMap::Triangle> birdTriangles =
          birdLevel.buildTrianglesWithCollapsedEdges(decimateEdgeBits, true);

      glm::u8vec4* trianglesLevel = trianglesAll + (lvl * MICRO_MESHLET_LOD_PRIMS) + MICRO_MESHLET_PRIMS * decimateEdgeBits;

      for(size_t t = 0; t < birdTriangles.size(); t++)
      {
        const baryutils::BaryLevelsMap::Triangle& tri = birdTriangles[t];
        if(tri.a != tri.b && tri.b != tri.c && tri.c != tri.a)
        {
          trianglesLevel[t].x = birdTriangles[t].a;
          trianglesLevel[t].y = birdTriangles[t].b;
          trianglesLevel[t].z = birdTriangles[t].c;
        }
        else
        {
          // zero degenerated triangles
          trianglesLevel[t].x = 0;
          trianglesLevel[t].y = 0;
          trianglesLevel[t].z = 0;
        }
      }
    }

    // when split into multiple partIDs, things are bit more complicated
    // generate index buffers for each meshlet and subdiv level config
    for(uint32_t lvl = 4; lvl <= 5; lvl++)
    {
      // each of these levels (4 or 5) requires a different number of
      // meshlets
      uint32_t numParts = 1 << (lvl - MICRO_PART_MAX_SUBDIV);
      numParts          = numParts * numParts;

      uint32_t subOffset = lvl == 4 ? 1 : 5;

      for(uint32_t partID = 0; partID < numParts; partID++, subOffset++)
      {
        const bary::BlockTriangle* partSplit = &triLevelNtoN[lvl][3][partID];

        glm::u8vec4* trianglesLevel = trianglesAll + (3 * MICRO_MESHLET_LOD_PRIMS) + (subOffset * MICRO_PART_MAX_PRIMITIVES)
                                 + MICRO_MESHLET_PRIMS * decimateEdgeBits;

        if(decimateEdgeBits == 0)
        {
          for(size_t t = 0; t < partLevel.triangles.size(); t++)
          {
            if(doPartFlip && partSplit->flipped)
            {
              trianglesLevel[t].x = partLevel.triangles[t].b;
              trianglesLevel[t].y = partLevel.triangles[t].a;
              trianglesLevel[t].z = partLevel.triangles[t].c;
            }
            else
            {
              trianglesLevel[t].x = partLevel.triangles[t].a;
              trianglesLevel[t].y = partLevel.triangles[t].b;
              trianglesLevel[t].z = partLevel.triangles[t].c;
            }
          }
          continue;
        }

        // decimation is complex because we need to figure out which of the part's
        // vertices got collapsed.
        // We cannot use birdPartLevel.buildTrianglesWithCollapsedEdges because
        // the hierarchical splitting changes the uv-coordinates of the
        // sub-triangle / part we operate in.

        std::vector<baryutils::BaryWUV_uint16>                                coordinates;
        std::unordered_map<baryutils::BaryLevelsMap::BaryCoordHash, uint32_t> coordMap;

        coordinates.reserve(MICRO_PART_MAX_VERTICES);

        // build new list of coordinates taking the local part UVs
        // into UVs of global space
        for(uint32_t v = 0; v < MICRO_PART_MAX_VERTICES; v++)
        {
          baryutils::BaryWUV_uint16 coord   = partLevel.coordinates[v];
          bary::BaryUV_uint16       coordUV = {coord.u, coord.v};

          // apply split transform to get from part UV to base UV
          coordUV = bary::baryBlockTriangleLocalToBaseUV(partSplit, coordUV);
          coord   = {uint16_t((1 << lvl) - coordUV.u - coordUV.v), coordUV.u, coordUV.v};

          coordinates.push_back(coord);
          coordMap.insert({baryutils::BaryLevelsMap::getHash(coord), v});
        }

        std::vector<baryutils::BaryLevelsMap::Triangle> birdTriangles;
        birdTriangles.reserve(MICRO_PART_MAX_PRIMITIVES);

        for(const baryutils::BaryLevelsMap::Triangle& triangle : partLevel.triangles)
        {
          // the joinVertex operates now in global UV space of subdivision "lvl",
          // which is the level we are generating triangle indices for.

          baryutils::BaryWUV_uint16 baryA = baryutils::BaryLevelsMap::joinVertex(coordinates[triangle.a], decimateEdgeBits, lvl);
          baryutils::BaryWUV_uint16 baryB = baryutils::BaryLevelsMap::joinVertex(coordinates[triangle.b], decimateEdgeBits, lvl);
          baryutils::BaryWUV_uint16 baryC = baryutils::BaryLevelsMap::joinVertex(coordinates[triangle.c], decimateEdgeBits, lvl);

          baryutils::BaryLevelsMap::Triangle tri;
          tri.a = coordMap.find(baryutils::BaryLevelsMap::getHash(baryA))->second;
          tri.b = coordMap.find(baryutils::BaryLevelsMap::getHash(baryB))->second;
          tri.c = coordMap.find(baryutils::BaryLevelsMap::getHash(baryC))->second;

          birdTriangles.push_back(tri);
        }

        for(size_t t = 0; t < birdTriangles.size(); t++)
        {
          const baryutils::BaryLevelsMap::Triangle& tri = birdTriangles[t];
          if(tri.a != tri.b && tri.b != tri.c && tri.c != tri.a)
          {
            if(doPartFlip && partSplit->flipped)
            {
              trianglesLevel[t].x = birdTriangles[t].b;
              trianglesLevel[t].y = birdTriangles[t].a;
              trianglesLevel[t].z = birdTriangles[t].c;
            }
            else
            {
              trianglesLevel[t].x = birdTriangles[t].a;
              trianglesLevel[t].y = birdTriangles[t].b;
              trianglesLevel[t].z = birdTriangles[t].c;
            }
          }
          else
          {
            // zero degenerated triangles
            trianglesLevel[t].x = 0;
            trianglesLevel[t].y = 0;
            trianglesLevel[t].z = 0;
          }
        }
      }
    }
  }
}

}  // namespace microdisp