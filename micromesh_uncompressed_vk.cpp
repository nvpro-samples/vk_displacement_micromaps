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

#include "micromesh_uncompressed_vk.hpp"
#include "resources_vk.hpp"
#include "parallel_work.hpp"

#include <nvh/misc.hpp>

#include "common.h"
#include "common_barymap.h"
#include "common_micromesh_uncompressed.h"

namespace microdisp {
namespace {

// Computes a bounding sphere of a mesh triangle with/without displacement bounds.
// minDisp and maxDisp are the minimum and maximum displacement over the triangle,
// before application of direction bounds.
// Returns (center, radius).
inline glm::vec4 computeSphere(const MeshSet& meshSet, uint64_t meshGlobalTriIdx, float minDisp, float maxDisp, bool directionBoundsAreUniform)
{
  const glm::uvec3* triIndices = reinterpret_cast<const glm::uvec3*>(meshSet.globalIndices.data());

  glm::vec4 sphere;

  glm::uvec3 indices = triIndices[meshGlobalTriIdx];
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
  vertExtents[0] = verts[0] + dirs[0] * minDisp;
  vertExtents[1] = verts[1] + dirs[1] * minDisp;
  vertExtents[2] = verts[2] + dirs[2] * minDisp;
  vertExtents[3] = verts[0] + dirs[0] * maxDisp;
  vertExtents[4] = verts[1] + dirs[1] * maxDisp;
  vertExtents[5] = verts[2] + dirs[2] * maxDisp;

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

// not particlulary fast, to do the switch at the end ;)
float getBaryMinMaxValue(bary::Format fmt, const void* data, size_t idx)
{
  switch(fmt)
  {
    case bary::Format::eR8_unorm:
      return float(reinterpret_cast<const uint8_t*>(data)[idx]) / float(0xFF);
    case bary::Format::eR16_unorm:
      return float(reinterpret_cast<const uint16_t*>(data)[idx]) / float(0xFFFF);
    case bary::Format::eR11_unorm_pack16:
    case bary::Format::eR11_unorm_packed_align32:
      return float(reinterpret_cast<const uint16_t*>(data)[idx]) / float(0x7FF);
    case bary::Format::eR32_sfloat:
      return float(reinterpret_cast<const float*>(data)[idx]);
    default:
      return 0.0f;
  }
}

// Simple wrapper around cmdToBuffer that acts like "Write this vector of data
// to this buffer starting at this offset".
// Note(nbickford): I've added this here rather than in Resources since it looks
// like this may be the only file in the program in which we do this.
// Moving this into Resources and integrating it into the simpleUploadBuffer...
// like approach is definitely possible, but then the new function
// (named e.g. simpleUploadBufferRange) becomes not as simple, which feels like
// it defeats the purpose, so I decided not to go there.
template <class T>
void uploadBufferRange(ResourcesVK& res, VkCommandBuffer cmd, const RBuffer& dst, const size_t offset_in_elements_of_T, const std::vector<T>& src)
{
  if(dst.addr)
  {
    assert((offset_in_elements_of_T + src.size()) * sizeof(T) <= dst.info.range);
  }
  res.m_allocator.getStaging()->cmdToBuffer(cmd, dst.buffer, offset_in_elements_of_T * sizeof(T),
                                            src.size() * sizeof(T), src.data());
}

}  // namespace

void MicromeshSetUncompressedVK::uploadBinding(ResourcesVK& res, VkCommandBuffer cmd, const MeshData& mdata) const
{
  MicromeshUncData data;
  data.distancesBits      = mdata.distances.addr;
  data.triangleBitsMinMax = mdata.baseTrianglesMinMax.addr;
  data.basetriangles      = mdata.baseTriangles.addr;
  data.basespheres        = mdata.baseTriangleSpheres.addr;
  data.attrNormals        = mdata.attrNormals.addr;
  res.m_allocator.getStaging()->cmdToBuffer(cmd, mdata.binding.buffer, 0, mdata.binding.info.range, &data);
}

static size_t RoundUpElementCountToU32Alignment(size_t count, size_t elementSize)
{
  const size_t unpaddedSize     = count * elementSize;
  const size_t paddedSizeInU32s = (unpaddedSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
  return (paddedSizeInU32s * sizeof(uint32_t)) / elementSize;
}

void MicromeshSetUncompressedVK::init(ResourcesVK& res, const MeshSet& meshSet, const BaryAttributesSet& barySet, bool withAttributes, uint32_t numThreads)
{
  if(barySet.displacements.empty())
    return;

  assert(meshDatas.empty());  // Don't call init() multiple times without calling deinit()!

  baryutils::BaryLevelsMap bmap = (barySet.uncompressedStats.maxSubdivLevel) ? barySet.makeBaryLevelsMapUncompressed() :
                                                                               barySet.makeBaryLevelsMapShading();
  baryMap.init(res, bmap);

  meshDatas.resize(meshSet.meshInfos.size());

  // allocation phase & smaller uploads
  VkCommandBuffer             cmd     = res.createTempCmdBuffer();
  nvvk::StagingMemoryManager* staging = res.m_allocator.getStaging();

  for(size_t m = 0; m < meshSet.meshInfos.size(); m++)
  {
    uint32_t        meshID   = uint32_t(m);
    const MeshInfo& mesh     = meshSet.meshInfos[m];
    MeshData&       meshData = meshDatas[m];

    if(mesh.displacementID == MeshSetID::INVALID)
      continue;

    // init buffers
    meshData.binding =
        res.createBuffer(sizeof(MicromeshUncData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    meshData.baseTriangles = res.createBuffer(sizeof(MicromeshUncBaseTri) * mesh.numPrimitives, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    if(barySet.displacements[mesh.displacementID].uncompressed)
    {
      const bary::BasicView basic = barySet.displacements[mesh.displacementID].uncompressed->getView();
      const bary::Group&    group = basic.groups[mesh.displacementGroup];

      // add safety margin
      meshData.distances = res.createBuffer(basic.valuesInfo->valueByteSize * group.valueCount + sizeof(uint32_t),
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      meshData.baseTriangleSpheres =
          res.createBuffer(sizeof(glm::vec4) * mesh.numPrimitives, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      meshData.baseTrianglesMinMax = res.createBuffer(basic.triangleMinMaxsInfo->elementByteSize * 2 * mesh.numPrimitives,
                                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

      staging->cmdToBuffer(cmd, meshData.baseTrianglesMinMax.buffer, 0, meshData.baseTrianglesMinMax.info.range,
                           basic.triangleMinMaxs + (basic.triangleMinMaxsInfo->elementByteSize * 2 * group.triangleFirst));
    }

    const BaryShadingAttribute* shadingAttr = barySet.getDisplacementShading(mesh.displacementID, SHADING_ATTRIBUTE_NORMAL_BIT);
    if(shadingAttr)
    {
      const bary::BasicView basic = shadingAttr->attribute->getView();
      const bary::Group&    group = basic.groups[mesh.displacementGroup];

      meshData.attrNormals =
          res.createBuffer(basic.valuesInfo->valueByteSize * group.valueCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    uploadBinding(res, cmd, meshData);

    uploadFlatTriangles(res, cmd, meshSet, barySet, meshID, numThreads);
  }

  res.tempSyncSubmit(cmd);

  // bigger uploads
  for(size_t m = 0; m < meshSet.meshInfos.size(); m++)
  {
    uint32_t        meshID   = uint32_t(m);
    const MeshInfo& mesh     = meshSet.meshInfos[m];
    const MeshData& meshData = meshDatas[m];

    if(mesh.displacementID == MeshSetID::INVALID)
      continue;

    if(barySet.displacements[mesh.displacementID].uncompressed)
    {
      const bary::BasicView basic = barySet.displacements[mesh.displacementID].uncompressed->getView();
      const bary::Group&    group = basic.groups[mesh.displacementGroup];

      res.simpleUploadBuffer(meshData.distances, basic.values + (basic.valuesInfo->valueByteSize * group.valueFirst));
    }

    const BaryShadingAttribute* shadingAttr = barySet.getDisplacementShading(mesh.displacementID, SHADING_ATTRIBUTE_NORMAL_BIT);
    if(shadingAttr)
    {
      const bary::BasicView basic = shadingAttr->attribute->getView();
      const bary::Group&    group = basic.groups[mesh.displacementGroup];

      res.simpleUploadBuffer(meshData.attrNormals, basic.values + (basic.valuesInfo->valueByteSize * group.valueFirst));
    }
  }
}

void MicromeshSetUncompressedVK::deinit(ResourcesVK& res)
{
  for(auto& meshData : meshDatas)
  {
    res.destroy(meshData.binding);
    res.destroy(meshData.baseTriangles);
    res.destroy(meshData.baseTriangleSpheres);
    res.destroy(meshData.baseTrianglesMinMax);
    res.destroy(meshData.distances);
    res.destroy(meshData.attrNormals);
  }

  baryMap.deinit(res);
  meshDatas.clear();
}


void MicromeshSetUncompressedVK::uploadFlatTriangles(ResourcesVK&             res,
                                                     VkCommandBuffer          cmd,
                                                     const MeshSet&           meshSet,
                                                     const BaryAttributesSet& barySet,
                                                     uint32_t                 meshID,
                                                     uint32_t                 numThreads)
{
  const MeshInfo& mesh             = meshSet.meshInfos[meshID];
  const MeshData& meshData         = meshDatas[meshID];
  const size_t    numMeshTriangles = mesh.numPrimitives;

  MicromeshUncBaseTri* baseTriData = res.m_allocator.getStaging()->cmdToBufferT<MicromeshUncBaseTri>(
      cmd, meshData.baseTriangles.buffer, meshData.baseTriangles.info.offset, meshData.baseTriangles.info.range);

  glm::vec4* sphereData =
      res.m_allocator.getStaging()->cmdToBufferT<glm::vec4>(cmd, meshData.baseTriangleSpheres.buffer,
                                                                meshData.baseTriangleSpheres.info.offset,
                                                                meshData.baseTriangleSpheres.info.range);

  const uint8_t* decimateEdgeFlags = meshSet.decimateEdgeFlags.empty() ? nullptr : meshSet.decimateEdgeFlags.data();

  // displacement
  {
    bary::BasicView    basic          = barySet.displacements[mesh.displacementID].uncompressed->getView();
    const bary::Format valueFormat    = basic.valuesInfo->valueFormat;
    const bary::Group& baryGroup      = basic.groups[mesh.displacementGroup];
    size_t             minMaxByteSize = basic.triangleMinMaxsInfo->elementByteSize * 2;

    assert(size_t(baryGroup.triangleCount) == numMeshTriangles);

    parallel_batches(
        numMeshTriangles,
        [&](uint64_t meshLocalTriIdx) {
          const size_t meshGlobalTriIdx  = (mesh.firstPrimitive) + meshLocalTriIdx;
          const size_t baryGlobalTriIdx  = baryGroup.triangleFirst + mesh.displacementMapOffset + meshLocalTriIdx;
          const bary::Triangle& baryPrim = basic.triangles[baryGlobalTriIdx];
          assert(baryGlobalTriIdx < basic.trianglesCount);

          // Compute the flat triangle
          MicromeshUncBaseTri& flat = baseTriData[meshLocalTriIdx];
          flat.subdivLevel          = baryPrim.subdivLevel;
          flat.firstValue           = baryPrim.valuesOffset;
          if (valueFormat == bary::Format::eR11_unorm_packed_align32)
          {
            flat.firstValue /= sizeof(uint32_t);
          }

          if(decimateEdgeFlags)
          {
            flat.topoBits = decimateEdgeFlags[meshGlobalTriIdx];
          }
          else
          {
            flat.topoBits = 0;
          }
          flat.meshletCount = uint32_t(baryMap.getLevel(flat.subdivLevel, flat.topoBits, MAX_BARYMAP_LEVELS).headersCount);

          // Compute the bounding sphere
          glm::vec4& sphere = sphereData[meshLocalTriIdx];

          const float valueBias  = baryGroup.floatBias.r;
          const float valueScale = baryGroup.floatScale.r;

          const float minDisp =
              getBaryMinMaxValue(basic.valuesInfo->valueFormat, basic.triangleMinMaxs, baryGlobalTriIdx * 2 + 0) * valueScale + valueBias;
          const float maxDisp =
              getBaryMinMaxValue(basic.valuesInfo->valueFormat, basic.triangleMinMaxs, baryGlobalTriIdx * 2 + 1) * valueScale + valueBias;

          sphere = computeSphere(meshSet, meshGlobalTriIdx, minDisp, maxDisp, mesh.directionBoundsAreUniform);
        },
        numThreads);
  }

  // shading
  if(barySet.getShading(mesh.baryNormalID))
  {
    bary::BasicView    basic     = barySet.shadings[mesh.baryNormalID].attribute->getView();
    const bary::Group& baryGroup = basic.groups[mesh.displacementGroup];

    parallel_batches(
        numMeshTriangles,
        [&](uint64_t meshLocalTriIdx) {
          const size_t meshGlobalTriIdx  = (mesh.firstPrimitive) + meshLocalTriIdx;
          const size_t baryGlobalTriIdx  = baryGroup.triangleFirst + mesh.displacementMapOffset + meshLocalTriIdx;
          const bary::Triangle& baryPrim = basic.triangles[baryGlobalTriIdx];
          assert(baryGlobalTriIdx < basic.trianglesCount);

          MicromeshUncBaseTri& flat = baseTriData[meshLocalTriIdx];
          assert(flat.subdivLevel == baryPrim.subdivLevel);
          flat.firstShadingValue = baryPrim.valuesOffset;
        },
        numThreads);
  }
}

}  // namespace microdisp
