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

#include "micromesh_compressed_vk.hpp"
#include "micromesh_decoder_utils_vk.hpp"
#include "resources_vk.hpp"
#include "parallel_work.hpp"

namespace microdisp {

void MicromeshSetCompressedVK::initBasics(ResourcesVK& res, const MeshSet& meshSet, const BaryAttributesSet& barySet, bool useBaseTriangles, bool useMips)
{
  memset(usedFormats, 0, sizeof(usedFormats));
  hasBaseTriangles = useBaseTriangles;

  meshDatas.clear();
  meshDatas.resize(meshSet.meshInfos.size());

  // allocation phase & smaller uploads
  nvvk::StagingMemoryManager* staging = res.m_allocator.getStaging();
  VkCommandBuffer             cmd     = res.createTempCmdBuffer();

  for(size_t m = 0; m < meshSet.meshInfos.size(); m++)
  {
    uint32_t        meshID   = uint32_t(m);
    const MeshInfo& mesh     = meshSet.meshInfos[m];
    MeshData&       meshData = meshDatas[m];

    if(mesh.displacementID == MeshSetID::INVALID)
      continue;

    assert(barySet.displacements[mesh.displacementID].compressed);

    const bary::BasicView     basic          = barySet.displacements[mesh.displacementID].compressed->getView();
    bary::Group               baryGroup      = basic.groups[mesh.displacementGroup];
    bary::GroupHistogramRange baryHistoGroup = basic.groupHistogramRanges[mesh.displacementGroup];

    // for now assume 1:1
    assert(baryGroup.triangleCount == (mesh.numPrimitives));

    meshData.combinedData = new MicromeshCombinedData;

    // init buffers
    meshData.binding = res.createBuffer(sizeof(MicromeshCombinedData),
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    for(uint32_t i = 0; i < baryHistoGroup.entryCount; i++)
    {
      usedFormats[basic.histogramEntries[i + baryHistoGroup.entryFirst].blockFormat] = true;
    }

    if(useBaseTriangles)
    {
      uint32_t baseTriangleCount = mesh.numPrimitives;

      meshData.baseTriangles = res.createBuffer(sizeof(MicromeshBaseTri) * baseTriangleCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      meshData.baseSpheres = res.createBuffer(sizeof(glm::vec4) * baseTriangleCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      meshData.microTriangleCount = baseTriangleCount;
    }
    else
    {
      assert(0 && "subtri codepath removed to lower complexity");
    }

    // add safety margin for out-of-bounds access
    meshData.distances =
        res.createBuffer(basic.valuesInfo->valueByteSize * baryGroup.valueCount + 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // only for visualization purposes, not required for actual rendering
    meshData.baseTriangleMinMaxs = res.createBuffer(basic.triangleMinMaxsInfo->elementByteSize * 2 * mesh.numPrimitives,
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    staging->cmdToBuffer(cmd, meshData.baseTriangleMinMaxs.buffer, 0, meshData.baseTriangleMinMaxs.info.range,
                         basic.triangleMinMaxs + (basic.triangleMinMaxsInfo->elementByteSize * 2 * baryGroup.triangleFirst));

    if(useMips)
    {
      assert(barySet.displacements[mesh.displacementID].compressedMisc);

      const bary::MiscView       misc         = barySet.displacements[mesh.displacementID].compressedMisc->getView();
      bary::GroupUncompressedMip baryMipGroup = misc.groupUncompressedMips[mesh.displacementGroup];

      meshData.mipDistances = res.createBuffer(misc.uncompressedMipsInfo->elementByteSize * baryMipGroup.mipCount,
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

      staging->cmdToBuffer(cmd, meshData.mipDistances.buffer, 0, meshData.mipDistances.info.range,
                           misc.uncompressedMips + (misc.uncompressedMipsInfo->elementByteSize * baryMipGroup.mipFirst));
    }

  }
  res.tempSyncSubmit(cmd);

  // slightly bigger uploads
  for(size_t m = 0; m < meshSet.meshInfos.size(); m++)
  {
    uint32_t        meshID   = uint32_t(m);
    const MeshInfo& mesh     = meshSet.meshInfos[m];
    const MeshData& meshData = meshDatas[m];

    if(mesh.displacementID == MeshSetID::INVALID)
      continue;


    const bary::BasicView basic = barySet.displacements[mesh.displacementID].compressed->getView();
    const bary::Group&    group = basic.groups[mesh.displacementGroup];

    res.simpleUploadBuffer(meshData.distances, basic.values + (basic.valuesInfo->valueByteSize * group.valueFirst));
  }
}
void MicromeshSetCompressedVK::uploadMeshDatasBinding(nvvk::StagingMemoryManager* staging, VkCommandBuffer cmd)
{
  for(const auto& meshData : meshDatas)
  {
    meshData.combinedData->fillAddresses(*this, meshData);

    staging->cmdToBuffer(cmd, meshData.binding.info.buffer, meshData.binding.info.offset, meshData.binding.info.range,
                         meshData.combinedData);
  }
}

void MicromeshSetCompressedVK::initAttributeNormals(ResourcesVK& res, const MeshSet& meshSet, const BaryAttributesSet& barySet, uint32_t numThreads)
{
  VkCommandBuffer cmd = res.createTempCmdBuffer();

  for(size_t m = 0; m < meshSet.meshInfos.size(); m++)
  {
    uint32_t        meshID   = uint32_t(m);
    const MeshInfo& mesh     = meshSet.meshInfos[m];
    MeshData&       meshData = meshDatas[m];

    if(mesh.displacementID == MeshSetID::INVALID)
      continue;

    const BaryShadingAttribute* shadingAttr = barySet.getDisplacementShading(mesh.displacementID, SHADING_ATTRIBUTE_NORMAL_BIT);
    if(!shadingAttr)
      continue;


    const bary::BasicView basic = shadingAttr->attribute->getView();
    const bary::Group&    group = basic.groups[mesh.displacementGroup];

    meshData.attrTriangles = res.createBuffer(sizeof(uint32_t) * mesh.numPrimitives, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    meshData.attrNormals = res.createBuffer(basic.valuesInfo->valueByteSize * group.valueCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // create per-triangle offsets for attributes, for every mesh triangle
    {
      uint32_t* flatData = res.m_allocator.getStaging()->cmdToBufferT<uint32_t>(
          cmd, meshData.attrTriangles.buffer, meshData.attrTriangles.info.offset, meshData.attrTriangles.info.range);

      const size_t numMeshTriangles = mesh.numPrimitives;

      assert(size_t(group.triangleCount) == numMeshTriangles);

      parallel_batches(
          numMeshTriangles,
          [&](uint64_t meshLocalTriIdx) {
            const size_t baryGlobalTriIdx  = group.triangleFirst + mesh.displacementMapOffset + meshLocalTriIdx;
            const bary::Triangle& baryPrim = basic.triangles[baryGlobalTriIdx];
            assert(baryGlobalTriIdx < basic.trianglesCount);

            // Compute the flat triangle
            flatData[meshLocalTriIdx] = baryPrim.valuesOffset;
          },
          numThreads);
    }
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

    const BaryShadingAttribute* shadingAttr = barySet.getDisplacementShading(mesh.displacementID, SHADING_ATTRIBUTE_NORMAL_BIT);
    if(shadingAttr)
    {
      const bary::BasicView basic = shadingAttr->attribute->getView();
      const bary::Group&    group = basic.groups[mesh.displacementGroup];

      res.simpleUploadBuffer(meshData.attrNormals, basic.values + (basic.valuesInfo->valueByteSize * group.valueFirst));
    }
  }
}

void MicromeshSetCompressedVK::deinit(ResourcesVK& res)
{
  for(MeshData& mdata : meshDatas)
  {
    res.destroy(mdata.subTriangles);
    res.destroy(mdata.subSpheres);
    res.destroy(mdata.baseTriangles);
    res.destroy(mdata.baseSpheres);
    res.destroy(mdata.distances);
    res.destroy(mdata.mipDistances);
    res.destroy(mdata.baseTriangleMinMaxs);
    res.destroy(mdata.binding);
    res.destroy(mdata.attrNormals);
    res.destroy(mdata.attrTriangles);

    if(mdata.combinedData)
    {
      delete mdata.combinedData;
    }
  }

  res.destroy(umajor2bmap);
  res.destroy(triangleIndices);
  res.destroy(vertices);
  res.destroy(descends);

  meshDatas.clear();
}

}  // namespace microdisp
