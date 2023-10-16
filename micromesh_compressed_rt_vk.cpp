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
#include "micromesh_compressed_rt_vk.hpp"
#include "resources_vk.hpp"
#include <nvh/alignment.hpp>
#include "parallel_work.hpp"

#include "micromesh_decoder_utils_vk.hpp"
#include "common.h"
#include "common_barymap.h"
#include "common_micromesh_compressed_rt.h"

namespace microdisp {

static_assert(sizeof(VkMicromapTriangleEXT) == sizeof(bary::Triangle), "sizeof mismatch VkMicromapTriangleEXT and bary::Triangle");
static_assert(offsetof(VkMicromapTriangleEXT, dataOffset) == offsetof(bary::Triangle, valuesOffset),
              "dataOffset mismatch VkMicromapTriangleNV and nv::BaryPrimitive");
static_assert(offsetof(VkMicromapTriangleEXT, subdivisionLevel) == offsetof(bary::Triangle, subdivLevel),
              "subdivLevel mismatch VkMicromapTriangleNV and nv::BaryPrimitive");
static_assert(offsetof(VkMicromapTriangleEXT, format) == offsetof(bary::Triangle, blockFormat),
              "format mismatch VkMicromapTriangleNV and nv::BaryPrimitive");

void MicromeshSetCompressedRayTracedVK::init(ResourcesVK& res, const MeshSet& meshSet, const BaryAttributesSet& barySet, uint32_t numThreads)
{
  VkResult result = VK_ERROR_UNKNOWN;

  struct BuildGroup
  {
    Group*   group;
    uint32_t displacementID;
    uint32_t displacementGroupID;

    VkDeviceSize inputSize;
    VkDeviceSize inputOffset;

    VkDeviceSize scratchSize;
    VkDeviceSize scratchOffset;
  };

  std::vector<BuildGroup> buildGroups;

  VkDeviceSize inputAlignment   = 256;
  VkDeviceSize scratchAlignment = 256;

  if(g_enableMicromeshRTExtensions)
  {
    VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accProps = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    props2.pNext = &accProps;
    vkGetPhysicalDeviceProperties2(res.m_physical, &props2);

    // latest spec and newer drivers no longer have dedicated alignment fields,
    // following behavior is in the spec
    inputAlignment   = 256;
    scratchAlignment = accProps.minAccelerationStructureScratchOffsetAlignment;
  }

  VkDeviceSize memMicromaps = 0;

  // setup allocation so we can have stable pointers to groups within buildGroups
  displacements.resize(barySet.displacements.size());
  for(size_t d = 0; d < barySet.displacements.size(); d++)
  {
    if(!barySet.displacements[d].compressed)
      continue;

    displacements[d].groups.resize(barySet.displacements[d].compressed->groups.size());

    for(size_t g = 0; g < displacements[d].groups.size(); g++)
    {
      Group&             group     = displacements[d].groups[g];
      bary::BasicView    baryDescr = barySet.displacements[d].compressed->getView();
      const bary::Group& baryGroup = baryDescr.groups[g];

      group.usages.resize(baryDescr.groupHistogramRanges[g].entryCount);
      const bary::HistogramEntry* histoEntries = baryDescr.histogramEntries + baryDescr.groupHistogramRanges[g].entryFirst;

      for(size_t i = 0; i < group.usages.size(); i++)
      {
        group.usages[i].count            = histoEntries[i].count;
        group.usages[i].format           = histoEntries[i].blockFormat;
        group.usages[i].subdivisionLevel = histoEntries[i].subdivLevel;
      }

      BuildGroup bgroup;
      bgroup.group               = &group;
      bgroup.displacementID      = uint32_t(d);
      bgroup.displacementGroupID = uint32_t(g);

      bgroup.inputSize = nvh::align_up(baryDescr.valuesInfo->valueByteSize * baryGroup.valueCount, inputAlignment)
                         + (sizeof(VkMicromapTriangleEXT) * baryGroup.triangleCount);

      // compute size
      VkMicromapBuildInfoEXT buildInfo      = {VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT};
      buildInfo.type                        = VK_MICROMAP_TYPE_DISPLACEMENT_MICROMAP_NV;
      buildInfo.flags                       = 0;
      buildInfo.mode                        = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
      buildInfo.dstMicromap                 = VK_NULL_HANDLE;
      buildInfo.usageCountsCount            = uint32_t(group.usages.size());
      buildInfo.pUsageCounts                = group.usages.data();
      buildInfo.data.deviceAddress          = 0ull;
      buildInfo.triangleArray.deviceAddress = 0ull;
      buildInfo.triangleArrayStride         = 0;

      VkMicromapBuildSizesInfoEXT sizeInfo = {VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT};
      if(g_enableMicromeshRTExtensions)
      {
        vkGetMicromapBuildSizesEXT(res.m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &sizeInfo);
        assert(sizeInfo.micromapSize && "sizeInfo.micromeshSize was zero");
        group.micromeshData = res.createBuffer(sizeInfo.micromapSize, VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT);
        bgroup.scratchSize  = std::max(sizeInfo.buildScratchSize, VkDeviceSize(4));
      }
      else
      {
        // code path for validation fallback, does some dummy work
        sizeInfo.micromapSize = bgroup.inputSize;
        group.micromeshData   = res.createBuffer(sizeInfo.micromapSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        bgroup.scratchSize    = 4;
      }

      memMicromaps += sizeInfo.micromapSize;

      // create micromesh
      VkMicromapCreateInfoEXT mmCreateInfo = {VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT};
      mmCreateInfo.createFlags             = 0;
      mmCreateInfo.buffer                  = group.micromeshData.buffer;
      mmCreateInfo.offset                  = 0;
      mmCreateInfo.size                    = sizeInfo.micromapSize;
      mmCreateInfo.type                    = VK_MICROMAP_TYPE_DISPLACEMENT_MICROMAP_NV;
      mmCreateInfo.deviceAddress           = 0ull;
      if(g_enableMicromeshRTExtensions)
      {
        result = vkCreateMicromapEXT(res.m_device, &mmCreateInfo, nullptr, &group.micromap);
        assert(result == VK_SUCCESS && "vkCreateMicromeshNV failed");
      }
      else
      {
        // code path for validation fallback, does some dummy work
        group.micromap = (VkMicromapEXT)((d + 1) * 1000 + g);
      }

      buildGroups.push_back(bgroup);
    }
  }

  LOGI("GeometryMM allocated size %d KB\n", memMicromaps / 1024);


  RBuffer      inputBuffer;
  VkDeviceSize inputSize      = 0;
  VkDeviceSize inputOffset    = 0;
  VkDeviceSize inputThreshold = 128 * 1024 * 1024;

  RBuffer      scratchBuffer;
  VkDeviceSize scratchSize      = 0;
  VkDeviceSize scratchOffset    = 0;
  VkDeviceSize scratchThreshold = 128 * 1024 * 1024;

  size_t tempFrom  = 0;
  size_t tempCount = 0;

  double buildTime = 0;

  for(size_t b = 0; b < buildGroups.size(); b++)
  {
    {
      BuildGroup& bgroup = buildGroups[b];

      bgroup.inputOffset   = inputOffset;
      bgroup.scratchOffset = scratchOffset;

      inputOffset = (inputOffset + bgroup.inputSize + inputAlignment - 1) & (~(inputAlignment - 1));
      inputSize   = bgroup.inputOffset + bgroup.inputSize;

      scratchOffset = (scratchOffset + bgroup.scratchSize + scratchAlignment - 1) & (~(scratchAlignment - 1));
      scratchSize   = bgroup.scratchOffset + bgroup.scratchSize;

      tempCount++;
    }

    // if > threshold or last
    if(inputSize > inputThreshold || scratchSize > scratchThreshold || (b == buildGroups.size() - 1))
    {
      if(inputBuffer.info.range < inputSize)
      {
        // recreate buffer with new size
        res.destroy(inputBuffer);
        if(g_enableMicromeshRTExtensions)
        {
          inputBuffer = res.createBuffer(inputSize, VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT);
        }
        else
        {
          // code path for validation fallback, does some dummy work
          inputBuffer = res.createBuffer(inputSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        }
      }

      if(scratchBuffer.info.range < scratchSize)
      {
        // recreate buffer with new size
        res.destroy(scratchBuffer);
        if(g_enableMicromeshRTExtensions)
        {
          scratchBuffer = res.createBuffer(scratchSize, VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT);
        }
        else
        {
          // code path for validation fallback, does some dummy work
          scratchBuffer = res.createBuffer(scratchSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        }
      }

      VkCommandBuffer cmd = res.createTempCmdBuffer();

      uint8_t* stagingMem =
          res.m_allocator.getStaging()->cmdToBufferT<uint8_t>(cmd, inputBuffer.buffer, 0, inputBuffer.info.range);

      // do input uploads
      for(size_t i = tempFrom; i < tempFrom + tempCount; i++)
      {
        BuildGroup&        bgroup    = buildGroups[i];
        bary::BasicView    baryDescr = barySet.displacements[bgroup.displacementID].compressed->getView();
        const bary::Group* baryGroup = baryDescr.groups + bgroup.displacementGroupID;

        size_t valuesSize = baryDescr.valuesInfo->valueByteSize * baryGroup->valueCount;
        size_t primOffset = nvh::align_up(valuesSize, inputAlignment);

        // values first then baryprims
        memcpy(stagingMem + bgroup.inputOffset,
               baryDescr.values + baryGroup->valueFirst * baryDescr.valuesInfo->valueByteSize, valuesSize);

        const VkMicromapTriangleEXT* groupPrims =
            reinterpret_cast<const VkMicromapTriangleEXT*>(baryDescr.triangles + baryGroup->triangleFirst);
        memcpy(stagingMem + bgroup.inputOffset + primOffset, groupPrims, sizeof(VkMicromapTriangleEXT) * baryGroup->triangleCount);
      }

      // barrier for upload finish
      VkMemoryBarrier2 memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
      memBarrier.srcStageMask     = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      memBarrier.srcAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      if(g_enableMicromeshRTExtensions)
      {
        memBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT;
        memBarrier.dstAccessMask = VK_ACCESS_2_MICROMAP_READ_BIT_EXT;
      }
      else
      {
        // code path for validation fallback, does some dummy work
        memBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
      }
      VkDependencyInfo depInfo   = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      depInfo.memoryBarrierCount = 1;
      depInfo.pMemoryBarriers    = &memBarrier;
      vkCmdPipelineBarrier2(cmd, &depInfo);

      // do builds
      std::vector<VkMicromapBuildInfoEXT> builds;
      builds.reserve(tempCount);

      res.tempStopWatch(cmd, true);

      for(size_t i = tempFrom; i < tempFrom + tempCount; i++)
      {
        BuildGroup&        bgroup    = buildGroups[i];
        Group&             group     = *bgroup.group;
        bary::BasicView    baryDescr = barySet.displacements[bgroup.displacementID].compressed->getView();
        const bary::Group* baryGroup = baryDescr.groups + bgroup.displacementGroupID;

        size_t valuesSize = baryDescr.valuesInfo->valueByteSize * baryGroup->valueCount;
        size_t primOffset = nvh::align_up(valuesSize, inputAlignment);

        VkMicromapBuildInfoEXT buildInfo      = {VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT};
        buildInfo.type                        = VK_MICROMAP_TYPE_DISPLACEMENT_MICROMAP_NV;
        buildInfo.flags                       = 0;
        buildInfo.mode                        = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
        buildInfo.dstMicromap                 = group.micromap;
        buildInfo.usageCountsCount            = uint32_t(group.usages.size());
        buildInfo.pUsageCounts                = group.usages.data();
        buildInfo.scratchData.deviceAddress   = scratchBuffer.addr + bgroup.scratchOffset;
        buildInfo.data.deviceAddress          = inputBuffer.addr + bgroup.inputOffset;
        buildInfo.triangleArray.deviceAddress = inputBuffer.addr + bgroup.inputOffset + primOffset;
        buildInfo.triangleArrayStride         = sizeof(VkMicromapTriangleEXT);

        builds.push_back(buildInfo);

        if(!g_enableMicromeshRTExtensions)
        {
          // code path for validation fallback, does some dummy work
          VkBufferCopy cpy;
          cpy.size      = valuesSize + sizeof(bary::Triangle) * baryGroup->triangleCount;
          cpy.srcOffset = bgroup.inputOffset;
          cpy.dstOffset = 0;
          vkCmdCopyBuffer(cmd, inputBuffer.buffer, group.micromeshData.buffer, 1, &cpy);
        }
      }
      if(g_enableMicromeshRTExtensions)
      {
        vkCmdBuildMicromapsEXT(cmd, uint32_t(builds.size()), builds.data());
      }

      res.tempStopWatch(cmd, false);

      // flush and wait
      res.tempSyncSubmit(cmd);

      buildTime += res.getStopWatchResult();

      tempFrom  = b + 1;
      tempCount = 0;
    }
  }

  LOGI("GeometryMM build time: ~ %d (us)\n", uint32_t(buildTime));

  res.destroy(inputBuffer);
  res.destroy(scratchBuffer);
}

void MicromeshSetCompressedRayTracedVK::deinit(ResourcesVK& res)
{
  for(auto& itdisp : displacements)
  {
    for(auto& itgroup : itdisp.groups)
    {
      if(itgroup.micromap && g_enableMicromeshRTExtensions)
      {
        vkDestroyMicromapEXT(res.m_device, itgroup.micromap, nullptr);
      }
      res.destroy(itgroup.micromeshData);
    }
  }

  for(auto& mesh : meshDatas)
  {
    res.destroy(mesh.attrNormals);
    res.destroy(mesh.attrTriangles);
  }

  res.destroy(umajor2bmap);
  res.destroy(instanceAttributes);
  res.destroy(binding);

  displacements.clear();
}

void MicromeshSetCompressedRayTracedVK::initAttributeNormals(ResourcesVK&             res,
                                                             const MeshSet&           meshSet,
                                                             const BaryAttributesSet& barySet,
                                                             uint32_t                 numThreads /*= 0*/)
{
  meshDatas.resize(meshSet.meshInfos.size());

  VkCommandBuffer             cmd     = res.createTempCmdBuffer();
  nvvk::StagingMemoryManager* staging = res.m_allocator.getStaging();

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

    meshData.attrTriangles = res.createBuffer(sizeof(MicromeshRtAttrTri) * mesh.numPrimitives, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    meshData.attrNormals = res.createBuffer(basic.valuesInfo->valueByteSize * group.valueCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // create per-triangle offsets for attributes, for every mesh triangle
    {
      MicromeshRtAttrTri* flatData = staging->cmdToBufferT<MicromeshRtAttrTri>(
          cmd, meshData.attrTriangles.buffer, meshData.attrTriangles.info.offset, meshData.attrTriangles.info.range);

      const size_t numMeshTriangles = mesh.numPrimitives;

      assert(size_t(group.triangleCount) == numMeshTriangles);

      parallel_batches(
          numMeshTriangles,
          [&](uint64_t meshLocalTriIdx) {
            const size_t          baryGlobalTriIdx = group.triangleFirst + mesh.displacementMapOffset + meshLocalTriIdx;
            const bary::Triangle& baryPrim         = basic.triangles[baryGlobalTriIdx];
            assert(baryGlobalTriIdx < basic.trianglesCount);

            // Compute the flat triangle
            flatData[meshLocalTriIdx].firstValue  = baryPrim.valuesOffset;
            flatData[meshLocalTriIdx].subdivLevel = baryPrim.subdivLevel;
          },
          numThreads);
    }
  }

  // instances buffer
  instanceAttributes =
      res.createBuffer(sizeof(MicromeshRtAttributes) * meshSet.meshInstances.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  {
    MicromeshRtAttributes* instanceAttributeData =
        staging->cmdToBufferT<MicromeshRtAttributes>(cmd, instanceAttributes.buffer, 0,
                                                     sizeof(MicromeshRtAttributes) * meshSet.meshInstances.size());
    for(size_t i = 0; i < meshSet.meshInstances.size(); i++)
    {
      MeshData& meshData                     = meshDatas[meshSet.meshInstances[i].meshID];
      instanceAttributeData[i].attrNormals   = meshData.attrNormals.addr;
      instanceAttributeData[i].attrTriangles = meshData.attrTriangles.addr;
    }
  }

  // binding buffer
  {
    // umajor2bmap table
    MicromeshRtData bindingData;
    initBmapIndices(umajor2bmap, bindingData.umajor2bmap, res, staging, cmd, barySet);
    binding = res.createBuffer(sizeof(MicromeshRtData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    staging->cmdToBuffer(cmd, binding.buffer, 0, sizeof(MicromeshRtData), &bindingData);
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

}  // namespace microdisp
