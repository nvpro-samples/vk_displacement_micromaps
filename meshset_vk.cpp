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
#include "resources_vk.hpp"

#include <glm/glm.hpp>
#include <nvvk/images_vk.hpp>
#include <stb_image.h>

#include "common.h"
#include "common_mesh.h"
#include "meshset_vk.hpp"
#include "octant_encoding.h"
#include "parallel_work.hpp"
#include "micromesh_compressed_rt_vk.hpp"
#include "vk_nv_micromesh.h"

namespace microdisp {
static void convertToOct(uint32_t* dst, const std::vector<glm::vec3>& vecs)
{
  const auto src = vecs.data();

  parallel_batches(
      vecs.size(),                                         //
      [&](uint64_t i) { dst[i] = vec_to_oct32(src[i]); },  //
      g_numThreads);
}

// Use with T == VkDeviceOrHostAddressConstKHR or T == VkDeviceOrHostAddressKHR
template <typename T>
T getBufferAddress(VkDevice device, VkBuffer buffer, VkDeviceSize offset = 0)
{
  VkBufferDeviceAddressInfo addressInfo = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  addressInfo.buffer                    = buffer;
  T address;
  address.deviceAddress = vkGetBufferDeviceAddress(device, &addressInfo) + offset;
  return address;
}


template <typename T>
void checkAllocSuccess(ResourcesVK& res, T handle)
{
  if(handle == VK_NULL_HANDLE)
  {
    VkDeviceSize allocatedSize, usedSize;
    res.m_memAllocator.getUtilization(allocatedSize, usedSize);
    LOGE("out of memory: used %d KB allocated %d KB\n", usedSize / 1024, allocatedSize / 1024);
    exit(-1);
  }
}

void MeshAttributesVK::init(ResourcesVK& res, const MeshSet& meshSet)
{
  if(meshSet.attributes.normals.size() > 0)
  {

    normals = res.createBuffer(sizeof(uint32_t) * meshSet.attributes.normals.size(),
                               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    checkAllocSuccess(res, normals.buffer);
  }

  if(meshSet.attributes.tangents.size() > 0)
  {
    tangents = res.createBuffer(sizeof(uint32_t) * meshSet.attributes.tangents.size(),
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    checkAllocSuccess(res, tangents.buffer);
  }
  if(meshSet.attributes.bitangents.size() > 0)
  {
    bitangents = res.createBuffer(sizeof(uint32_t) * meshSet.attributes.bitangents.size(),
                                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    checkAllocSuccess(res, bitangents.buffer);
  }
  if(meshSet.attributes.texcoords0.size() > 0)
  {
    tex0s = res.createBuffer(sizeof(glm::vec2) * meshSet.attributes.texcoords0.size(),
                             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    checkAllocSuccess(res, tex0s.buffer);
  }


  nvvk::StagingMemoryManager* staging = res.m_allocator.getStaging();
  res.tempResetResources();

  VkCommandBuffer cmd = res.createTempCmdBuffer();
  convertToOct(staging->cmdToBufferT<uint32_t>(cmd, normals.buffer, 0, normals.info.range), meshSet.attributes.normals);
  convertToOct(staging->cmdToBufferT<uint32_t>(cmd, tangents.buffer, 0, tangents.info.range), meshSet.attributes.tangents);
  convertToOct(staging->cmdToBufferT<uint32_t>(cmd, bitangents.buffer, 0, bitangents.info.range), meshSet.attributes.bitangents);

  res.tempSyncSubmit(cmd);

  res.simpleUploadBuffer(tex0s, meshSet.attributes.texcoords0.data());
}

void MeshAttributesVK::deinit(ResourcesVK& res)
{
  res.destroy(normals);
  res.destroy(tangents);
  res.destroy(bitangents);
  res.destroy(tex0s);
}

void MeshSetVK::init(ResourcesVK& res, const MeshSet& meshSet)
{
  positions = res.createBuffer(sizeof(glm::vec3) * meshSet.attributes.positions.size(),
                               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                   | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  checkAllocSuccess(res, positions.buffer);
  indices = res.createBuffer(sizeof(uint32_t) * meshSet.indices.size(),
                             VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                 | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  checkAllocSuccess(res, indices.buffer);
  instances = res.createBuffer(sizeof(InstanceData) * meshSet.meshInstances.size(),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  checkAllocSuccess(res, instances.buffer);

  binding = res.createBuffer(sizeof(MeshData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  checkAllocSuccess(res, binding.buffer);

  attr.init(res, meshSet);

  res.simpleUploadBuffer(indices, meshSet.indices.data());
  res.simpleUploadBuffer(positions, meshSet.attributes.positions.data());

  updateInstances(res, meshSet);
}


void MeshSetVK::initNormalMaps(ResourcesVK& res, const MeshSet& meshSet)
{
  materialTextures.resize(meshSet.textures.size());

  for(size_t i = 0; i < meshSet.textures.size(); i++)
  {
    int     width;
    int     height;
    int     components;
    uint8_t defaultData[4] = {127, 127, 128, 128};
    defaultData;

    stbi_uc* data = meshSet.textures[i].filename == "defaultNormalMap" ?
                        nullptr :
                        stbi_load(meshSet.textures[i].filename.c_str(), &width, &height, &components, STBI_rgb_alpha);
    if(!data)
    {
      width  = 1;
      height = 1;
    }

    // note, this is a very basic implementation not for quality/performance.
    // proper renderer would use BC compressed textures with mipmaps.

    VkCommandBuffer cmd       = res.createTempCmdBuffer();
    VkDeviceSize    imgSize   = VkDeviceSize(width) * height * 4;
    VkExtent2D      imgExtent = {uint32_t(width), uint32_t(height)};
    VkImageCreateInfo imgInfo = nvvk::makeImage2DCreateInfo(imgExtent, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
    VkSamplerCreateInfo samplerInfo = nvvk::makeSamplerCreateInfo();
    samplerInfo.addressModeU        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    materialTextures[i] = res.m_allocator.createTexture(cmd, imgSize, data ? data : defaultData, imgInfo, samplerInfo);
    res.tempSyncSubmit(cmd);

    if(data)
    {
      stbi_image_free(data);
    }
  }
}

void MeshSetVK::initDisplacementDirections(ResourcesVK& res, const MeshSet& meshSet)
{
  res.destroy(displacementDirections);

  if(!meshSet.attributes.directions.empty())
  {
    displacementDirections = res.createBuffer(meshSet.attributes.directions.size() * sizeof(f16vec4),
                                              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    checkAllocSuccess(res, displacementDirections.buffer);

    nvvk::StagingMemoryManager* staging = res.m_allocator.getStaging();
    res.tempResetResources();
    {
      VkCommandBuffer cmd = res.createTempCmdBuffer();
      f16vec4* dirs = staging->cmdToBufferT<f16vec4>(cmd, displacementDirections.buffer, 0, displacementDirections.info.range);

      for(size_t i = 0; i < meshSet.attributes.directions.size(); i++)
      {
        glm::vec3 nrm = meshSet.attributes.directions[i];
        dirs[i]       = {float16_t(nrm.x), float16_t(nrm.y), float16_t(nrm.z), 0};
      }

      res.tempSyncSubmit(cmd);
    }
  }
  updateBinding(res);
}

void MeshSetVK::initDisplacementBounds(ResourcesVK& res, const MeshSet& meshSet)
{
  res.destroy(displacementDirectionBounds);

  if(!meshSet.attributes.directionBounds.empty())
  {
    displacementDirectionBounds = res.createBuffer(meshSet.attributes.directionBounds.size() * sizeof(boundsVec2),
                                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

#if BOUNDS_AS_FP32
    res.simpleUploadBuffer(displacementDirectionBounds, meshSet.attributes.directionBounds.data());
#else
    nvvk::StagingMemoryManager* staging = res.m_allocator.getStaging();
    res.tempResetResources();
    {
      VkCommandBuffer cmd = res.createTempCmdBuffer();
      f16vec2* bounds =
          staging->cmdToBufferT<f16vec2>(cmd, displacementDirectionBounds.buffer, 0, displacementDirectionBounds.info.range);

      for(size_t i = 0; i < meshSet.attributes.directionBounds.size(); i++)
      {
        bounds[i] = {float16_t(meshSet.attributes.directionBounds[i].x), float16_t(meshSet.attributes.directionBounds[i].y)};
      }

      res.tempSyncSubmit(cmd);
    }
#endif
  }

  updateBinding(res);
}

void MeshSetVK::initDisplacementEdgeFlags(ResourcesVK& res, const MeshSet& meshSet)
{
  res.destroy(displacementEdgeFlags);

  if(meshSet.decimateEdgeFlags.size())
  {
    displacementEdgeFlags = res.createBuffer(meshSet.decimateEdgeFlags.size() * sizeof(uint8_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    checkAllocSuccess(res, displacementEdgeFlags.buffer);

    res.simpleUploadBuffer(displacementEdgeFlags, meshSet.decimateEdgeFlags.data());
  }

  updateBinding(res);
}

void MeshSetVK::deinit(ResourcesVK& resources)
{
  if(!positions.buffer)
    return;

  attr.deinit(resources);

  resources.destroy(positions);
  resources.destroy(indices);
  resources.destroy(instances);

  resources.destroy(displacementDirections);
  resources.destroy(displacementDirectionBounds);
  resources.destroy(displacementEdgeFlags);

  resources.destroy(binding);

  deinitRayTracing(resources);

  for(auto& it : materialTextures)
  {
    resources.destroy(it);
  }

  materialTextures.clear();
}

void MeshSetVK::deinitRayTracing(ResourcesVK& resources)
{
  for(MeshRT& it : rtMeshes)
  {
    resources.m_allocator.destroy(it.blas);
  }

  rtMeshes = std::vector<MeshRT>();

  resources.m_allocator.destroy(sceneTlas);
}

void MeshSetVK::updateBinding(ResourcesVK& res)
{
  MeshData mesh;
  mesh.indices             = indices.addr;
  mesh.positions           = positions.addr;
  mesh.instances           = instances.addr;
  mesh.normals             = attr.normals.addr;
  mesh.tangents            = attr.tangents.buffer ? attr.tangents.addr : attr.normals.addr;
  mesh.bitangents          = attr.bitangents.buffer ? attr.bitangents.addr : attr.normals.addr;
  mesh.tex0s               = attr.tex0s.buffer ? attr.tex0s.addr : positions.addr;
  mesh.dispDirections      = displacementDirections.buffer ? displacementDirections.addr : attr.normals.addr;
  mesh.dispDirectionBounds = displacementDirectionBounds.buffer ? displacementDirectionBounds.addr : 0;
  mesh.dispDecimateFlags   = displacementEdgeFlags.buffer ? displacementEdgeFlags.addr : 0;

  res.simpleUploadBuffer(binding, &mesh);
}

void MeshSetVK::updateInstances(ResourcesVK& res, const MeshSet& meshSet)
{
  res.destroy(instances);

  instances = res.createBuffer(sizeof(InstanceData) * meshSet.meshInstances.size(),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  checkAllocSuccess(res, instances.buffer);

  nvvk::StagingMemoryManager* staging = res.m_allocator.getStaging();
  res.tempResetResources();

  {
    VkCommandBuffer cmd   = res.createTempCmdBuffer();
    InstanceData*   insts = staging->cmdToBufferT<InstanceData>(cmd, instances.buffer, 0, instances.info.range);

    for(const MeshInstance& it : meshSet.meshInstances)
    {
      const MeshInfo& mesh = meshSet.meshInfos[it.meshID];

      if(mesh.materialID != MeshSetID::INVALID && mesh.materialID < meshSet.materials.size())
      {
        insts->color       = glm::vec4(meshSet.materials[mesh.materialID].diffuse, 1.0f);
        insts->normalMapID = meshSet.materials[mesh.materialID].normalMapTextureID;
      }
      else
      {
        insts->color       = vec4(1);
        insts->normalMapID = ~0;
      }
      // meshSet.transformNodes[it.nodeId == tools::common::INVALID_ID ? 0 : it.nodeId].ctm
      insts->worldMatrix  = it.xform;
      insts->worldMatrixI = glm::inverse(it.xform);
      insts->firstIndex   = mesh.firstIndex;
      insts->firstVertex  = mesh.firstVertex;
      insts->bboxMin      = glm::vec4(mesh.bbox.mins, 0.0f);
      insts->bboxMax      = glm::vec4(mesh.bbox.maxs, 0.0f);
      insts->lodSphere    = (insts->bboxMax + insts->bboxMin) * 0.5f;
      insts->lodSphere.w  = mesh.longestEdge * 0.5f;
      insts->lodSubdiv    = mesh.displacementMaxSubdiv;
      insts->lodRange     = glm::length(glm::vec3(insts->bboxMax - insts->bboxMin)) * 0.5f;
      insts++;
    }

    MeshData* ubo = staging->cmdToBufferT<MeshData>(cmd, binding.buffer, 0, binding.info.range);

    res.tempSyncSubmit(cmd);
  }

  updateBinding(res);
}

void MeshSetVK::initRayTracingGeometry(ResourcesVK& res, const MeshSet& meshSet, const MicromeshSetCompressedRayTracedVK* micromeshSetRT)
{
  VkResult        result;
  VkDevice        device = res.m_device;
  nvvk::DebugUtil debugUtil(device);

  size_t numMeshes = meshSet.meshInfos.size();
  rtMeshes.resize(numMeshes);

  VkDeviceSize memRegular = 0;
  VkDeviceSize memCompact = 0;

  size_t                                  tempStart     = 0;
  size_t                                  tempIndex     = 0;
  VkDeviceSize                            tempMem       = 0;
  VkDeviceSize                            tempThreshold = 512 * 1024 * 1024;
  std::vector<VkAccelerationStructureKHR> tempAccsKHR(numMeshes);
  std::vector<nvvk::AccelKHR>             tempAccs(numMeshes);
  std::vector<VkDeviceSize>               tempCompactSize(numMeshes);
  std::vector<VkDeviceSize>               tempScratchSize(numMeshes);
  VkDeviceSize                            tempScratch = 0;

  RBuffer scratch;

  auto scratchResize = [&](VkDeviceSize newSize) {
    // resize scratch buffer if necessary
    if(newSize > scratch.info.range)
    {
      if(scratch.buffer)
      {
        res.destroy(scratch);
      }
      scratch = res.createBuffer(newSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      checkAllocSuccess(res, scratch.buffer);
      debugUtil.setObjectName(scratch.buffer, "RTGeometryScratch");
    }
  };

  VkQueryPool queryPool;
  {
    VkQueryPoolCreateInfo queryCreateInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    queryCreateInfo.queryCount = uint32_t(numMeshes);
    queryCreateInfo.queryType  = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;


    result = vkCreateQueryPool(device, &queryCreateInfo, nullptr, &queryPool);
    assert(result == VK_SUCCESS);
  }

  double buildTime = 0;

  size_t pIndex = 0;
  for(size_t m = 0; m < numMeshes; m++)
  {
    const MeshInfo& mesh   = meshSet.meshInfos[m];
    MeshRT&         meshRT = rtMeshes[m];

    uint32_t vertexCount = mesh.numVertices;
    bool     isLast      = (m == numMeshes - 1);


    meshRT.geometry                    = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    meshRT.geometry.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;
    meshRT.geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    meshRT.geometry.geometry.triangles = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};

    meshRT.blasRange.primitiveCount  = mesh.numPrimitives;
    meshRT.blasRange.primitiveOffset = uint32_t(indices.info.offset + sizeof(uint32_t) * mesh.firstIndex);
    meshRT.blasRange.firstVertex     = mesh.firstVertex;
    meshRT.blasRange.transformOffset = 0;

    meshRT.geometry.geometry.triangles.vertexData = getBufferAddress<VkDeviceOrHostAddressConstKHR>(device, positions.buffer);

    meshRT.geometry.geometry.triangles.vertexStride = sizeof(glm::vec3);
    meshRT.geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    meshRT.geometry.geometry.triangles.maxVertex    = vertexCount;

    meshRT.geometry.geometry.triangles.indexData = getBufferAddress<VkDeviceOrHostAddressConstKHR>(device, indices.buffer);
    meshRT.geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;

    VkBuildAccelerationStructureFlagsKHR buildFlags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR;


    meshRT.geometryDisplacement = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_DISPLACEMENT_MICROMAP_NV};
    if(micromeshSetRT && mesh.displacementID != MeshSetID::INVALID
       && !micromeshSetRT->displacements[mesh.displacementID].groups.empty()
       && (micromeshSetRT->displacements[mesh.displacementID].groups[mesh.displacementGroup].micromap))
    {
      const MicromeshSetCompressedRayTracedVK::Group& micromeshGroup =
          micromeshSetRT->displacements[mesh.displacementID].groups[mesh.displacementGroup];

      meshRT.geometryDisplacement.micromap         = micromeshGroup.micromap;
      meshRT.geometryDisplacement.usageCountsCount = uint32_t(micromeshGroup.usages.size());
      meshRT.geometryDisplacement.pUsageCounts     = micromeshGroup.usages.data();

      if(displacementEdgeFlags.addr)
      {
        meshRT.geometryDisplacement.displacedMicromapPrimitiveFlags.deviceAddress =
            displacementEdgeFlags.addr + mesh.firstPrimitive * sizeof(uint8_t);
        meshRT.geometryDisplacement.displacedMicromapPrimitiveFlagsStride = sizeof(uint8_t);
      }

      // driver should apply mesh.firstVertex offset
      meshRT.geometryDisplacement.displacementVectorBuffer.deviceAddress       = displacementDirections.addr;
      meshRT.geometryDisplacement.displacementVectorStride                     = sizeof(f16vec4);
      meshRT.geometryDisplacement.displacementVectorFormat                     = VK_FORMAT_R16G16B16A16_SFLOAT;
      meshRT.geometryDisplacement.displacementBiasAndScaleBuffer.deviceAddress = displacementDirectionBounds.addr;
      meshRT.geometryDisplacement.displacementBiasAndScaleStride               = sizeof(float) * 2;
      meshRT.geometryDisplacement.displacementBiasAndScaleFormat               = VK_FORMAT_R32G32_SFLOAT;

      if(g_enableMicromeshRTExtensions)
      {
        meshRT.geometry.geometry.triangles.pNext = &meshRT.geometryDisplacement;
      }

      //buildFlags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_EXT;
    }

    // create acc
    VkAccelerationStructureBuildGeometryInfoKHR accCreateInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    accCreateInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    accCreateInfo.flags         = buildFlags | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    accCreateInfo.geometryCount = 1;
    accCreateInfo.pGeometries   = &meshRT.geometry;
    meshRT.geometryInfo         = accCreateInfo;

    VkAccelerationStructureBuildSizesInfoKHR buildSize = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &accCreateInfo,
                                            &meshRT.blasRange.primitiveCount, &buildSize);

    tempScratchSize[tempIndex] = buildSize.buildScratchSize;
    tempScratch += alignedSize(tempScratchSize[tempIndex], 16);

    VkAccelerationStructureCreateInfoKHR createInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.offset                               = 0;
    createInfo.size                                 = buildSize.accelerationStructureSize;
    createInfo.type                                 = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    tempAccs[tempIndex]                             = res.m_allocator.createAcceleration(createInfo);

    if(tempAccs[tempIndex].accel == VK_NULL_HANDLE)
    {
      LOGE("Cannot create temporary raytracing bottom-level acceleration structure\n");
      exit(-1);
    }
    std::string debugName = "RTGeometryTemp:" + std::to_string(pIndex);
    debugUtil.setObjectName(tempAccs[tempIndex].accel, debugName);

    tempAccsKHR[tempIndex] = tempAccs[tempIndex].accel;

    meshRT.geometryInfo.dstAccelerationStructure = tempAccs[tempIndex].accel;

    memRegular += buildSize.accelerationStructureSize;
    tempMem += buildSize.accelerationStructureSize;

    if(tempMem > tempThreshold || (isLast))
    {
      size_t tempCount = tempIndex + 1;
      assert(tempCount == (pIndex - tempStart + 1));

      scratchResize(tempScratch);

      // trigger builds
      VkCommandBuffer cmd           = res.createTempCmdBuffer();
      VkDeviceSize    scratchOffset = 0;

      debugUtil.setObjectName(cmd, "RTGeometryCmd");

      vkCmdResetQueryPool(cmd, queryPool, 0, uint32_t(tempCount));

      for(size_t i = 0; i < tempCount; i++)
      {
        size_t  pIndexTemp = tempStart + i;
        MeshRT& rtMesh     = rtMeshes[pIndexTemp];

        // trigger builds
        rtMesh.geometryInfo.scratchData = getBufferAddress<VkDeviceOrHostAddressKHR>(device, scratch.buffer, scratchOffset);
        VkAccelerationStructureBuildRangeInfoKHR* r = &rtMesh.blasRange;
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &rtMesh.geometryInfo, &r);


        scratchOffset += alignedSize(tempScratchSize[i], 16);
      }

      {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, 0, 0, 0);

        vkCmdWriteAccelerationStructuresPropertiesKHR(cmd, uint32_t(tempCount), tempAccsKHR.data(),
                                                      VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool, 0);
      }

      res.tempStopWatch(cmd, false);
      res.tempSyncSubmit(cmd);

      buildTime += res.getStopWatchResult();

      // readback queries
      {
        vkGetQueryPoolResults(device, queryPool, 0, uint32_t(tempCount), sizeof(VkDeviceSize) * tempCount,
                              tempCompactSize.data(), sizeof(VkDeviceSize), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
      }

      cmd           = res.createTempCmdBuffer();
      scratchOffset = 0;

      for(size_t i = 0; i < tempCount; i++)
      {
        size_t  pIndexTemp = tempStart + i;
        MeshRT& rtMesh     = rtMeshes[pIndexTemp];

        // create final allocations
        // create acc
        VkAccelerationStructureBuildGeometryInfoKHR accCreateInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        accCreateInfo.type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        accCreateInfo.flags = buildFlags;

        accCreateInfo.geometryCount = 1;
        accCreateInfo.pGeometries   = &rtMesh.geometry;


        VkAccelerationStructureCreateInfoKHR createInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        createInfo.offset                               = 0;
        createInfo.size                                 = tempCompactSize[i];
        createInfo.type                                 = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

        rtMesh.blas = res.m_allocator.createAcceleration(createInfo);

        if(rtMesh.blas.accel == VK_NULL_HANDLE)
        {
          LOGE("Cannot create raytracing bottom-level acceleration structure\n");
          exit(-1);
        }


        std::string debugName = "RTGeometry:" + std::to_string(pIndexTemp);
        debugUtil.setObjectName(rtMesh.blas.accel, debugName);

        memCompact += tempCompactSize[i];

        // trigger compaction
        VkCopyAccelerationStructureInfoKHR copyInfo = {VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
        copyInfo.dst                                = rtMesh.blas.accel;
        copyInfo.src                                = tempAccs[i].accel;
        copyInfo.mode                               = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;

        vkCmdCopyAccelerationStructureKHR(cmd, &copyInfo);
      }

      res.tempSyncSubmit(cmd);

      // cleanup temp resources
      for(size_t i = 0; i < tempCount; i++)
      {
        res.m_allocator.destroy(tempAccs[i]);
      }

      tempStart   = pIndex + 1;
      tempIndex   = 0;
      tempScratch = 0;
      tempMem     = 0;
    }
    else
    {
      tempIndex++;
    }

    pIndex++;
  }


  LOGI("GeometryAS build size %d KB / compact %d KB (%d)\n", memRegular / 1024, memCompact / 1024, (memCompact * 100) / memRegular);
  LOGI("GeometryAS build time: ~ %d (us)\n", uint32_t(buildTime));

  // Clean up temporary resources
  vkDestroyQueryPool(device, queryPool, nullptr);

  res.destroy(scratch);
}

void MeshSetVK::initRayTracingScene(ResourcesVK& res, const MeshSet& meshSet, const MicromeshSetCompressedRayTracedVK* micromeshSetRT)
{
  res.m_allocator.destroy(sceneTlas);

  VkDevice        device = res.m_device;
  nvvk::DebugUtil debugUtil(device);

  uint32_t numInstances = uint32_t(meshSet.meshInstances.size());
  size_t   numMeshes    = meshSet.meshInfos.size();

  RBuffer scratch;

  {
    nvvk::StagingMemoryManager* staging = res.m_allocator.getStaging();
    res.tempResetResources();
    VkCommandBuffer cmd = res.createTempCmdBuffer();

    RBuffer rtInstancesScene = res.createBuffer(sizeof(VkAccelerationStructureInstanceKHR) * numInstances,
                                                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                                    | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    checkAllocSuccess(res, rtInstancesScene.buffer);

    VkAccelerationStructureInstanceKHR* instancesScene =
        staging->cmdToBufferT<VkAccelerationStructureInstanceKHR>(cmd, rtInstancesScene.buffer, 0, rtInstancesScene.info.range);
    for(uint32_t i = 0; i < numInstances; i++)
    {
      VkAccelerationStructureInstanceKHR& instance = instancesScene[i];
      const MeshInstance&                 minst    = meshSet.meshInstances[i];
      const MeshInfo&                     mesh     = meshSet.meshInfos[minst.meshID];
      MeshRT&                             meshRT   = rtMeshes[minst.meshID];

      instance.transform.matrix[0][0] = minst.xform[0][0];
      instance.transform.matrix[0][1] = minst.xform[0][1];
      instance.transform.matrix[0][2] = minst.xform[0][2];
      instance.transform.matrix[0][3] = minst.xform[0][3];

      instance.transform.matrix[1][0] = minst.xform[1][0];
      instance.transform.matrix[1][1] = minst.xform[1][1];
      instance.transform.matrix[1][2] = minst.xform[1][2];
      instance.transform.matrix[1][3] = minst.xform[1][3];

      instance.transform.matrix[2][0] = minst.xform[2][0];
      instance.transform.matrix[2][1] = minst.xform[2][1];
      instance.transform.matrix[2][2] = minst.xform[2][2];
      instance.transform.matrix[2][3] = minst.xform[2][3];

      instance.flags               = 0;
      instance.mask                = 0xFF;
      instance.instanceCustomIndex = i;

      // use different SBT offset if displaced
      instance.instanceShaderBindingTableRecordOffset = meshRT.geometryDisplacement.micromap ? 1 : 0;

      VkAccelerationStructureDeviceAddressInfoKHR accelAddressInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
      accelAddressInfo.accelerationStructure  = meshRT.blas.accel;
      instance.accelerationStructureReference = vkGetAccelerationStructureDeviceAddressKHR(device, &accelAddressInfo);
    }

    res.tempSyncSubmit(cmd);


    VkAccelerationStructureGeometryKHR geometry = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometry.instances = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    geometry.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;

    geometry.geometry.instances.data = getBufferAddress<VkDeviceOrHostAddressConstKHR>(device, rtInstancesScene.buffer);
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;


    VkAccelerationStructureBuildGeometryInfoKHR accCreateInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    accCreateInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    accCreateInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    accCreateInfo.geometryCount = 1;
    accCreateInfo.pGeometries   = &geometry;
    accCreateInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &accCreateInfo,
                                            &numInstances, &sizesInfo);

    VkAccelerationStructureCreateInfoKHR createInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.offset                               = 0;
    createInfo.size                                 = sizesInfo.accelerationStructureSize;
    createInfo.type                                 = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    sceneTlas                                       = res.m_allocator.createAcceleration(createInfo);
    if(sceneTlas.accel == VK_NULL_HANDLE)
    {
      LOGE("Cannot create raytracing scene top-level acceleration structure\n");
      exit(-1);
    }

    scratch = res.createBuffer(sizesInfo.buildScratchSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                                               | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    checkAllocSuccess(res, scratch.buffer);
    debugUtil.setObjectName(scratch.buffer, "RTSceneScratch");

    sceneTlasInfo                            = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    sceneTlasInfo.accelerationStructureCount = 1;
    sceneTlasInfo.pAccelerationStructures    = &sceneTlas.accel;

    cmd = res.createTempCmdBuffer();

    {
      accCreateInfo.dstAccelerationStructure = sceneTlas.accel;
      accCreateInfo.scratchData              = getBufferAddress<VkDeviceOrHostAddressKHR>(device, scratch.buffer);
      VkAccelerationStructureBuildRangeInfoKHR buildRange = {};
      buildRange.primitiveCount                           = numInstances;

      VkAccelerationStructureBuildRangeInfoKHR* r = &buildRange;
      vkCmdBuildAccelerationStructuresKHR(cmd, 1, &accCreateInfo, &r);
    }

    res.tempSyncSubmit(cmd);
    res.m_allocator.destroy(rtInstancesScene);
  }

  res.destroy(scratch);
}

}  // namespace microdisp
