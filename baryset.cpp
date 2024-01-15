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

#include <nvh/nvprint.hpp>

#include "parallel_work.hpp"
#include "baryset.hpp"
#include "config.h"

const BaryShadingAttribute* BaryAttributesSet::getDisplacementShading(uint32_t dispID, ShadingAttributeBit attributeBit) const
{
  if(shadings.empty() || dispID == MeshSetID::INVALID)
    return nullptr;

  for(auto& shading : shadings)
  {
    if(shading.displacementID == dispID && shading.attributeFlags == attributeBit && shading.attribute)
    {
      return &shading;
    }
  }

  return nullptr;
}

static void fillDummyTriangleMinMaxs(baryutils::BaryBasicData& baryData)
{
  baryData.triangleMinMaxsInfo.elementCount         = uint32_t(2 * baryData.triangles.size());
  baryData.triangleMinMaxsInfo.elementByteAlignment = 4;

  switch(baryData.valuesInfo.valueFormat)
  {
    case bary::Format::eDispC1_r11_unorm_block:
    case bary::Format::eR11_unorm_pack16:
    case bary::Format::eR11_unorm_packed_align32:
      baryData.triangleMinMaxsInfo.elementByteSize = uint32_t(sizeof(uint16_t));
      baryData.triangleMinMaxsInfo.elementFormat   = bary::Format::eR11_unorm_pack16;
      baryData.triangleMinMaxs.resize(baryData.triangleMinMaxsInfo.elementByteSize * baryData.triangleMinMaxsInfo.elementCount);
      {
        uint16_t* minMaxs = reinterpret_cast<uint16_t*>(baryData.triangleMinMaxs.data());
        for(size_t i = 0; i < baryData.triangles.size(); i++)
        {
          minMaxs[i * 2 + 0] = 0;
          minMaxs[i * 2 + 1] = 0x7FF;
        }
      }
      break;
    case bary::Format::eR16_unorm:
      baryData.triangleMinMaxsInfo.elementByteSize = uint32_t(sizeof(uint16_t));
      baryData.triangleMinMaxsInfo.elementFormat   = bary::Format::eR16_unorm;
      baryData.triangleMinMaxs.resize(baryData.triangleMinMaxsInfo.elementByteSize * baryData.triangleMinMaxsInfo.elementCount);
      {
        uint16_t* minMaxs = reinterpret_cast<uint16_t*>(baryData.triangleMinMaxs.data());
        for(size_t i = 0; i < baryData.triangles.size(); i++)
        {
          minMaxs[i * 2 + 0] = 0;
          minMaxs[i * 2 + 1] = 0xFFFF;
        }
      }
      break;
    case bary::Format::eR8_unorm:
      baryData.triangleMinMaxsInfo.elementByteSize = uint32_t(sizeof(uint8_t));
      baryData.triangleMinMaxsInfo.elementFormat   = bary::Format::eR8_unorm;
      baryData.triangleMinMaxs.resize(baryData.triangleMinMaxsInfo.elementByteSize * baryData.triangleMinMaxsInfo.elementCount);
      {
        uint8_t* minMaxs = reinterpret_cast<uint8_t*>(baryData.triangleMinMaxs.data());
        for(size_t i = 0; i < baryData.triangles.size(); i++)
        {
          minMaxs[i * 2 + 0] = 0;
          minMaxs[i * 2 + 1] = 0xFF;
        }
      }
      break;
    case bary::Format::eR32_sfloat:
      baryData.triangleMinMaxsInfo.elementByteSize = uint32_t(sizeof(float));
      baryData.triangleMinMaxsInfo.elementFormat   = bary::Format::eR32_sfloat;
      baryData.triangleMinMaxs.resize(baryData.triangleMinMaxsInfo.elementByteSize * baryData.triangleMinMaxsInfo.elementCount);
      {
        float* minMaxs = reinterpret_cast<float*>(baryData.triangleMinMaxs.data());
        for(size_t i = 0; i < baryData.triangles.size(); i++)
        {
          minMaxs[i * 2 + 0] = 0.0f;
          minMaxs[i * 2 + 1] = 1.0f;
        }
      }
      break;
  }
}

uint32_t BaryAttributesSet::loadDisplacement(const char* baryFilePathCStr, baryutils::BaryFile& bfile, baryutils::BaryFileOpenOptions* fileOpenOptions)
{
  uint32_t displacementId = uint32_t(displacements.size());

  bary::Result baryResult = bfile.open(baryFilePathCStr, fileOpenOptions);
  if(baryResult != bary::Result::eSuccess)
  {
    LOGE("BaryAttributesSet::loadDisplacement: Loading BARY file %s failed with code %s\n", baryFilePathCStr,
         bary::baryResultGetName(baryResult));
    return MeshSetID::INVALID;
  }

  baryResult = bfile.validate(bary::ValueSemanticType::eDisplacement);
  if(baryResult != bary::Result::eSuccess)
  {
    LOGE("BaryAttributesSet::loadDisplacement: Validating Displacement BARY file %s failed with code %s\n",
         baryFilePathCStr, bary::baryResultGetName(baryResult));
    return MeshSetID::INVALID;
  }

  if(!bfile.getBasic().triangleMinMaxsInfo || bfile.getBasic().triangleMinMaxsInfo->elementCount != bfile.getBasic().trianglesCount * 2)
  {
    LOGW("BaryAttributesSet::loadDisplacement: BARY file %s did not have triangle min/maxs (will get worst-case values)\n",
         baryFilePathCStr);
  }

  // Start by determining whether the .bary file uses the compressed format.
  BaryDisplacementAttribute baryDisplacement;

  switch(bfile.m_content.basic.valuesInfo->valueFormat)
  {
    case bary::Format::eDispC1_r11_unorm_block:
      baryDisplacement.compressed = std::make_unique<baryutils::BaryBasicData>();
      baryDisplacement.compressed->setData(bfile.m_content.basic);
      // check if mips in file, if so load
      if(bfile.getMisc().groupUncompressedMipsCount && bfile.getMisc().triangleUncompressedMipsCount && bfile.getMisc().uncompressedMipsInfo)
      {
        baryDisplacement.compressedMisc = std::make_unique<baryutils::BaryMiscData>();
        baryDisplacement.compressedMisc->setData(bfile.m_content.misc);
      }

      if(baryDisplacement.compressed->triangleMinMaxs.empty())
      {
        fillDummyTriangleMinMaxs(*baryDisplacement.compressed);
      }

      displacements.push_back(std::move(baryDisplacement));
      return displacementId;
    case bary::Format::eR11_unorm_pack16:
    case bary::Format::eR11_unorm_packed_align32:
    case bary::Format::eR8_unorm:
    case bary::Format::eR16_unorm:
    case bary::Format::eR32_sfloat:
      baryDisplacement.uncompressed = std::make_unique<baryutils::BaryBasicData>();
      baryDisplacement.uncompressed->setData(bfile.m_content.basic);

      if(baryDisplacement.uncompressed->triangleMinMaxs.empty())
      {
        fillDummyTriangleMinMaxs(*baryDisplacement.uncompressed);
      }

      displacements.push_back(std::move(baryDisplacement));
      return displacementId;
    default:
      assert(!"Unknown or unsupported bary valueFormat!");
      LOGE("BaryAttributesSet::loadDisplacement: The BARY file %s used unknown valueFormat %d.\n", baryFilePathCStr,
           bfile.m_content.basic.valuesInfo->valueFormat);
      return MeshSetID::INVALID;
  }
}

uint32_t BaryAttributesSet::loadAttribute(const char* baryFilePathCStr, baryutils::BaryFile& bfile, baryutils::BaryFileOpenOptions* fileOpenOptions)
{
  uint32_t attributeId = uint32_t(shadings.size());

  bary::Result baryResult = bfile.open(baryFilePathCStr, fileOpenOptions);
  if(baryResult != bary::Result::eSuccess)
  {
    LOGE("BaryAttributesSet::loadAttribute: Loading BARY file %s failed with code %s\n", baryFilePathCStr,
         bary::baryResultGetName(baryResult));
    return MeshSetID::INVALID;
  }

  baryResult = bfile.validate(bary::ValueSemanticType::eGeneric);
  if(baryResult != bary::Result::eSuccess)
  {
    LOGE("BaryAttributesSet::loadAttribute: Validating Generic BARY file %s failed with code %s\n", baryFilePathCStr,
         bary::baryResultGetName(baryResult));
    return MeshSetID::INVALID;
  }

  BaryShadingAttribute shadingAttr;
  shadingAttr.attribute = std::make_unique<baryutils::BaryBasicData>();
  shadingAttr.attribute->setData(bfile.m_content.basic);
  shadings.push_back(std::move(shadingAttr));
  return attributeId;
}

void BaryAttributesSet::updateStats()
{
  compressedStats   = {};
  uncompressedStats = {};
  shadingStats      = {};

  compressedMipByteSize = 0;

  for(size_t d = 0; d < displacements.size(); d++)
  {
    const BaryDisplacementAttribute& displacementAttr = displacements[d];

    if(displacementAttr.compressed)
    {
      compressedStats.append(displacementAttr.compressed->getView());
      if(displacementAttr.compressedMisc)
      {
        compressedMipByteSize += displacementAttr.compressedMisc->uncompressedMipsInfo.elementCount
                                 * displacementAttr.compressedMisc->uncompressedMipsInfo.elementByteSize;
      }
    }
    if(displacementAttr.uncompressed)
    {
      uncompressedStats.append(displacementAttr.uncompressed->getView());
    }
  }
  for(size_t s = 0; s < shadings.size(); s++)
  {
    const BaryShadingAttribute& shadingAttr = shadings[s];

    if(shadingAttr.attribute)
    {
      shadingStats.append(shadingAttr.attribute->getView());
    }
  }
}

static float getSignedTriangleVolume(glm::vec3 a, glm::vec3 b, glm::vec3 c)
{
  // http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf
  return (1.0f / 6.0f)
         * (-(c.x * b.y * a.z) + (b.x * c.y * a.z) + (c.x * a.y * b.z) - (a.x * c.y * b.z) - (b.x * a.y * c.z) + (a.x * b.y * c.z));
}

static float getBoundsVolume(glm::vec3 min0, glm::vec3 min1, glm::vec3 min2, glm::vec3 max0, glm::vec3 max1, glm::vec3 max2)
{
  // prismoid shape made out of
  // 1 bottom tri, 3 side quads, 1 top tri

  return (
      // bottom (reverse winding)
      getSignedTriangleVolume(min2, min1, min0) +

      // sides
      getSignedTriangleVolume(min0, min1, max0) + getSignedTriangleVolume(max0, min1, max1)
      + getSignedTriangleVolume(min2, min0, max2) + getSignedTriangleVolume(max2, min0, max0)
      + getSignedTriangleVolume(min1, min2, max1) + getSignedTriangleVolume(max1, min2, max2) +

      // top
      getSignedTriangleVolume(max0, max1, max2));
}

double BaryAttributesSet::computeShellVolume(const MeshSet& meshSet, bool perferDirectionBounds, bool preferUncompressed, uint32_t numThreads /*= 0*/) const
{
  uint32_t            usedThreadCount = std::max(numThreads, 1u);
  std::vector<double> perThreadMinVolumes(usedThreadCount, 0);
  std::vector<double> perThreadVolumes(usedThreadCount, 0);

  for(size_t i = 0; i < meshSet.meshInfos.size(); i++)
  {
    const MeshInfo& mesh = meshSet.meshInfos[i];

    if(mesh.displacementID == MeshSetID::INVALID)
      continue;

    const BaryDisplacementAttribute& attr = displacements[mesh.displacementID];
    const baryutils::BaryBasicData*  baryData =
        (!attr.compressed || (attr.uncompressed && preferUncompressed)) ? attr.uncompressed.get() : attr.compressed.get();

    if(!baryData)
      continue;

    const bary::Group& baryGroup = baryData->groups[mesh.displacementGroup];

    const glm::vec3* positions  = &meshSet.attributes.positions[mesh.firstVertex];
    const glm::vec3* directions = &meshSet.attributes.directions[mesh.firstVertex];
    const glm::vec2* directionsBounds = mesh.directionBoundsAreUniform || meshSet.attributes.directionBounds.empty() ?
                                                nullptr :
                                                &meshSet.attributes.directionBounds[mesh.firstVertex];

    const uint32_t* indices = &meshSet.indices[mesh.firstIndex];

    float dir_min = baryGroup.floatBias.r;
    float dir_max = baryGroup.floatBias.r + baryGroup.floatScale.r;

    parallel_batches(
        mesh.numPrimitives,
        [&](uint64_t triIdx, uint32_t threadIdx) {
          uint32_t idxA = indices[triIdx * 3 + 0];
          uint32_t idxB = indices[triIdx * 3 + 1];
          uint32_t idxC = indices[triIdx * 3 + 2];

          glm::vec3 posA = positions[idxA];
          glm::vec3 posB = positions[idxB];
          glm::vec3 posC = positions[idxC];

          glm::vec3 dirA = directions[idxA];
          glm::vec3 dirB = directions[idxB];
          glm::vec3 dirC = directions[idxC];

          if(directionsBounds)
          {
            glm::vec2 boundsA = directionsBounds[idxA];
            glm::vec2 boundsB = directionsBounds[idxB];
            glm::vec2 boundsC = directionsBounds[idxC];

            posA = posA + dirA * boundsA.x;
            posB = posB + dirB * boundsB.x;
            posC = posC + dirC * boundsC.x;

            dirA = dirA * boundsA.y;
            dirB = dirB * boundsB.y;
            dirC = dirC * boundsC.y;
          }

          float vol = getBoundsVolume(posA + dirA * dir_min, posB + dirB * dir_min, posC + dirC * dir_min,
                                      posA + dirA * dir_max, posB + dirB * dir_max, posC + dirC * dir_max);

          perThreadVolumes[threadIdx] += double(vol);
        },
        numThreads);
  }

  double total = 0;
  for(uint32_t i = 0; i < usedThreadCount; i++)
  {
    total += perThreadVolumes[i];
  }

  return total;
}

void BaryAttributesSet::fillUniformDirectionBounds(MeshSet& meshSet) const
{
  assert(meshSet.attributes.directionBounds.size() == meshSet.attributes.positions.size());

  for(MeshInfo& mesh : meshSet.meshInfos)
  {
    if(mesh.displacementID == MeshSetID::INVALID || !mesh.directionBoundsAreUniform)
      continue;

    const BaryDisplacementAttribute& attr = displacements[mesh.displacementID];
    const baryutils::BaryBasicData*  baryData =
        (!attr.compressed || (attr.uncompressed)) ? attr.uncompressed.get() : attr.compressed.get();

    if(!baryData)
      continue;

    const bary::Group& baryGroup = baryData->groups[mesh.displacementGroup];

    glm::vec2* bounds = meshSet.attributes.directionBounds.data() + mesh.firstVertex;

    glm::vec2 biasScale = {baryGroup.floatBias.r, baryGroup.floatScale.r};
    std::fill(bounds, bounds + mesh.numVertices, biasScale);
  }
}

bool BaryAttributesSet::supportsCompressedMips() const
{
  if(displacements.empty())
    return false;

  for(size_t d = 0; d < displacements.size(); d++)
  {
    const BaryDisplacementAttribute& displacementAttr = displacements[d];

    // without mip data cannot use this
    if(!displacementAttr.compressedMisc)
    {
      return false;
    }

    bary::BasicView basic = displacementAttr.compressed->getView();

    uint32_t minSubdivLevel;
    uint32_t maxSubdivLevel;
    bary::baryBasicViewGetMinMaxSubdivLevels(&basic, &minSubdivLevel, &maxSubdivLevel);

    if(maxSubdivLevel > 5)
      return false;
  }

  return true;
}
