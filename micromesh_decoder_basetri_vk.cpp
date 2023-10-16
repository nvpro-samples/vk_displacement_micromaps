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

#include "micromesh_decoder_basetri_vk.hpp"
#include "micromesh_decoder_utils_vk.hpp"

namespace microdisp {

static_assert((MICRO_PART_MAX_SUBDIV) == (MICRO_MIP_SUBDIV + 1), "MICRO_MIP_SUBDIV and one vertex level must match MICRO_PART_MAX_SUBDIV");
static_assert((MICRO_PART_MAX_SUBDIV + MICRO_MIP_SUBDIV) == (MICRO_MAX_SUBDIV),
              "MICRO_MIP_SUBDIV and MICRO_PART_MAX_SUBDIV must match MICRO_MAX_SUBDIV");

bool MicromeshBaseTriangleDecoderVK::init(ResourcesVK& res, const MeshSet& meshSet, const BaryAttributesSet& barySet, bool withAttributes, uint32_t numThreads)
{
  if(barySet.displacements.empty() || !barySet.supportsCompressedMips())
    return false;

  m_micro.initBasics(res, meshSet, barySet, true, true);

  if(withAttributes)
  {
    initAttributes(m_micro, res, meshSet, barySet, numThreads);
  }

  // common data, independent of actual micromesh

  m_micro.triangleIndices =
      res.createBuffer(sizeof(uint32_t) * (MICRO_MESHLET_PRIMS * MICRO_MESHLET_TOPOS), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  // one set of vertices for each partMicro config, plus zeroed dummy to allow safe out of bounds access
  m_micro.vertices = res.createBuffer(sizeof(MicromeshBTriVertex) * MICRO_BTRI_VTX_COUNT, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  nvvk::StagingMemoryManager* staging = res.m_allocator.getStaging();
  VkCommandBuffer             cmd     = res.createTempCmdBuffer();
  m_micro.uploadMeshDatasBinding(staging, cmd);

  for(size_t m = 0; m < meshSet.meshInfos.size(); m++)
  {
    uint32_t        meshID = uint32_t(m);
    const MeshInfo& mesh   = meshSet.meshInfos[m];

    if(mesh.displacementID == MeshSetID::INVALID)
      continue;

    uploadMicroBaseTriangles(staging, cmd, meshSet, barySet, meshID, numThreads);
  }

  {
    MicroSplitParts splits;

    // setup indices
    splits.uploadTriangleIndices(staging, cmd, m_micro.triangleIndices);

    // setup vertices
    uploadVertices(staging, cmd, splits);
  }

  res.tempSyncSubmit(cmd);

  return true;
}

void MicromeshBaseTriangleDecoderVK::uploadMicroBaseTriangles(nvvk::StagingMemoryManager* staging,
                                                              VkCommandBuffer             cmd,
                                                              const MeshSet&              meshSet,
                                                              const BaryAttributesSet&    barySet,
                                                              uint32_t                    meshID,
                                                              uint32_t                    numThreads)
{
  const MicromeshSetCompressedVK::MeshData& meshData         = m_micro.meshDatas[meshID];
  const MeshInfo&                           mesh             = meshSet.meshInfos[meshID];
  const size_t                              numMeshTriangles = mesh.numPrimitives;

  const BaryDisplacementAttribute&  displacementAttr = barySet.displacements[mesh.displacementID];
  bary::MiscView                    misc             = displacementAttr.compressedMisc->getView();
  bary::BasicView                   basic            = displacementAttr.compressed->getView();
  const bary::GroupUncompressedMip& mipGroup         = misc.groupUncompressedMips[mesh.displacementGroup];
  const bary::Group&                baryGroup        = basic.groups[mesh.displacementGroup];
  const size_t                      minMaxByteSize   = basic.triangleMinMaxsInfo->elementByteSize * 2;

  MicromeshBaseTri* baseTriData =
      staging->cmdToBufferT<MicromeshBaseTri>(cmd, meshData.baseTriangles.buffer, meshData.baseTriangles.info.offset,
                                              meshData.baseTriangles.info.range);

  nvmath::vec4f* sphereData =
      staging->cmdToBufferT<nvmath::vec4f>(cmd, meshData.baseSpheres.buffer, meshData.baseSpheres.info.offset,
                                           meshData.baseSpheres.info.range);

  const uint8_t* decimateEdgeFlags = meshSet.decimateEdgeFlags.empty() ? nullptr : meshSet.decimateEdgeFlags.data();

  const size_t mipSize = (((MICRO_MIP_VERTICES * 11) + 31) / 32) * sizeof(uint32_t);

  parallel_batches(
      numMeshTriangles,
      [&](uint64_t meshLocalTriIdx) {
        const size_t          meshGlobalTriIdx = (mesh.firstPrimitive) + meshLocalTriIdx;
        const size_t          baryGlobalTriIdx = baryGroup.triangleFirst + mesh.displacementMapOffset + meshLocalTriIdx;
        const bary::Triangle& baseTri          = basic.triangles[baryGlobalTriIdx];
        const uint16_t* baseMinMaxs = reinterpret_cast<const uint16_t*>(basic.triangleMinMaxs + baryGlobalTriIdx * minMaxByteSize);

        uint32_t formatIndex = getFormatIndex(baseTri.blockFormat);

        float valueBias  = baryGroup.floatBias.r;
        float valueScale = baryGroup.floatScale.r;

        MicromeshBaseTri& micro  = baseTriData[meshLocalTriIdx];
        nvmath::vec4f&    sphere = sphereData[meshLocalTriIdx];

        const int32_t upperValueBound = 0x7FF;
        const int32_t upperCullBound  = MICRO_BASE_CULLDIST_MASK;

        float minDisp = float(baseMinMaxs[0]) / float(upperValueBound) * valueScale + valueBias;
        float maxDisp = float(baseMinMaxs[1]) / float(upperValueBound) * valueScale + valueBias;

        float   delta    = !mesh.directionBoundsAreUniform && meshSet.attributes.directionBounds.empty() ? 0.0f : -0.5f;
        float   absDist  = std::max(std::abs(minDisp + delta), std::abs(maxDisp + delta));
        int32_t cullDist = int32_t(((absDist - valueBias) / valueScale) * float(upperCullBound));
        cullDist         = std::min(std::max(cullDist, 1), upperCullBound);

        const uint32_t baseTopo = (decimateEdgeFlags ? decimateEdgeFlags[meshGlobalTriIdx] : 0);

        bary::TriangleUncompressedMip primMip       = misc.triangleUncompressedMips[baryGlobalTriIdx];
        uint32_t                      primMipOffset = primMip.mipOffset;
        if(baseTri.blockFormatDispC1 == bary::BlockFormatDispC1::eR11_unorm_lvl3_pack512)
        {
          // ignore for flat compression
          primMipOffset = ~0;
        }

        uint32_t mipOffset = primMipOffset == ~0 ? 0 : ((primMipOffset) / uint32_t(mipSize));

        assert(primMipOffset == ~0 || primMip.subdivLevel >= MICRO_MIP_SUBDIV);

        uint32_t mipOffsetLo = mipOffset & ((1u << MICRO_BASE_MIPLO_WIDTH) - 1);
        uint32_t mipOffsetHi = mipOffset >> MICRO_BASE_MIPLO_WIDTH;

        assert(mipOffset < MICRO_BASE_MIP_MAX);

        micro.dataOffset = ((baseTri.valuesOffset) / sizeof(uint32_t)) / MICRO_BASE_DATA_VALUE_MUL;

        assert(micro.dataOffset <= MICRO_BASE_DATA_MASK);

        micro.dataOffset |= packBits(mipOffsetHi, MICRO_BASE_DATA_MIPHI_SHIFT, MICRO_BASE_DATA_MIPHI_WIDTH);

        micro.packedBits = 0;
        micro.packedBits |= packBits(baseTri.subdivLevel, MICRO_BASE_LVL_SHIFT, MICRO_BASE_LVL_WIDTH);
        micro.packedBits |= packBits(baseTopo, MICRO_BASE_TOPO_SHIFT, MICRO_BASE_TOPO_WIDTH);
        micro.packedBits |= packBits(formatIndex, MICRO_BASE_FMT_SHIFT, MICRO_BASE_FMT_WIDTH);
        micro.packedBits |= packBits(cullDist, MICRO_BASE_CULLDIST_SHIFT, MICRO_BASE_CULLDIST_WIDTH);
        micro.packedBits |= packBits(mipOffsetLo, MICRO_BASE_MIPLO_SHIFT, MICRO_BASE_MIPLO_WIDTH);

        sphere = computeSphere(micro, meshSet, meshGlobalTriIdx, minDisp, maxDisp, mesh.directionBoundsAreUniform);
      },
      numThreads);
}

void MicromeshBaseTriangleDecoderVK::uploadVertices(nvvk::StagingMemoryManager* staging, VkCommandBuffer cmd, const MicroSplitParts& splits)
{
  MicromeshBTriVertex* verticesAll =
      staging->cmdToBufferT<MicromeshBTriVertex>(cmd, m_micro.vertices.buffer, m_micro.vertices.info.offset,
                                                 m_micro.vertices.info.range);

  memset(verticesAll, 0, m_micro.vertices.info.range);

  MicromeshFormatInfo formatInfos;

  for(uint32_t formatIdx = 0; formatIdx < MICRO_MAX_FORMATS; formatIdx++)
  {
    bool                    isFlat = false;
    uint32_t                blockSubdiv;
    uint32_t                blockBits;
    bary::BlockFormatDispC1 baryFormat;
    if(formatIdx == MICRO_FORMAT_64T_512B)
    {
      baryFormat  = bary::BlockFormatDispC1::eR11_unorm_lvl3_pack512;
      blockSubdiv = 3;
      blockBits   = 512;
      isFlat      = true;
    }
    else if(formatIdx == MICRO_FORMAT_256T_1024B)
    {
      baryFormat  = bary::BlockFormatDispC1::eR11_unorm_lvl4_pack1024;
      blockSubdiv = 4;
      blockBits   = 1024;
    }
    else if(formatIdx == MICRO_FORMAT_1024T_1024B)
    {
      baryFormat  = bary::BlockFormatDispC1::eR11_unorm_lvl5_pack1024;
      blockSubdiv = 5;
      blockBits   = 1024;
    }
    else
    {
      assert(0 && "format not supported");
      return;
    }

    for(uint32_t baseSubdiv = 0; baseSubdiv <= MICRO_MAX_SUBDIV; baseSubdiv++)
    {
      // skip unsupported configs
      if(!isFlat && baseSubdiv < blockSubdiv)
        continue;

      for(uint32_t targetSubdiv = 0; targetSubdiv <= baseSubdiv; targetSubdiv++)
      {
        // don't go over max part subdiv, in that case we use more parts
        const baryutils::BaryLevelsMap::Level& usedLevel =
            splits.map.getLevel(std::min(uint32_t(MICRO_PART_MAX_SUBDIV), targetSubdiv));
        uint32_t numParts = 1 << ((std::max(uint32_t(MICRO_PART_MAX_SUBDIV), targetSubdiv) - MICRO_PART_MAX_SUBDIV) * 2);

        for(uint32_t partID = 0; partID < numParts; partID++)
        {
          MicromeshBTriVertex* vertices = verticesAll + MICRO_BTRI_VTX_OFFSET(partID, targetSubdiv, baseSubdiv, formatIdx);

          uint32_t numVertices = static_cast<uint32_t>(usedLevel.coordinates.size());

          // our index buffers are based on triangle splits, so need to convert part UVs based on those
          for(uint32_t v = 0; v < numVertices; v++)
          {
            baryutils::BaryWUV_uint16 localCoord   = usedLevel.coordinates[v];
            bary::BaryUV_uint16       localCoordUV = {localCoord.u, localCoord.v};

            baryutils::BaryWUV_uint16 baseCoord   = localCoord;
            bary::BaryUV_uint16       baseCoordUV = localCoordUV;

            if(numParts > 1)
            {
              // apply split transform
              baseCoordUV = bary::baryBlockTriangleLocalToBaseUV(&splits.triLevel5to3[partID], localCoordUV);
            }

            // the coordinate is now in targetSubdiv, but we actually need to find the vertex in the baseLevel
            // so we need to shift it into the baseSubdiv level
            baseCoordUV.u <<= (baseSubdiv - targetSubdiv);
            baseCoordUV.v <<= (baseSubdiv - targetSubdiv);
            baseCoord = {uint16_t((1 << baseSubdiv) - baseCoordUV.u - baseCoordUV.v), baseCoordUV.u, baseCoordUV.v};

            MicromeshBTriVertex& mvtx = vertices[v];
            mvtx.packed               = 0;
            mvtx.uv.x                 = uint8_t(baseCoord.u);
            mvtx.uv.y                 = uint8_t(baseCoord.v);
            mvtx.parents.x            = splits.partVertexMergeIndices[v].a;
            mvtx.parents.y            = splits.partVertexMergeIndices[v].b;

            if(isFlat)
            {
              mvtx.packed |= MICRO_BTRI_VTX_UNSIGNED;
            }

            // figure out which block / or if we use mip

            uint32_t baseVertexLevel;
            uint32_t baseVertexIndex;
            bary::baryBirdLayoutGetVertexLevelInfo(baseCoord.u, baseCoord.v, baseSubdiv, &baseVertexLevel, &baseVertexIndex);

            // unless flat or no need for mip, first MICRO_MIP_SUBDIV levels
            // will always look up values in mip block
            if(!isFlat && baseVertexLevel <= MICRO_MIP_SUBDIV && baseSubdiv >= MICRO_MIP_MIN_SUBDIV)
            {
              uint32_t mipVertexBitPos = formatInfos.getBlockIndex(MICRO_FORMAT_64T_512B, baseVertexLevel, baseVertexIndex);
              uint32_t mipVertexBitNum = formatInfos.getWidth(MICRO_FORMAT_64T_512B, baseVertexLevel);

              // use mip, store as level 0, vertextype won't matter
              mvtx.packed |= MICRO_BTRI_VTX_MIP;
              mvtx.packed |= MICRO_BTRI_VTX_UNSIGNED;
              mvtx.packed |= packBits(mipVertexBitNum, MICRO_BTRI_VTX_BITNUM_SHIFT, MICRO_BTRI_VTX_BITNUM_WIDTH);
              mvtx.packed |= packBits(mipVertexBitPos, MICRO_BTRI_VTX_BITPOS_SHIFT, MICRO_BTRI_VTX_BITPOS_WIDTH);
            }
            else if(baseSubdiv <= blockSubdiv)
            {
              // single block
              uint32_t blockVertexBitPos = formatInfos.getBlockIndex(formatIdx, baseVertexLevel, baseVertexIndex);
              uint32_t blockVertexBitNum = formatInfos.getWidth(formatIdx, baseVertexLevel);
              uint32_t blockVertexType   = formatInfos.getVertexType(baseCoord);
              uint32_t blockCorrMask     = (1u << formatInfos.getCorrWidth(formatIdx, baseVertexLevel)) - 1;
              uint32_t blockCorrPos      = formatInfos.getCorrIndex(formatIdx, baseVertexLevel, blockVertexType);

              if(baseVertexLevel == 0)
              {
                mvtx.packed |= MICRO_BTRI_VTX_UNSIGNED;
              }
              mvtx.packed |= packBits(blockCorrMask, MICRO_BTRI_VTX_CORRMASK_SHIFT, MICRO_BTRI_VTX_CORRMASK_WIDTH);
              mvtx.packed |= packBits(blockCorrPos, MICRO_BTRI_VTX_CORRPOS_SHIFT, MICRO_BTRI_VTX_CORRPOS_WIDTH);
              mvtx.packed |= packBits(blockVertexBitNum, MICRO_BTRI_VTX_BITNUM_SHIFT, MICRO_BTRI_VTX_BITNUM_WIDTH);
              mvtx.packed |= packBits(blockVertexBitPos, MICRO_BTRI_VTX_BITPOS_SHIFT, MICRO_BTRI_VTX_BITPOS_WIDTH);
            }
            else
            {
              // need to find which block we are in
              uint32_t                  blockIndex = ~0;
              baryutils::BaryWUV_uint16 blockCoord;

              if(targetSubdiv == baseSubdiv && blockSubdiv == MICRO_PART_MAX_SUBDIV)
              {
                // easiest case can deduce block based on part
                blockCoord = localCoord;
                blockIndex = partID;
              }
              else
              {
                // check all splits that are relevant
                uint32_t                   numSplits   = 1 << ((baseSubdiv - blockSubdiv) * 2);
                const bary::BlockTriangle* splitLevels = splits.triLevelNtoN[baseSubdiv][blockSubdiv];
                for(uint32_t s = 0; s < numSplits; s++)
                {
                  // find which block fits
                  bary::BaryUV_uint16 blockCoordUV = bary::baryBlockTriangleBaseToLocalUV(splitLevels + s, baseCoordUV);
                  if(bary::baryUVisValid(blockCoordUV, blockSubdiv))
                  {
                    blockCoord.w = (1 << blockSubdiv) - blockCoordUV.u - blockCoordUV.v;
                    blockCoord.u = blockCoordUV.u;
                    blockCoord.v = blockCoordUV.v;
                    blockIndex   = s;
                  }
                }

                assert(blockIndex != ~0);
              }
              uint32_t blockVertexLevel;
              uint32_t blockVertexIndex;
              bary::baryBirdLayoutGetVertexLevelInfo(blockCoord.u, blockCoord.v, blockSubdiv, &blockVertexLevel, &blockVertexIndex);
              uint32_t blockVertexBitPos =
                  formatInfos.getBlockIndex(formatIdx, blockVertexLevel, blockVertexIndex) + blockIndex * blockBits;
              uint32_t blockVertexBitNum = formatInfos.getWidth(formatIdx, blockVertexLevel);
              uint32_t blockVertexType   = formatInfos.getVertexType(blockCoord);
              uint32_t blockCorrMask     = (1u << formatInfos.getCorrWidth(formatIdx, blockVertexLevel)) - 1;
              uint32_t blockCorrPos      = formatInfos.getCorrIndex(formatIdx, blockVertexLevel, blockVertexType);

              if(baseVertexLevel == 0)
              {
                mvtx.packed |= MICRO_BTRI_VTX_UNSIGNED;
              }
              mvtx.packed |= packBits(blockCorrMask, MICRO_BTRI_VTX_CORRMASK_SHIFT, MICRO_BTRI_VTX_CORRMASK_WIDTH);
              mvtx.packed |= packBits(blockCorrPos, MICRO_BTRI_VTX_CORRPOS_SHIFT, MICRO_BTRI_VTX_CORRPOS_WIDTH);
              mvtx.packed |= packBits(blockVertexBitNum, MICRO_BTRI_VTX_BITNUM_SHIFT, MICRO_BTRI_VTX_BITNUM_WIDTH);
              mvtx.packed |= packBits(blockVertexBitPos, MICRO_BTRI_VTX_BITPOS_SHIFT, MICRO_BTRI_VTX_BITPOS_WIDTH);
            }
          }
        }
      }
    }
  }
}

}  // namespace microdisp
