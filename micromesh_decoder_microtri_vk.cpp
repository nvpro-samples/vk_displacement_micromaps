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

#include "micromesh_decoder_microtri_vk.hpp"
#include "micromesh_decoder_utils_vk.hpp"

namespace microdisp {


bool MicromeshMicroTriangleDecoderVK::init(ResourcesVK&             res,
                                           const MeshSet&           meshSet,
                                           const BaryAttributesSet& barySet,
                                           bool                     withAttributes,
                                           bool                     useIntrinsic,
                                           uint32_t                 numThreads)
{
  if(barySet.displacements.empty())
    return false;

  // check support for this renderer

  for(size_t d = 0; d < barySet.displacements.size(); d++)
  {
    const BaryDisplacementAttribute& displacementAttr = barySet.displacements[d];
    bary::BasicView                  basic            = displacementAttr.compressed->getView();

    uint32_t minSubdivLevel;
    uint32_t maxSubdivLevel;
    bary::baryBasicViewGetMinMaxSubdivLevels(&basic, &minSubdivLevel, &maxSubdivLevel);

    if(maxSubdivLevel > 5)
      return false;
  }

  m_micro.initBasics(res, meshSet, barySet, true, false);

  if(withAttributes)
  {
    initAttributes(m_micro, res, meshSet, barySet, numThreads);
  }

  // common data, independent of actual micromesh

  // buffer not really needed when intrinsic is used, but simpler code if we have some dummy allocation here
  m_micro.descends = res.createBuffer(useIntrinsic ? 16 : sizeof(MicromeshMTriDescend) * MICRO_MTRI_DESCENDS_COUNT,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  m_micro.triangleIndices =
      res.createBuffer(sizeof(uint32_t) * (MICRO_MESHLET_PRIMS * MICRO_MESHLET_TOPOS), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  // one set of vertices for each partMicro config, plus zeroed dummy to allow safe out of bounds access
  m_micro.vertices = res.createBuffer(sizeof(MicromeshMTriVertex) * MICRO_MTRI_VTX_COUNT, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

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

    // when intrinsic is used we don't need any information to decode the compressed values
    if(!useIntrinsic)
    {
      // setup descend info
      uploadDescends(staging, cmd, splits);
    }
  }

  res.tempSyncSubmit(cmd);
  res.tempResetResources();

  return true;
}


void MicromeshMicroTriangleDecoderVK::uploadMicroBaseTriangles(nvvk::StagingMemoryManager* staging,
                                                               VkCommandBuffer             cmd,
                                                               const MeshSet&              meshSet,
                                                               const BaryAttributesSet&    barySet,
                                                               uint32_t                    meshID,
                                                               uint32_t                    numThreads)
{
  const MicromeshSetCompressedVK::MeshData& meshData         = m_micro.meshDatas[meshID];
  const MeshInfo&                           mesh             = meshSet.meshInfos[meshID];
  const size_t                              numMeshTriangles = mesh.numPrimitives;

  const BaryDisplacementAttribute& displacementAttr = barySet.displacements[mesh.displacementID];
  bary::BasicView                  basic            = displacementAttr.compressed->getView();
  const bary::Group&               baryGroup        = basic.groups[mesh.displacementGroup];
  const size_t                     minMaxByteSize   = basic.triangleMinMaxsInfo->elementByteSize * 2;

  MicromeshBaseTri* micromeshData =
      staging->cmdToBufferT<MicromeshBaseTri>(cmd, meshData.baseTriangles.buffer, meshData.baseTriangles.info.offset,
                                              meshData.baseTriangles.info.range);

  glm::vec4* sphereData =
      staging->cmdToBufferT<glm::vec4>(cmd, meshData.baseSpheres.buffer, meshData.baseSpheres.info.offset,
                                           meshData.baseSpheres.info.range);

  const uint8_t* decimateEdgeFlags = meshSet.decimateEdgeFlags.empty() ? nullptr : meshSet.decimateEdgeFlags.data();

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

        MicromeshBaseTri& micro  = micromeshData[meshLocalTriIdx];
        glm::vec4&    sphere = sphereData[meshLocalTriIdx];

        const int32_t upperValueBound = 0x7FF;
        const int32_t upperCullBound  = MICRO_BASE_CULLDIST_MASK;

        float minDisp = float(baseMinMaxs[0]) / float(upperValueBound) * valueScale + valueBias;
        float maxDisp = float(baseMinMaxs[1]) / float(upperValueBound) * valueScale + valueBias;

        float   delta    = !mesh.directionBoundsAreUniform && meshSet.attributes.directionBounds.empty() ? 0.0f : -0.5f;
        float   absDist  = std::max(std::abs(minDisp + delta), std::abs(maxDisp + delta));
        int32_t cullDist = int32_t(((absDist - valueBias) / valueScale) * float(upperCullBound));
        cullDist         = std::min(std::max(cullDist, 1), upperCullBound);

        const uint32_t baseTopo = (decimateEdgeFlags ? decimateEdgeFlags[meshGlobalTriIdx] : 0);

        micro.dataOffset = (baseTri.valuesOffset) / sizeof(uint32_t);

        micro.packedBits = 0;
        micro.packedBits |= packBits(baseTri.subdivLevel, MICRO_BASE_LVL_SHIFT, MICRO_BASE_LVL_WIDTH);
        micro.packedBits |= packBits(baseTopo, MICRO_BASE_TOPO_SHIFT, MICRO_BASE_TOPO_WIDTH);
        micro.packedBits |= packBits(formatIndex, MICRO_BASE_FMT_SHIFT, MICRO_BASE_FMT_WIDTH);
        micro.packedBits |= packBits(cullDist, MICRO_BASE_CULLDIST_SHIFT, MICRO_BASE_CULLDIST_WIDTH);

        sphere = computeSphere(micro, meshSet, meshGlobalTriIdx, minDisp, maxDisp, mesh.directionBoundsAreUniform);
      },
      numThreads);
}

static inline uint32_t getVertexCorner(const baryutils::BaryLevelsMap::Triangle& tri, uint32_t vertexIndex)
{
  if(tri.a == vertexIndex)
    return 0;
  else if(tri.b == vertexIndex)
    return 1;
  else if(tri.c == vertexIndex)
    return 2;
  else
  {
    assert(0);
    return ~0;
  }
}

struct VertexTriangles
{
  // max valence in barycentric subdiv
  static const uint32_t MAX_TRIANGLES = 6;
  uint32_t              numTriangles  = 0;
  uint32_t              triangles[MAX_TRIANGLES];
};

static inline std::vector<VertexTriangles> buildVertexTriangles(uint32_t                                  numVertices,
                                                                uint32_t                                  numTriangles,
                                                                const baryutils::BaryLevelsMap::Triangle* triangles)
{
  std::vector<VertexTriangles> vertexTriangles(numVertices);

  for(uint32_t i = 0; i < numTriangles; i++)
  {
    const baryutils::BaryLevelsMap::Triangle& triangle = triangles[i];
    assert(triangle.a < numVertices && triangle.b < numVertices && triangle.c < numVertices);

    VertexTriangles& vertexTriangleA = vertexTriangles[triangle.a];
    VertexTriangles& vertexTriangleB = vertexTriangles[triangle.b];
    VertexTriangles& vertexTriangleC = vertexTriangles[triangle.c];

    assert(vertexTriangleA.numTriangles < VertexTriangles::MAX_TRIANGLES);
    assert(vertexTriangleB.numTriangles < VertexTriangles::MAX_TRIANGLES);
    assert(vertexTriangleC.numTriangles < VertexTriangles::MAX_TRIANGLES);
    vertexTriangleA.triangles[vertexTriangleA.numTriangles++] = i;
    vertexTriangleB.triangles[vertexTriangleB.numTriangles++] = i;
    vertexTriangleC.triangles[vertexTriangleC.numTriangles++] = i;
  }

  return vertexTriangles;
}


void MicromeshMicroTriangleDecoderVK::uploadVertices(nvvk::StagingMemoryManager* staging, VkCommandBuffer cmd, const MicroSplitParts& splits)
{
  MicromeshMTriVertex* verticesAll =
      staging->cmdToBufferT<MicromeshMTriVertex>(cmd, m_micro.vertices.buffer, m_micro.vertices.info.offset,
                                                 m_micro.vertices.info.range);

  for(uint32_t baseSubdiv = 0; baseSubdiv <= MICRO_MAX_SUBDIV; baseSubdiv++)
  {
    // Any basesubdiv < 3 is still stored within an uncompressed basesubdiv 3 layout.
    // This adjustment is relevant to ensure that we index the "descendVertices" properly
    uint32_t storageSubdiv = std::max(3u, baseSubdiv);

    const baryutils::BaryLevelsMap::Level& storageLevel = splits.map.getLevel(storageSubdiv);
    std::vector<VertexTriangles>           storageVertexTriangles =
        buildVertexTriangles(uint32_t(storageLevel.coordinates.size()), uint32_t(storageLevel.triangles.size()),
                             storageLevel.triangles.data());

    for(uint32_t targetSubdiv = 0; targetSubdiv <= baseSubdiv; targetSubdiv++)
    {
      // don't go over max part subdiv, in that case we use more parts
      const baryutils::BaryLevelsMap::Level& usedLevel =
          splits.map.getLevel(std::min(uint32_t(MICRO_PART_MAX_SUBDIV), targetSubdiv));
      uint32_t numParts = 1 << ((std::max(uint32_t(MICRO_PART_MAX_SUBDIV), targetSubdiv) - MICRO_PART_MAX_SUBDIV) * 2);

      for(uint32_t partID = 0; partID < numParts; partID++)
      {
        MicromeshMTriVertex* vertices = verticesAll + MICRO_MTRI_VTX_OFFSET(partID, targetSubdiv, baseSubdiv);

        uint32_t numVertices = static_cast<uint32_t>(usedLevel.coordinates.size());

        // our index buffers are based on triangle splits, so need to convert part UVs based on those
        for(uint32_t v = 0; v < numVertices; v++)
        {
          baryutils::BaryWUV_uint16 coord   = usedLevel.coordinates[v];
          bary::BaryUV_uint16       coordUV = {coord.u, coord.v};


          if(numParts > 1)
          {
            // apply split transform
            coordUV = bary::baryBlockTriangleLocalToBaseUV(&splits.triLevel5to3[partID], coordUV);
          }

          // for "descendVertices" lookup the storageLevel is relevant.
          // the coordinate is now in targetSubdiv, but we actually need to find the vertex in the storageLevel
          // so we need to shift it into the storageSubdiv level
          bary::BaryUV_uint16 storageCoordUV = coordUV;
          storageCoordUV.u <<= (storageSubdiv - targetSubdiv);
          storageCoordUV.v <<= (storageSubdiv - targetSubdiv);
          baryutils::BaryWUV_uint16 storageCoord = {uint16_t((1 << storageSubdiv) - storageCoordUV.u - storageCoordUV.v),
                                                    storageCoordUV.u, storageCoordUV.v};

          uint32_t storageVertexIndex = storageLevel.getCoordIndex(storageCoord);
          assert(storageVertexIndex != ~0);
          // use the first triangle that has this
          uint32_t triIndex = storageVertexTriangles[storageVertexIndex].triangles[0];
          uint32_t corner   = getVertexCorner(storageLevel.triangles[triIndex], storageVertexIndex);

          // for rendering we do want it in baseLevel
          // the coordinate is now in targetSubdiv, but we actually need to find the vertex in the baseLevel
          // so we need to shift it into the baseSubdiv level
          coordUV.u <<= (baseSubdiv - targetSubdiv);
          coordUV.v <<= (baseSubdiv - targetSubdiv);
          coord = {uint16_t((1 << baseSubdiv) - coordUV.u - coordUV.v), coordUV.u, coordUV.v};

          MicromeshMTriVertex& mvtx = vertices[v];
          mvtx.packed               = 0;
          mvtx.packed |= packBits(coord.u, MICRO_MTRI_VTX_U_SHIFT, MICRO_MTRI_VTX_UV_WIDTH);
          mvtx.packed |= packBits(coord.v, MICRO_MTRI_VTX_V_SHIFT, MICRO_MTRI_VTX_UV_WIDTH);
          mvtx.packed |= packBits(corner, MICRO_MTRI_VTX_CORNER_SHIFT, MICRO_MTRI_VTX_CORNER_WIDTH);
          mvtx.packed |= packBits(triIndex, MICRO_MTRI_VTX_MTRI_SHIFT, MICRO_MTRI_VTX_MTRI_WIDTH);
        }
      }
    }
  }
}

static_assert(MICRO_MTRI_DESCEND_VERTEX_WIDTH <= sizeof(uint16_t) * 8, "MICRO_MTRI_DESCEND_VERTEX_WIDTH size too big");

static void packInto(MicromeshMTriDescend&      descend,
                     uint32_t                   idx,
                     uint32_t                   blockSubdiv,
                     uint32_t                   blockFormat,
                     const MicromeshFormatInfo& formatInfos,
                     baryutils::BaryWUV_uint16  coord,
                     uint32_t                   descendLevel)
{
  uint32_t index;
  uint32_t data;
  uint32_t level;
  uint32_t type;

  bary::baryBirdLayoutGetVertexLevelInfo(coord.u, coord.v, blockSubdiv, &level, &index);
  data = formatInfos.getBlockIndex(blockFormat, level, index);

  assert(descendLevel == ~0 || level == descendLevel);

  type = formatInfos.getVertexType(coord);

  uint32_t packed = 0;
  packed |= packBits(data, MICRO_MTRI_DESCEND_VERTEX_DATA_SHIFT, MICRO_MTRI_DESCEND_VERTEX_DATA_WIDTH);
  packed |= packBits(level, MICRO_MTRI_DESCEND_VERTEX_LVL_SHIFT, MICRO_MTRI_DESCEND_VERTEX_LVL_WIDTH);
  packed |= packBits(type, MICRO_MTRI_DESCEND_VERTEX_TYPE_SHIFT, MICRO_MTRI_DESCEND_VERTEX_TYPE_WIDTH);

  descend.vertices[idx] = uint16_t(packed);
}

void MicromeshMicroTriangleDecoderVK::uploadDescends(nvvk::StagingMemoryManager* staging, VkCommandBuffer cmd, const MicroSplitParts& splits)
{
  MicromeshMTriDescend* descendsAll =
      staging->cmdToBufferT<MicromeshMTriDescend>(cmd, m_micro.descends.buffer, m_micro.descends.info.offset,
                                                  m_micro.descends.info.range);

  const MicromeshFormatInfo info;

  for(uint32_t blockFormat = 0; blockFormat < MICRO_MAX_FORMATS; blockFormat++)
  {
    uint32_t                numTrianglesPerBlock;
    uint32_t                numDescendLevels;
    uint32_t                blockSubdiv;
    uint32_t                blockBits;
    bary::BlockFormatDispC1 baryFormat;
    if(blockFormat == MICRO_FORMAT_64T_512B)
    {
      baryFormat           = bary::BlockFormatDispC1::eR11_unorm_lvl3_pack512;
      numTrianglesPerBlock = 64;
      numDescendLevels     = 1;
      blockSubdiv          = 3;
      blockBits            = 512;
    }
    else if(blockFormat == MICRO_FORMAT_256T_1024B)
    {
      baryFormat           = bary::BlockFormatDispC1::eR11_unorm_lvl4_pack1024;
      numTrianglesPerBlock = 256;
      numDescendLevels     = 5;
      blockSubdiv          = 4;
      blockBits            = 1024;
    }
    else if(blockFormat == MICRO_FORMAT_1024T_1024B)
    {
      baryFormat           = bary::BlockFormatDispC1::eR11_unorm_lvl5_pack1024;
      numTrianglesPerBlock = 1024;
      numDescendLevels     = 6;
      blockSubdiv          = 5;
      blockBits            = 1024;
    }
    else
    {
      assert(0 && "format not supported");
      return;
    }

    const baryutils::BaryLevelsMap::Level& baseLevel  = splits.map.getLevel(MICRO_MAX_SUBDIV);
    const baryutils::BaryLevelsMap::Level& blockLevel = splits.map.getLevel(blockSubdiv);

    if(blockFormat == MICRO_FORMAT_64T_512B)
    {
      for(uint32_t tri = 0; tri < numTrianglesPerBlock; tri++)
      {
        MicromeshMTriDescend&                     descend = descendsAll[MICRO_MTRI_DESCENDS_INDEX(tri, blockFormat)];
        const baryutils::BaryLevelsMap::Triangle& blockTriangle = blockLevel.triangles[tri];

        packInto(descend, 0, blockSubdiv, blockFormat, info, blockLevel.coordinates[blockTriangle.a], ~0);
        packInto(descend, 1, blockSubdiv, blockFormat, info, blockLevel.coordinates[blockTriangle.b], ~0);
        packInto(descend, 2, blockSubdiv, blockFormat, info, blockLevel.coordinates[blockTriangle.c], ~0);
      }
    }
    else
    {
      // create the outermost triangle and then split towards the target

      std::vector<bary::BlockTriangle> triStates[2];
      triStates[0].resize(numTrianglesPerBlock);
      triStates[1].resize(numTrianglesPerBlock);

      {
        bary::BlockTriangle& triState = triStates[0][0];
        triState.flipped              = 0;
        triState.vertices[0]          = baryutils::makeUV(1 << (blockSubdiv), 0, 0);
        triState.vertices[1]          = baryutils::makeUV(0, 1 << (blockSubdiv), 0);
        triState.vertices[2]          = baryutils::makeUV(0, 0, 1 << (blockSubdiv));

        MicromeshMTriDescend& descend = descendsAll[MICRO_MTRI_DESCENDS_INDEX(0, blockFormat)];
        packInto(descend, 0, blockSubdiv, blockFormat, info, baryutils::makeWUV(triState.vertices[0], blockSubdiv), 0);
        packInto(descend, 1, blockSubdiv, blockFormat, info, baryutils::makeWUV(triState.vertices[1], blockSubdiv), 0);
        packInto(descend, 2, blockSubdiv, blockFormat, info, baryutils::makeWUV(triState.vertices[2], blockSubdiv), 0);
      }

      uint32_t descendOffset = 1;
      for(uint32_t descendLevel = 1; descendLevel < numDescendLevels; descendLevel++)
      {
        std::vector<bary::BlockTriangle>& writeStates = triStates[(descendLevel & 1)];
        std::vector<bary::BlockTriangle>& readStates  = triStates[(descendLevel & 1) ^ 1];

        uint numSplits = 1 << ((descendLevel - 1) * 2);
        for(uint32_t split = 0; split < numSplits; split++)
        {
          MicromeshMTriDescend& descend = descendsAll[MICRO_MTRI_DESCENDS_INDEX(descendOffset + split, blockFormat)];
          const bary::BlockTriangle* previousState = &readStates[split];
          bary::BlockTriangle*       splitStates   = &writeStates[split * 4];

          //
          //               V
          //              / \ 
          //             / 3 \ 
          //            c0____c1
          //           / \ 1 / \ 
          //          / 0 \ / 2 \ 
          //         W ___ c2___ U
          //

          // perform split
          bary::baryBlockTriangleSplitDispC1(previousState, splitStates, 1);

          {
            packInto(descend, 0, blockSubdiv, blockFormat, info,
                     baryutils::makeWUV(splitStates[0].vertices[2], blockSubdiv), descendLevel);
            packInto(descend, 1, blockSubdiv, blockFormat, info,
                     baryutils::makeWUV(splitStates[2].vertices[2], blockSubdiv), descendLevel);
            packInto(descend, 2, blockSubdiv, blockFormat, info,
                     baryutils::makeWUV(splitStates[0].vertices[1], blockSubdiv), descendLevel);
          }
        }

        descendOffset += numSplits;
      }
    }
  }
}

}  // namespace microdisp
