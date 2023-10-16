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

#include "meshset.hpp"
#include "meshset_utils.hpp"
#include "parallel_work.hpp"

bool MeshSet::hasContiguousIndices() const
{
  for(const MeshInfo& it : meshInfos)
  {
    if(it.firstIndex % 3 || it.numIndices % 3)
      return false;
  }

  return true;
}

void MeshSet::setupProcessingGlobals(uint32_t numThreads)
{
  // this makes iterating over triangles easier
  globalIndices = getIndicesWithFirstVertex(*this, numThreads);
  assert(!globalIndices.empty());

  globalTriangleToMeshID = buildTriangleMeshs(*this);

  // create a second set of triangle indices that resolve the vertices to be unique positions
  // This undos the splitting of vertices when their texcoords or normals differ but positions
  globalUniquePosMap     = buildUniquePositionMap(*this, numThreads);
  globalUniquePosIndices = buildMappedIndices(globalUniquePosMap, globalIndices, numThreads);
}

void MeshSet::setupDirectionBoundsGlobals(uint32_t numThreads)
{
  globalUniquePosDirMap     = buildUniquePositionDirectionMap(*this, numThreads);
  globalUniquePosDirIndices = buildMappedIndices(globalUniquePosDirMap, globalIndices, numThreads);
}

void MeshSet::clearDirectionBoundsGlobals()
{
  globalUniquePosDirIndices.clear();
  globalUniquePosDirMap.clear();
}

void MeshSet::setupInstanceGrid(size_t numOrig, size_t copies, uint32_t axis, nvmath::vec3f refShift)
{
  srand(2342);
  size_t sq      = 1;
  int    numAxis = 0;
  if(!axis)
    axis = 3;

  for(int i = 0; i < 3; i++)
  {
    numAxis += (axis & (1 << i)) ? 1 : 0;
  }

  switch(numAxis)
  {
    case 1:
      sq = copies;
      break;
    case 2:
      while(sq * sq < copies)
      {
        sq++;
      }
      break;
    case 3:
      while(sq * sq * sq < copies)
      {
        sq++;
      }
      break;
  }


  meshInstances.resize(numOrig * copies);
  for(size_t c = 1; c < copies; c++)
  {
    nvmath::vec3f shift = refShift;

    float u = 0;
    float v = 0;
    float w = 0;

    switch(numAxis)
    {
      case 1:
        u = float(c);
        break;
      case 2:
        u = float(c % sq);
        v = float(c / sq);
        break;
      case 3:
        u = float(c % sq);
        v = float((c / sq) % sq);
        w = float(c / (sq * sq));
        break;
    }

    float use = u;

    if(axis & (1 << 0))
    {
      shift.x *= -use;
      if(numAxis > 1)
        use = v;
    }
    else
    {
      shift.x = 0;
    }

    if(axis & (1 << 1))
    {
      shift.y *= use;
      if(numAxis > 2)
        use = w;
      else if(numAxis > 1)
        use = v;
    }
    else
    {
      shift.y = 0;
    }

    if(axis & (1 << 2))
    {
      shift.z *= -use;
    }
    else
    {
      shift.z = 0;
    }

    for(size_t i = 0; i < numOrig; i++)
    {
      MeshInstance& minst = meshInstances[i + c * numOrig];
      minst               = meshInstances[i];

      nvmath::vec3f translation;
      nvmath::mat3f rot;
      nvmath::vec3f scale;
      minst.xform.get_translation(translation);
      if (axis & (8|16|32))
      {
        minst.xform.set_translation(nvmath::vec3f(0, 0, 0));
        nvmath::vec3f mask = { axis & 8 ? 1.0f : 0.0f, axis & 16 ? 1.0f : 0.0f, axis & 32 ? 1.0f : 0.0f};
        nvmath::vec3f dir(float(rand())/float(RAND_MAX), float(rand())/float(RAND_MAX), float(rand())/float(RAND_MAX));
        dir = nvmath::nv_max(dir * mask,mask * 0.00001f);
        float angle = (float(rand())/float(RAND_MAX)) * nv_pi * 2.0f;
        dir.normalize();
        minst.xform.rotate(angle, dir);
      }
      minst.xform.set_translation(translation + shift);
    }
  }
}

void MeshSet::setupLargestInstance()
{
  for(MeshInfo& mesh : meshInfos)
  {
    mesh.largestInstanceID = MeshSetID::INVALID;
  }

  std::vector<float> magnitude(meshInfos.size(), 0);

  for(size_t i = 0; i < meshInstances.size(); i++)
  {
    uint32_t meshInfoId = meshInstances[i].meshID;
    float    mag        = nvmath::length(meshInstances[i].bbox.diagonal());
    if(mag > magnitude[meshInfoId])
    {
      magnitude[meshInfoId]                   = mag;
      meshInfos[meshInfoId].largestInstanceID = uint32_t(i);
    }
  }
}

void MeshSet::setupEdgeLengths(uint32_t numThreads /*= 0*/)
{
  for(MeshInfo& mesh : meshInfos)
  {
    uint32_t numPerThread = std::max(numThreads, 1u);

    std::vector<float>    maxSizes(numPerThread, 0);
    std::vector<double>   avgSizes(numPerThread, 0);
    std::vector<uint32_t> avgCounts(numPerThread, 0);

    parallel_batches(
        mesh.numPrimitives,
        [&](uint64_t idx, uint32_t thread) {
          float size = 0;

          nvmath::vec3f va = attributes.positions[indices[idx * 3 + mesh.firstIndex + 0] + mesh.firstVertex];
          nvmath::vec3f vb = attributes.positions[indices[idx * 3 + mesh.firstIndex + 1] + mesh.firstVertex];
          nvmath::vec3f vc = attributes.positions[indices[idx * 3 + mesh.firstIndex + 2] + mesh.firstVertex];

          size = std::max(std::max((va - vb).norm(), (va - vc).norm()), (vb - vc).norm());

          maxSizes[thread] = std::max(maxSizes[thread], size);
          avgSizes[thread] += double(size);
          avgCounts[thread]++;
        },
        numThreads);

    double   average      = 0;
    uint32_t averageCount = 0;
    mesh.longestEdge      = 0;

    for(uint32_t i = 0; i < numPerThread; i++)
    {
      mesh.longestEdge = std::max(maxSizes[i], mesh.longestEdge);
      average += avgSizes[i];
      averageCount += avgCounts[i];
    }

    mesh.averageEdge = float(average / double(averageCount));
  }
}

std::vector<uint32_t> MeshSet::getDisplacementGroupOrderedMeshs(uint32_t displacementID) const
{
  uint32_t groupsCount = 0;
  for(size_t meshIdx = 0; meshIdx < meshInfos.size(); meshIdx++)
  {
    const MeshInfo& mesh = meshInfos[meshIdx];

    if(mesh.displacementID == displacementID)
    {
      groupsCount = std::max(mesh.displacementGroup + 1, groupsCount);
    }
  }

  std::vector<uint32_t> localMeshGroups(groupsCount, MeshSetID::INVALID);

  for(size_t meshIdx = 0; meshIdx < meshInfos.size(); meshIdx++)
  {
    const MeshInfo& mesh = meshInfos[meshIdx];

    if(mesh.displacementID == displacementID)
    {
      localMeshGroups[mesh.displacementGroup] = uint32_t(meshIdx);
    }
  }

  return localMeshGroups;
}

uint32_t MeshSet::findDisplacementGroupMesh(uint32_t displacementID, uint32_t groupID) const
{
  for(size_t meshIdx = 0; meshIdx < meshInfos.size(); meshIdx++)
  {
    const MeshInfo& mesh = meshInfos[meshIdx];
    if(mesh.displacementID == displacementID && mesh.displacementGroup == groupID)
    {
      return uint32_t(meshIdx);
    }
  }

  return MeshSetID::INVALID;
}
