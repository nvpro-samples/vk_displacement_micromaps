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

#include "meshset_utils.hpp"
#include "parallel_work.hpp"
#include <cassert>
#include <cstring> // memcmp
#include <glm/glm.hpp>
#include <unordered_set>

static inline uint32_t murmurHash2A(const void* key, size_t len, uint32_t seed)
{
  if(len == 0)
    return seed;

#define mmix(h, k)                                                                                                     \
  {                                                                                                                    \
    k *= m;                                                                                                            \
    k ^= k >> r;                                                                                                       \
    k *= m;                                                                                                            \
    h *= m;                                                                                                            \
    h ^= k;                                                                                                            \
  }

  const uint32_t m = 0x5bd1e995;
  const int32_t  r = 24;
  size_t         l = len;

  const uint8_t* data = (const uint8_t*)key;

  uint32_t h = seed;
  uint32_t t = 0;

  while(len >= 4)
  {
    uint32_t k = *(uint32_t*)data;

    mmix(h, k);

    data += 4;
    len -= 4;
  }


  switch(len)
  {
    case 3:
      t ^= data[2] << 16;
    case 2:
      t ^= data[1] << 8;
    case 1:
      t ^= data[0];
  };

  mmix(h, t);
  mmix(h, l);

  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;

  return h;

#undef mmix
}

void resizeAttributes(MeshAttributes& out, const MeshAttributes& attribsRef, size_t numVertices)
{
  out.positions.resize(numVertices);
  out.normals.resize(attribsRef.normals.empty() ? 0 : numVertices);
  out.tangents.resize(attribsRef.tangents.empty() ? 0 : numVertices);
  out.bitangents.resize(attribsRef.bitangents.empty() ? 0 : numVertices);
  out.texcoords0.resize(attribsRef.texcoords0.empty() ? 0 : numVertices);
}

// based on vertexMap keep indices not equal to ~0
// output vector must be sized correctly
template <typename T>
static void compactAttributeVector(std::vector<T>& attribOut, const std::vector<T>& attribIn, const std::vector<uint32_t>& vertexMap, uint32_t numThreads)
{
  parallel_batches(
      attribIn.size(),
      [&](uint64_t v) {
        uint32_t idx = vertexMap[v];
        if(idx != ~0u)
        {
          attribOut[idx] = attribIn[v];
        }
      },
      numThreads);
}

void compactAttributes(MeshAttributes& attribsOut, const MeshAttributes& attribsRef, const std::vector<uint32_t>& vertexMap, uint32_t numThreads /*= 0*/)
{
  compactAttributeVector(attribsOut.positions, attribsRef.positions, vertexMap, numThreads);
  compactAttributeVector(attribsOut.normals, attribsRef.normals, vertexMap, numThreads);
  compactAttributeVector(attribsOut.tangents, attribsRef.tangents, vertexMap, numThreads);
  compactAttributeVector(attribsOut.bitangents, attribsRef.bitangents, vertexMap, numThreads);
  compactAttributeVector(attribsOut.texcoords0, attribsRef.texcoords0, vertexMap, numThreads);
}

void copyNoGeometryData(MeshSet& meshSetOut, const MeshSet& meshSetIn)
{
  meshSetOut.bbox          = meshSetIn.bbox;
  meshSetOut.materials     = meshSetIn.materials;
  meshSetOut.meshInfos     = meshSetIn.meshInfos;
  meshSetOut.meshInstances = meshSetIn.meshInstances;
}

std::vector<size_t> getUniqueIndexMeshs(const MeshSet& meshSet, bool nonOverlap)
{
  size_t              numMeshes = meshSet.meshInfos.size();
  std::vector<size_t> uniqueMeshs;

  std::unordered_map<uint64_t, uint32_t> uniqueStorageMap;
  uniqueStorageMap.reserve(numMeshes);
  uniqueMeshs.reserve(numMeshes);

  for(size_t i = 0; i < numMeshes; i++)
  {
    const MeshInfo& info = meshSet.meshInfos[i];
    uint64_t        hash = uint64_t(info.firstIndex) | (nonOverlap ? 0 : uint64_t(info.numIndices) << 32);
    auto            it   = uniqueStorageMap.find(hash);
    if(it == uniqueStorageMap.end())
    {
      uniqueMeshs.push_back(i);
      uniqueStorageMap.insert({hash, info.numIndices});
    }
    else if(nonOverlap && it->second != info.numIndices)
    {
      return {};
    }
  }

  return uniqueMeshs;
}

std::vector<size_t> getUniqueVertexMeshs(const MeshSet& meshSet, bool nonOverlappingRanges)
{
  size_t              numMeshes = meshSet.meshInfos.size();
  std::vector<size_t> uniqueMeshs;

  std::unordered_map<uint64_t, uint32_t> uniqueStorageMap;
  uniqueStorageMap.reserve(numMeshes);
  uniqueMeshs.reserve(numMeshes);

  for(size_t i = 0; i < numMeshes; i++)
  {
    const MeshInfo& info = meshSet.meshInfos[i];
    uint64_t        hash = uint64_t(info.firstVertex) | (nonOverlappingRanges ? 0 : uint64_t(info.numVertices) << 32);
    auto            it   = uniqueStorageMap.find(hash);
    if(it == uniqueStorageMap.end())
    {
      uniqueMeshs.push_back(i);
      uniqueStorageMap.insert({hash, info.numVertices});
    }
    else if(nonOverlappingRanges && it->second != info.numVertices)
    {
      return {};
    }
  }

  return uniqueMeshs;
}

std::vector<uint32_t> getIndicesWithFirstVertex(const MeshSet& meshSet, uint32_t numThreads /*= 0*/)
{
  std::vector<uint32_t> indices(meshSet.indices.size(), ~0u);

  for(const MeshInfo& it : meshSet.meshInfos)
  {
    bool nonConsistent = false;

    parallel_batches(
        it.numIndices,
        [&](uint64_t idx) {
          uint32_t newIndex = it.firstVertex + meshSet.indices[idx + it.firstIndex];
          if(indices[it.firstIndex + idx] == ~0u)
          {
            indices[it.firstIndex + idx] = newIndex;
          }
          else if(indices[it.firstIndex + idx] != newIndex)
          {
            nonConsistent = true;
          }
        },
        numThreads);

    if(nonConsistent)
    {
      return {};
    }
  }

  return indices;
}


std::vector<uint32_t> buildUniquePositionMap(const MeshSet& meshSet, uint32_t numThreads /*= 0*/)
{
  struct VertexHashEntry
  {
    uint32_t numVertices = 0;
    uint32_t firstVertex = 0;
  };

  size_t                numVertices = meshSet.attributes.positions.size();
  std::vector<uint32_t> vertexUniquePosMap(numVertices);

  for(size_t i = 0; i < numVertices; i++)
  {
    vertexUniquePosMap[i] = uint32_t(i);
  }

  // don't build globally, but locally within meshInfos
  for(const MeshInfo& mesh : meshSet.meshInfos)
  {
    const uint32_t               hashTableMask = 0xFFFFF;
    const uint32_t               hashTableSize = hashTableMask + 1;
    std::vector<VertexHashEntry> hashEntries(hashTableSize, VertexHashEntry());
    std::vector<uint32_t>        vertexHashes(mesh.numVertices);

    std::vector<std::atomic_uint32_t> vertexHashCounts(hashTableSize);
    for(auto& it : vertexHashCounts)
    {
      it = 0;
    }

    parallel_batches(
        mesh.numVertices,
        [&](uint64_t idx) {
          uint32_t hash =
              murmurHash2A(meshSet.attributes.positions.data() + idx + mesh.firstVertex, sizeof(glm::vec3), 127) & hashTableMask;
          vertexHashes[idx] = hash;
          vertexHashCounts[hash] += 1;
        },
        numThreads);

    uint32_t offset = 0;
    for(uint32_t i = 0; i < hashTableSize; i++)
    {
      hashEntries[i].firstVertex = offset;
      hashEntries[i].numVertices = vertexHashCounts[i];
      offset += hashEntries[i].numVertices;
      hashEntries[i].numVertices = 0;
    }

    std::vector<uint32_t> vertexHashIndices(mesh.numVertices);
    for(size_t i = 0; i < mesh.numVertices; i++)
    {
      uint32_t hash                                                                        = vertexHashes[i];
      vertexHashIndices[hashEntries[hash].firstVertex + (hashEntries[hash].numVertices++)] = uint32_t(i) + mesh.firstVertex;
    }

    parallel_batches(
        mesh.numVertices,
        [&](uint64_t idx) {
          uint32_t               hash  = vertexHashes[idx];
          const VertexHashEntry& entry = hashEntries[hash];
          for(uint32_t i = 0; i < entry.numVertices; i++)
          {
            uint32_t iOther = vertexHashIndices[entry.firstVertex + i];
            uint64_t iSelf  = idx + mesh.firstVertex;
            if(memcmp(meshSet.attributes.positions.data() + iOther, meshSet.attributes.positions.data() + iSelf, sizeof(glm::vec3)) == 0)
            {
              vertexUniquePosMap[iSelf] = iOther;
              break;
            }
          }
        },
        numThreads);
  }

  return vertexUniquePosMap;
}

std::vector<uint32_t> buildTriangleMeshs(const MeshSet& meshSet)
{
  std::vector<uint32_t> triMeshs;
  triMeshs.resize(meshSet.indices.size() / 3);

  for(size_t m = 0; m < meshSet.meshInfos.size(); m++)
  {
    size_t first = meshSet.meshInfos[m].firstPrimitive;
    for(size_t i = 0; i < meshSet.meshInfos[m].numPrimitives; i++)
    {
      triMeshs[first + i] = uint32_t(m);
    }
  }

  return triMeshs;
}

std::vector<uint32_t> buildUniquePositionDirectionMap(const MeshSet& meshSet, uint32_t numThreads /*= 0*/)
{
  struct VertexHashEntry
  {
    uint32_t numVertices = 0;
    uint32_t firstVertex = 0;
  };

  size_t                numVertices = meshSet.attributes.positions.size();
  std::vector<uint32_t> vertexUniquePosDirMap(numVertices);

  for(size_t i = 0; i < numVertices; i++)
  {
    vertexUniquePosDirMap[i] = uint32_t(i);
  }

  // don't build globally, but locally within meshInfos
  for(const MeshInfo& mesh : meshSet.meshInfos)
  {
    const uint32_t               hashTableMask = 0xFFFFF;
    const uint32_t               hashTableSize = hashTableMask + 1;
    std::vector<VertexHashEntry> hashEntries(hashTableSize, VertexHashEntry());
    std::vector<uint32_t>        vertexHashes(mesh.numVertices);

    std::vector<std::atomic_uint32_t> vertexHashCounts(hashTableSize);
    for(auto& it : vertexHashCounts)
    {
      it = 0;
    }

    parallel_batches(
        mesh.numVertices,
        [&](uint64_t idx) {
          uint32_t hash =
              murmurHash2A(meshSet.attributes.directions.data() + idx + mesh.firstVertex, sizeof(glm::vec3),
                           murmurHash2A(meshSet.attributes.positions.data() + idx + mesh.firstVertex, sizeof(glm::vec3), 127))
              & hashTableMask;
          vertexHashes[idx] = hash;
          vertexHashCounts[hash] += 1;
        },
        numThreads);

    uint32_t offset = 0;
    for(uint32_t i = 0; i < hashTableSize; i++)
    {
      hashEntries[i].firstVertex = offset;
      hashEntries[i].numVertices = vertexHashCounts[i];
      offset += hashEntries[i].numVertices;
      hashEntries[i].numVertices = 0;
    }

    std::vector<uint32_t> vertexHashIndices(mesh.numVertices);
    for(size_t i = 0; i < mesh.numVertices; i++)
    {
      uint32_t hash                                                                        = vertexHashes[i];
      vertexHashIndices[hashEntries[hash].firstVertex + (hashEntries[hash].numVertices++)] = uint32_t(i) + mesh.firstVertex;
    }

    parallel_batches(
        mesh.numVertices,
        [&](uint64_t idx) {
          uint32_t               hash  = vertexHashes[idx];
          const VertexHashEntry& entry = hashEntries[hash];
          for(uint32_t i = 0; i < entry.numVertices; i++)
          {
            uint32_t iOther = vertexHashIndices[entry.firstVertex + i];
            uint32_t iSelf  = uint32_t(idx) + mesh.firstVertex;
            if(memcmp(meshSet.attributes.positions.data() + iOther, meshSet.attributes.positions.data() + iSelf, sizeof(glm::vec3)) == 0
               && memcmp(meshSet.attributes.directions.data() + iOther, meshSet.attributes.directions.data() + iSelf,
                         sizeof(glm::vec3))
                      == 0)
            {
              vertexUniquePosDirMap[iSelf] = iOther;
              break;
            }
          }
        },
        numThreads);
  }

  return vertexUniquePosDirMap;
}

std::vector<uint32_t> buildMappedIndices(const std::vector<uint32_t>& vertexMap, const std::vector<uint32_t>& indices, uint32_t numThreads /*= 0*/)
{
  std::vector<uint32_t> newIndices;
  newIndices.resize(indices.size());

  parallel_batches(
      indices.size(), [&](uint64_t idx) { newIndices[idx] = vertexMap[indices[idx]]; }, numThreads);

  return newIndices;
}



std::vector<glm::vec3> buildSmoothVertexNormals(const MeshSet& meshSet, float areaWeight /*= 0.0f*/, uint32_t numThreads /*= 0*/)
{
  if(meshSet.globalUniquePosIndices.empty() || meshSet.globalUniquePosMap.empty())
  {
    return {};
  }

  size_t numVertices = meshSet.attributes.positions.size();
  size_t numTris     = meshSet.indices.size() / 3;

  const uint32_t*       uniques   = meshSet.globalUniquePosMap.data();
  const glm::uvec3* indices   = reinterpret_cast<const glm::uvec3*>(meshSet.globalUniquePosIndices.data());
  const glm::vec3*  positions = reinterpret_cast<const glm::vec3*>(meshSet.attributes.positions.data());

  std::vector<glm::vec3> triNormals(numTris);

  parallel_batches(
      numTris,
      [&](uint64_t idx) {
        glm::uvec3 triIndices = indices[idx];

        glm::vec3 v0 = positions[triIndices.x];
        glm::vec3 v1 = positions[triIndices.y];
        glm::vec3 v2 = positions[triIndices.z];

        glm::vec3 e0 = v1 - v0;
        glm::vec3 e1 = v2 - v0;

        glm::vec3 nrm = glm::cross(e0, e1);
        float         len = glm::length(nrm);
        len               = (0.5f * areaWeight) + (1.0f - areaWeight) * len;
        nrm               = nrm / len;

        triNormals[idx] = nrm;
      },
      numThreads);


  std::vector<glm::vec3> smoothNormals(numVertices, {0, 0, 0});
  for(size_t i = 0; i < numTris; i++)
  {
    glm::uvec3 triIndices = indices[i];

    smoothNormals[triIndices.x] += triNormals[i];
    smoothNormals[triIndices.y] += triNormals[i];
    smoothNormals[triIndices.z] += triNormals[i];
  }

  for(uint32_t i = 0; i < uint32_t(numVertices); i++)
  {
    uint32_t map = uniques[i];
    if(map == i)
    {
      smoothNormals[i] = glm::normalize(smoothNormals[i]);
    }
    else
    {
      smoothNormals[i] = smoothNormals[map];
    }
  }

  return smoothNormals;
}

