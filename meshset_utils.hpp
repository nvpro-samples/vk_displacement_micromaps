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

#pragma once

#include "meshset.hpp"

// resizes attribute vectors if non-empty in meshRef
void resizeAttributes(MeshAttributes& attribsOut, const MeshAttributes& attribsRef, size_t numVertices);

void compactAttributes(MeshAttributes&              attribsOut,
                       const MeshAttributes&        attribsRef,
                       const std::vector<uint32_t>& vertexMap,
                       uint32_t                     numThreads = 0);

// copies all but indices/attributes
void copyNoGeometryData(MeshSet& meshSetOut, const MeshSet& meshSetIn);

// return indices of meshinfos with unique firstIndex/numIndices pairing
// if nonOverlappingRanges is true, can return empty if firstIndex is double booked
// under different numIndices
std::vector<size_t> getUniqueIndexMeshs(const MeshSet& meshSet, bool nonOverlappingRanges);

// return indices of meshinfos with unique firstVertex/numVertices pairing
// if nonOverlappingRanges is true, can return empty if firstVertex is double booked
// under different numVertices
std::vector<size_t> getUniqueVertexMeshs(const MeshSet& meshSet, bool nonOverlappingRanges);

// returns array of global indices with mesh.firstVertex already applied,
// empty if firstVertex values are inconsistent among meshInfos
std::vector<uint32_t> getIndicesWithFirstVertex(const MeshSet& meshSet, uint32_t numThreads = 0);

std::vector<uint32_t> buildTriangleMeshs(const MeshSet& meshSet);

// output map contains index to self (if unique) or index to first vertex with same position
std::vector<uint32_t> buildUniquePositionMap(const MeshSet& meshSet, uint32_t numThreads = 0);

// output map contains index to self (if unique) or index to first vertex with same position/direction
std::vector<uint32_t> buildUniquePositionDirectionMap(const MeshSet& meshSet, uint32_t numThreads = 0);

std::vector<uint32_t> buildMappedIndices(const std::vector<uint32_t>& vertexMap,
                                         const std::vector<uint32_t>& indices,
                                         uint32_t                     numThreads = 0);

// based on vertex uniqueMap creates smooth per-vertex normals
// areaWeight allows to influence the vertex normalization by triangle area
// otherwise we normalize over affected triangles
// returns empty if prerequisites not met (vtxUniqueMap and glocalUniqueIndices must exist)
std::vector<glm::vec3> buildSmoothVertexNormals(const MeshSet& meshSet, float areaWeight = 0.0f, uint32_t numThreads = 0);


template <typename T>
inline glm::vec<2, T, glm::qualifier::defaultp> getInterpolated(const std::vector<glm::vec<2, T, glm::qualifier::defaultp>>& attributes,
                                                                glm::vec3  coord,
                                                                glm::uvec3 indices)
{
  return attributes[indices.x] * coord.x + attributes[indices.y] * coord.y + attributes[indices.z] * coord.z;
}

template <typename T>
inline T getInterpolated(const std::vector<T>& attributes, glm::vec3 coord, glm::uvec3 indices)
{
  return attributes[indices.x] * coord.x + attributes[indices.y] * coord.y + attributes[indices.z] * coord.z;
}

template <typename T>
inline T getInterpolated(const T* attributes, glm::vec3 coord, glm::uvec3 indices)
{
  return attributes[indices.x] * coord.x + attributes[indices.y] * coord.y + attributes[indices.z] * coord.z;
}

template <typename T>
inline T getInterpolated(const T* attributes, glm::vec3 coord)
{
  return attributes[0] * coord.x + attributes[1] * coord.y + attributes[2] * coord.z;
}

// Add 3 values by first sorting them from smallest to largest, so the result is order-independent
template <typename T>
inline glm::vec<3, T, glm::qualifier::defaultp> add3Sorted(glm::vec<3, T, glm::qualifier::defaultp> a,
                                                           glm::vec<3, T, glm::qualifier::defaultp> b,
                                                           glm::vec<3, T, glm::qualifier::defaultp> c)
{
  T aL = a[0] + a[1] + a[2];
  T bL = b[0] + b[1] + b[2];
  T cL = c[0] + c[1] + c[2];

  if(aL > bL)
    std::swap(a, b);
  if(bL > cL)
    std::swap(b, c);
  if(aL > bL)
    std::swap(a, b);

  return a + b + c;
}

template <typename T>
inline glm::vec<2, T, glm::qualifier::defaultp> add3Sorted(glm::vec<2, T, glm::qualifier::defaultp> a,
                                                           glm::vec<2, T, glm::qualifier::defaultp> b,
                                                           glm::vec<2, T, glm::qualifier::defaultp> c)
{
  T aL = a[0] + a[1];
  T bL = b[0] + b[1];
  T cL = c[0] + c[1];

  if(aL > bL)
    std::swap(a, b);
  if(bL > cL)
    std::swap(b, c);
  if(aL > bL)
    std::swap(a, b);

  return a + b + c;
}

template <typename T>
inline T add3Sorted(T a, T b, T c)
{
  if(a > b)
    std::swap(a, b);
  if(b > c)
    std::swap(b, c);
  if(a > b)
    std::swap(a, b);
  return a + b + c;
}


// Order-independent interpolation using sorting (see add3Sorted above)
template <typename T>
inline glm::vec<2, T, glm::qualifier::defaultp> getInterpolatedSorted(const std::vector<glm::vec<2, T, glm::qualifier::defaultp>>& attributes,
                                                                      glm::vec3  coord,
                                                                      glm::uvec3 indices)
{
  return add3Sorted(attributes[indices.x] * coord.x, attributes[indices.y] * coord.y, attributes[indices.z] * coord.z);
}

template <typename T>
inline T getInterpolatedSorted(const std::vector<T>& attributes, glm::vec3 coord, glm::uvec3 indices)
{
  return add3Sorted(attributes[indices.x] * coord.x, attributes[indices.y] * coord.y, attributes[indices.z] * coord.z);
}

template <typename T>
inline T getInterpolatedSorted(const T* attributes, glm::vec3 coord, glm::uvec3 indices)
{
  return add3Sorted(attributes[indices.x] * coord.x, attributes[indices.y] * coord.y, attributes[indices.z] * coord.z);
}

template <typename T>
inline T getInterpolatedSorted(const T* attributes, glm::vec3 coord)
{
  return add3Sorted(attributes[0] * coord.x, attributes[1] * coord.y, attributes[2] * coord.z);
}
