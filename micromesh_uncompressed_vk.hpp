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

#include "barymap_vk.hpp"
#include "baryset.hpp"
#include "meshset.hpp"
#include "resources_base_vk.hpp"

namespace microdisp {

// This is an analogue to BaryAttributesSet, except it stores its data in Vulkan
// buffers instead of the CPU. Instead of sets of buffers for each BaryMesh,
// BaryMeshSetVK contains buffers which store all BaryAttributes data concatenated.
// This also includes a struct for getting the ranges of data used by each mesh.
struct MicromeshSetUncompressedVK
{
  BaryLevelsMapVK baryMap;

  struct MeshData
  {
    RBuffer binding;    // Contains pointers to the start of each buffer.
    RBuffer distances;  // Array of uint32_ts storing the raw bits of micro-vertex displacement values.

    RBuffer baseTriangles;        // Array of MicromeshUncBaseTri. 1 per mesh triangle.
    RBuffer baseTriangleSpheres;  // Array of vec4s containing information about bounding spheres for renderers. 1 per mesh triangle.
    RBuffer baseTrianglesMinMax;  // Type of array depends on bary format. 2 per bary triangle.

    RBuffer attrNormals; // per micro-vertex octant encoded normals (optional)
  };

  // 1:1 to meshSet.meshInfos
  std::vector<MeshData> meshDatas;

  void init(ResourcesVK& resources, const MeshSet& meshSet, const BaryAttributesSet& barySet, bool withAttributes, uint32_t numThreads);
  void deinit(ResourcesVK& resources);

private:
  // Updates the values of `binding` with pointers to the start of each Vulkan buffer.
  void uploadBinding(ResourcesVK& resources, VkCommandBuffer cmd, const MeshData& meshData) const;

  void uploadFlatTriangles(ResourcesVK& resources, VkCommandBuffer cmd, const MeshSet& meshSet, const BaryAttributesSet& barySet, uint32_t meshID, uint32_t numThreads);
};

}  // namespace microdisp
