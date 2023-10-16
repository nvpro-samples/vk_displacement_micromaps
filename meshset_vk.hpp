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
#include "resources_base_vk.hpp"
#include "vk_nv_micromesh.h"

namespace microdisp {
struct TriDisplacementVK;
struct MicromeshSetCompressedRayTracedVK;

struct MeshAttributesVK
{
  RBuffer normals;
  RBuffer tangents;
  RBuffer bitangents;
  RBuffer tex0s;

  void init(ResourcesVK& res, const MeshSet& meshSet);
  void deinit(ResourcesVK& res);
};

class MeshSetVK
{
public:
  RBuffer binding;

  RBuffer positions;
  RBuffer indices;
  RBuffer instances;

  RBuffer displacementDirections;
  RBuffer displacementDirectionBounds;
  RBuffer displacementEdgeFlags;

  MeshAttributesVK attr;

  struct MeshRT
  {
    nvvk::AccelKHR                                         blas;
    VkAccelerationStructureGeometryKHR                     geometry;
    VkAccelerationStructureBuildGeometryInfoKHR            geometryInfo;
    VkAccelerationStructureTrianglesDisplacementMicromapNV geometryDisplacement;
    VkAccelerationStructureBuildRangeInfoKHR               blasRange;
  };

  std::vector<MeshRT>                          rtMeshes;
  nvvk::AccelKHR                               sceneTlas;
  VkWriteDescriptorSetAccelerationStructureKHR sceneTlasInfo;

  std::vector<RTextureR> materialTextures;

  void init(ResourcesVK& resources, const MeshSet& meshSet);
  void initNormalMaps(ResourcesVK& resources, const MeshSet& meshSet);

  // if the meshSet doesn't have the appropriate content, existing resource will be destroyed
  // and binding updated to reflect that
  void initDisplacementDirections(ResourcesVK& resources, const MeshSet& meshSet);
  void initDisplacementBounds(ResourcesVK& resources, const MeshSet& meshSet);
  void initDisplacementEdgeFlags(ResourcesVK& resources, const MeshSet& meshSet);

  void initRayTracingGeometry(ResourcesVK&                             resources,
                              const MeshSet&                           meshSet,
                              const MicromeshSetCompressedRayTracedVK* micromeshSetRT = nullptr);
  void initRayTracingScene(ResourcesVK& resources, const MeshSet& meshSet, const MicromeshSetCompressedRayTracedVK* micromeshSetRT = nullptr);
  void deinit(ResourcesVK& resources);
  void deinitRayTracing(ResourcesVK& resources);

  void updateBinding(ResourcesVK& resources);
  void updateInstances(ResourcesVK& resources, const MeshSet& meshSet);
};

}  // namespace microdisp