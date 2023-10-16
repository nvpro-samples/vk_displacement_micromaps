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
#include "baryset.hpp"
#include "resources_base_vk.hpp"
#include "vk_nv_micromesh.h"

namespace microdisp {

struct MicromeshSetCompressedRayTracedVK
{
  RBuffer umajor2bmap;
  RBuffer instanceAttributes;
  RBuffer binding;

  // each bary::Group is mapped to one VkMicromapEXT
  struct Group
  {
    VkMicromapEXT micromap = nullptr;
    // referenced by above
    RBuffer micromeshData;

    std::vector<VkMicromapUsageEXT> usages;
  };

  struct Info
  {
    std::vector<Group> groups;
  };

  // maps 1:1 to BaryAttributesSet
  std::vector<Info> displacements;

  // maps 1:1 to meshes
  struct MeshData
  {
    RBuffer attrNormals;
    RBuffer attrTriangles;
  };
  std::vector<MeshData> meshDatas;

  void init(ResourcesVK& res, const MeshSet& meshSet, const BaryAttributesSet& barySet, uint32_t numThreads);
  void deinit(ResourcesVK& res);

  // creates buffers & uploads micro vertex attribute normals
  void initAttributeNormals(ResourcesVK& res, const MeshSet& meshSet, const BaryAttributesSet& barySet, uint32_t numThreads = 0);

};

}  // namespace microdisp
