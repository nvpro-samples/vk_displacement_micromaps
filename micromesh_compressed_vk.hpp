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

#include <baryutils/baryutils.h>
#include "meshset.hpp"
#include "baryset.hpp"
#include "resources_base_vk.hpp"

namespace microdisp {

struct MicromeshCombinedData;

struct MicromeshSetCompressedVK
{
  RBuffer umajor2bmap;

  RBuffer vertices;
  RBuffer descends;
  RBuffer triangleIndices;

  struct MeshData
  {
    RBuffer                binding;
    MicromeshCombinedData* combinedData = nullptr;

    // Either base- or sub-triangle data is used and
    // never both. We kept separate variables
    // for clarity

    RBuffer baseTriangles;
    RBuffer baseSpheres;

    RBuffer subTriangles;
    RBuffer subSpheres;

    RBuffer distances;
    RBuffer mipDistances;

    RBuffer attrNormals;
    RBuffer attrTriangles;

    // just for visualization purposes not for rendering
    RBuffer baseTriangleMinMaxs;

    // either sub or base triangle count
    uint32_t microTriangleCount;
  };

  std::vector<MeshData> meshDatas;

  bool hasBaseTriangles = false;
  bool usedFormats[uint32_t(bary::BlockFormatDispC1::eR11_unorm_lvl5_pack1024) + 1];

  // creates buffers & uploads typical data that is agnostic of the specific
  // rasterization decoder chosen.
  // see various `micromesh_decoder_...` files for the full init sequence

  void initBasics(ResourcesVK& res, const MeshSet& meshSet, const BaryAttributesSet& barySet, bool useBaseTriangles, bool useMips);

  // creates buffers & uploads micro vertex attribute normals
  void initAttributeNormals(ResourcesVK& res, const MeshSet& meshSet, const BaryAttributesSet& barySet, uint32_t numThreads = 0);

  void deinit(ResourcesVK& res);

  // updates the state of `MeshData::combinedData` to retrieve most buffer addresses and store them
  // in the binding buffer
  void uploadMeshDatasBinding(nvvk::StagingMemoryManager* staging, VkCommandBuffer cmd);
};

}  // namespace microdisp