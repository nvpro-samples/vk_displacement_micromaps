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

#include "micromesh_compressed_vk.hpp"

namespace microdisp {

class MicroSplitParts;

class MicromeshBaseTriangleDecoderVK
{
public:
  MicromeshBaseTriangleDecoderVK(MicromeshSetCompressedVK& microSet)
      : m_micro(microSet)
  {
  }

  bool init(ResourcesVK& res, const MeshSet& meshSet, const BaryAttributesSet& barySet, bool withAttributes, uint32_t numThreads);

private:
  MicromeshSetCompressedVK& m_micro;

  void uploadMicroBaseTriangles(nvvk::StagingMemoryManager* staging,
                                VkCommandBuffer             cmd,
                                const MeshSet&              meshSet,
                                const BaryAttributesSet&    barySet,
                                uint32_t                    meshID,
                                uint32_t                    numThreads);

  void uploadVertices(nvvk::StagingMemoryManager* staging, VkCommandBuffer cmd, const MicroSplitParts& splits);
};


}  // namespace microdisp
