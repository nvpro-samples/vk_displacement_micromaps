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

#include <algorithm>
#include <platform.h>
#include <nvh/nvprint.hpp>
#include <nvvk/buffers_vk.hpp>
#include <nvvk/commands_vk.hpp>
#include <nvvk/debug_util_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/memorymanagement_vk.hpp>
#include <nvvk/pipeline_vk.hpp>
#include <nvvk/shadermodulemanager_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>

namespace microdisp {

class ResourcesVK;

typedef nvvk::Texture RTextureR;

struct RBuffer : nvvk::Buffer
{
  VkDescriptorBufferInfo info = {VK_NULL_HANDLE};
  VkDeviceAddress        addr = 0;
};

struct RTextureRW : nvvk::Texture
{
  VkDescriptorImageInfo descriptorImage;
  VkExtent3D            extent;
};

inline void cmdCopyBuffer(VkCommandBuffer cmd, const RBuffer& src, const RBuffer& dst)
{
  VkBufferCopy cpy = {src.info.offset, dst.info.offset, src.info.range};
  vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &cpy);
}

}  // namespace microdisp
