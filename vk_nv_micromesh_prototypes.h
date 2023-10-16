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

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//
// WARNING: VK_NV_displacement_micromap is still in beta
//
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "vk_nv_micromesh.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef VULKAN_NV_DEFINED_EXT_opacity_micromap
#ifndef VK_NO_PROTOTYPES
void load_VK_EXT_opacity_micromap_prototypes(VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr);
#else
typedef struct VK_EXT_opacity_micromap_functions
{
  PFN_vkCreateMicromapEXT                 pfn_vkCreateMicromapEXT;
  PFN_vkDestroyMicromapEXT                pfn_vkDestroyMicromapEXT;
  PFN_vkCmdBuildMicromapsEXT              pfn_vkCmdBuildMicromapsEXT;
  PFN_vkBuildMicromapsEXT                 pfn_vkBuildMicromapsEXT;
  PFN_vkCopyMicromapEXT                   pfn_vkCopyMicromapEXT;
  PFN_vkCopyMicromapToMemoryEXT           pfn_vkCopyMicromapToMemoryEXT;
  PFN_vkCopyMemoryToMicromapEXT           pfn_vkCopyMemoryToMicromapEXT;
  PFN_vkWriteMicromapsPropertiesEXT       pfn_vkWriteMicromapsPropertiesEXT;
  PFN_vkCmdCopyMicromapEXT                pfn_vkCmdCopyMicromapEXT;
  PFN_vkCmdCopyMicromapToMemoryEXT        pfn_vkCmdCopyMicromapToMemoryEXT;
  PFN_vkCmdCopyMemoryToMicromapEXT        pfn_vkCmdCopyMemoryToMicromapEXT;
  PFN_vkCmdWriteMicromapsPropertiesEXT    pfn_vkCmdWriteMicromapsPropertiesEXT;
  PFN_vkGetDeviceMicromapCompatibilityEXT pfn_vkGetDeviceMicromapCompatibilityEXT;
  PFN_vkGetMicromapBuildSizesEXT          pfn_vkGetMicromapBuildSizesEXT;
} VK_EXT_opacity_micromap_functions;

void load_VK_EXT_opacity_micromap_functions(VK_EXT_opacity_micromap_functions* fns, VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr);
#endif
#else   // ^^^ #ifdef VULKAN_NV_DEFINED_EXT_opacity_micromap
// When the Vulkan SDK provides VK_EXT_opacity_micromap, extensions_vk.cpp loads it for us.
inline void load_VK_EXT_opacity_micromap_prototypes(VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr){};
#endif  // #ifdef VULKAN_NV_DEFINED_EXT_opacity_micromap

#ifdef __cplusplus
}
#endif

// there are no extra function prototypes for displacement