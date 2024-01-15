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

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <cctype>
#include <filesystem>
#include <nvh/misc.hpp>
#include <nvvk/images_vk.hpp>
#include <platform.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <thread>

#include "config.h"
#include "meshset_gltf.hpp"
#include "meshset_utils.hpp"
#include "parallel_work.hpp"
#include "resources_vk.hpp"
#include "scene_vk.hpp"
#include "vk_nv_micromesh.h"


namespace microdisp {

void SceneVK::init(ResourcesVK& res) {}

void SceneVK::deinit(ResourcesVK& res)
{
  meshSetLoVK.deinit(res);
}


void SceneVK::updateGrid(ResourcesVK& res, uint32_t copies, uint32_t axis, float spacing, bool gpu)
{
  copies = std::max(copies, 1u);

  // Handle the case where only meshLo exists (e.g. when viewing a file with displacement):
  glm::vec3 diagonal = meshSetLo->bbox.diagonal() * spacing;
  if(spacing < 0)
  {
    diagonal = glm::vec3(-spacing);
  }

  meshSetLo->setupInstanceGrid(meshSetLoOrigInstanceCount, copies, axis, diagonal);

  if(gpu)
  {
    meshSetLoVK.updateInstances(res, *meshSetLo);
  }
}

void SceneVK::updateLow(ResourcesVK& res)
{
  meshSetLoVK.deinit(res);
  meshSetLoVK.init(res, *meshSetLo);
  meshSetLoVK.initDisplacementDirections(res, *meshSetLo);
  meshSetLoVK.initDisplacementEdgeFlags(res, *meshSetLo);
  meshSetLoVK.initDisplacementBounds(res, *meshSetLo);
  meshSetLoVK.initNormalMaps(res, *meshSetLo);
}

}  // namespace microdisp
