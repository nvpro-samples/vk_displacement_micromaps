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

#include "scene.hpp"
#include "meshset_vk.hpp"

namespace microdisp {

class SceneVK : public Scene
{
public:
  // vk resources for lo-res meshset
  MeshSetVK meshSetLoVK;

  void init(ResourcesVK& res);
  void deinit(ResourcesVK& res);

  // Uploads the lo-res mesh (`meshLo`) to `meshLoVK`.
  void updateLow(ResourcesVK& res);
  // Updates the grid of instances of the lo-res and hi-res mesh.
  void updateGrid(ResourcesVK& res, uint32_t copies, uint32_t axis, float spacing, bool gpu = true);

private:
  void finalizeMeshLo();
  // After loading displacement, computes necessary fields of a BaryMesh object,
  // based on the lo-res meshes.
  // Can set hasCustomBary, hasCustomUncompressed, and hasCustomCompressed.
  bool finalizeLoadedBarySet();
};
}  // namespace microdisp
