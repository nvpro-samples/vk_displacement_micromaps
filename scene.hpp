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

#include <stdint.h>
#include <string>
#include <vector>
#include <memory>

#include "config.h"
#include "meshset.hpp"
#include "baryset.hpp"

namespace microdisp {

class Scene
{
public:
  bool hasCompressedDisplacement   = false;
  bool hasUncompressedDisplacement = false;
  bool hasDirectionBounds       = false;

  // lo-resolution meshset that gets the displacement applied
  std::unique_ptr<MeshSet> meshSetLo;
  size_t                   meshSetLoOrigInstanceCount = 0;

  // stores barycentric attributes and displacements for meshLo
  // can be loaded from disk or filled via baryBaker
  BaryAttributesSet          barySet;
  double                     barySetShellVolume = 0;

  // Loads a scene from filename
  bool load(const char* filenameLo);

private:
  void finalizeMeshLo();
  // After loading displacement, computes necessary fields of a BaryMesh object,
  // based on the lo-res meshes.
  // Can set hasCustomBary, hasCustomUncompressed, and hasCustomCompressed.
  bool finalizeLoadedBarySet();
};
}  // namespace microdisp
