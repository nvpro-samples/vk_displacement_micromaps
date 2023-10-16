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

#include "resources_base_vk.hpp"
#include <baryutils/baryutils.h>

namespace microdisp {

struct BaryLevelsMapVK
{
  struct Level
  {
    size_t coordsOffset  = 0;
    size_t headersOffset = 0;
    size_t dataOffset    = 0;

    size_t firstHeader  = 0;
    size_t headersCount = 0;
    size_t firstData    = 0;
    size_t dataCount    = 0;
  };

  RBuffer binding;
  RBuffer data;

  std::vector<Level> levels;

  const Level& getLevel(uint32_t subdivLevel, uint32_t topoBits, uint32_t maxLevelCount) const
  {
    return levels[subdivLevel + topoBits * maxLevelCount];
  }

  void init(ResourcesVK& res, const baryutils::BaryLevelsMap& bmap);
  void deinit(ResourcesVK& res);
};

}  // namespace microdisp
