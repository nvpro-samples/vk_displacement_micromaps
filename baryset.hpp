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
#include <baryutils/baryutils.h>

#include <memory>

struct BaryDisplacementAttribute
{
  // displacement
  // The BARY representation of uncompressed data
  std::unique_ptr<baryutils::BaryBasicData> uncompressed = nullptr;

  // The BARY representation of compressed data
  std::unique_ptr<baryutils::BaryBasicData> compressed = nullptr;
  std::unique_ptr<baryutils::BaryMiscData>  compressedMisc = nullptr;
};

enum ShadingAttributeBit : uint32_t
{
  SHADING_ATTRIBUTE_NORMAL_BIT = 1
};

struct BaryShadingAttribute
{
  // this is a bit of a hack to allow the baking tool and viewer
  // to use per-micro vertex shading attributes
  std::unique_ptr<baryutils::BaryBasicData> attribute = nullptr;

  uint32_t attributeFlags = 0;
  // which displacement this shading attribute is used with
  uint32_t displacementID = MeshSetID::INVALID;
};

// Scenes can have multiple sources of displacement information. This class
// stores all those sources, as well as information about how each mesh indexes
// into each source. To render a mesh set with displacement, you need both a
// MeshSet ("base mesh and mapping information") and
// a BaryAttributesSet ("set of displacements")
struct BaryAttributesSet
{
  // We have one of these for each source of displacement (e.g. .bary file).
  // These are indexed using each Mesh's displacementId member.

  std::vector<BaryDisplacementAttribute> displacements;
  std::vector<BaryShadingAttribute>      shadings;

  // shading attribute may not exist
  const BaryShadingAttribute* getShading(size_t idx) const { return shadings.empty() || idx == MeshSetID::INVALID ? nullptr : &shadings[idx]; }
  BaryShadingAttribute*       getShading(size_t idx) { return shadings.empty() || idx == MeshSetID::INVALID ? nullptr : &shadings[idx]; }

  const BaryShadingAttribute* getDisplacementShading(uint32_t dispID, ShadingAttributeBit attributeBit) const;

  // returns MeshSetID::INVALID on fail otherwise slot in `displacements`
  uint32_t loadDisplacement(const char* baryFilePathCStr, baryutils::BaryFile& bfile, baryutils::BaryFileOpenOptions* fileOpenOptions = nullptr);

  // returns MeshSetID::INVALID on fail otherwise slot in `shading attributes`
  uint32_t loadAttribute(const char* baryFilePathCStr, baryutils::BaryFile& bfile, baryutils::BaryFileOpenOptions* fileOpenOptions = nullptr);



  // these depend on updateStats() called
  baryutils::BaryStats compressedStats;
  baryutils::BaryStats uncompressedStats;
  baryutils::BaryStats shadingStats;

  uint64_t compressedMipByteSize = 0;

  // whenever the contents of displacements change, call this
  // function
  void updateStats();

  // these require updateStats to be called before
  baryutils::BaryLevelsMap makeBaryLevelsMapShading() const
  {
    return baryutils::BaryLevelsMap(shadingStats.valueOrder, shadingStats.maxSubdivLevel);
  }
  baryutils::BaryLevelsMap makeBaryLevelsMapUncompressed() const
  {
    return baryutils::BaryLevelsMap(uncompressedStats.valueOrder, uncompressedStats.maxSubdivLevel);
  }
  baryutils::BaryLevelsMap makeBaryLevelsMapCompressed() const
  {
    return baryutils::BaryLevelsMap(compressedStats.valueOrder, compressedStats.maxSubdivLevel);
  }

  double computeShellVolume(const MeshSet& baseMeshSet, bool perferDirectionBounds, bool preferUncompressed, uint32_t numThreads = 0) const;

  void fillUniformDirectionBounds(MeshSet& meshSet) const;

  bool supportsCompressedMips() const;

};