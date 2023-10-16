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

//////////////////////////////////////////////////////////////////////////
// MicroBinPack
//
// For dynamic level of detail handling and to improve efficiency for
// micromeshes with low subdivsion levels we allow packing multiple
// decoding jobs in a single subgroup.
// A single decoding job is however targeting one subdivision level
// uniformly, so this struct will be filled in such fashion that
// it bins decoding jobs based on target subdivision level.
// See `micromesh_binpack.glsl`
//
// Typically used as output of task-shader and input for mesh-shader

struct MicroBinPack 
{
  // arrays are ordered by bins of micromeshes
  // with equal packSize
  // within a bin subIDs are ordered ascending
  //
  // the arrays may contain ranges for subdiv level 7
  // which equals to being invalid and these make no
  // contributions to meshlet counts

  // pack 
  // --------------
  // exclusive prefix where each meshlet starts
  //    for packSize == 1: multiple meshlets may be created
  //                       per input micromesh
  //    otherwise: packSize many are packed per meshlet
  //
  // 15 bit : meshlet workgroup offset 
  //    for subdiv limited to 5: 
  //        worst-case is (SUBGROUP_SIZE-1) * 16, all previous threads requiring 16 meshlets
  //        SUBGROUP_SIZE == 32:  9 bits
  //        SUBGROUP_SIZE == 64: 10 bits
  //    for subdiv limited to 7: 
  //        worst-case is  (SUBGROUP_SIZE-1) * (highest resolution per base triangle/64)
  //        SUBGROUP_SIZE == 32:  13 bits  and highest res is 16 384 tris
  //        SUBGROUP_SIZE == 64:  14 bits  and highest res is 16 384 tris
  //
  // 2 bit : pack size (1,4,8,16) (actual) how many micromeshes per meshlet
  //                    0,1,2,3   (encoded)
  //
  // info
  // -------------
  // 6 bit : microID relative to baseID
  //        SUBGROUP_SIZE == 32:  5 bits
  //        SUBGROUP_SIZE == 64:  6 bits
  //
  // 4 bit : target subdiv level
  //         needs to be able to store MICRO_BIN_INVALID_SUBDIV

  uint32_t  pack_info[SUBGROUP_SIZE];
  uint32_t  baseID;
};

#define MICRO_BIN_PACK_OFFSET_MASK  ((1<<15)-1)
#define MICRO_BIN_PACK_SIZE_MASK    ((1<<2)-1)
#define MICRO_BIN_PACK_SIZE_SHIFT   15
#define MICRO_BIN_INFO_SHIFT        17
#define MICRO_BIN_INFO_ID_MASK      ((1<<6)-1)
#define MICRO_BIN_INFO_LVL_SHIFT    6
#define MICRO_BIN_INVALID_SUBDIV    15
#define MICRO_BIN_SPLIT_SUBDIV      3
