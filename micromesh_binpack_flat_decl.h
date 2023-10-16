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
// MicroBinPackFlat
//
// Encodes the task performed by one decoding warp.
// One flat packing is either up to 16 packed decoding jobs
// or a single. If single (target subdiv >= 3) then it needs
// to provide which partition within the primitive it is
// decoding.

struct MicroBinPackFlat
{
  uint32_t  pack;
  //    4: target subdiv level 
  // if target subdiv >= 3
  //   28: final ID
  // else (target subdiv < 3)
  //   28: base ID
  
  uint32_t  partOrMask;
  // if target subdiv >= 3
  //   32: partID
  // else (target subdiv < 3)
  //   32: relative ID bitmask
  //       bit set for each relative ID active in packing
  //       only works for SUBGROUP_SIZE == 32, otherwise need 64 bit here and adjust a few more
  
  uint32_t instanceID;
};

#define MICRO_BIN_FLAT_PACK_LVL_MASK          ((1<<4)-1)
#define MICRO_BIN_FLAT_PACK_BASE_SHIFT        4
#define MICRO_BIN_SPLIT_SUBDIV                3
#define MICRO_BIN_INVALID_SUBDIV              15

#if MAX_BASE_SUBDIV != 5
  #error "bits "
#endif