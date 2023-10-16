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
 
// MicroBinPackFlatSplit without bin packing!!
// -------------------------------------------
//
// Do NOT USE, only here to show perf benefits of binpacking.
//
// Variant of the `micromesh_binpack_flat.glsl` which uses the "flat" encoding
// only for >= MICRO_BIN_SPLIT_SUBDIV and otherwise passes smaller triangles to MICROBINPACK_PROCESS
// one at a time per subgroup. No binning is performed.
//
// passing the MicroBinPack as in/out struct
// breaks things because of shading language semantics (copy in/out struct),
// we need to reference the storage directly (could be task input/output or shared memory or global memory)
//
// user must specify the variable we modify by macro
// All require:
//
//  MICROBINPACK_USE_MESHLETCOUNT            0/1
//    1: the numbers of meshlets is provided explictly
//    0: the numbers of meshlets is derived from targetSubdiv
//
// MicroBinPackFlatSplit_subgroupPack requires
// - MICROBINPACK_OUT and MICROBINPACK_OUT_ATOM and MICROBINPACK_OUT_MAX and MICROBINPACK_PROCESS

// these functions are illustrated in the rasterization pdf document

#if defined (MICROBINPACK_OUT) && defined(MICROBINPACK_OUT_ATOM) && defined(MICROBINPACK_OUT_MAX) && defined(MICROBINPACK_PROCESS)

void MicroBinPackFlatSplit_subgroupPack( 
                                uint baseID, 
                                uint relativeID, 
                                uint targetSubdiv,
                                uint targetMeshletCount,
                                bool valid,
                                uint instanceID
                                )
{
  uint laneID          = gl_SubgroupInvocationID;
  
  if (subgroupAll(!valid)){
    return;
  }
  
  // NO BINPACKING
  // ----------------
  
  // bin packs based on targetlevel
  // level 3,4,5 use 1 or more meshlets and can go in same bin
  uint  packSubdiv  = min(MICRO_BIN_SPLIT_SUBDIV, targetSubdiv);
  
  bool isCompact  = packSubdiv < MICRO_BIN_SPLIT_SUBDIV;
  bool isFlat     = !isCompact && valid;
  
  // # meshlet outputs
  // ------------
  // figure out how many meshlets we output to flat

  uint meshletCount = 0;
  if (isFlat) {
#if MICROBINPACK_USE_MESHLETCOUNT
    meshletCount      = targetMeshletCount;
#else
    // packSubdiv == 3, needs 1 or more meshlets due to splitting
    uint splitSubdiv  = uint(max(int(targetSubdiv) - MICRO_BIN_SPLIT_SUBDIV, 0));
    meshletCount      = (1 << (splitSubdiv * 2));
#endif
  }

  // enqueue large into flat output
  uint offsetFlat = subgroupInclusiveAdd(meshletCount);
  
  uint startBin   = 0;
  if (laneID == SUBGROUP_SIZE-1)
  {
    startBin = atomicAdd(MICROBINPACK_OUT_ATOM, offsetFlat);
  #if USE_STATS
    atomicAdd(stats.meshlets, offsetFlat);
  #endif
  }
  
  // was inclusiveAdd, need exclusive
  offsetFlat -= meshletCount;
  startBin = subgroupShuffle(startBin, SUBGROUP_SIZE-1);
  
  if (isFlat && startBin < MICROBINPACK_OUT_MAX)
  {
    MicroBinPackFlat flatPack;
    flatPack.instanceID = instanceID;
    flatPack.pack       = targetSubdiv | ((baseID + relativeID) << MICRO_BIN_FLAT_PACK_BASE_SHIFT);\
    
    for (uint32_t i = 0; i < meshletCount; i++)
    {
      flatPack.partOrMask = i;
      MICROBINPACK_OUT[offsetFlat + i + startBin] = flatPack;
    }
  }
  
  // directly rasterize all compact packs
  bool doProcess    = (isCompact && valid);
  uvec4 processVote = subgroupBallot(doProcess);
  uint processCount = subgroupBallotBitCount(processVote);
  
  if (processCount == 0){
    return;
  }
  
  if (laneID == 0)
  {
  #if USE_STATS
    atomicAdd(stats.meshlets, processCount);
  #endif
  }
  
  uint processPrefix = subgroupBallotExclusiveBitCount(processVote);
  for (uint p = 0; p < processCount; p++)
  {
    uvec4 batchVote = subgroupBallot(p == processPrefix);
    uint  readID    = subgroupBallotFindMSB(batchVote);
    
    uint sub_microID      = subgroupShuffle(baseID + relativeID, readID);
    uint sub_targetSubdiv = subgroupShuffle(targetSubdiv, readID);
    
    MicroDecoderConfig cfg;
    
    cfg.microID      = sub_microID;
    cfg.partID       = 0; // < split subdiv doesn't have parts
    
    cfg.targetSubdiv = sub_targetSubdiv;
    cfg.partSubdiv   = sub_targetSubdiv;
      
    cfg.packID       = 0;
    cfg.packThreads  = SUBGROUP_SIZE;
    cfg.packThreadID = laneID;
    
    cfg.valid        = true;
    
    MICROBINPACK_PROCESS(cfg);
  }
}
#endif
