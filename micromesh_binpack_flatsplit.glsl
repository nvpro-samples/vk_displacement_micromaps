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
 
// MicroBinPackFlatSplit
// ----------------
// Variant of the `micromesh_binpack_flat.glsl` which uses the "flat" encoding
// only for >= MICRO_BIN_SPLIT_SUBDIV and otherwise packs them to smaller bins
// and sends each bin to MICROBINPACK_PROCESS one at a time per subgroup.
//
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

shared uint s_binCompact[SUBGROUP_SIZE * MICRO_FLAT_SPLIT_TASK_GROUPS];

#if MICRO_FLAT_SPLIT_TASK_GROUPS > 1
uint c_binCompact_offset = gl_LocalInvocationID.y * SUBGROUP_SIZE;
#else
uint c_binCompact_offset = 0;
#endif

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

  // COMMON BINPACK LOGIC
  // --------------------
  
  // bin packs based on targetlevel
  // level 3,4,5 use 1 or more meshlets and can go in same bin
  uint  packSubdiv  = min(MICRO_BIN_SPLIT_SUBDIV, targetSubdiv);
  uint  packSizeC   = MICRO_BIN_SPLIT_SUBDIV - packSubdiv;
  
  // compute partition masks for each different bin
  // vote sets a bit for all threads using the same packSubdiv
  uvec4 packVote    = subgroupPartitionNV(valid ? packSubdiv : MICRO_BIN_INVALID_SUBDIV);

  uint  packFirst   = subgroupBallotFindLSB(packVote);
  uint  packLast    = subgroupBallotFindMSB(packVote);
  uint  packCount   = subgroupBallotBitCount(packVote);
  uint  packPrefix  = subgroupBallotExclusiveBitCount(packVote);
  
  // bins need to operate globally across entire subgroup
  // but let only lead thread participate meaningful  
  bool  isPackFirst = packFirst == laneID;
  
  // output index computation
  // ------------------------
  // all firsts contribute to finding an offset in output arrays
  // all outputs are binned tightly
  // use shuffle to redistribute value among pack
  uint binStartIdx  = subgroupExclusiveAdd(isPackFirst ? packCount : 0);
  binStartIdx       = subgroupShuffle(binStartIdx, packFirst);
  
  
  // fill outputs according to bins
  //
  // there is no clear ordering of bins as such, it depends on 
  // thread ordering, but it is guaranteed that each bin
  // is packed tightly in relativeID ordering.
  uint outID        = binStartIdx + packPrefix;
  
  // # meshlet outputs
  // ------------
  // figure out how many meshlets we need to spawn in total
  // as well as where each bin's meshlets start
  // ensure invalid contributions will later be safely skipped
  // over

  uint meshletCount = 0;
  uint packSize     = 1;
  if (packSubdiv < MICRO_BIN_SPLIT_SUBDIV) {
    uint packSubCount = packCount;
    packSize          = (1 << packSizeC) * (packSizeC == 0 ? 1 : 2);
    meshletCount      = (packSubCount + packSize - 1) / packSize;
    meshletCount      = isPackFirst && valid ? meshletCount : 0;
  }
  else {
#if MICROBINPACK_USE_MESHLETCOUNT
    meshletCount      = valid ? targetMeshletCount : 0;
#else
    // packSubdiv == 3, needs 1 or more meshlets due to splitting
    uint splitSubdiv  = uint(max(int(targetSubdiv) - MICRO_BIN_SPLIT_SUBDIV, 0));
    meshletCount      = valid ? (1 << (splitSubdiv * 2)) : 0;
#endif
  }

  // -------------------------
  

  // enqueue large into flat output
  bool isCompact  = packSubdiv < MICRO_BIN_SPLIT_SUBDIV;
  bool isFlat     = !isCompact && valid;
  uint countFlat  = isFlat ? meshletCount : 0;
  uint offsetFlat = subgroupInclusiveAdd(countFlat);
  
  uint startBin = 0;
  if (laneID == SUBGROUP_SIZE-1)
  {
    startBin = atomicAdd(MICROBINPACK_OUT_ATOM, offsetFlat);
  #if USE_STATS
    atomicAdd(stats.meshlets, offsetFlat);
  #endif
  }
  // was inclusiveAdd, need exclusive
  offsetFlat -= countFlat;
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
  // for compacted, we start a decode that fits in a single subgroup every packSize-many
  bool doProcess    = isCompact && valid && (((packPrefix) % packSize) == 0);
  uvec4 processVote = subgroupBallot(doProcess);
  uint processCount = subgroupBallotBitCount(processVote);
  
  if (processCount == 0){
    return;
  }
  
  // We need to find the basetriangles (baseID + relativeID) later through linear indexing.
  // This array stores the IDs within bins tightly.
  // Each bin starts at binStartIdx and then all
  // basetriangle IDs within that bin are stored linearly.
  // There is no further ordering guarantees.
  s_binCompact[outID + c_binCompact_offset] = baseID + relativeID;
    
  memoryBarrierShared();
  barrier();
  
  if (laneID == 0)
  {
  #if USE_STATS
    atomicAdd(stats.meshlets, processCount);
  #endif
  }
  
  uint processPrefix = subgroupBallotExclusiveBitCount(processVote);
  for (uint p = 0; p < processCount; p++)
  {
    // This loop iterates over all threads whose data needs to be
    // processed.
    // We find the n-th thread index using the algorithm below.
    
    // example:
    // bit            0 1 2 3 4 5 6 7
    // 
    // processVote:   0 0 1 1 0 1 0 1
    // processCount:  4
    // processPrefix: 0 0 0 1 2 2 3 3
    //
    // loop 
    // p == 0:
    //    batchVote:  1 1 1 0 0 0 0 0
    //    readIdx:        2       
    // p == 1:
    //    batchVote:  0 0 0 1 0 0 0 0
    //    readIdx:          3   
    // p == 2:
    //    batchVote:  0 0 0 0 1 1 0 0
    //    readIdx:              5
    // p == 3:
    //    batchVote:  0 0 0 0 0 0 1 1
    //    readIdx:                  7
    
    uvec4 batchVote = subgroupBallot(p == processPrefix);
    uint  readID   = subgroupBallotFindMSB(batchVote);
    
    // Processing is done subgroup-wide, so send the thread's data
    // to the entire subgroup, prior kicking off the processing
    // callback.
    
    uint sub_binStartIdx  = subgroupShuffle(binStartIdx, readID);
    uint sub_targetSubdiv = subgroupShuffle(targetSubdiv,readID);
    uint sub_packPrefix   = subgroupShuffle(packPrefix,  readID);
    uint sub_packCount    = subgroupShuffle(packCount,   readID);
    
    uint packThreads     = 2 << sub_targetSubdiv;
    uint packID          = laneID >> (sub_targetSubdiv+1); //  == laneID / packThreads
    uint packThreadID    = laneID & (packThreads-1);
    
    valid = sub_packPrefix + packID < sub_packCount;
    
    MicroDecoderConfig cfg;
    
    cfg.microID      = valid ? s_binCompact[sub_binStartIdx + sub_packPrefix + packID + c_binCompact_offset] : 0;
    cfg.partID       = 0; // < split subdiv doesn't have parts
    
    cfg.targetSubdiv = sub_targetSubdiv;
    cfg.partSubdiv   = sub_targetSubdiv;
      
    cfg.packID       = packID;
    cfg.packThreads  = packThreads;
    cfg.packThreadID = packThreadID;
    
    cfg.valid        = valid;
    
    MICROBINPACK_PROCESS(cfg);
  }
}
#endif
