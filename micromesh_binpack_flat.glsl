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
 
// MicroBinPackFlat
// ----------------
// Variant of the regular `micromesh_binpack.glsl` which uses the "flat" encoding
// to store binpacks individually in global memory.
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
// MicroBinPackFlat_subgroupPack requires
// - MICROBINPACK_OUT and MICROBINPACK_OUT_ATOM and MICROBINPACK_OUT_MAX
// MicroBinPackFlat_subgroupUnpack requires:
// - MICROBINPACK_IN

#if SUBGROUP_SIZE != 32 && (defined (MICROBINPACK_OUT) || (defined (MICROBINPACK_IN) && !defined(MICROBINPACK_NO_SUBSPLIT)))
// partOrMask is uint32
#error "SUBGROUP_SIZE != 32 currently not supported"
#endif

// these functions are illustrated in the rasterization pdf document

#if defined (MICROBINPACK_OUT) && defined(MICROBINPACK_OUT_ATOM) && defined(MICROBINPACK_OUT_MAX)

shared uint s_binFlat[SUBGROUP_SIZE * MICRO_FLAT_TASK_GROUPS];

#if MICRO_FLAT_TASK_GROUPS > 1
uint c_binFlat_offset = gl_LocalInvocationID.y * SUBGROUP_SIZE;
#else
uint c_binFlat_offset = 0;
#endif

void MicroBinPackFlat_subgroupPack( 
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
  
  // offsets within 
  uint packOffsetM = subgroupPartitionedExclusiveAddNV(meshletCount, packVote);
  uint binTotalM   = subgroupShuffle(packOffsetM + meshletCount, packLast);
  uint binOffsetM  = subgroupExclusiveAdd(isPackFirst ? binTotalM : 0);
  binOffsetM       = subgroupShuffle(binOffsetM, packFirst);
  
  
  // We need to find the basetriangles (baseID + relativeID) later through linear indexing.
  // This array stores the IDs within bins tightly.
  // Each bin starts at binStartIdx and then all
  // relativeIDs (base-triangles within bin) are stored linearly.
  // There is no further ordering guarantees.
  
  s_binFlat[outID + c_binFlat_offset] = relativeID;
  
  uint startBin = 0;
  if (outID == SUBGROUP_SIZE-1)
  {
    uint outCount   = binOffsetM + binTotalM;
    startBin        = atomicAdd(MICROBINPACK_OUT_ATOM, outCount);
    
  #if USE_STATS
    atomicAdd(stats.meshlets, outCount);
  #endif
  }
  
  startBin = subgroupMax(startBin);
  
  barrier();
  memoryBarrierShared();
  
  if (startBin >= MICROBINPACK_OUT_MAX) return;
  
  // not the greatest looping logic for now:
  // we go wide for compact bins
  // but individual loop for non-compact
  // individual loops are <= 16 
  
  MicroBinPackFlat flatPack;
  
  
  bool isCompact = packSubdiv < MICRO_BIN_SPLIT_SUBDIV;
  
  // for non-compact bins, we emit every basetriangle's meshlets individually.
  // for compacted bins, we do the emit that fits in a single subgroup every packSize-many
  if (valid && (!isCompact || (((packPrefix) % packSize) == 0)))
  {    
    flatPack.instanceID = instanceID;
    flatPack.pack       = targetSubdiv | ((baseID + (isCompact ? 0 : relativeID)) << MICRO_BIN_FLAT_PACK_BASE_SHIFT);

    uint partOrMask = 0;
    if (isCompact) {
      uint packIndex = packPrefix / packSize;
      // binOffsetM needs to be adjusted for the sub-bin we are in
      binOffsetM += packIndex;
    
      uint lastID  = outID + min(packSize, packCount - packPrefix) - 1;
      uint nextBit = s_binFlat[lastID + c_binFlat_offset] + 1;
      
      // mask should have all the bits of the current sub-bin
      
      uint prevMask = ((1 << (relativeID)) -1);
      uint nextMask = nextBit == 32 ? 0xFFFFFFFFu : ((1 << nextBit)-1);
      
      partOrMask = packVote.x & (prevMask ^ nextMask);
      
      //stats.debugB[laneID] = prevMask;
      //stats.debugC[laneID] = nextMask;
    }
    else {
      binOffsetM += packOffsetM;
    }
    uint localCount = isCompact ? 1 : meshletCount;

    // compact will output only a single item
    // non compact will output multiple and iterate partOrMask
    for (uint m = 0; m < localCount; m++) {
      flatPack.partOrMask = partOrMask;
      partOrMask++;
    
      MICROBINPACK_OUT[(m + binOffsetM + startBin)] = flatPack;
    }
  }
}
#endif

#ifdef MICROBINPACK_IN
MicroDecoderConfig MicroBinPackFlat_subgroupUnpack(MicroBinPackFlat flatPack)
{
  // a micromesh can be either packed with others in the same meshlet
  // or use multiple meshlets (aka parts)
  //
  // find original micromesh we are from, and packing information
  uint laneID       = gl_SubgroupInvocationID;
  
  uint targetSubdiv = flatPack.pack & MICRO_BIN_FLAT_PACK_LVL_MASK;
  uint microID      = flatPack.pack >> MICRO_BIN_FLAT_PACK_BASE_SHIFT;
  uint partID;
  
  uint packID;
  uint packThreads;
  uint packThreadID = laneID;
  bool valid;
  
#ifndef MICROBINPACK_NO_SUBSPLIT
  // compact
  if (targetSubdiv < MICRO_BIN_SPLIT_SUBDIV)
  {
    packThreads     = 2 << targetSubdiv;
    packID          = laneID >> (targetSubdiv+1); //  == laneID / packThreads
    packThreadID    = laneID & (packThreads-1);
    
    partID     = 0;
    
    uint mask  = flatPack.partOrMask;

    uint bit   = 0;
    uint count = 0;
    // search idx n'th bit in mask
    uint idx   = packID + 1;
  
    for (int s = 4; s >= 0; s--){
      uint i     = 1 << s;
      uint count = bitCount(mask & ((1<<i)-1));
      bool upper = idx > count;
      bit   = upper ? bit + i : bit;
      mask  = upper ? mask >> i : mask;
      idx   = upper ? idx - count : idx;
    }
    
    valid = mask != 0 && idx == 1;
    microID += valid ? bit : 0;
  }
  else
#endif
  {
    packThreads     = SUBGROUP_SIZE;
    packID          = 0;
    packThreadID    = laneID;
    
    partID          = flatPack.partOrMask;
    
    valid = true;
  }
  
  MicroDecoderConfig cfg;
  
  cfg.microID      = microID;
  cfg.partID       = partID;
  
  cfg.targetSubdiv = targetSubdiv;
  cfg.partSubdiv   = min(MICRO_BIN_SPLIT_SUBDIV, targetSubdiv);
    
  cfg.packID       = packID;
  cfg.packThreads  = packThreads;
  cfg.packThreadID = packThreadID;
  
  cfg.valid        = valid;
  
  return cfg;
}
#endif