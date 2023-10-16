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

// MicroBinPack
// ------------
// This bin packer stores a single array output struct.
//
// passing the MicroBinPack as in/out struct
// breaks things because of shading language semantics (copy in/out struct),
// we need to reference the storage directly (could be task input/output or shared memory)
//
// user must specify the variable we modify by macro
// All require:
//
//  MICROBINPACK_USE_MESHLETCOUNT            0/1
//    1: the numbers of meshlets is provided explictly
//    0: the numbers of meshlets is derived from targetSubdiv
//
// MicroBinPack_subgroupPack requires
// - MICROBINPACK_OUT and MICROBINPACK_OUT_COUNT
// MicroBinPack_subgroupUnpack requires:
// - MICROBINPACK_IN

// these functions are illustrated in the rasterization pdf document

#if defined (MICROBINPACK_OUT) && defined(MICROBINPACK_OUT_COUNT)
void MicroBinPack_subgroupPack( uint baseID, 
                                uint relativeID, 
                                uint targetSubdiv,
                                uint targetMeshletCount,
                                bool valid
                                )
{
  uint laneID          = gl_SubgroupInvocationID;
  
  if (subgroupAll(!valid)){
    MICROBINPACK_OUT_COUNT = 0;
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

  MICROBINPACK_OUT.pack_info[outID] = ((relativeID | (targetSubdiv << MICRO_BIN_INFO_LVL_SHIFT)) << MICRO_BIN_INFO_SHIFT) | 
                                      ((packOffsetM + binOffsetM) | (packSizeC << MICRO_BIN_PACK_SIZE_SHIFT));
  
  if (outID == SUBGROUP_SIZE-1)
  {
    uint count              = binOffsetM + binTotalM;
    MICROBINPACK_OUT_COUNT  = count;
    MICROBINPACK_OUT.baseID = baseID;

  #if USE_STATS
    atomicAdd(stats.meshlets, count);
  #endif
  }
}
#endif

#ifdef MICROBINPACK_IN
MicroDecoderConfig MicroBinPack_subgroupUnpack(uint wgroupID)
{
  // a micromesh can be either packed with others in the same meshlet
  // or use multiple meshlets (aka parts)
  //
  // find original micromesh we are from, and packing information
  uint laneID      = gl_SubgroupInvocationID;
  
  // compare against offsets using warp
  uint  offsetM    = uint(MICROBINPACK_IN.pack_info[laneID]) & MICRO_BIN_PACK_OFFSET_MASK;
  uvec4 vote       = subgroupBallot(wgroupID >= offsetM);
  uint inID        = subgroupBallotFindMSB(vote);
  uint partID      = wgroupID - subgroupShuffle(offsetM, inID);
  
  // example for input arrays
  // 9 micromeshes at different target subdiv levels
  // and packings (subdiv 7 tags invalid)
  
  // output is 
  //  2 meshlets subdiv 2
  //  1 meshlet  subdiv 3
  // 16 meshlet parts subdiv 3 (from subdiv 5)
  
  // idx:       0  1  2  3  4  5  6  7  8
  //
  // packSize:  4  4  4  4  4  4  1  1  1
  // subdiv  :  2  2  2  2  2  2  7  3  5
  // offsetM :  0  2  2  2  2  2  2  2  3
  //
  // wgroupID:  0  1  2  3 ... 18
  // inID:      0  0  7  8 ... 8
  //
  //
  //  example wgroupID == 1 
  //    inID is 0 after the ballotFindMSB above
  //    partID is 1
  //
  //    because packSize is != 1,
  //    we rebase inID to 4 first (second meshlet for packSize 4: partID * 4)
  //    and then, the 1st to 4th threads use inID 4, the 5th to 8th thread use inID 5
  //    further threads will be marked invalid, as their subdiv level mismatches
  //    the first in the warp (inID 4).
  //
  //  example wgroupID == 5
  //
  //    inID is 8 after the ballotFindMSB 
  //      (it implictly skips over the subdiv 7 invalid entry)
  //    partID is 2
  //  
  //    we can leave that as is for packSize == 1
  
  uint pack = MICROBINPACK_IN.pack_info[inID];
  
  // get pack configuration
  // how many micromeshes per meshlet
  uint packSizeC  = (pack >> MICRO_BIN_PACK_SIZE_SHIFT) & MICRO_BIN_PACK_SIZE_MASK;
  uint packSize   = (1 << packSizeC) * (packSizeC == 0 ? 1 : 2);
    // how many threads each micromesh gets
  uint packThreads  = SUBGROUP_SIZE / packSize;
  // which pack we are
  uint packID       = laneID  / packThreads;
  uint packThreadID = laneID  & (packThreads-1);
  
  // need to change input index when we pack more than one
  // into warp
  if (packSize != 1) {
    // find bin's start index in the input array 
    uint packStart  = inID; // (pack >> MICRO_BIN_PACK_START_SHIFT) & MICRO_BIN_PACK_START_MASK;
    
    // within bin, compute new offset based on how many meshlets already were 
    // done for this bin (partID many), and then the local packID we are
    inID             = min( packStart + (partID * packSize) + packID, SUBGROUP_SIZE-1);
  }

  uint info         = MICROBINPACK_IN.pack_info[inID] >> MICRO_BIN_INFO_SHIFT;
  uint microID      = MICROBINPACK_IN.baseID + (info & MICRO_BIN_INFO_ID_MASK);
  uint targetSubdiv = info >> (MICRO_BIN_INFO_LVL_SHIFT);

  // the info array packs tightly, so our packed meshlet might pull in values
  // from the next bin
  uint subdivUni    = subgroupShuffle(targetSubdiv, 0);
  bool valid        = subdivUni == targetSubdiv;
  targetSubdiv      = subdivUni;
  
  MicroDecoderConfig cfg;
  // these don't have parts
  if (targetSubdiv <= MICRO_BIN_SPLIT_SUBDIV) partID = 0;
  
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