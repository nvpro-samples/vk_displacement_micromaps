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




// must ensure that lvl 0 or flat uses isUnsigned
int microdec_decodeCorrectionBits(inout MicroDecoder dec, uint lvl, uint bitPos, uint vertexType, bool isUnsigned)
{
  uint corrBits = microdec_getNumCorrBits(dec, lvl);
  uint raw      = microdata_readDataBits(dec.dataOffset, bitPos, corrBits);
  return (isUnsigned) ? int(raw) : microdec_decodePredictionCorrection(dec, vertexType, microdata_convertSigned(raw, corrBits));
}

// MICRO_MTRI_USE_MATH 
// == 1 evaluates the bird curve on the fly (much slower)
// == 0 uses precomputed table
#ifndef MICRO_MTRI_USE_MATH
#define MICRO_MTRI_USE_MATH 0
#endif

#if MICRO_MTRI_USE_MATH

/////////////////////////////////////////////////////

int decodeDescendVertex(inout MicroDecoder dec, ivec3 descendVtx, uint vertexLvl, uint subVertex, uint dbgVertex)
{
  ivec3 quadVtx  = descendVtx & (~1);
  uint vertexIdx = bird_getTripletIndex(quadVtx.x, quadVtx.y, quadVtx.z, vertexLvl) * 3;
  vertexIdx += subVertex;

  microdec_setCurrentSubdivisionLevel(dec, vertexLvl, 0);
  
  uint corrBits   = microdec_getNumCorrBits(dec, vertexLvl);
  uint bitPos     = microdec_getStartPos(dec, vertexLvl) + vertexIdx * corrBits;
  uint vertexType = microdata_getVertexType(descendVtx);
  int correction  = microdec_decodeCorrectionBits(dec, vertexLvl, bitPos, vertexType, false);

  return correction;
}
#else
int decodeDescendVertex(inout MicroDecoder dec, uint descendVtx, uint vertexLvl, uint dbgVertex)
{
  uint vertexType = bitfieldExtract(descendVtx, MICRO_MTRI_DESCEND_VERTEX_TYPE_SHIFT, MICRO_MTRI_DESCEND_VERTEX_TYPE_WIDTH);
  uint bitPos     = bitfieldExtract(descendVtx, MICRO_MTRI_DESCEND_VERTEX_DATA_SHIFT, MICRO_MTRI_DESCEND_VERTEX_DATA_WIDTH);
  //     vertexLvl  = bitfieldExtract(descendVtx, MICRO_MTRI_DESCEND_VERTEX_LVL_SHIFT,  MICRO_MTRI_DESCEND_VERTEX_LVL_WIDTH);

  microdec_setCurrentSubdivisionLevel(dec, vertexLvl, 0);
  int correction = microdec_decodeCorrectionBits(dec, vertexLvl, bitPos, vertexType, false);

  return correction;
}
#endif

int decodeDescendVertexBegin(inout MicroDecoder dec, uint descendVtx, uint dbgVertex)
{
  uint vertexType = bitfieldExtract(descendVtx, MICRO_MTRI_DESCEND_VERTEX_TYPE_SHIFT, MICRO_MTRI_DESCEND_VERTEX_TYPE_WIDTH);
  uint bitPos     = bitfieldExtract(descendVtx, MICRO_MTRI_DESCEND_VERTEX_DATA_SHIFT, MICRO_MTRI_DESCEND_VERTEX_DATA_WIDTH);
  uint vertexLvl  = bitfieldExtract(descendVtx, MICRO_MTRI_DESCEND_VERTEX_LVL_SHIFT,  MICRO_MTRI_DESCEND_VERTEX_LVL_WIDTH);

  microdec_setCurrentSubdivisionLevel(dec, vertexLvl, 0);
  int correction = microdec_decodeCorrectionBits(dec, vertexLvl, bitPos, vertexType, true);

  return correction;
}

ivec3 decodeDescend(inout MicroDecoder dec, uint blockTri, uint formatIdx, out bool isFlipped, uint dbgVertex)
{
  isFlipped = false;
  bool isFlat    = formatIdx == MICRO_FORMAT_64T_512B;
  uint triOffset = isFlat ? blockTri : 0;
  
  MicromeshMTriDescend descend = microdata_loadMicromeshDescend(MICRO_MTRI_DESCENDS_INDEX(triOffset, formatIdx));
  
  ivec3 displacements;
  
  displacements.x = decodeDescendVertexBegin(dec, uint(descend.vertices.x), dbgVertex);
  displacements.y = decodeDescendVertexBegin(dec, uint(descend.vertices.y), dbgVertex);
  displacements.z = decodeDescendVertexBegin(dec, uint(descend.vertices.z), dbgVertex);
  
#if MICRO_SUPPORTED_FORMAT_BITS == (1<<MICRO_FORMAT_64T_512B)
  // only flat
  return displacements;
#else
  // else hierarchical decoding
  
  if (isFlat) return displacements;
  
  uint levelShift = formatIdx == MICRO_FORMAT_256T_1024B ? 8 : 10;
  
  ivec3 vertexW = ivec3(1,0,0);
  ivec3 vertexU = ivec3(0,1,0);
  ivec3 vertexV = ivec3(0,0,1);
  
  triOffset = 1;

  [[unroll]]
  for (uint level = 1; level < 6; level++)
  {
    if (formatIdx != MICRO_FORMAT_1024T_1024B && level == 5) continue;
  
    //     
    //               V
    //              / \  
    //             / 3 \  
    //           c0_____c1
    //           / \ 1 / \ 
    //          / 0 \ / 2 \ 
    //         W ___c2 ___ U
    //         

    const uint W    = 0;
    const uint U    = 1;
    const uint V    = 2;
    
    const uint VW   = 0;
    const uint UV   = 1;
    const uint UW   = 2;    
    
    ivec3 splits;
    
    // find parent triangle of blockTri in other levels:
    // which triangle blockTri is a child of within this level
    uint levelTri = blockTri >> levelShift;
    // which triangle blockTri is a child of or equal to within the next level
    // allows us to determine which of the 4 children it is of current level
    uint splitTri = blockTri >> (levelShift - 2);
    uint splitIdx = splitTri & 3;
    
  #if MICRO_MTRI_USE_MATH
    vertexW <<= 1;
    vertexU <<= 1;
    vertexV <<= 1;
    
    ivec3 vertexVW = (vertexV + vertexW) / 2;
    ivec3 vertexUW = (vertexU + vertexW) / 2;
    ivec3 vertexUV = (vertexU + vertexV) / 2;
    
    splits[VW] = decodeDescendVertex(dec, vertexVW, level, isFlipped ? 1 : 0, dbgVertex);
    splits[UV] = decodeDescendVertex(dec, vertexUV, level, isFlipped ? 0 : 1, dbgVertex);
    splits[UW] = decodeDescendVertex(dec, vertexUW, level, 2,                 dbgVertex);
  #else
    descend    = microdata_loadMicromeshDescend(MICRO_MTRI_DESCENDS_INDEX(triOffset + levelTri, formatIdx));
    
    splits[VW] = decodeDescendVertex(dec, uint(descend.vertices.x), level, dbgVertex);
    splits[UV] = decodeDescendVertex(dec, uint(descend.vertices.y), level, dbgVertex);
    splits[UW] = decodeDescendVertex(dec, uint(descend.vertices.z), level, dbgVertex);
  #endif
    
    ivec3 previous = displacements;
    
    splits[VW] = microdec_compute(dec, previous[V], previous[W], splits[VW]);
    splits[UV] = microdec_compute(dec, previous[U], previous[V], splits[UV]);
    splits[UW] = microdec_compute(dec, previous[U], previous[W], splits[UW]);
    
    switch(splitIdx) {
    case 0:
      displacements[W] = displacements[W];
      displacements[U] = splits[UW];
      displacements[V] = splits[VW];
    #if MICRO_MTRI_USE_MATH
      vertexU = vertexUW;
      vertexV = vertexVW;
    #endif
      break;
    case 1:
      isFlipped = !isFlipped;
      displacements[W] = splits[VW];
      displacements[U] = splits[UV];
      displacements[V] = splits[UW];
    #if MICRO_MTRI_USE_MATH
      vertexW = vertexVW;
      vertexU = vertexUV;
      vertexV = vertexUW;
    #endif
      break;
    case 2:
      displacements[W] = splits[UW];
      displacements[U] = displacements[U];
      displacements[V] = splits[UV];
    #if MICRO_MTRI_USE_MATH
      vertexW = vertexUW;
      vertexV = vertexUV;
    #endif
      break;
    case 3:
      isFlipped = !isFlipped;
      displacements[W] = splits[UV];
      displacements[U] = splits[VW];
      displacements[V] = displacements[V];
    #if MICRO_MTRI_USE_MATH
      vertexW = vertexUV;
      vertexU = vertexVW;
    #endif
      break;
    }
    
    // next iteration
    // increase resolution granularity
    levelShift -= 2;
  #if !MICRO_MTRI_USE_MATH
    // shift by number of split-triangles in this level
    triOffset  += (1 << ((level-1) * 2));
  #endif
  }
  
  return displacements;
#endif
}

int smicrodec_getVertexDisplacement(inout SubgroupMicromeshDecoder sdec, inout MicromeshMTriVertex localVtx, uint outIndex)
{
  uint  baseMtri     = microvertex_getMTri(localVtx);    
  uint  corner       = microvertex_getCorner(localVtx);

  uint  formatIdx    = microdec_getFormatIdx(sdec.dec);
  uint  formatSubdiv = microdec_getFormatSubdiv(sdec.dec);
  uint  formatSize   = microdec_getBitSize(sdec.dec) / 32;
  uint  block        = baseMtri >> (formatSubdiv * 2);
  uint  blockTri     = baseMtri & ((1u << (formatSubdiv * 2))-1);

  microdec_setDataOffset(sdec.dec, sdec.firstData + formatSize * block);

  bool blockFlipped = ((bitCount(bird_extractEvenBits(block)) % 2) == 1);
  bool isFlipped;
  ivec3 displacements = decodeDescend(sdec.dec, blockTri, formatIdx, isFlipped, outIndex);

  if (isFlipped BOOL_XOR blockFlipped)
  {
    displacements.xy = displacements.yx;
  }
  
  return displacements[corner];
}


