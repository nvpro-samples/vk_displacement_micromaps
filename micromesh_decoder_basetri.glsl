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

//////////////////////////////////////////////////////////////
// BaseTriangle Mip Decoder
// MICRO_DECODER == MICRO_DECODER_BASETRI_MIP_SHUFFLE
//
// This decoder makes use of shuffle to gather the displacement values of the two
// vertices whose edge was split to create a new vertex. That new
// vertex computes its displacement using a signed correction value that 
// is applied on the average displacement of the edge vertices.
//
// The decode process iteratively adds more vertices that are computed
// for each subdivision level, relying on the results of the
// previous level.
//
// A base triangle can contain multiple compressed blocks. To aid decoding
// acorss the entire base triangle we store uncompressed a few mip levels over 
// the entire base triangle.
// This yields better performance, especially when rendering at low lod, as we
// don't need to touch multiple blocks.

struct SubgroupMicromeshDecoder
{
  MicroDecoder        dec;
  MicroDecoderConfig  cfg;
  
  MicromeshBaseTri     microBaseTri;
  
  uint baseID;
  
  // offset into threads for shuffle access
  // influenced by packID when multiple decoders live in same
  // workgroup
  uint                threadOffset;
  
  // index buffer offset based on lod and topology
  uint                primOffset;
  
  // offset into vertices array
  // influenced by partID state (if micromesh split into multiple meshlets)
  uint                vertexOffset;
  
  // from when vertices are updated in the current decode iteration
  // previous vertex thtreads required active for shuffle access to their results
  uint                decodeVertStart;
  
  // per-thread micro vertex info used during decoding
  // vtx info is accurate for first 2 levels, 3rd level is partial, and needs extra warp iteration
  // to load vertices 32...44
  MicromeshBTriVertex  localVtx;
  // encoded correction value (signed delta or flat)
  // in flat case this matches displacement value
  int                  localCorrection;
  
  MicromeshBTriDescend localDescend;
  
  // final displacement value, used to apply delta to, fetched via shuffle
  int                  displacement;
};


#include "micromesh_decoder_api.glsl"

// MicromeshBTriVertex
ivec2 microvertex_getUV(MicromeshBTriVertex vtx) {
  return ivec2(vtx.uv.x, vtx.uv.y);
}
uvec2 microvertex_getAB(MicromeshBTriVertex vtx) {
  return ivec2(vtx.parents.x, vtx.parents.y);
}
uint microvertex_getCorrMask(MicromeshBTriVertex vtx) {
  return bitfieldExtract(vtx.packed, MICRO_BTRI_VTX_CORRMASK_SHIFT, MICRO_BTRI_VTX_CORRMASK_WIDTH);
}
uint microvertex_getCorrPos(MicromeshBTriVertex vtx) {
  return bitfieldExtract(vtx.packed, MICRO_BTRI_VTX_CORRPOS_SHIFT, MICRO_BTRI_VTX_CORRPOS_WIDTH);
}
uint microvertex_getBitNum(MicromeshBTriVertex vtx) {
  return bitfieldExtract(vtx.packed, MICRO_BTRI_VTX_BITNUM_SHIFT, MICRO_BTRI_VTX_BITNUM_WIDTH);
}
uint microvertex_getBitPos(MicromeshBTriVertex vtx) {
  return vtx.packed >> MICRO_BTRI_VTX_BITPOS_SHIFT;
}
bool microvertex_isMip(MicromeshBTriVertex vtx) {
  return (vtx.packed & MICRO_BTRI_VTX_MIP) != 0;
}
bool microvertex_isUnsigned(MicromeshBTriVertex vtx) {
  return (vtx.packed & MICRO_BTRI_VTX_UNSIGNED) != 0;
}

int microdec_decodeCorrectionBitsBVtx(inout MicroDecoder dec, MicromeshBTriVertex vtx, uint vert)
{
  uint corrMask   = microvertex_getCorrMask(vtx);
  uint corrPos    = microvertex_getCorrPos(vtx);
  uint bitNum     = microvertex_getBitNum(vtx);
  uint bitPos     = microvertex_getBitPos(vtx);
  bool isMip      = microvertex_isMip(vtx);
  bool isUnsigned = microvertex_isUnsigned(vtx);
  
  uint raw        = isMip ?
                     microdata_readMipBits (dec.mipOffset,  bitPos, bitNum) :
                     microdata_readDataBits(dec.dataOffset, bitPos, bitNum);
  
  // all unsigned is marked as lvl 0
  if (isUnsigned)
  {
    return int(raw);
  }
  else {
    // figure out which block
    uint  blockIdx = bitPos >> microdec_getBitShift(dec);
    // for now all compressed blocks are 1024
    uint  blockU2s  = true ? (1024/64) : (512/64);
    // load last 64 bit of block, it contains correction shfit values
    uvec2 blockEnd2 = microdata_loadDistance2((dec.dataOffset/2) + ((1 + blockIdx) * blockU2s) - 1);
    uint64_t blockShifts = packUint2x32(blockEnd2);
    
    uint shift = uint(blockShifts >> corrPos) & corrMask;
    int corr = microdata_convertSigned(raw, bitNum);
    corr <<= shift;
    return corr;
  }
}

int smicrodec_subgroupGetVertexDisplacement(inout SubgroupMicromeshDecoder sdec, 
                                            uint  vert, 
                                            uvec2 abIdx,
                                            int   correction)
{
  // safe to use shuffle for accessing first vertices
  int  a      = subgroupShuffle(sdec.displacement, abIdx.x + sdec.threadOffset);
  int  b      = subgroupShuffle(sdec.displacement, abIdx.y + sdec.threadOffset);

  // if this vertex was already decoded (< decodeVertStart) then use existing displacement
  // otherwise compute new displacement value using averages and correction.
  
  int disp = vert < sdec.decodeVertStart ? sdec.displacement : 
              (microdec_compute(sdec.dec, a, b, correction));
  
  return disp;
}

//////////////////////////////////////
// public api

// the principle operations of the decoding process is illustrated in the rasterization pdf document

uint smicrodec_getThreadCount(uint partSubdiv)
{
#if SUBGROUP_SIZE == 32
  return partSubdiv == 3 ? 32u : (1u << (1+partSubdiv));
#elif SUBGROUP_SIZE == 64
  return partSubdiv == 3 ? 64u : (2u << (1+partSubdiv));
#else
  #error "unspported SUBGROUP_SIZE"
#endif
}

void smicrodec_subgroupInit(inout SubgroupMicromeshDecoder sdec,
                            MicroDecoderConfig             cfg,
                            MicromeshBaseTri               microBaseTri,
                            uint                           firstMicro,
                            uint                           firstData,
                            uint                           firstMipData)
{
  uint packThreadID  = cfg.packThreadID;
  uint packThreads   = cfg.packThreads;
  uint packID        = cfg.packID;
  uint microID       = cfg.microID;
  uint partID        = cfg.partID;
  uint subdivTarget  = cfg.targetSubdiv;
  uint formatIdx     = micromesh_getFormat(microBaseTri);

  sdec.microBaseTri  = microBaseTri;
  sdec.baseID        = microID;
  
  sdec.cfg           = cfg;

  microdec_init(sdec.dec, formatIdx, 
    firstData    + micromesh_getDataOffset(microBaseTri),
    firstMipData + micromesh_getMipOffset(microBaseTri));

  uint subdivBase         = micromesh_getBaseSubdiv(microBaseTri);
  bool isFlat             = formatIdx == MICRO_FORMAT_64T_512B || subdivTarget <= MICRO_MIP_SUBDIV;
  
  // offset into meshlet configurations
  // influenced by partID state (if micromesh split into multiple meshlets)
  // there are 3 partMicro configrations 
  // 1 meshlet  for level 3 or less
  // 4 meshlets for level 4
  // 5 meshlets for level 5
  uint partOffset    = subdiv_getPartOffset(subdivTarget, partID);
  
  sdec.threadOffset  = packID * packThreads;
  sdec.vertexOffset  = MICRO_BTRI_VTX_OFFSET(partID, subdivTarget, subdivBase, formatIdx);
  
  sdec.primOffset    = MICRO_MESHLET_LOD_PRIMS * cfg.partSubdiv;
#if USE_NON_UNIFORM_SUBDIV
  uint topo          = micromesh_getBaseTopo(microBaseTri);
#else
  uint topo          = 0;
#endif
  sdec.primOffset    += partOffset * MICRO_PART_MAX_PRIMITIVES + topo * MICRO_MESHLET_PRIMS;
  
  
  // always read initial values
  sdec.localVtx          = microdata_loadMicromeshVertex (packThreadID + sdec.vertexOffset);  
  sdec.localCorrection   = microdec_decodeCorrectionBitsBVtx(sdec.dec, sdec.localVtx, packThreadID);

#if MICRO_SUPPORTED_FORMAT_BITS != (1<<MICRO_FORMAT_64T_512B)
  if (isFlat)
  {
    return;
  }
  
  // we can only reach this if targetSubdiv is >= 3
  // meaning all vertices are required
  
  // subdivTarget    : the target subdivision level for the displaced patch
  //                   subdivDescend + subdivMerge + subdivVertex
  // the hierarchy is as follows:
  // subdivDescend    2: number of levels provided via mips (0..2)
  // subdivMerge      x: number of levels resolved by computing displacement values through merging
  //                     maximum is 2
  // subdivVertex     1: last level is computed at vertex time based on merging
  //

  // this yields
  // subdivTarget 5  subdivMerge 2
  // subdivTarget 4  subdivMerge 1
  // subdivTarget 3  subdivMerge 0
  uint subdivMip     = MICRO_MIP_SUBDIV;
  uint subdivVertex  = 1; // last level reads from sdec.displacement
  uint subdivMerge   = uint(max(0,int(subdivTarget) - int(subdivMip + subdivVertex)));
  
  // add merge levels
  // subd  | # vertices |
  // subd 0| 3          | anchor/ post descend level
  // subd 1| 6          | mergeLevel 0
  // subd 2| 15         | mergeLevel 1
  // subd 3| 45         | vtx level
  
  // if no merging is done, only hast level is computed
 
  
  // this will generate 
  // subdivMerge 2 decodeVertStart 3
  // subdivMerge 1 decodeVertStart 6
  // subdivMerge 0 decodeVertStart 15
  uint revMerge = (2 - subdivMerge);
  // ((0 or 1 or 4) + 1) * 3 == (3 or 6 or 15)
  sdec.decodeVertStart = ((revMerge * revMerge) + 1) * 3;
  
  sdec.displacement = sdec.localCorrection;
  for (uint merge = 0; merge < subdivMerge; merge++)
  {
    // this will generate
    // for subdivMerge 2 
    //  merge 0 numVerts 6
    //  merge 1 numVerts 15
    // for subdivMerge 1
    //  merge 0 numVerts 15
    
    uint numVerts = 6 + (subdivMerge == 1 ? 1 : merge) * 9;
    
    if (packThreadID < numVerts)
    {
      sdec.displacement = smicrodec_subgroupGetVertexDisplacement(sdec, packThreadID, microvertex_getAB(sdec.localVtx), sdec.localCorrection);
    }
    
    sdec.decodeVertStart = numVerts;
  }
#endif
}

uint smicrodec_getIterationCount()
{
#if SUBGROUP_SIZE == 32
  return 2;
#elif SUBGROUP_SIZE == 64
  return 1;
#else
  #error "unspported SUBGROUP_SIZE"
#endif
}

uint smicrodec_getPackID(inout SubgroupMicromeshDecoder sdec)
{
  return sdec.cfg.packID;
}

uint smicrodec_getDataIndex(inout SubgroupMicromeshDecoder sdec, uint iterationIndex)
{
  return sdec.cfg.packThreadID + iterationIndex * sdec.cfg.packThreads;
}

uint smicrodec_getNumTriangles(inout SubgroupMicromeshDecoder sdec)
{
  return subdiv_getNumTriangles(sdec.cfg.partSubdiv);  
}
uint smicrodec_getNumVertices (inout SubgroupMicromeshDecoder sdec)
{
  return subdiv_getNumVerts(sdec.cfg.partSubdiv);  
}
uint smicrodec_getMeshTriangle(inout SubgroupMicromeshDecoder sdec)
{
  return sdec.baseID;
}
uint smicrodec_getBaseSubdiv  (inout SubgroupMicromeshDecoder sdec)
{
  return micromesh_getBaseSubdiv(sdec.microBaseTri);
}
uint smicrodec_getFormatIdx   (inout SubgroupMicromeshDecoder sdec)
{
  return microdec_getFormatIdx(sdec.dec);
}
uint smicrodec_getMicroSubdiv (inout SubgroupMicromeshDecoder sdec)
{
  // intentional full subdiv here
  return micromesh_getBaseSubdiv(sdec.microBaseTri);
}

MicroDecodedVertex smicrodec_subgroupGetVertex (inout SubgroupMicromeshDecoder sdec, uint iterationIndex)
{
  uint numVerts  = subdiv_getNumVerts(sdec.cfg.partSubdiv); 
  uint packID    = sdec.cfg.packID;
  
  uint vert      = smicrodec_getDataIndex(sdec, iterationIndex);
  bool vertValid = vert < numVerts && sdec.cfg.valid;
  bool isFlat    = microdec_isFlat(sdec.dec) || sdec.cfg.targetSubdiv <= MICRO_MIP_SUBDIV;
  
#if SUBGROUP_SIZE == 32
  #if 0
    MicromeshBTriVertex localVtx = iterationIndex > 0 ? microdata_loadMicromeshVertex(vert + sdec.vertexOffset) : sdec.localVtx;
  #else
    // compiler bug WAR
    // https://github.com/KhronosGroup/glslang/issues/2843
    MicromeshBTriVertex localVtx;
    if (iterationIndex > 0)
    {
      localVtx = microdata_loadMicromeshVertex(vert + sdec.vertexOffset);
    }
    else {
      localVtx = sdec.localVtx;
    }
  #endif
  int localCorrection = iterationIndex > 0 ? microdec_decodeCorrectionBitsBVtx(sdec.dec, localVtx, vert) : sdec.localCorrection;
#else
  MicromeshBTriVertex localVtx = sdec.localVtx;
  int localCorrection          = sdec.localCorrection;
#endif

  MicroDecodedVertex outVertex; 
  
  if (isFlat) {
    outVertex.displacement = localCorrection;
  }
  else {
    outVertex.displacement = smicrodec_subgroupGetVertexDisplacement(sdec, vert, microvertex_getAB(localVtx), localCorrection);
  }
  
  outVertex.valid      = vertValid;
  outVertex.localIndex = vert;
  outVertex.outIndex   = vert + packID * numVerts;
  
  if (vertValid) 
  {
    // Compute barycentrics
    outVertex.uv = microvertex_getUV(localVtx);
    outVertex.bary.yz = vec2(outVertex.uv) / float(subdiv_getNumSegments( micromesh_getBaseSubdiv(sdec.microBaseTri) ));
    outVertex.bary.x = 1.0f - outVertex.bary.y - outVertex.bary.z;
  }

  return outVertex;
}

MicroDecodedTriangle smicrodec_getTriangle   ( inout SubgroupMicromeshDecoder sdec, uint iterationIndex)
{
  uint numVerts     = subdiv_getNumVerts(sdec.cfg.partSubdiv); 
  uint numTriangles = subdiv_getNumTriangles(sdec.cfg.partSubdiv);  
  uint packID       = sdec.cfg.packID;

  uint prim = smicrodec_getDataIndex(sdec, iterationIndex);
  bool primValid = prim < numTriangles && sdec.cfg.valid;
  
  MicroDecodedTriangle outPrim;
  outPrim.valid      = primValid;
  outPrim.localIndex = prim;
  outPrim.outIndex   = prim + packID * numTriangles;
  
  if (primValid) 
  {
    uint localTri = microdata_loadTriangleIndices(prim + sdec.primOffset);
    uvec3 indices;
    indices.x = (localTri >> 0)  & 0xFF;
    indices.y = (localTri >> 8)  & 0xFF;
    indices.z = (localTri >> 16) & 0xFF;
    
    outPrim.indices = indices + packID * numVerts;
  }
  
  return outPrim;
}

