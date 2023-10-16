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
// MicroTriangle Decoder
// MICRO_DECODER == MICRO_DECODER_MICROTRI_THREAD
//
// This decoder operates on a single thread (no shuffle)
// and using a lookup table identifies which micro-vertex
// needs to be decoded.

// If MICRO_MTRI_USE_INTRINSIC == 1
// The micro-vertex is fetched directly through and intrinsic
// from the BLAS that uses a micromap.
// This fetch is implemented in `smicrodec_getVertexPos`
// in `micromesh_decoder_api.glsl`
//
// If MICRO_MTRI_USE_INTRINSIC == 0
// Performs the hierarchical descend into the block compressed
// displacement data on a per-thread basis.
// Each thread targets one micro-vertex. In this process it subdivides
// the triangle and decodes its displacement at each level. Then
// picks which of the four child triangles are relevant to the target
// micro-vertex and continues the procedure until the appropriate level in which
// the micro-vertex exists is reached.
// There are two variants:
//  MICRO_MTRI_USE_MATH == 1 does the descending logic through ALU
//  MICRO_MTRI_USE_MATH == 0 uses a precomputed table for the hierarchical
//                           decoding path for each micro-vertex

struct SubgroupMicromeshDecoder
{
  MicroDecoder        dec;
  MicroDecoderConfig  cfg;
  
  MicromeshBaseTri    microBaseTri;
  
  uint baseID;
  uint firstData;
  
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
};


#include "micromesh_decoder_api.glsl"


// MicromeshMTriVertex
ivec2 microvertex_getUV(MicromeshMTriVertex vtx) {
  return ivec2( bitfieldExtract(vtx.packed, MICRO_MTRI_VTX_U_SHIFT, MICRO_MTRI_VTX_UV_WIDTH),
                bitfieldExtract(vtx.packed, MICRO_MTRI_VTX_V_SHIFT, MICRO_MTRI_VTX_UV_WIDTH));
}

uint microvertex_getMTri(MicromeshMTriVertex vtx) {
  return vtx.packed >> MICRO_MTRI_VTX_MTRI_SHIFT;
}

uint microvertex_getCorner(MicromeshMTriVertex vtx) {
  return bitfieldExtract(vtx.packed, MICRO_MTRI_VTX_CORNER_SHIFT, MICRO_MTRI_VTX_CORNER_WIDTH);
}

#if !MICRO_MTRI_USE_INTRINSIC
#include "micromesh_decoder_microtri_eval.glsl"
#endif

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
                            MicroDecoderConfig            cfg,
                            MicromeshBaseTri              microBaseTri,
                            uint                          firstMicro,
                            uint                          firstData,
                            uint                          firstMipData)
{
  uint packThreadID  = cfg.packThreadID;
  uint packThreads   = cfg.packThreads;
  uint packID        = cfg.packID;
  uint microID       = cfg.microID;
  uint partID        = cfg.partID;
  uint subdivTarget  = cfg.targetSubdiv;
  uint subdivBase    = micromesh_getBaseSubdiv(microBaseTri);

  sdec.microBaseTri  = microBaseTri;
  sdec.firstData     = firstData + microBaseTri.dataOffset;
  sdec.baseID        = microID;

  sdec.cfg           = cfg;
  
  microdec_init(sdec.dec, micromesh_getFormat(microBaseTri), 0, 0);
  
  // offset into meshlet configurations
  // influenced by partID state (if micromesh split into multiple meshlets)
  // there are 3 partMicro configrations 
  // 1 meshlet  for level 3 or less
  // 4 meshlets for level 4
  // 5 meshlets for level 5
  uint partOffset    = subdiv_getPartOffset(subdivTarget, partID);
  
  sdec.threadOffset  = packID * packThreads;
  sdec.vertexOffset  = MICRO_MTRI_VTX_OFFSET(partID, subdivTarget, subdivBase);
  
  sdec.primOffset    = MICRO_MESHLET_LOD_PRIMS * cfg.partSubdiv;
#if USE_NON_UNIFORM_SUBDIV
  uint topo          = micromesh_getBaseTopo(microBaseTri);
#else
  uint topo          = 0;
#endif
  sdec.primOffset    += partOffset * MICRO_PART_MAX_PRIMITIVES + topo * MICRO_MESHLET_PRIMS;
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
  bool isFlat    = microdec_isFlat(sdec.dec);
  

  MicromeshMTriVertex localVtx = microdata_loadMicromeshVertex(vert + sdec.vertexOffset);

  MicroDecodedVertex outVertex;
  outVertex.valid      = vertValid;
  outVertex.localIndex = vert;
  outVertex.outIndex   = vert + packID * numVerts;

  if (vertValid) 
  {
    ivec2 baseUV       = microvertex_getUV(localVtx);
    
#if MICRO_MTRI_USE_INTRINSIC
    // required as input to the intrinsic
    outVertex.blasBasePrimitiveID = smicrodec_getMeshTriangle(sdec);
#else
    outVertex.displacement = smicrodec_getVertexDisplacement(sdec, localVtx, outVertex.outIndex);
#endif

    // Compute barycentrics
    outVertex.uv = baseUV;
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


