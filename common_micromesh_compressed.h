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

// this file is included by C++ and GLSL

#ifndef _COMMON_MICROMESH_COMPRESSED_H_
#define _COMMON_MICROMESH_COMPRESSED_H_

#include "common_micromesh.h"

//////////////////////////////////////////////////////////////////////////

// Data structures to render block-compressed barycentric micromesh displacements.

#ifdef __cplusplus
namespace microdisp {
using namespace glm;
#endif

// binding information for descriptor set

#define DRAWCOMPRESSED_UBO_VIEW 0
#define DRAWCOMPRESSED_SSBO_STATS 1
#define DRAWCOMPRESSED_UBO_MESH 2
#define DRAWCOMPRESSED_UBO_COMPRESSED 3
#define DRAWCOMPRESSED_UBO_SCRATCH 4
#define DRAWCOMPRESSED_TEX_HIZ 5
#define DRAWCOMPRESSED_IMG_ATOMIC 6
#define DRAWCOMPRESSED_ACC 7

// these are set via RendererVK::getShaderPrepend()
#ifndef SHADING_UMAJOR
#define SHADING_UMAJOR 1
#endif

#ifndef MICRO_DECODER
#define MICRO_DECODER MICRO_DECODER_BASETRI_MIP_SHUFFLE
#endif

#ifndef MICRO_SUPPORTED_FORMAT_BITS
#define MICRO_SUPPORTED_FORMAT_BITS 7
#endif

//////////////////////////////////////////////////////////////////////////
// Micromesh
//
// We currently decode on a sub-triangle basis.
// Each sub-triangle matches one compressed block.
// Therefore the term "micromesh" is now often used
// when referring to such a block / sub-triangle.


// level of decoding

// per base-triangle decoding with mip data via shuffle
#define MICRO_DECODER_BASETRI_MIP_SHUFFLE 2
// per micro-triangle decoding per thread
#define MICRO_DECODER_MICROTRI_THREAD 3

#define MICRO_UNORM_BITS 11

#define MICRO_FORMAT_64T_512B 0
#define MICRO_FORMAT_256T_1024B 1
#define MICRO_FORMAT_1024T_1024B 2
#define MICRO_MAX_FORMATS 3

#define MICRO_FORMAT_SUPPORTED(fmt) ((MICRO_SUPPORTED_FORMAT_BITS & (1 << fmt)) != 0)

#define MICRO_FORMAT_MIN_SUBDIV 3
#define MICRO_FORMAT_MAX_SUBDIV 5
#define MICRO_FORMAT_MAX_LEVELS (MICRO_FORMAT_MAX_SUBDIV + 1)
#define MICRO_FORMAT_MAX_TRIANGLES (1u << (MICRO_FORMAT_MAX_SUBDIV * 2))

#define MICRO_MIP_SUBDIV 2
#define MICRO_MIP_MIN_SUBDIV 4
#define MICRO_MIP_VERTICES 15

#define MICRO_PART_MAX_SUBDIV 3
#define MICRO_PART_MAX_PRIMITIVES 64
#define MICRO_PART_MAX_VERTICES 45

// 45 rounded to 48 for better alignment
#define MICRO_PART_VERTICES_STRIDE 48

// meshlet config
#define MICRO_MESHLET_VERTICES 64
#define MICRO_MESHLET_PRIMITIVES 64

// due to splitting of micro into multiple meshlets aka subparts
//  1: subd <= 3
//  4: subd == 4
// 16: subd == 5
#define MICRO_MESHLET_PARTS (1 + 4 + 16)

#define MICRO_MESHLET_TOPOS 8
#define MICRO_MESHLET_LOD_PRIMS 16
#define MICRO_MESHLET_PRIMS (MICRO_PART_MAX_PRIMITIVES * MICRO_MESHLET_PARTS + MICRO_MESHLET_LOD_PRIMS * 3)

//////////////////////////////////////////////////////////////////////////
// MicromeshBaseTri compact information
//
// Requires that all sub-triangles use the same format
// and makes use of a specialized decoder

#define MICRO_BASE_LVL_SHIFT 0
#define MICRO_BASE_LVL_WIDTH 3
#define MICRO_BASE_LVL_MASK ((1 << MICRO_BASE_LVL_WIDTH) - 1)
#define MICRO_BASE_TOPO_SHIFT 3
#define MICRO_BASE_TOPO_WIDTH 3
#define MICRO_BASE_TOPO_MASK ((1 << MICRO_BASE_TOPO_WIDTH) - 1)
#define MICRO_BASE_FMT_SHIFT 6
#define MICRO_BASE_FMT_WIDTH 2
#define MICRO_BASE_FMT_MASK ((1 << MICRO_BASE_FMT_WIDTH) - 1)
#define MICRO_BASE_CULLDIST_SHIFT 8
#define MICRO_BASE_CULLDIST_WIDTH 8
#define MICRO_BASE_CULLDIST_MASK ((1 << MICRO_BASE_CULLDIST_WIDTH) - 1)

#define MICRO_BASE_MIPLO_SHIFT 16
#define MICRO_BASE_MIPLO_WIDTH 16
#define MICRO_BASE_MIPLO_MASK ((1 << MICRO_BASE_MIPLO_WIDTH) - 1)

// this special data offset encoding is only relevant to mip decoder
// data offsets are minimum aligned to 512 bit / 64 byte / 16 u32s
// meaning 4GB can hold (32 - 6) many blocks
#define MICRO_BASE_DATA_VALUE_MUL 16
#define MICRO_BASE_DATA_SHIFT 0
#define MICRO_BASE_DATA_WIDTH 26
#define MICRO_BASE_DATA_MASK ((1 << MICRO_BASE_DATA_WIDTH) - 1)
#define MICRO_BASE_DATA_MIPHI_SHIFT 26
#define MICRO_BASE_DATA_MIPHI_WIDTH 6
#define MICRO_BASE_DATA_MIPHI_MASK ((1 << MICRO_BASE_DATA_MIPHI_WIDTH) - 1)

// 22 bits for mip-offsets
#define MICRO_BASE_MIP_WIDTH (MICRO_BASE_MIPLO_WIDTH + MICRO_BASE_DATA_MIPHI_WIDTH)
#define MICRO_BASE_MIP_MAX (1 << (MICRO_BASE_MIP_WIDTH))
// 192 bits for subdiv 2 == 6 * 32
#define MICRO_BASE_MIP_VALUE_MUL 6

struct MicromeshBaseTri
{
#if defined(__cplusplus) && defined(_DEBUG)
  union
  {
    struct
    {
      uint32_t baseLevel : 3;  // max subdiv level 5
      uint32_t baseTopo : 3;   // 0..7 for various edge decimate combinations
      uint32_t fmt : 2;        // compression block subdiv level
      uint32_t cullDist : 8;
      uint32_t mipLo : 16;
    };

    uint32_t packedBits;
  };
#else
  uint32_t packedBits;
#endif
  uint32_t dataOffset;
};
#ifndef __cplusplus
layout(buffer_reference, buffer_reference_align = 8, scalar) restrict readonly buffer MicromeshBaseTris_in
{
  MicromeshBaseTri d[];
};
#endif


//////////////////////////////////////////////////////////////////////////
// MicromeshBaseTriangleDecoder related
//
// This decoder operates on base-triangles
// It computes vertex displacements within a meshlet-part
// by making use of shuffle to use existing values in other
// threads for the prediction to compute the new values with corrections.
//

#define MICRO_BTRI_VTX_UNSIGNED (1 << 0)
#define MICRO_BTRI_VTX_MIP (1 << 1)
#define MICRO_BTRI_VTX_CORRPOS_SHIFT 2
#define MICRO_BTRI_VTX_CORRPOS_WIDTH 6
#define MICRO_BTRI_VTX_CORRMASK_SHIFT 8
#define MICRO_BTRI_VTX_CORRMASK_WIDTH 4
#define MICRO_BTRI_VTX_BITNUM_SHIFT 12
#define MICRO_BTRI_VTX_BITNUM_WIDTH 4
#define MICRO_BTRI_VTX_BITPOS_SHIFT 16
#define MICRO_BTRI_VTX_BITPOS_WIDTH 16


// max parts within one decoding op
// due to splitting of micro into multiple meshlets
#define MICRO_BTRI_MAX_MESHLET_PARTS 16

// this is a table precomputing the exact vertices used by each
// meshlet partition. The table size and indexing in theory could be altered to make it
// more compact, as the MICRO_MAX_LEVELS * MICRO_MAX_LEVELS is actually only used in
// half and MICRO_BTRI_MAX_MESHLET_PARTS for each level is also too much.
// We need it also per-format, as the formats have different block sizes / bit positions
// of vertices.

#define MICRO_BTRI_VTX_COUNT                                                                                           \
  (MICRO_BTRI_MAX_MESHLET_PARTS * MICRO_MAX_LEVELS * MICRO_MAX_LEVELS * MICRO_MAX_FORMATS * MICRO_MESHLET_VERTICES)
#define MICRO_BTRI_VTX_OFFSET(partID, targetSubdiv, baseSubdiv, formatIdx)                                                        \
  (((partID) + (MICRO_BTRI_MAX_MESHLET_PARTS * (targetSubdiv)) + (MICRO_BTRI_MAX_MESHLET_PARTS * MICRO_MAX_LEVELS) * (baseSubdiv) \
    + ((formatIdx) * MICRO_BTRI_MAX_MESHLET_PARTS * MICRO_MAX_LEVELS * MICRO_MAX_LEVELS))                                         \
   * MICRO_MESHLET_VERTICES)


struct MicromeshBTriVertex
{

  // uv coords are local to sub-triangle
  //
  // level and index are used in combination with encoding format bit info
  // to calculate final bit position for uncompressed / correction values
  //
  // a and b are merge indices used in shuffle to access the parent
  // vertices involved in the splitting.

#if defined(__cplusplus) && defined(_DEBUG)
  union
  {
    struct
    {
      uint32_t isUnsigned : 1;
      uint32_t isMip : 1;
      uint32_t corrPos : 6;   // 0..63
      uint32_t corrMask : 4;  // 0..15 (max 4 bit wide)
      uint32_t bitnum : 4;    // 0..11
      uint32_t bitpos : 16;   // 0..((16 * 1024) -1) // bit position across multiple blocks
    };
#endif
    uint32_t packed;
#if defined(__cplusplus) && defined(_DEBUG)
  };
#endif
#if defined(__cplusplus)
  glm::u8vec2 uv;
  glm::u8vec2 parents;
#else
  u8vec2 uv;
  u8vec2 parents;
#endif
};
#ifndef __cplusplus
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict readonly buffer MicromeshBTriVertexs_in
{
  MicromeshBTriVertex d[];
};
#endif


// static pairing table, used for all vertices
#define MICRO_BTRI_DESCENDS_COUNT (MICRO_MESHLET_VERTICES)

#define MICRO_BTRI_DESCEND_A_SHIFT 0
#define MICRO_BTRI_DESCEND_B_SHIFT 4

// contains the local shuffle indices of parents
// for all vertices within a part

#define MicromeshBTriDescend u8vec2
#define MicromeshBTriDescends_in u8vec2s_in


//////////////////////////////////////////////////////////////////////////
// MicromeshMicroTriangleDecoder related
//
// This decoder gets the displacements per vertex
// by decoding one micro-triangle that the vertex
// belongs to and then picking the right corner
// vertex of that micro-triangle.
//
// MicromeshMTriVertex hence stores which micro-
// triangle and which corner it is.
//
// MicromeshMTriDescend stores the information
// to decode one micro-triangle by doing the
// hierarchial decoding in multiple descend
// operations.

#define MICRO_MTRI_VTX_U_SHIFT 0
#define MICRO_MTRI_VTX_V_SHIFT 6
#define MICRO_MTRI_VTX_UV_WIDTH 6
#define MICRO_MTRI_VTX_CORNER_SHIFT 12
#define MICRO_MTRI_VTX_CORNER_WIDTH 2
#define MICRO_MTRI_VTX_MTRI_SHIFT 14
#define MICRO_MTRI_VTX_MTRI_WIDTH 10

// max parts within one decoding op
// due to splitting of micro into multiple meshlets
#define MICRO_MAX_MTRI_MESHLET_PARTS 16

#define MICRO_MTRI_VTX_COUNT                                                                                           \
  (MICRO_MAX_MTRI_MESHLET_PARTS * MICRO_MAX_LEVELS * MICRO_MAX_LEVELS * MICRO_PART_VERTICES_STRIDE)
#define MICRO_MTRI_VTX_OFFSET(partID, targetSubdiv, baseSubdiv)                                                                    \
  (((partID) + (MICRO_MAX_MTRI_MESHLET_PARTS * (targetSubdiv)) + (MICRO_MAX_MTRI_MESHLET_PARTS * MICRO_MAX_LEVELS) * (baseSubdiv)) \
   * MICRO_PART_VERTICES_STRIDE)

struct MicromeshMTriVertex
{
  // This vertex operates in base-triangle space, not like the MicromeshSTriVertex above.
  // UVs are relative to base-triangle.
  // The vertex is also format specific, so we hardcode the
  // location of the correction bits across all blocks relative
  // to the base-triangle. Including the special mip-block.

#if defined(__cplusplus) && defined(_DEBUG)
  union
  {
    struct
    {
      uint32_t u : 6;       // 0..32 (level 5 max) relative to base-triangle
      uint32_t v : 6;       // 0..32 (level 5 max)
      uint32_t corner : 2;  // 0..2 subvertex within micro-tri (w,u,v)
      uint32_t mtriID : 9;  // 0..1023 (1024 max micro tri for subdiv-5)
    };
#endif
    uint32_t packed;
#if defined(__cplusplus) && defined(_DEBUG)
  };
#endif
};
#ifndef __cplusplus
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict readonly buffer MicromeshMTriVertexs_in
{
  MicromeshMTriVertex d[];
};
#endif


#define MICRO_MTRI_DESCEND_VERTEX_LVL_SHIFT 0
#define MICRO_MTRI_DESCEND_VERTEX_LVL_WIDTH 3
#define MICRO_MTRI_DESCEND_VERTEX_TYPE_SHIFT 3
#define MICRO_MTRI_DESCEND_VERTEX_TYPE_WIDTH 2
#define MICRO_MTRI_DESCEND_VERTEX_DATA_SHIFT 5
#define MICRO_MTRI_DESCEND_VERTEX_DATA_WIDTH 11
#define MICRO_MTRI_DESCEND_VERTEX_WIDTH 16
// 16 * 3 = 48

// we use one big descend table for each format a
// micro-triangle may require max MICRO_FORMAT_MAX_LEVELS
// many steps to descend to reach the final resolution.

#define MICRO_MTRI_DESCENDS_COUNT (MICRO_FORMAT_MAX_TRIANGLES * MICRO_FORMAT_MAX_LEVELS * MICRO_MAX_FORMATS)

// indexing is tuned so MICRO_FORMAT_64T_512B has locality regards blockTri (and ignores level)
// compressed formats have locality for levels on same blockTri, given they need to fetch all levels in the end
#define MICRO_MTRI_DESCENDS_INDEX(blockTri, format)                                                                    \
  ((blockTri) + (MICRO_FORMAT_MAX_TRIANGLES * MICRO_FORMAT_MAX_LEVELS) * (format))


struct MicromeshMTriDescend
{
  // defines the per-vertex split/descend operation
  // input is 3 vertices, and output is 3 vertices
  // with deltas applied
  // perform appropriate split, applies delta

#if defined(__cplusplus) && defined(_DEBUG)
  union
  {
    struct
    {
      // 3 vertices
      uint16_t v0_lvl : 3;    // 0..5 level max 5
      uint16_t v0_type : 2;   // 0..3 vertex type (edge / interior)
      uint16_t v0_data : 11;  // 0..1023 bit

      uint16_t v1_lvl : 3;    // 0..5 level max 5
      uint16_t v1_type : 2;   // 0..3 vertex type (edge / interior)
      uint16_t v1_data : 11;  // 0..1023 bit

      uint16_t v2_lvl : 3;    // 0..5 level max 5
      uint16_t v2_type : 2;   // 0..3 vertex type (edge / interior)
      uint16_t v2_data : 11;  // 0..1023 bit

      uint16_t _pad;
    };
#endif
#if defined(__cplusplus)
    glm::u16vec4 vertices;
#else
  u16vec4 vertices;
#endif

#if defined(__cplusplus) && defined(_DEBUG)
  };
#endif
};
#ifndef __cplusplus
layout(buffer_reference, buffer_reference_align = 8, scalar) restrict readonly buffer MicromeshMTriDescends_in
{
  MicromeshMTriDescend d[];
};
#endif

//////////////////////////////////////////////////////////////////////////
// MicromeshFormatDescr
//
// compression format details (we could hardcode these when formats are frozen)

#define MICRO_FORMATINFO_CORR_WIDTH 4
#define MICRO_FORMATINFO_CORR_MASK ((1 << MICRO_FORMATINFO_CORR_WIDTH) - 1)
#define MICRO_FORMATINFO_START_SHIFT MICRO_FORMATINFO_CORR_WIDTH

struct MicromeshFormatDescr
{
  // 6 required (level 5 + anchor), padded to 8
  uint16_t width_start[8];  // 4  bits correction width
                            // 12 bits start
};

#ifndef __cplusplus
layout(buffer_reference, buffer_reference_align = 16, scalar) restrict readonly buffer MicromeshFormatDescrs_in
{
  MicromeshFormatDescr d[];
};
#endif

//////////////////////////////////////////////////////////////////////////
// MicromeshData
//
// Main data container that contains compressed displacement distance data
// as well as all meta-information to render the micromesh.

struct MicromeshData
{
  // fixed static lookup tables, data independent
  BUFFER_REF(MicromeshFormatDescrs_in) formats;
#if MICRO_DECODER == MICRO_DECODER_BASETRI_MIP_SHUFFLE
  BUFFER_REF(MicromeshBTriDescends_in) descendInfos;
  BUFFER_REF(MicromeshBTriVertexs_in) vertices;
#elif MICRO_DECODER == MICRO_DECODER_MICROTRI_THREAD
  BUFFER_REF(MicromeshMTriDescends_in) descendInfos;
  BUFFER_REF(MicromeshMTriVertexs_in) vertices;
#else
  uint64_t descendInfos;
  uint64_t vertices;
#endif

  // index buffers
  // MAX_MICRO_MESHLET_TOPOS * MAX_MICRO_MESHLET_PRIMS
  BUFFER_REF(uints_in) triangleIndices;

  // only relevant for other barycentric attributes
  BUFFER_REF(uints_in) umajor2bmap[MICRO_MAX_LEVELS];

  // data/mesh-dependent

  // flattened triangles (resolved indirection from mesh tri to micromap tri)
  BUFFER_REF(MicromeshBaseTris_in) basetriangles;  // only available if all child subtriangles of same format
  BUFFER_REF(vec4s_in) basespheres;                // only available if all child subtriangles of same format

  BUFFER_REF(uints_in) distances;
  BUFFER_REF(uints_in) mipDistances;  // only available for MICRO_DECODER_BASETRI_MIP_SHUFFLE

  BUFFER_REF(uints_in) attrTriangleOffsets;
  BUFFER_REF(uints_in) attrNormals;

  // used solely for visualization
  BUFFER_REF(uint16s_in) basetriangleMinMaxs;
};

//////////////////////////////////////////////////////////////////////////
// per-draw info

struct DrawMicromeshPushData
{
  uint    firstVertex;
  uint    firstTriangle;
  uint    instanceID;
  f16vec2 scale_bias;

  uint     microMax;
  uint     _pad;
  uint64_t binding;
};

#ifndef __cplusplus
layout(buffer_reference, buffer_reference_align = 16, scalar) restrict readonly buffer DrawMicromeshPushData_in
{
  DrawMicromeshPushData d[];
};
#endif

//////////////////////////////////////////////////////////////////////////

struct MicromeshScratchData
{
  BUFFER_REF(uints_inout) atomicCounter;
  BUFFER_REF(DrawMicromeshPushData_in) instancePushDatas;
  BUFFER_REF(uints_in) scratchData;

  uint maxCount;  // always power of 2
  uint maxMask;
};


#ifdef __cplusplus
}
#endif

#endif
