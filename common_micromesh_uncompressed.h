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

#ifndef _COMMON_MICROMESH_UNCOMPRESSED_H_
#define _COMMON_MICROMESH_UNCOMPRESSED_H_

#include "common_micromesh.h"

//////////////////////////////////////////////////////////////////////////

// Data structures to render uncompressed barycentric micromesh displacements.
// May contain per-micro-vertex attributes

#ifdef __cplusplus
namespace microdisp {
using namespace nvmath;
#endif

// binding information for descriptor set

#define DRAWUNCOMPRESSED_UBO_VIEW         0
#define DRAWUNCOMPRESSED_SSBO_STATS       1
#define DRAWUNCOMPRESSED_UBO_MESH         2
#define DRAWUNCOMPRESSED_UBO_MAP          3
#define DRAWUNCOMPRESSED_UBO_UNCOMPRESSED 4
#define DRAWUNCOMPRESSED_UBO_SCRATCH      5
#define DRAWUNCOMPRESSED_TEX_HIZ          6
#define DRAWUNCOMPRESSED_IMG_ATOMIC       7


#ifndef __cplusplus

    // these are set via RendererVK::getShaderPrepend()
    #ifndef UNCOMPRESSED_UMAJOR
    #define UNCOMPRESSED_UMAJOR 1
    #endif

    #ifndef UNCOMPRESSED_DISPLACEMENT_BITS
    #define UNCOMPRESSED_DISPLACEMENT_BITS          32
    #endif
    
    // setup types and decode macros
    #if   UNCOMPRESSED_DISPLACEMENT_BITS == 32
      #define dbits_in  floats_in
      #define dbitsminmax_in floats_in
      #define dbitsUnpack(v)  (v)
    #elif UNCOMPRESSED_DISPLACEMENT_BITS == 16
      #define dbits_in  uint16s_in
      #define dbitsminmax_in uint16s_in
      #define dbitsUnpack(v)  (float((v)) / float(0xFFFF))
    #elif UNCOMPRESSED_DISPLACEMENT_BITS == 11
      #define dbits_in  uint16s_in
      #define dbitsminmax_in uint16s_in
      #define dbitsUnpack(v)  (float((v)) / float(0x7FF))
    #elif UNCOMPRESSED_DISPLACEMENT_BITS == -11
      #define dbits_in uints_in
      #define dbitsminmax_in uint16s_in
      #define dbitsUnpack(v) (float((v)) / float(0x7FF))
    #elif UNCOMPRESSED_DISPLACEMENT_BITS == 8
      #define dbits_in  uint8s_in
      #define dbitsminmax_in uint8s_in
      #define dbitsUnpack(v)  (float((v)) / float(0xFF))
    #else
      #error "invalid UNCOMPRESSED_DISPLACEMENT_BITS"
    #endif
    
    float loadUncompressedDisplacement(dbits_in distances, uint firstValue, uint idx)
    {
    #if UNCOMPRESSED_DISPLACEMENT_BITS == -11
      uint base = ((idx * 11) / 32) + firstValue;
      uint rest = ((idx * 11) % 32);
      uint rawLo = distances.d[base];
      uint rawHi = distances.d[base + 1]; // must be safe to overfetch
      uint64_t raw64 = packUint2x32(uvec2(rawLo,rawHi));
      uint value = uint(raw64 >> rest) & 0x7FF;
      return dbitsUnpack(value);
    #else
      return dbitsUnpack(distances.d[firstValue + idx]);
    #endif
    }

#endif


////////////////////////////////////////////////////////////////////
// BaryFlatTriangle
//
// flattened information for easier rendering

struct MicromeshUncBaseTri
{
  uint16_t subdivLevel;
  uint16_t topoBits;
  uint     firstValue;
  uint     firstShadingValue;
  uint     meshletCount;
};
#ifndef __cplusplus
layout(buffer_reference, buffer_reference_align = 16, scalar) restrict readonly buffer MicromeshUncBaseTri_in
{
  MicromeshUncBaseTri d[];
};
#endif

////////////////////////////////////////////////////////////////////
// BaryMeshData
//
// primary container holding pointers to all data required
// to render an uncompressed bary displaced mesh.

struct MicromeshUncData
{
  // quantized distances
  BUFFER_REF(dbits_in)               distancesBits;
  BUFFER_REF(dbitsminmax_in)         triangleBitsMinMax;

  // flattened triangles (resolved indirection from mesh tri to micromap tri)
  BUFFER_REF(MicromeshUncBaseTri_in) basetriangles;
  BUFFER_REF(vec4s_in)               basespheres;

  // per micro-vertex attributes
  BUFFER_REF(uints_in)               attrNormals;
};

///////////////////////////////////////////////////////////////////
// per-draw info

struct DrawMicromeshUncPushData
{
  uint firstVertex;
  uint firstTriangle;
  
  uint instanceID;
  uint triangleMax;
  vec2 scale_bias;

  uint64_t binding;
};

#ifndef __cplusplus
layout(buffer_reference, buffer_reference_align = 16, scalar) restrict readonly buffer DrawMicromeshUncompressedPushData_in
{
  DrawMicromeshUncPushData d[];
};
#endif

//////////////////////////////////////////////////////////////////////////

struct MicromeshUncScratchData
{
  BUFFER_REF(uints_inout)                           atomicCounter;
  BUFFER_REF(DrawMicromeshUncompressedPushData_in)  instancePushDatas;
  BUFFER_REF(uints_in)                              scratchData;
  
  uint maxCount;  // always power of 2
  uint maxMask;
};


#ifdef __cplusplus
}
#endif

#endif
