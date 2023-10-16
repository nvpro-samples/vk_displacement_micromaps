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
 
/////////////////////////////////////////////////////

// Several generic helpers when dealing with micromesh data


/////////////////////////////////////////////////////
// subdivision mesh helpers
//
// lua: function subdiv_getNumVerts(subd) local numVertsPerEdge = math.pow(2,subd) + 1; return (numVertsPerEdge * (numVertsPerEdge + 1)) / 2; end

uint subdiv_getNumVertsPerEdge  (uint subdiv)   { return (1u << subdiv) + 1; }
uint subdiv_getNumTriangles     (uint subdiv)   { return (1u << (subdiv * 2));}
uint subdiv_getNumSegments      (uint subdiv)   { return (1u << subdiv); }
uint subdiv_getNumVerts         (uint subdiv)   { uint numVertsPerEdge = subdiv_getNumVertsPerEdge(subdiv); return (numVertsPerEdge * (numVertsPerEdge + 1)) / 2; }
uint subdiv_getNumMeshlets      (uint subdiv)   { return (1u << ((max(3, subdiv)-3) * 2)); }

// larger micromeshes may be split due to meshlet being able to do 64 triangles max
// precomputed data for those cases use different offsets
// offset is 0 for level 3 or less:  1 meshlet
//           1 for level 4        :  4 meshlets
//           5 for level 5        : 16 meshlets
uint subdiv_getPartOffset(uint subdiv, uint partID) {
  return ((subdiv == 5 ? 5 : (subdiv == 4 ? 1 : 0)) + partID);
}

// HLSL doesn't like ^^
#ifndef BOOL_XOR
#define BOOL_XOR ^^
#endif

/////////////////////////////////////////////////////
// Bird Curve Indexing

// Compute 2 16-bit prefix XORs in a 32-bit register
uint bird_prefixEor2(uint x)
{
  x ^= (x >> 1) & 0x7fff7fff;
  x ^= (x >> 2) & 0x3fff3fff;
  x ^= (x >> 4) & 0x0fff0fff;
  x ^= (x >> 8) & 0x00ff00ff;

  return x;
}

// Extract even bits
uint bird_extractEvenBits(uint x)
{
  x &= 0x55555555;
  x = (x | (x >> 1)) & 0x33333333;
  x = (x | (x >> 2)) & 0x0f0f0f0f;
  x = (x | (x >> 4)) & 0x00ff00ff;
  x = (x | (x >> 8)) & 0x000fffff;

  return x;
}

// Interleave 16 even bits from x with 16 odd bits from y
uint bird_interleaveBits2(uint x, uint y)
{
  x = (x & 0xffff) | (y << 16);
  x = ((x >> 8) & 0x0000ff00) | ((x << 8) & 0x00ff0000) | (x & 0xff0000ff);
  x = ((x >> 4) & 0x00f000f0) | ((x << 4) & 0x0f000f00) | (x & 0xf00ff00f);
  x = ((x >> 2) & 0x0c0c0c0c) | ((x << 2) & 0x30303030) | (x & 0xc3c3c3c3);
  x = ((x >> 1) & 0x22222222) | ((x << 1) & 0x44444444) | (x & 0x99999999);

  return x;
}

// Compute index of a single triplet of compression coefficients from triangle's barycentric coordinates
// Assumes u, v and w have only 16 valid bits in the lsbs (good for subdivision depths up to 64K segments per edge)
uint bird_getTripletIndex(uint w, uint u, uint v, uint level)
{
  const uint coordMask = ((1U << level) - 1);

  uint b0 = ~(u ^ w) & coordMask;
  uint t  = (u ^ v) & b0;
  uint c  = (((u & v & w) | (~u & ~v & ~w)) & coordMask) << 16;
  uint f  = bird_prefixEor2(t | c) ^ u; 
  uint b1 = (f & ~b0) | t;
  uint dist = bird_interleaveBits2(b0, b1);

  // Adjust computed distance accounting for "skipped" triangles on the bird curve
  f >>= 16;
  b0 <<= 1;
  return (dist + (b0 & ~f) - (b0 & f)) >> 3; 
}

uvec3 bird_getUVW(uint dist)
{
  uint b0 = bird_extractEvenBits(dist);
  uint b1 = bird_extractEvenBits(dist >> 1);

  uint fx = bird_prefixEor2(b0);
  uint fy = bird_prefixEor2(b0 & ~b1);

  uvec3 uvw;
  uint t = fy ^ b1;
  uvw.x = (fx & ~t) | (b0 & ~t) | (~b0 & ~fx & t);
  uvw.y = fy ^ b0;
  uvw.z = (~fx & ~t) | (b0 & ~t) | (~b0 & fx & t);
  
  return uvw;
}

/////////////////////////////////////////////////////
// u-major indexing
/* 
 *  w/a - e2 - v/c
 *   |       /
 *   |      /
 *   e0    e1
 *   |    /
 *   |   /
 *   |  /
 *   u/b
 *
 * Elements are stored in this order, with n=5:
 *   00 01 02 03 04
 *   05 06 07 08
 *   09 10 11
 *   12 13
 *   14
 */

uint umajorUV_toLinear(uint numVertsPerEdge, ivec2 uv)
{
  uint x = uv.y;
  uint y = uv.x;
  uint trinum = (y * (y + 1)) / 2;
  return y * (numVertsPerEdge + 1) - trinum + x;
}

ivec2 umajorUV_fromLinear(uint numVertsPerEdge, uint idx)
{
  uint y = uint(floor((-sqrt((2 * numVertsPerEdge + 1) * (2 * numVertsPerEdge + 1) - 8 * float(idx)) + 2 * numVertsPerEdge + 1) / 2));
  uint x = idx - y * (2 * numVertsPerEdge - y + 1) / 2;
  return ivec2(y, x);
}


/////////////////////////////////////////////////////

uint micromesh_getFormatSubdiv(uint formatIdx)
{
  return formatIdx + 3;
}

/////////////////////////////////////////////////////
// micromesh MicromeshBaseTri/MicromeshSubTri

uint  micromesh_getDataSize   (in MicromeshBaseTri baseTri) { return ((uint(baseTri.packedBits) >> MICRO_BASE_FMT_SHIFT) & MICRO_BASE_FMT_MASK) > 0 ? 32 : 16; }
uint  micromesh_getBaseSubdiv (in MicromeshBaseTri baseTri) { return (uint(baseTri.packedBits) >> MICRO_BASE_LVL_SHIFT)  & MICRO_BASE_LVL_MASK; }
uint  micromesh_getBaseTopo   (in MicromeshBaseTri baseTri) { return (uint(baseTri.packedBits) >> MICRO_BASE_TOPO_SHIFT) & MICRO_BASE_TOPO_MASK; }
uint  micromesh_getFormat     (in MicromeshBaseTri baseTri) { return (uint(baseTri.packedBits) >> MICRO_BASE_FMT_SHIFT)  & MICRO_BASE_FMT_MASK; }
uint  micromesh_getSubdiv     (in MicromeshBaseTri baseTri) { return min(micromesh_getFormatSubdiv(micromesh_getFormat(baseTri)),micromesh_getBaseSubdiv(baseTri)); }
 int  micromesh_getCullDist   (in MicromeshBaseTri baseTri)
{ 
  int d = int((uint(baseTri.packedBits) >> MICRO_BASE_CULLDIST_SHIFT) & MICRO_BASE_CULLDIST_MASK);
  int bitsGap = MICRO_UNORM_BITS - MICRO_BASE_CULLDIST_WIDTH;
  return (d << bitsGap) + ((1 << bitsGap) - 1);
}
#if MICRO_DECODER == MICRO_DECODER_BASETRI_MIP_SHUFFLE
uint  micromesh_getDataOffset (in MicromeshBaseTri baseTri) { return ((baseTri.dataOffset >> MICRO_BASE_DATA_SHIFT) & MICRO_BASE_DATA_MASK) * MICRO_BASE_DATA_VALUE_MUL; }
uint  micromesh_getMipOffset  (in MicromeshBaseTri baseTri)
{
  return (bitfieldExtract(baseTri.packedBits, MICRO_BASE_MIPLO_SHIFT, MICRO_BASE_MIPLO_WIDTH) | 
         (bitfieldExtract(baseTri.dataOffset, MICRO_BASE_DATA_MIPHI_SHIFT, MICRO_BASE_DATA_MIPHI_WIDTH) << MICRO_BASE_MIPLO_WIDTH)) * MICRO_BASE_MIP_VALUE_MUL; 
}
#endif

////////////////////////////////////////////////////////////


#if USE_FP16_DISPLACEMENT_MATH 
  #define dispVec2  f16vec2
#else
  #define dispVec2  vec2
#endif

dispVec2 micromesh_concatScaleBias(dispVec2 a, dispVec2 b)
{
  return dispVec2(a.x * b.x, a.y * b.x + b.y);
}


float micromesh_getFloatDisplacement(in int displacement, dispVec2 scale_bias)
{
  uint limit = (1 << MICRO_UNORM_BITS) - 1;
#if USE_FP16_DISPLACEMENT_MATH    
  float16_t unbiased_disp = scale_bias.x * float16_t(float(displacement) * (1.f / float(limit)));
#else
  float     unbiased_disp = float(scale_bias.x) * (float(displacement) * (1.f / float(limit)));
#endif
  float     disp          = float(unbiased_disp) + float(scale_bias.y);
  return disp;
}

