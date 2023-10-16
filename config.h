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

//////////////////////////////////////////////////////////////////////////
// this file is included by C++ and GLSL

#ifndef _CONFIG_H_
#define _CONFIG_H_

/////////////////////////////////////////////////////

#define API_SUPPORTED_SETUP_ONLY  1

// warning raising this beyond 5 has consequences on storage bits
// and needs manual changes in code
#define MAX_BASE_SUBDIV           5

#define BOUNDS_AS_FP32            1

#define ATOMIC_LAYERS  2

// must not change
#define SUBGROUP_SIZE  32

// Different surface visualization modes.
#define SURFACEVIS_SHADING      0       // Default shading.
#define SURFACEVIS_ANISOTROPY   1       // gl_PrimitiveID is not used; additional pervertexNV attributes; batlow coloring.
#define SURFACEVIS_BASETRI      2       // gl_PrimitiveID holds base triangle index; colorizePrimitive coloring.
#define SURFACEVIS_MICROTRI     3       // gl_PrimitiveID holds unique index per microtriangle; colorizePrimitive coloring. 0 for standard renderer.
#define SURFACEVIS_LOCALTRI     4       // gl_PrimitiveID holds local index of meshlet triangle; colorizePrimitive coloring. 0 for standard renderer.
#define SURFACEVIS_FORMAT       5       // gl_PrimitiveID holds index of encoding format used; colorizePrimitive coloring. 0 for non-umesh renderers.
#define SURFACEVIS_LODBIAS      6       // gl_PrimitiveID holds lod bias. custom hue2rgb coloring; valid for umesh-lod-renderers only.
#define SURFACEVIS_VALUERANGE   7       // gl_PrimitiveID holds effective base triangle range compared to mesh value range.
#define SURFACEVIS_BASESUBDIV   8       // gl_PrimitiveID holds base triangle subdiv level
#define SURFACEVIS_LODSUBDIV    9       // gl_PrimitiveID holds post lod base triangle subdiv level

#define CLEAR_COLOR     0.1, 0.13, 0.15,0

///////////////////////////////////////////////////
#if defined(__cplusplus)

  #include "half.hpp"
  #include <stddef.h>
  
  enum ModelType
  {
    MODEL_LO,
    MODEL_DISPLACED,
    NUM_MODELTYPES,
    MODEL_SHELL,
  };

  // few more status prints
  extern bool     g_verbose;
  // allow enabling raytracing extension for micromesh
  // if true then codepaths assume native extension exists and rely on it
  // if false we still do some fake setup work but the image will be the basemesh alone
  extern bool     g_enableMicromeshRTExtensions;
  // number of default processing threads
  extern uint32_t g_numThreads;

  class float16_t
  {
  private:
    glm::detail::hdata h = 0;

  public:
    float16_t() {}
    float16_t(float f) { h = glm::detail::toFloat16(f); }

    operator float() const { return glm::detail::toFloat32(h); }
  };

  struct f16vec2
  {
    float16_t x;
    float16_t y;
  };

  struct f16vec4
  {
    float16_t x;
    float16_t y;
    float16_t z;
    float16_t w;
  };

  struct u16vec2
  {
    uint16_t x;
    uint16_t y;
  };
  struct u16vec4
  {
    uint16_t x;
    uint16_t y;
    uint16_t z;
    uint16_t w;

    uint16_t&  operator[](size_t i) { return (&x)[i]; }
  };
  struct u8vec2
  {
    uint8_t x;
    uint8_t y;

    uint8_t&  operator[](size_t i) { return (&x)[i]; }
  };
  struct u8vec4
  {
    uint8_t x;
    uint8_t y;
    uint8_t z;
    uint8_t w;

    uint8_t&  operator[](size_t i) { return (&x)[i]; }
  };
#else

  uint encodeMinMaxFp32(float val) {
    uint bits = floatBitsToUint(val); 
    bits ^= (int(bits) >> 31) | 0x80000000u; 
    return bits;
  }
  
  float decodeMinMaxFp32(uint bits) {
    bits ^= ~(int(bits) >> 31) | 0x80000000u;
    return uintBitsToFloat(bits);
  }
  
  const float FLT_MAX = 3.402823466e+38F;
  const float FLT_EPSILON = 1.192092896e-07F;

#endif

#endif


