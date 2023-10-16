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

#ifndef _COMMON_MESH_H_
#define _COMMON_MESH_H_

//////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
namespace microdisp {
using namespace nvmath;
#endif

///////////////////////////////////////////////////////////////////
// InstanceData

struct InstanceData
{
  mat4  worldMatrix;
  mat4  worldMatrixI;
  
  vec4  color;
  uvec3 _pad;
  uint  normalMapID;
  
  uint  firstVertex;
  uint  firstIndex;
  uint  lodSubdiv;
  float lodRange;
  
  vec4  bboxMin;
  vec4  bboxMax;
  vec4  lodSphere;
};
#ifndef __cplusplus
layout(buffer_reference, buffer_reference_align = 16, scalar) restrict readonly buffer InstanceDatas_in
{
    InstanceData d[];
};
#endif

///////////////////////////////////////////////////////////////////
// MeshData

struct MeshData
{
    BUFFER_REF(uints_in)          indices;
    BUFFER_REF(vec3s_in)          positions;
    BUFFER_REF(uints_in)          normals;
    BUFFER_REF(uints_in)          tangents;
    BUFFER_REF(uints_in)          bitangents;
    BUFFER_REF(vec2s_in)          tex0s;
    BUFFER_REF(f16vec4s_in)       dispDirections;
    BUFFER_REF(boundsVec2s_in)    dispDirectionBounds;
    BUFFER_REF(uint8s_in)         dispDecimateFlags;
    BUFFER_REF(InstanceDatas_in)  instances;
};


///////////////////////////////////////////////////////////////////

struct DrawPushData
{
  uint  firstVertex;
  uint  firstIndex;
  uint  instanceID;
  uint  triangleMax;
  uint  shellDir;
  float shellMin;
  float shellMax;
};


#ifdef __cplusplus
}
#endif
#endif