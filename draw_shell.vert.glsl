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

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : require

#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_buffer_reference2 : enable

#extension GL_NV_fragment_shader_barycentric : enable

#include "common.h"
#include "common_mesh.h"

////////////////////////////////////////////////////////////////

layout(scalar, binding = DRAWSTD_UBO_VIEW) uniform sceneBuffer {
  SceneData scene;
  SceneData sceneLast;
};
layout(scalar, binding = DRAWSTD_SSBO_STATS) coherent buffer statsBuffer {
  ShaderStats stats;
};
layout(scalar, binding = DRAWSTD_UBO_MESH) uniform meshBuffer {
  MeshData mesh;
};


layout(push_constant) uniform pushDraw {
  DrawPushData push;
};

////////////////////////////////////////////////////////////////

layout(location=0) out Interpolants {
  float shell;
} OUT;

////////////////////////////////////////////////////////////////

// to avoid multiple shaders the shell rendering is done as follows
// - we render triangles with polygon mode lines
// - object is rendered in two passes with the same shader, but in different modes
// - first, the object is drawn as triangles with indexbuffer and instancing twice
//   gl_InstanceIndex 0 is the min shell, 1 the max shell
// - second, the object is drawn as triangles, no indexbuffer, one per vertex
//   this is the pass that generates the lines along direction vectors
//   we create one triangle per direction vector where 1 vertex is atop
//   and two at bottom. We nudge the bottom vertex by one pixel


void main()
{
  uint  vertexID;
  float shell;
  float nudge = 0;
  
  if (push.shellDir != 0)
  {
    vertexID = push.firstVertex + (gl_VertexIndex/3);
    shell    = float(gl_VertexIndex & 1);
    // * 2 because of -1,1 clipspace
    nudge    = (gl_VertexIndex % 3) == 2 ? 2.0/float(scene.viewportf.x) : 0.0f;
  }
  else
  {
    vertexID = push.firstVertex + gl_VertexIndex;
    shell    = float(gl_InstanceIndex);
  }

  vec3 oPos    = mesh.positions.d[vertexID];
  f16vec3 oDir = mesh.dispDirections.d[vertexID].xyz;
  
#if USE_DIRECTION_BOUNDS
  boundsVec2 bounds0 = mesh.dispDirectionBounds.d[vertexID];
  
  oPos = oPos + oDir * bounds0.x;
  oDir = oDir * float16_t(bounds0.y);
#endif

  mat4 worldMatrix   = mesh.instances.d[push.instanceID].worldMatrix;
  
  float disp = mix(push.shellMin, push.shellMax, shell);
  oPos += oDir * disp;
  
  vec4 wPos = worldMatrix * vec4(oPos,1);

  OUT.shell = shell;
  gl_Position = scene.viewProjMatrix * wPos;
  // in the shellDir case we have 3 vertices per triangle (one triangle per line)
  // vertex 0 is bottom, vertex 1 is tip, vertex 2 is bottom again
  // we nudge vertex 2 by one pixel
  // * hPos.w to get nudge from screenspace into clipspace  
  gl_Position.x += nudge * gl_Position.w;
}