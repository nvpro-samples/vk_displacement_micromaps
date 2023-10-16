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
  vec3 wPos;
  vec3 wNormal;
  vec2 tex;
#if USE_TEXTURE_NORMALS
  vec3 wTangent;
  vec3 wBitangent;
#endif
} OUT;

#if SURFACEVIS == SURFACEVIS_ANISOTROPY
layout(location=5) out PerVertex {
  vec3 vwPos; // Copy of wPos; aliasing with wPos is not permitted
} OUTvtx;
#endif

////////////////////////////////////////////////////////////////

void main()
{
  mat4 worldMatrix = mesh.instances.d[push.instanceID].worldMatrix;
  mat4 worldMatrixIT = transpose(inverse(worldMatrix));
  vec4 wPos   = worldMatrix * vec4(mesh.positions.d[push.firstVertex + gl_VertexIndex],1);
  OUT.wPos    = wPos.xyz;
  OUT.wNormal = mat3(worldMatrixIT) * unpackUvec(mesh.normals.d[push.firstVertex + gl_VertexIndex]);
  OUT.tex     = mesh.tex0s.d[push.firstVertex + gl_VertexIndex];
#if USE_TEXTURE_NORMALS
  OUT.wTangent   = mat3(worldMatrixIT) * unpackUvec(mesh.tangents.d[push.firstVertex + gl_VertexIndex]);
  OUT.wBitangent = mat3(worldMatrixIT) * unpackUvec(mesh.bitangents.d[push.firstVertex + gl_VertexIndex]);
#endif
#if SURFACEVIS == SURFACEVIS_ANISOTROPY
  OUTvtx.vwPos = wPos.xyz;
#endif
  gl_Position = scene.viewProjMatrix * wPos;
}