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
#extension GL_EXT_nonuniform_qualifier : require

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
#if USE_TEXTURE_NORMALS
layout(binding = 0, set=DSET_TEXTURES) uniform sampler2D tex2Ds[];
#endif

layout(push_constant) uniform pushDraw {
  DrawPushData push;
};

////////////////////////////////////////////////////////////////

layout(location=0) in Interpolants {
  vec3 wPos;
  vec3 wNormal;
  vec2 tex;
#if USE_TEXTURE_NORMALS
  vec3 wTangent;
  vec3 wBitangent;
#endif
} IN;

#if SURFACEVIS == SURFACEVIS_ANISOTROPY
layout(location=5) in pervertexNV PerVertex {
  vec3 vwPos;
} INvtx[3];
#endif

////////////////////////////////////////////////////////////////

layout(location=0,index=0) out vec4 out_Color;

////////////////////////////////////////////////////////////////

#include "draw_shading.glsl"

void main()
{
#if SURFACEVIS == SURFACEVIS_SHADING
#if USE_FACET_SHADING
  vec3 wNormal = -cross(dFdx(IN.wPos), dFdy(IN.wPos));
#else
  vec3 wNormal = IN.wNormal;
#endif

  out_Color = shading(push.instanceID, gl_PrimitiveID, IN.wPos, IN.tex, wNormal
#if USE_TEXTURE_NORMALS
  ,IN.wTangent ,IN.wBitangent
#endif
  );
#else // Surface debug visualization
  out_Color = surfaceVisShading(
#if SURFACEVIS == SURFACEVIS_ANISOTROPY
      INvtx[0].vwPos, INvtx[1].vwPos, INvtx[2].vwPos,
#endif
      gl_PrimitiveID);
#endif
}