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
#extension GL_NV_mesh_shader : enable

#include "common.h"
#include "common_mesh.h"
#include "common_barymap.h"
#include "common_micromesh_uncompressed.h"

////////////////////////////////////////////////////////////////

layout(scalar, binding = DRAWUNCOMPRESSED_UBO_VIEW) uniform sceneBuffer {
  SceneData scene;
  SceneData sceneLast;
};
layout(scalar, binding = DRAWUNCOMPRESSED_SSBO_STATS) coherent buffer statsBuffer {
  ShaderStats stats;
};
layout(scalar, binding = DRAWUNCOMPRESSED_UBO_MESH) uniform meshBuffer {
  MeshData mesh;
};
layout(scalar, binding = DRAWUNCOMPRESSED_UBO_UNCOMPRESSED) uniform baryBuffer {
  MicromeshUncData microdata;
};
#if USE_TEXTURE_NORMALS
layout(binding = 0, set=DSET_TEXTURES) uniform sampler2D tex2Ds[];
#endif

layout(push_constant) uniform pushDraw {
  DrawMicromeshUncPushData push;
};

////////////////////////////////////////////////////////////////

#if SURFACEVIS == SURFACEVIS_SHADING
layout(location = 0) in Interpolants
{
  vec3      wPos;
  vec3      bary;
  flat uint tri;
}
IN;
#endif

#if (SURFACEVIS == SURFACEVIS_SHADING && USE_MICROVERTEX_NORMALS) \
    || (SURFACEVIS == SURFACEVIS_ANISOTROPY)
layout(location = 3) in pervertexNV PerVertex
{
#if (SURFACEVIS == SURFACEVIS_SHADING && USE_MICROVERTEX_NORMALS)
  uint vidx;
#endif
#if SURFACEVIS == SURFACEVIS_ANISOTROPY
  vec3 vwPos;  // Copy of wPos; aliasing with wPos is unfortunately invalid.
#endif
}
INvtx[3];
#endif

////////////////////////////////////////////////////////////////

layout(location=0,index=0) out vec4 out_Color;

////////////////////////////////////////////////////////////////

#include "draw_shading.glsl"

void main()
{
#if SURFACEVIS == SURFACEVIS_SHADING
  {
    uint tri         = IN.tri;
    uvec3 triIndices = uvec3( mesh.indices.d[tri * 3 + 0],
                              mesh.indices.d[tri * 3 + 1],
                              mesh.indices.d[tri * 3 + 2]) + push.firstVertex;
    
    // Generate vertices
    mat4 worldMatrix = mesh.instances.d[push.instanceID].worldMatrix;
    mat4 worldMatrixIT = transpose(inverse(worldMatrix));

    vec3 wPos    = IN.wPos;
    vec2 tex     = getInterpolatedArray(mesh.tex0s.d, triIndices, IN.bary);
    
  #if USE_FACET_SHADING == 1
    vec3 wNormal      = -cross(dFdx(IN.wPos), dFdy(IN.wPos));
  #elif USE_TEXTURE_NORMALS
    vec3 wNormal      = mat3(worldMatrixIT) * getInterpolatedArrayUvec(mesh.normals.d,    triIndices, IN.bary);
    vec3 wTangent     = mat3(worldMatrixIT) * getInterpolatedArrayUvec(mesh.tangents.d,   triIndices, IN.bary);
    vec3 wBitangent   = mat3(worldMatrixIT) * getInterpolatedArrayUvec(mesh.bitangents.d, triIndices, IN.bary);
  #elif USE_MICROVERTEX_NORMALS
    uvec3 baryIndices = uvec3(INvtx[0].vidx, INvtx[1].vidx, INvtx[2].vidx);
    vec3 wNormal      = mat3(worldMatrixIT) * getInterpolatedArrayUvec(microdata.attrNormals.d, baryIndices, vec3(gl_BaryCoordNV));
  #else
    vec3 wNormal      = mat3(worldMatrixIT) * getInterpolatedArrayUvec(mesh.normals.d,    triIndices, IN.bary);
  #endif

    out_Color = shading(push.instanceID, gl_PrimitiveID, wPos, tex, wNormal
    #if USE_TEXTURE_NORMALS
      , wTangent, wBitangent
    #endif
    );
  }
#else // Surface debug visualization
  out_Color = surfaceVisShading(
  #if SURFACEVIS == SURFACEVIS_ANISOTROPY
      INvtx[0].vwPos, INvtx[1].vwPos, INvtx[2].vwPos,
  #endif
      gl_PrimitiveID);

#endif
}