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

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference2 : enable

#if SUPPORTS_MICROMESH_RT
#extension GL_NV_displacement_micromap : enable
#endif

#include "common.h"
#include "common_mesh.h"

#if USE_MICROVERTEX_NORMALS
#include "common_micromesh_compressed_rt.h"
#endif

////////////////////////////////////////////////////////////////

layout(scalar, binding = DRAWRAY_UBO_VIEW) uniform sceneBuffer {
  SceneData scene;
  SceneData sceneLast;
};
layout(scalar, binding = DRAWRAY_SSBO_STATS) coherent buffer statsBuffer {
  ShaderStats stats;
};

layout(scalar, binding = DRAWRAY_UBO_MESH) uniform meshBuffer {
  MeshData mesh;
};

#if USE_TEXTURE_NORMALS
layout(binding = 0, set=DSET_TEXTURES) uniform sampler2D tex2Ds[];
#endif

#if USE_MICROVERTEX_NORMALS
layout(scalar, binding = DRAWRAY_SSBO_RTINSTANCES) readonly buffer rtdataInstBuffer {
  MicromeshRtAttributes instanceAttributes[];
};
layout(scalar, binding = DRAWRAY_SSBO_RTDATA) readonly buffer rtdataBuffer {
  MicromeshRtData data;
};
#endif

////////////////////////////////////////////////////////////////

hitAttributeEXT vec2 hitAttribs;

layout(location = 0) rayPayloadInEXT uvec2 rayHit;

////////////////////////////////////////////////////////////////

#define DISABLE_SHADING_SPECIALS 1
#include "draw_shading.glsl"

#if USE_MICROVERTEX_NORMALS
vec2 baseToMicro(vec2 barycentrics[3], vec2 p)
{
    vec2  ap   = p - barycentrics[0];
    vec2  ab   = barycentrics[1] - barycentrics[0];
    vec2  ac   = barycentrics[2] - barycentrics[0];
    float rdet = 1.f / ( ab.x * ac.y - ab.y * ac.x );
    return vec2(ap.x * ac.y - ap.y * ac.x,
                ap.y * ab.x - ap.x * ab.y) * rdet;
}

uint subdiv_getNumVertsPerEdge  (uint subdiv)   { return (1u << subdiv) + 1; }

uint umajorUV_toLinear(uint numVertsPerEdge, ivec2 uv)
{
  uint x = uv.y;
  uint y = uv.x;
  uint trinum = (y * (y + 1)) / 2;
  return y * (numVertsPerEdge + 1) - trinum + x;
}

uint bary2index(vec2 uv, uint level)
{
  uint valueIdx = umajorUV_toLinear(subdiv_getNumVertsPerEdge(level), ivec2(uv * (1 << level)));
  #if !UNCOMPRESSED_UMAJOR
  valueIdx      = data.umajor2bmap[level].d[valueIdx];
  #endif
  return valueIdx;
}
#endif

vec4 raytraceShading(uint instanceID, uint primID, vec2 baryattribs, inout vec3 wPos)
{
  bool isMicro = false;
  
#if SUPPORTS_MICROMESH_RT
  
  // we use the SBT instance offset to ensure this shader permutation
  // is only used with displaced micromeshes
  // look for `instance.instanceShaderBindingTableRecordOffset` in
  // `meshset_vk.cpp`
  isMicro = true;
  
  
  /*
    // an alternative would be a runtime check
  
    uint hitKind = gl_HitKindEXT;
    if (hitKind == gl_HitKindFrontFacingMicroTriangleNV || hitKind == gl_HitKindBackFacingMicroTriangleNV)
    {
      isMicro = true;
    }
  */
#endif

  InstanceData inst  = mesh.instances.d[instanceID];
  mat4 worldMatrix   = inst.worldMatrix;
  mat3 worldMatrixIT = mat3(transpose(inverse(worldMatrix)));
  
  const vec3 baryBase = vec3(1.0 - baryattribs.x - baryattribs.y, baryattribs.x, baryattribs.y);
  
  uvec3 triIndices;
  triIndices.x = mesh.indices.d[primID * 3 + 0 + inst.firstIndex] + inst.firstVertex;
  triIndices.y = mesh.indices.d[primID * 3 + 1 + inst.firstIndex] + inst.firstVertex;
  triIndices.z = mesh.indices.d[primID * 3 + 2 + inst.firstIndex] + inst.firstVertex;

  vec3  v0;
  vec3  v1;
  vec3  v2;
  
#if SUPPORTS_MICROMESH_RT
  if (isMicro)
  {
    v0   = gl_HitMicroTriangleVertexPositionsNV[0];
    v1   = gl_HitMicroTriangleVertexPositionsNV[1];
    v2   = gl_HitMicroTriangleVertexPositionsNV[2];
  }
  else
#endif
  {
    v0   = mesh.positions.d[triIndices.x];
    v1   = mesh.positions.d[triIndices.y];
    v2   = mesh.positions.d[triIndices.z];
  }
  
  float dummy = isMicro ? 0.0000001 : 0.0;

#if SURFACEVIS != SURFACEVIS_SHADING
  // Do surface debug visualization
  return surfaceVisShading(
  #if SURFACEVIS == SURFACEVIS_ANISOTROPY
      v0, v1, v2,
  #endif
      int(primID)) + dummy;
#else 
  // Do regular shading
  vec3 n0 = unpackUvec(mesh.normals.d[triIndices.x]);
  vec3 n1 = unpackUvec(mesh.normals.d[triIndices.y]);
  vec3 n2 = unpackUvec(mesh.normals.d[triIndices.z]);
  
#if USE_FACET_SHADING
  vec3 wNormal    = worldMatrixIT * (normalize(cross((v1 - v0), (v2 - v0))));
  vec2 tex        = vec2(0);
#elif USE_MICROVERTEX_NORMALS && SUPPORTS_MICROMESH_RT
  vec3 oNormal;
  if (isMicro)
  {
    vec2 baryMicro     = baseToMicro(gl_HitMicroTriangleVertexBarycentricsNV, baryattribs);
    
    MicromeshRtAttributes    rtAttribs = instanceAttributes[instanceID];
    MicromeshRtAttrTri       attribTri = rtAttribs.attrTriangles.d[primID];
    
    uint subdivLevel   = attribTri.subdivLevel;
    uint firstValue    = attribTri.firstValue;
    uvec3 microIndices = uvec3(bary2index(gl_HitMicroTriangleVertexBarycentricsNV[0], subdivLevel),
                               bary2index(gl_HitMicroTriangleVertexBarycentricsNV[1], subdivLevel),
                               bary2index(gl_HitMicroTriangleVertexBarycentricsNV[2], subdivLevel));
    uvec3 baryIndices  = uvec3(firstValue) + microIndices;
    oNormal = getInterpolatedArrayUvec(rtAttribs.attrNormals.d, baryIndices, vec3(1.0 - baryMicro.x - baryMicro.y, baryMicro));
  }
  else {
    oNormal = getInterpolated(n0, n1, n2, baryBase);
  }
  vec3 wNormal       = worldMatrixIT * oNormal;
  vec2 tex           = vec2(0);
  
#elif USE_TEXTURE_NORMALS
  vec2 tex        = getInterpolatedArray(mesh.tex0s.d, triIndices, baryBase);
  vec3 wNormal    = worldMatrixIT * getInterpolated(n0, n1, n2, baryBase);
  vec3 wTangent   = worldMatrixIT * getInterpolatedArrayUvec(mesh.tangents.d, triIndices, baryBase);
  vec3 wBitangent = worldMatrixIT * getInterpolatedArrayUvec(mesh.bitangents.d, triIndices, baryBase);
#else
  vec3 wNormal    = worldMatrixIT * getInterpolated(n0, n1, n2, baryBase);
  vec2 tex        = vec2(0);
#endif

  return shading(instanceID, int(primID), wPos, tex, wNormal
#if USE_TEXTURE_NORMALS
  , wTangent, wBitangent
#endif
  );
#endif
}

void main()
{ 
  float t      = gl_HitTEXT;
  vec3 wPos    = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * t;
  vec4 color   = raytraceShading(gl_InstanceID, gl_PrimitiveID, hitAttribs, wPos);
  vec4 hPos    = (scene.viewProjMatrix * vec4(wPos,1));
  float depth  = (hPos.z/hPos.w);
  
  rayHit.x = packUnorm4x8(color);
  rayHit.y = floatBitsToUint(depth);
}


