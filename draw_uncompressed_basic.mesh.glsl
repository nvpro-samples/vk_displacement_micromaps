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

#extension GL_NV_mesh_shader : enable

#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_8bit_storage : enable
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_shader_atomic_int64 : enable

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require

#include "common.h"
#include "common_mesh.h"
#include "common_barymap.h"
#include "common_micromesh_uncompressed.h"

layout(local_size_x=MICRO_GROUP_SIZE) in;
layout(max_primitives = MAX_BARYMAP_PRIMITIVES, max_vertices = MAX_BARYMAP_VERTICES, triangles) out;

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
layout(scalar, binding = DRAWUNCOMPRESSED_UBO_MAP) uniform bmapBuffer {
  BaryMapData bmap;
};
layout(scalar, binding = DRAWUNCOMPRESSED_UBO_UNCOMPRESSED) uniform baryBuffer {
  MicromeshUncData microdata;
};

layout(push_constant) uniform pushDraw {
  DrawMicromeshUncPushData push;
};


//////////////////////////////////////////////////////////////////////////
// INPUT

taskNV in Task
{
  uint      baseID;
  uint16_t  prefix[MICRO_TRI_PER_TASK];
} IN;

//////////////////////////////////////////////////////////////////////////
// OUTPUT

#if SURFACEVIS == SURFACEVIS_SHADING
layout(location = 0) out Interpolants
{
  vec3      wPos;
  vec3      bary;
  flat uint tri;
}
OUT[];
#endif

#if (SURFACEVIS == SURFACEVIS_SHADING && USE_MICROVERTEX_NORMALS) \
    || (SURFACEVIS == SURFACEVIS_ANISOTROPY)
layout(location = 3) out PerVertex
{
#if (SURFACEVIS == SURFACEVIS_SHADING && USE_MICROVERTEX_NORMALS)
  uint vidx;
#endif
#if SURFACEVIS == SURFACEVIS_ANISOTROPY
  vec3 vwPos;  // Copy of wPos; aliasing with wPos is unfortunately invalid.
#endif
}
OUTvtx[];
#endif

//////////////////////////////////////////////////////////////////////////

#include "draw_culling.glsl"

//////////////////////////////////////////////////////////////////////////

  // This shader does not do any packing of micromeshes with low subdivision
  // into a single mesh-shader invocation. This yields less performance as we
  // unterutilize the hardware this way (an entire mesh workgroup may generate 
  // only a single triangle).
  // Look at the draw_micromesh_lod shaders which use a more complex setup
  // that does this.

// debug min.max cage
// set to 0 or 1
// or undefine
//#define OVERLAY_MINMAX  1

void main()
{
  uint baseID = gl_WorkGroupID.x;
  uint laneID = gl_SubgroupInvocationID;

  // task emits N meshlets for multiple triangles
  // find original triangle we are from
  uint  prefix = IN.prefix[laneID];
  uvec4 voteID = subgroupBallot(baseID >= prefix);
  uint subID   = subgroupBallotFindMSB(voteID);
  uint meshID  = baseID - subgroupShuffle(prefix, subID);
  
  uint triLocal  = IN.baseID + subID;
  uint tri       = triLocal + push.firstTriangle;
  
  MicromeshUncBaseTri baseTri = microdata.basetriangles.d[triLocal];
  uint baseSubdiv        = baseTri.subdivLevel;
  uint firstValue        = baseTri.firstValue;
  uint firstShadingValue = baseTri.firstShadingValue;
  
#if USE_NON_UNIFORM_SUBDIV
  uint baseConfig = baseSubdiv + MAX_BARYMAP_LEVELS * uint(baseTri.topoBits);
#else
  uint baseConfig = baseSubdiv;
#endif
  
  BaryMapMeshlet header = bmap.levelsUni[baseConfig].meshletHeaders.d[meshID];
  uint numVertices      = header.numVertices;
  uint numPrimitives    = header.numPrimitives;
  uint offsetPrims      = header.offsetPrims;
  uint offsetVertices   = header.offsetVertices;
  uint vertMax          = numVertices-1;
  uint primMax          = numPrimitives-1;
  
  uint  baryMax = (1 << baseSubdiv);
  float baryRcp = 1.0 / float(1 << baseSubdiv);
  
  mat4 worldMatrix = mesh.instances.d[push.instanceID].worldMatrix;
  mat4 worldMatrixIT = transpose(inverse(worldMatrix));


  uvec3 triIndices = uvec3( mesh.indices.d[tri * 3 + 0],
                            mesh.indices.d[tri * 3 + 1],
                            mesh.indices.d[tri * 3 + 2]) + push.firstVertex;
                              
  vec3 v0 = mesh.positions.d[triIndices.x];
  vec3 v1 = mesh.positions.d[triIndices.y];
  vec3 v2 = mesh.positions.d[triIndices.z];
  
  f16vec3 d0 = mesh.dispDirections.d[triIndices.x].xyz;
  f16vec3 d1 = mesh.dispDirections.d[triIndices.y].xyz;
  f16vec3 d2 = mesh.dispDirections.d[triIndices.z].xyz;
  
#if USE_DIRECTION_BOUNDS
  boundsVec2 bounds0 = mesh.dispDirectionBounds.d[triIndices.x];
  boundsVec2 bounds1 = mesh.dispDirectionBounds.d[triIndices.y];
  boundsVec2 bounds2 = mesh.dispDirectionBounds.d[triIndices.z];
  
  v0 = v0 + d0 * bounds0.x;
  v1 = v1 + d1 * bounds1.x;
  v2 = v2 + d2 * bounds2.x;
  
  d0 = d0 * float16_t(bounds0.y);
  d1 = d1 * float16_t(bounds1.y);
  d2 = d2 * float16_t(bounds2.y);
#endif
  
  for (uint32_t runs = 0; runs < uint( MAX_BARYMAP_VERTICES / MICRO_GROUP_SIZE); runs++)
  {
    uint vert     = laneID + runs * MICRO_GROUP_SIZE;
    uint vertRead = min(vert,  vertMax);
    
    uint vertPacked = bmap.levels.d[baseConfig].meshletData.d[offsetVertices + vertRead];
    uvec2 vertUV    = uvec2(vertPacked & 0xFF, (vertPacked >> 8) & 0xFF);
    vec3 bary       = vec3(baryMax - vertUV.x - vertUV.y, vertUV.x, vertUV.y) * baryRcp;
    
    uint  vidx      = vertPacked >> 16;
    
    float dispDistance = loadUncompressedDisplacement(microdata.distancesBits, firstValue, vidx) 
                         * push.scale_bias.x + push.scale_bias.y;
    
    // for tweakable scaling (not compatible with RT)
  #if USE_OVERLAY && defined(OVERLAY_MINMAX)
    dispDistance = OVERLAY_MINMAX * push.scale_bias.x + push.scale_bias.y;
  #endif
    dispDistance = dispDistance * scene.disp_scale + scene.disp_bias;
    
    // compute interpolation
    vec3 dispDirection = getInterpolated(d0, d1, d2, bary);

    vec3 pos      = getInterpolated(v0, v1, v2, bary) + vec3(dispDirection) * dispDistance;
    vec4 wPos     = worldMatrix * vec4(pos,1);
    gl_MeshVerticesNV[vert].gl_Position = scene.viewProjMatrix * wPos;
    
    // tag misses
    //if (floatBitsToUint(dispDistance) == 0x80000000) wPos = vec4(1.0f/0.0f);
  #if SURFACEVIS == SURFACEVIS_SHADING
    OUT[vert].wPos       = wPos.xyz;
    OUT[vert].bary       = bary;
    OUT[vert].tri        = tri;
    #if USE_MICROVERTEX_NORMALS
      OUTvtx[vert].vidx  = firstShadingValue + vidx;
    #endif
  #elif SURFACEVIS == SURFACEVIS_ANISOTROPY
    OUTvtx[vert].vwPos   = wPos.xyz;
  #endif
  }
  
#if SURFACEVIS == SURFACEVIS_VALUERANGE
  float valueRange = getValueRange( dbitsUnpack(microdata.triangleBitsMinMax.d[triLocal * 2 + 0]), 
                                    dbitsUnpack(microdata.triangleBitsMinMax.d[triLocal * 2 + 1]));
#else
  float valueRange = 0;
#endif

#if USE_PRIMITIVE_CULLING
  uint numTrianglesOut = 0;
#else
  uint numTrianglesOut = numPrimitives;
#endif
  for (uint32_t runs = 0; runs < uint( (MAX_BARYMAP_PRIMITIVES + MICRO_GROUP_SIZE - 1)  / MICRO_GROUP_SIZE); runs++)
  {
    uint prim     = laneID + runs * MICRO_GROUP_SIZE;
    uint primRead = min(prim,  primMax);
    
    uint  pidxPacked = bmap.levels.d[baseConfig].meshletData.d[offsetPrims + primRead];
    uvec3 pidx       = uvec3(pidxPacked & 0xFF, (pidxPacked >> 8) & 0xFF, (pidxPacked >> 16));
    
    bool visible = prim <= primMax;
  #if USE_PRIMITIVE_CULLING
    if (visible) {
      RasterVertex a = getRasterVertex(gl_MeshVerticesNV[pidx.x].gl_Position);
      RasterVertex b = getRasterVertex(gl_MeshVerticesNV[pidx.y].gl_Position);
      RasterVertex c = getRasterVertex(gl_MeshVerticesNV[pidx.z].gl_Position);
      
      visible = testTriangle(a,b,c, 1.0);
      //visible = true;
    }
    uvec4 voteVis = subgroupBallot(visible);
    uint  primOut = numTrianglesOut + subgroupBallotExclusiveBitCount(voteVis);
    
    numTrianglesOut += subgroupBallotBitCount(voteVis);
  #else
    uint primOut = prim;
  #endif
    
    if (visible) 
    {
      gl_PrimitiveIndicesNV[primOut * 3 + 0] = pidx.x;
      gl_PrimitiveIndicesNV[primOut * 3 + 1] = pidx.y;
      gl_PrimitiveIndicesNV[primOut * 3 + 2] = pidx.z;
    
      chooseSurfaceVisOutput(gl_MeshPrimitivesNV[primOut].gl_PrimitiveID, 
          tri, prim + meshID * numPrimitives, 0, 0, valueRange, baseSubdiv, baseSubdiv);
    }
  }

  if (laneID == 0)
  {
    gl_PrimitiveCountNV = numTrianglesOut;
  #if USE_STATS
    atomicAdd(stats.triangles, numTrianglesOut);
  #endif
  }
}