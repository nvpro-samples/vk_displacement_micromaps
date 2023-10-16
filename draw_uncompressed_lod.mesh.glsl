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
#include "micromesh_binpack_decl.glsl"
#include "micromesh_decoder_config.glsl"

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
  MicroBinPack binpack;
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

// MicroBinPack functions need to reference target variables directly via macros
#define MICROBINPACK_IN             IN.binpack
#include "micromesh_binpack.glsl"

//////////////////////////////////////////////////////////////////////////

uint subdiv_getNumVertsPerEdge  (uint subdiv)   { return (1 << subdiv) + 1; }
uint subdiv_getNumTriangles     (uint subdiv)   { return (1 << (subdiv * 2));}
uint subdiv_getNumSegments      (uint subdiv)   { return (1 << subdiv); }
uint subdiv_getNumVerts         (uint subdiv)   { uint numVertsPerEdge = subdiv_getNumVertsPerEdge(subdiv); return (numVertsPerEdge * (numVertsPerEdge + 1)) / 2; }

void main()
{
  //////////////////////////////////////
  // Initial Configuration Phase
  
  uint wgroupID   = gl_WorkGroupID.x;
  uint laneID     = gl_SubgroupInvocationID;
  
  MicroDecoderConfig cfg = MicroBinPack_subgroupUnpack(wgroupID);
  uint targetSubdiv      = cfg.targetSubdiv;
  
#if !USE_PRIMITIVE_CULLING
  uint  validCount = 1;
  if (targetSubdiv < 3){
    uvec4 validVote  = subgroupBallot(cfg.packThreadID == 0 && cfg.valid);
    validCount       = subgroupBallotBitCount(validVote);
  }
#endif

  uint triLocal    = cfg.microID;
  uint tri         = triLocal + push.firstTriangle;
  uint meshID      = cfg.partID;
  
  MicromeshUncBaseTri baseTri = microdata.basetriangles.d[triLocal];
  uint baseSubdiv        = baseTri.subdivLevel;
  uint firstValue        = baseTri.firstValue;
  uint firstShadingValue = baseTri.firstShadingValue;
  
#if USE_NON_UNIFORM_SUBDIV && 1
  uint targetConfig = targetSubdiv + ((baseSubdiv == targetSubdiv) ? MAX_BARYMAP_LEVELS * uint(baseTri.topoBits) : 0);
  uint baseConfig   = baseSubdiv   + ((baseSubdiv == targetSubdiv) ? MAX_BARYMAP_LEVELS * uint(baseTri.topoBits) : 0);
#else
  uint targetConfig = targetSubdiv;
  uint baseConfig   = baseSubdiv;
#endif
  
  BaryMapMeshlet header = bmap.levels.d[targetConfig].meshletHeaders.d[meshID];
  uint numVertices      = header.numVertices;
  uint numPrimitives    = header.numPrimitives;
  uint offsetPrims      = header.offsetPrims;
  uint offsetVertices   = header.offsetVertices;
  uint vertMax          = numVertices-1;
  uint primMax          = numPrimitives-1;
  
  uint vertStride       = targetSubdiv < 3 ? subdiv_getNumVerts(targetSubdiv) : 0;
  
  // vertex coords are pushed out by the subdiv delta
  offsetVertices += (baseSubdiv - targetSubdiv) * numVertices;
  
  uint  baryMax = 1 << baseSubdiv;
  float baryRcp = 1.0 / float(baryMax);
  
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

  uint iterationCount = SUBGROUP_SIZE == 32 ? 2 : 1;
  
  for (uint32_t vertIter = 0; vertIter < iterationCount; vertIter++)
  {
    uint vert      = cfg.packThreadID + vertIter * cfg.packThreads;
    uint vertRead  = min(vert,  vertMax);
    uint vertOut   = vert + cfg.packID * vertStride;
    bool vertValid = vert < numVertices && cfg.valid;
    
    if (!vertValid) continue;
    
    uint vertPacked = bmap.levels.d[targetConfig].meshletData.d[offsetVertices + vertRead];
    uvec2 vertUV    = uvec2(vertPacked & 0xFF, (vertPacked >> 8) & 0xFF);
    vec3 bary       = vec3(baryMax - vertUV.x - vertUV.y, vertUV.x, vertUV.y) * baryRcp;
    
    uint  vidx      = vertPacked >> 16;
    
    float dispDistance = loadUncompressedDisplacement(microdata.distancesBits, firstValue, vidx)
                         * push.scale_bias.x + push.scale_bias.y;
   
    // for tweakable scaling (not compatible with RT)
    dispDistance = dispDistance * scene.disp_scale + scene.disp_bias;
    
    // compute interpolation
    vec3 dispDirection = getInterpolated(d0, d1, d2, bary);

    vec3 pos      = getInterpolated(v0, v1, v2, bary) + vec3(dispDirection) * dispDistance;
    vec4 wPos     = worldMatrix * vec4(pos,1);
    gl_MeshVerticesNV[vertOut].gl_Position = scene.viewProjMatrix * wPos;
    
  #if SURFACEVIS == SURFACEVIS_SHADING
    OUT[vertOut].wPos       = wPos.xyz;
    OUT[vertOut].bary       = bary;
    OUT[vertOut].tri        = tri;
    #if USE_MICROVERTEX_NORMALS
      OUTvtx[vertOut].vidx  = firstShadingValue + vidx;
    #endif
  #elif SURFACEVIS == SURFACEVIS_ANISOTROPY
    OUTvtx[vertOut].vwPos   = wPos.xyz;
  #endif
  }
  
#if SURFACEVIS == SURFACEVIS_VALUERANGE
  float valueRange = getValueRange( dbitsUnpack(microdata.triangleBitsMinMax.d[triLocal * 2 + 0]), 
                                    dbitsUnpack(microdata.triangleBitsMinMax.d[triLocal * 2 + 1]));
#else
  float valueRange = 0;
#endif

  uint numTrianglesOut = 0;

  for (uint32_t primIter = 0; primIter < iterationCount; primIter++)
  {
    uint prim     = cfg.packThreadID + primIter * cfg.packThreads;
    uint primRead = min(prim,  primMax);
    
    uint  pidxPacked = bmap.levels.d[targetConfig].meshletData.d[offsetPrims + primRead];
    uvec3 pidx       = uvec3(pidxPacked & 0xFF, (pidxPacked >> 8) & 0xFF, (pidxPacked >> 16));
    pidx            += (cfg.packID * vertStride);
    
    bool visible = prim <= primMax && cfg.valid;
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
    uint primOut = prim + cfg.packID * numPrimitives;
  #endif
    
    if (visible) 
    {
      gl_PrimitiveIndicesNV[primOut * 3 + 0] = pidx.x;
      gl_PrimitiveIndicesNV[primOut * 3 + 1] = pidx.y;
      gl_PrimitiveIndicesNV[primOut * 3 + 2] = pidx.z;
    
      chooseSurfaceVisOutput(gl_MeshPrimitivesNV[primOut].gl_PrimitiveID, 
          tri, prim + meshID * numPrimitives, 0, baseSubdiv - cfg.targetSubdiv, valueRange, baseSubdiv, cfg.targetSubdiv);
    }
  }
  
  if (laneID == 0)
  {
  #if !USE_PRIMITIVE_CULLING
    numTrianglesOut = validCount * numPrimitives;
  #endif

    gl_PrimitiveCountNV = numTrianglesOut;
  #if USE_STATS
    atomicAdd(stats.triangles, numTrianglesOut);
  #endif
  }
}