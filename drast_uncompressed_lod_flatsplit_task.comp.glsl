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
#extension GL_EXT_control_flow_attributes : enable

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
#extension GL_EXT_shader_image_int64 : enable

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require

#extension GL_NV_shader_subgroup_partitioned : require

#include "common.h"
#include "common_mesh.h"
#include "common_barymap.h"
#include "common_micromesh_uncompressed.h"
#include "micromesh_binpack_flat_decl.h"

layout(buffer_reference, buffer_reference_align = 4) restrict buffer writeonly MicroBinPackFlats_out {
  MicroBinPackFlat d[];
};

////////////////////////////////////////////////////////////////
// BINDINGS

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

layout(scalar, binding = DRAWUNCOMPRESSED_UBO_SCRATCH) uniform scratchBuffer {
  MicromeshUncScratchData  scratch;
};

#if USE_OCCLUSION_CULLING
  #define SUPPORTS_HIZ 1
  layout(binding = DRAWUNCOMPRESSED_TEX_HIZ)  uniform sampler2D texHizFar;
#endif

layout(binding=DRAWUNCOMPRESSED_IMG_ATOMIC, r64ui) uniform coherent u64image2DArray imgVisBuffer;

layout(push_constant) uniform pushDraw {
  DrawMicromeshUncPushData push;
};

//////////////////////////////////////////////////////////////////////////

#if IS_CONCAT_TASK

layout(local_size_x=SUBGROUP_SIZE) in;

void main()
{
#if MICRO_FLAT_MESH_GROUPS > 1
  if (gl_LocalInvocationIndex == 0) {
    uint outCount   = min(scratch.atomicCounter.d[0], scratch.maxCount);
    uint alignCount = (outCount + MICRO_FLAT_MESH_GROUPS - 1) / MICRO_FLAT_MESH_GROUPS;
    
    scratch.atomicCounter.d[0] = alignCount;
    
    // append some tail dummy data
    MicroBinPackFlats_out outData = MicroBinPackFlats_out(scratch.scratchData);
    
    MicroBinPackFlat flatPackDummy;
    flatPackDummy.pack       = MICRO_BIN_INVALID_SUBDIV;
    flatPackDummy.partOrMask = 0;
    
    for (uint i = 0; i < MICRO_FLAT_MESH_GROUPS; i++) {
      outData.d[i + outCount] = flatPackDummy;
    }
  }
#endif
}

#else

layout(local_size_x=MICRO_GROUP_SIZE, local_size_y=MICRO_FLAT_SPLIT_TASK_GROUPS) in;

//////////////////////////////////////////////////////////////////////////

#include "draw_culling.glsl"
#include "drast_utils.glsl"
#include "micromesh_decoder_config.glsl"
#include "micromesh_culling_uncompressed.glsl"

void processMicromesh(MicroDecoderConfig cfg);

#define MICROBINPACK_USE_MESHLETCOUNT             1
#define MICROBINPACK_OUT                          MicroBinPackFlats_out(scratch.scratchData).d
#define MICROBINPACK_OUT_ATOM                     scratch.atomicCounter.d[0]
#define MICROBINPACK_OUT_MAX                      scratch.maxCount
#define MICROBINPACK_PROCESS                      processMicromesh

#if 1
// the standard bin packing
#include "micromesh_binpack_flatsplit.glsl"
#else
// This is a variant that doesn't actually do bin packing, but processes
// base triangles one at a time. This takes more iterations and on low subdiv
// means poor SIMD/threads in subgroup utilization.
// Only exists to showcase benefit of binpacking. Do not use.
#include "micromesh_nopack_flatsplit.glsl"
#endif

//////////////////////////////////////////////////////////////////////////

shared RasterVertex s_vertices[MAX_BARYMAP_VERTICES * MICRO_FLAT_SPLIT_TASK_GROUPS];
#if MICRO_FLAT_SPLIT_TASK_GROUPS > 1
uint                c_vertices_offset = MAX_BARYMAP_VERTICES * gl_LocalInvocationID.y;
#else
uint                c_vertices_offset = 0;
#endif

//////////////////////////////////////////////////////////////////////////

uint subdiv_getNumVertsPerEdge  (uint subdiv)   { return (1 << subdiv) + 1; }
uint subdiv_getNumTriangles     (uint subdiv)   { return (1 << (subdiv * 2));}
uint subdiv_getNumSegments      (uint subdiv)   { return (1 << subdiv); }
uint subdiv_getNumVerts         (uint subdiv)   { uint numVertsPerEdge = subdiv_getNumVertsPerEdge(subdiv); return (numVertsPerEdge * (numVertsPerEdge + 1)) / 2; }

//////////////////////////////////////////////////////////////////////////

void processMicromesh(MicroDecoderConfig cfg)
{
  uint laneID = gl_SubgroupInvocationID;

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
  uint baseSubdiv          = baseTri.subdivLevel;
  uint firstValue          = baseTri.firstValue;
  
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
    s_vertices[vertOut + c_vertices_offset] = getRasterVertex(scene.viewProjMatrix * wPos);
  }
  
  //////////////////////////////////////
  // Primitive Iteration Phase

  uint warpTriangles = 0;
  for (uint32_t primIter = 0; primIter < iterationCount; primIter++)
  {
    uint prim     = cfg.packThreadID + primIter * cfg.packThreads;
    uint primRead = min(prim,  primMax);
    
    uint  pidxPacked = bmap.levels.d[targetConfig].meshletData.d[offsetPrims + primRead];
    uvec3 indices    = uvec3(pidxPacked & 0xFF, (pidxPacked >> 8) & 0xFF, (pidxPacked >> 16));
    indices          += (cfg.packID * vertStride);
    
    if (prim <= primMax && cfg.valid)    
    {
      RasterVertex a = (s_vertices[indices.x + c_vertices_offset]);
      RasterVertex b = (s_vertices[indices.y + c_vertices_offset]);
      RasterVertex c = (s_vertices[indices.z + c_vertices_offset]);
      
      vec2 pixelMin;
      vec2 pixelMax;
      float triArea;
      
      bool visible = testTriangle(a,b,c, 1.0, pixelMin, pixelMax, triArea);
      if (visible) 
      {      
        vec2 pixelDim  = uvec2(pixelMax - pixelMin);
        for (uint py = 0; py < pixelDim.y; py++)
        {
          for (uint px = 0; px < pixelDim.x; px++)
          {
            vec2 pixel = pixelMin + vec2(0.5) + vec2(px,py);
            
            rasterTriangle(pixel, tri, prim + cfg.partID * numPrimitives, indices, a, b, c, triArea, 1.0);
          }
        }
      }
    #if USE_STATS
      warpTriangles += subgroupBallotBitCount(subgroupBallot(visible));
    #endif
    }
  }
  
#if USE_STATS
  if (gl_SubgroupInvocationID == 0) {
    atomicAdd(stats.triangles, warpTriangles);
  }
#endif
}

//////////////////////////////////////////////////////////////////////////

void processTasks(uint baseID)
{
  uint laneID = gl_SubgroupInvocationID;
  
  uint triMax       = push.triangleMax;
  
  //if (baseID != 0) return;
  //triMax = 0;
  
  bool valid        = baseID + laneID <= triMax;
  uint relativeID   = valid ? laneID : 0;  
  uint triID        = baseID + relativeID;
  
  MicromeshUncBaseTri baseTri = microdata.basetriangles.d[min(triID, triMax)];
  
#if MICRO_FLAT_SPLIT_TASK_GROUPS > 1
  if (baseID > push.triangleMax) return;
#endif
  
#if USE_LOD
  uint meshletCount = baseTri.meshletCount;
  uint baseSubdiv   = baseTri.subdivLevel;
  
  uint targetSubdiv = cullAndLodBaseTriangle(triID, baseSubdiv, valid);
  
  #if 0
    uint targetMax = subgroupMax(valid ? targetSubdiv : 0);
    targetSubdiv   = valid ? targetMax : targetSubdiv;
  #endif

  if (valid && targetSubdiv < baseSubdiv) {
    meshletCount = bmap.levels.d[targetSubdiv].meshletCount;
  }
#else
  uint meshletCount = baseTri.meshletCount;
  uint targetSubdiv = baseTri.subdivLevel;
#endif
  
  MicroBinPackFlatSplit_subgroupPack(baseID, relativeID, targetSubdiv, meshletCount, valid, push.instanceID);
}

void main()
{
  ////////////////////////
  // "task-shading" phase

#if MICRO_FLAT_SPLIT_TASK_GROUPS > 1
  uint baseID = ((gl_WorkGroupID.x * MICRO_FLAT_SPLIT_TASK_GROUPS) + gl_LocalInvocationID.y) * MICRO_TRI_PER_TASK;
#else
  uint baseID = gl_WorkGroupID.x * MICRO_TRI_PER_TASK;
#endif

  processTasks(baseID);
}

#endif