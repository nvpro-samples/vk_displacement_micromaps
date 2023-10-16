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

#if MICRO_DECODER == MICRO_DECODER_MICROTRI_THREAD && MICRO_MTRI_USE_INTRINSIC
#extension GL_NV_displacement_micromap : require
#extension GL_EXT_ray_query : require
#endif

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
#include "common_micromesh_compressed.h"
#include "micromesh_binpack_flat_decl.h"

layout(buffer_reference, buffer_reference_align = 4) restrict buffer writeonly MicroBinPackFlats_out {
  MicroBinPackFlat d[];
};

////////////////////////////////////////////////////////////////
// BINDINGS

layout(scalar, binding = DRAWCOMPRESSED_UBO_VIEW) uniform sceneBuffer {
  SceneData scene;
  SceneData sceneLast;
};
layout(scalar, binding = DRAWCOMPRESSED_SSBO_STATS) coherent buffer statsBuffer {
  ShaderStats stats;
};
layout(scalar, binding = DRAWCOMPRESSED_UBO_COMPRESSED) uniform microBuffer {
  MicromeshData microdata;
};
layout(scalar, binding = DRAWCOMPRESSED_UBO_MESH) uniform meshBuffer {
  MeshData mesh;
};

layout(scalar, binding = DRAWCOMPRESSED_UBO_SCRATCH) uniform scratchBuffer {
  MicromeshScratchData  scratch;
};

#if USE_OCCLUSION_CULLING
  #define SUPPORTS_HIZ 1
  layout(binding = DRAWCOMPRESSED_TEX_HIZ)  uniform sampler2D texHizFar;
#endif

#if MICRO_DECODER == MICRO_DECODER_MICROTRI_THREAD && MICRO_MTRI_USE_INTRINSIC
layout(binding=DRAWCOMPRESSED_ACC) uniform accelerationStructureEXT sceneTlas;
#endif

layout(binding=DRAWCOMPRESSED_IMG_ATOMIC, r64ui) uniform coherent u64image2DArray imgVisBuffer;

layout(push_constant) uniform pushDraw {
  DrawMicromeshPushData push;
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
#include "micromesh_utils.glsl"
#include "micromesh_decoder.glsl"
#include "micromesh_culling_compressed.glsl"

void processMicromesh(MicroDecoderConfig cfg);

#define MICROBINPACK_USE_MESHLETCOUNT   0
#define MICROBINPACK_OUT                MicroBinPackFlats_out(scratch.scratchData).d
#define MICROBINPACK_OUT_ATOM           scratch.atomicCounter.d[0]
#define MICROBINPACK_OUT_MAX            scratch.maxCount
#define MICROBINPACK_PROCESS            processMicromesh

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

shared RasterVertex s_vertices[MICRO_MESHLET_VERTICES * MICRO_FLAT_SPLIT_TASK_GROUPS];
#if MICRO_FLAT_SPLIT_TASK_GROUPS > 1
uint                c_vertices_offset = MICRO_MESHLET_VERTICES * gl_LocalInvocationID.y;
#else
uint                c_vertices_offset = 0;
#endif

//////////////////////////////////////////////////////////////////////////

void processMicromesh(MicroDecoderConfig cfg)
{
  uint laneID = gl_SubgroupInvocationID;

  // Load
  MicromeshBaseTri microTri = microdata.basetriangles.d[cfg.microID];

  //////////////////////////////////////
  // Initial Decoding Phase

  SubgroupMicromeshDecoder sdec;
  smicrodec_subgroupInit(sdec, cfg, microTri, 0, 0, 0);
  
  //////////////////////////////////////
  // Mesh Preparation Phase
  
  mat4 worldMatrix   = mesh.instances.d[push.instanceID].worldMatrix;
  mat4 worldMatrixIT = transpose(inverse(worldMatrix));
  
  uint  triLocal   = smicrodec_getMeshTriangle(sdec);
  uint  tri        = triLocal + push.firstTriangle;
  uvec3 triIndices = uvec3( mesh.indices.d[tri * 3 + 0],
                            mesh.indices.d[tri * 3 + 1],
                            mesh.indices.d[tri * 3 + 2]) + push.firstVertex;
  
  // Generate vertices  
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
  
  //////////////////////////////////////
  // Vertex Iteration Phase
  
  dispVec2 scale_bias = micromesh_concatScaleBias(push.scale_bias, dispVec2(scene.disp_scale, scene.disp_bias));

  for (uint vertIter = 0; vertIter < smicrodec_getIterationCount(); vertIter++)
  {
    MicroDecodedVertex decoded = smicrodec_subgroupGetVertex(sdec, vertIter);
    uint vertOut               = decoded.outIndex;

    // safe to early out post shuffle 
    // This thread may not be valid, but a valid one before it may need to acces its data for shuffle in
    // smicrodec_subgroupGetLocalVertex
    
    if (!decoded.valid) continue;
    
    vec4 wPos = smicrodec_getVertexPos(decoded, v0, v1, v2, d0, d1, d2,
                                       worldMatrix,
                                       push.instanceID,
                                       scale_bias);
    s_vertices[vertOut + c_vertices_offset] = getRasterVertex(scene.viewProjMatrix * wPos);
  }
  
  //////////////////////////////////////
  // Primitive Iteration Phase

  uint numTriangles = smicrodec_getNumTriangles(sdec); 
  
#if 0
  uint inTriangles = subgroupBallotBitCount(subgroupBallot(cfg.packThreadID == 0 && valid)) * numTriangles;
  if (gl_SubgroupInvocationID == 0){
    atomicAdd(stats.debugUI, inTriangles);
  }
#endif

  // Generate primitives
  uint warpTriangles = 0;
  for (uint primIter = 0; primIter < smicrodec_getIterationCount(); primIter++)
  {
    MicroDecodedTriangle decoded = smicrodec_getTriangle(sdec, primIter);    
    if (!decoded.valid) continue;
    
    {
      RasterVertex a = (s_vertices[decoded.indices.x + c_vertices_offset]);
      RasterVertex b = (s_vertices[decoded.indices.y + c_vertices_offset]);
      RasterVertex c = (s_vertices[decoded.indices.z + c_vertices_offset]);
      
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
            
            rasterTriangle(pixel, tri, decoded.localIndex + cfg.partID * numTriangles, decoded.indices, a, b, c, triArea, 1.0);
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
  uint laneID       = gl_SubgroupInvocationID;
  
  uint microMax     = push.microMax;
  
  //microMax = 16;
  //if (gl_WorkGroupID.x > 0) return;
  
  bool valid        = baseID + laneID <= microMax;
  uint relativeID   = valid ? laneID : 0;  
  uint microID      = baseID + relativeID;
  
#if MICRO_FLAT_TASK_GROUPS > 1
  if (baseID > push.microMax) return;
#endif

#if USE_LOD
    uint targetSubdiv = cullAndLodMicroBaseTriangle(microID, valid);
#else
    MicromeshBaseTri microBaseTri = microdata.basetriangles.d[microID];
    uint targetSubdiv = micromesh_getBaseSubdiv(microBaseTri);
#endif

  MicroBinPackFlatSplit_subgroupPack(baseID, relativeID, targetSubdiv, 0, valid, push.instanceID);
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