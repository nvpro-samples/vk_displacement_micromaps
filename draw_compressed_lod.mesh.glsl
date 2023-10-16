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
#include "common_micromesh_compressed.h"
#include "micromesh_binpack_decl.glsl"

//////////////////////////////////////////////////////////////////////////

layout(local_size_x=MICRO_GROUP_SIZE) in;
layout(max_primitives = MICRO_MESHLET_PRIMITIVES, max_vertices = MICRO_MESHLET_VERTICES, triangles) out;

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
#if MICRO_DECODER == MICRO_DECODER_MICROTRI_THREAD && MICRO_MTRI_USE_INTRINSIC
layout(binding=DRAWCOMPRESSED_ACC) uniform accelerationStructureEXT sceneTlas;
#endif

layout(push_constant) uniform pushDraw {
  DrawMicromeshPushData push;
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

#include "micromesh_utils.glsl"
#include "micromesh_decoder.glsl"
#include "draw_culling.glsl"

// MicroBinPack functions need to reference target variables directly via macros
#define MICROBINPACK_USE_MESHLETCOUNT   0
#define MICROBINPACK_IN                 IN.binpack
#include "micromesh_binpack.glsl"

//////////////////////////////////////////////////////////////////////////

void main()
{
  //////////////////////////////////////
  // Decoder Configuration Phase
  
  uint wgroupID   = gl_WorkGroupID.x;
  uint laneID     = gl_SubgroupInvocationID;
  
  MicroDecoderConfig cfg   = MicroBinPack_subgroupUnpack(wgroupID);
#if !USE_PRIMITIVE_CULLING
  uint  validCount = 1;
  if (cfg.targetSubdiv < 3){
    uvec4 validVote  = subgroupBallot(cfg.packThreadID == 0 && cfg.valid);
    validCount       = subgroupBallotBitCount(validVote);
  }
#endif

  // Load microTri
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

#if USE_MICROVERTEX_NORMALS
  uint firstValue = microdata.attrTriangleOffsets.d[triLocal];
#endif

  uint baseSubdiv = smicrodec_getBaseSubdiv(sdec);
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
    gl_MeshVerticesNV[vertOut].gl_Position = scene.viewProjMatrix * wPos;

  #if SURFACEVIS == SURFACEVIS_SHADING
    OUT[vertOut].wPos    = wPos.xyz;
    OUT[vertOut].bary    = decoded.bary;
    OUT[vertOut].tri     = tri;
    #if USE_MICROVERTEX_NORMALS
      uint valueIdx      = umajorUV_toLinear(subdiv_getNumVertsPerEdge(baseSubdiv), decoded.uv);
      #if !SHADING_UMAJOR
           valueIdx      = microdata.umajor2bmap[baseSubdiv].d[valueIdx];
      #endif
      OUTvtx[vertOut].vidx  = firstValue + valueIdx;
    #endif
  #elif SURFACEVIS == SURFACEVIS_ANISOTROPY
    OUTvtx[vertOut].vwPos   = wPos.xyz;
  #endif
  }
  
  //////////////////////////////////////
  // Primitive Iteration Phase
  
#if SURFACEVIS == SURFACEVIS_VALUERANGE
  // FIXME tri index can actually be wrong here
  // okay for the baking app
  float valueRange = getValueRange( float(microdata.basetriangleMinMaxs.d[tri * 2 + 0]) / float (0x7FF), 
                                    float(microdata.basetriangleMinMaxs.d[tri * 2 + 1]) / float (0x7FF));
#else
  float valueRange = 0;
#endif

  uint numTriangles = smicrodec_getNumTriangles(sdec);
  uint formatIdx    = smicrodec_getFormatIdx(sdec);
  uint microSubdiv  = smicrodec_getMicroSubdiv(sdec);

  // Generate primitives
#if USE_PRIMITIVE_CULLING
  uint numTrianglesOut = 0;
#else
  uint numTrianglesOut = numTriangles;
#endif
  for (uint primIter = 0; primIter < smicrodec_getIterationCount(); primIter++)
  {
    MicroDecodedTriangle decoded = smicrodec_getTriangle(sdec, primIter); 
    bool visible                 = decoded.valid;
  #if USE_PRIMITIVE_CULLING
    if (visible) {
      
      RasterVertex a = getRasterVertex(gl_MeshVerticesNV[decoded.indices.x].gl_Position);
      RasterVertex b = getRasterVertex(gl_MeshVerticesNV[decoded.indices.y].gl_Position);
      RasterVertex c = getRasterVertex(gl_MeshVerticesNV[decoded.indices.z].gl_Position);
      
      visible = testTriangle(a,b,c, 1.0);
    }
    uvec4 voteVis = subgroupBallot(visible);
    uint  primOut = numTrianglesOut + subgroupBallotExclusiveBitCount(voteVis);
    
    numTrianglesOut += subgroupBallotBitCount(voteVis);
  #else
    uint primOut = decoded.outIndex;
  #endif
    
    if (visible) 
    {
      gl_PrimitiveIndicesNV[primOut * 3 + 0] = decoded.indices.x;
      gl_PrimitiveIndicesNV[primOut * 3 + 1] = decoded.indices.y;
      gl_PrimitiveIndicesNV[primOut * 3 + 2] = decoded.indices.z;

      uint id = decoded.localIndex + cfg.partID * numTriangles;
      chooseSurfaceVisOutput(gl_MeshPrimitivesNV[primOut].gl_PrimitiveID, 
          tri, id, formatIdx, microSubdiv - cfg.targetSubdiv, valueRange, baseSubdiv, cfg.targetSubdiv);
    }
  }
  
  if (laneID == 0)
  {
  #if !USE_PRIMITIVE_CULLING
    numTrianglesOut = validCount * numTriangles;
  #endif
  
    gl_PrimitiveCountNV = numTrianglesOut;
  #if USE_STATS
    atomicAdd(stats.triangles, numTrianglesOut);
  #endif
  }
}