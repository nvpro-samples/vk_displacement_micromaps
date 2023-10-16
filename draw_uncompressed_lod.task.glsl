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
#extension GL_NV_shader_subgroup_partitioned : require

#include "common.h"
#include "common_mesh.h"
#include "common_barymap.h"
#include "common_micromesh_uncompressed.h"
#include "micromesh_binpack_decl.glsl"

layout(local_size_x=MICRO_TRI_PER_TASK) in;

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
#if USE_OCCLUSION_CULLING
  #define SUPPORTS_HIZ 1
  layout(binding = DRAWUNCOMPRESSED_TEX_HIZ)  uniform sampler2D texHizFar;
#endif

layout(push_constant) uniform pushDraw {
  DrawMicromeshUncPushData push;
};

//////////////////////////////////////////////////////////////////////////
// OUTPUT

taskNV out Task
{
  MicroBinPack binpack;
} OUT;


//////////////////////////////////////////////////////////////////////////

#include "draw_culling.glsl"
#include "micromesh_culling_uncompressed.glsl"

// MicroBinPack functions need to reference target variables directly via macros
#define MICROBINPACK_USE_MESHLETCOUNT             1
#define MICROBINPACK_OUT                          OUT.binpack
#define MICROBINPACK_OUT_COUNT                    gl_TaskCountNV
#include "micromesh_binpack.glsl"

//////////////////////////////////////////////////////////////////////////

void main()
{
  uint baseID = gl_WorkGroupID.x * MICRO_TRI_PER_TASK;
  uint laneID = gl_SubgroupInvocationID;
  
  uint triMax       = push.triangleMax;
  
  //if (baseID != 0) return;
  //triMax = 1;
  
  bool valid        = baseID + laneID <= triMax;
  uint relativeID   = valid ? laneID : 0;  
  uint triID        = baseID + relativeID;
  
  MicromeshUncBaseTri baseTri = microdata.basetriangles.d[min(triID, triMax)];
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
  
  MicroBinPack_subgroupPack(baseID, relativeID, targetSubdiv, meshletCount, valid);
  
  //gl_TaskCountNV *= scene.dbgUint;
}