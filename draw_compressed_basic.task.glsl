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

#include "common.h"
#include "common_mesh.h"
#include "common_barymap.h"
#include "common_micromesh_uncompressed.h"
#include "common_micromesh_compressed.h"

//////////////////////////////////////////////////////////////////////////

layout(local_size_x=MICRO_GROUP_SIZE) in;

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
layout(push_constant) uniform pushDraw {
  DrawMicromeshPushData push;
};

//////////////////////////////////////////////////////////////////////////
// OUTPUT

taskNV out Task
{
  uint      baseID;
  uint16_t  prefix[MICRO_TRI_PER_TASK];
} OUT;


//////////////////////////////////////////////////////////////////////////

#include "micromesh_utils.glsl"

//////////////////////////////////////////////////////////////////////////

  // This shader does not do any packing of micromeshes with low subdivision
  // into a single mesh-shader invocation. This yields less performance as we
  // unterutilize the hardware this way (an entire mesh workgroup may generate 
  // only a single triangle).
  // Look at the draw_micromesh_lod shaders which use a more complex setup
  // that does this.

void main()
{
  uint baseID = gl_WorkGroupID.x * MICRO_TRI_PER_TASK;
  uint laneID = gl_SubgroupInvocationID;

  OUT.baseID   = baseID;
  
  uint microID = baseID + laneID;
  
  MicromeshBaseTri microBaseTri = microdata.basetriangles.d[min(microID, push.microMax)];
  uint microSubdiv              = micromesh_getBaseSubdiv(microBaseTri);
  
  uint partMicroMeshlets = subdiv_getNumMeshlets(microSubdiv);
  uint microMax          = push.microMax;
  
  bool valid = true;
  
  // debugging
  //valid = microID == 0;
  //partMicroMeshlets = 1;
  
  uint meshletCount       = microID <= microMax && valid ? partMicroMeshlets : 0;
  
  uint prefix         = subgroupExclusiveAdd(meshletCount);
  OUT.prefix[laneID]  = uint16_t(prefix);
  
  if (laneID == MICRO_TRI_PER_TASK-1)
  {
    gl_TaskCountNV = prefix + meshletCount;
  #if USE_STATS
    atomicAdd(stats.meshlets, prefix + meshletCount);
  #endif
  }
}