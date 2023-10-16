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

layout(local_size_x=MICRO_GROUP_SIZE, local_size_y=MICRO_FLAT_TASK_GROUPS) in;

//////////////////////////////////////////////////////////////////////////

#include "draw_culling.glsl"
#include "micromesh_utils.glsl"
#include "micromesh_culling_compressed.glsl"

#define MICROBINPACK_USE_MESHLETCOUNT   0
#define MICROBINPACK_OUT                MicroBinPackFlats_out(scratch.scratchData).d
#define MICROBINPACK_OUT_ATOM           scratch.atomicCounter.d[0]
#define MICROBINPACK_OUT_MAX            scratch.maxCount

#include "micromesh_binpack_flat.glsl"

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

  MicroBinPackFlat_subgroupPack(baseID, relativeID, targetSubdiv, 0, valid, push.instanceID);
}

void main()
{
  ////////////////////////
  // "task-shading" phase

#if MICRO_FLAT_TASK_GROUPS > 1
  uint baseID = ((gl_WorkGroupID.x * MICRO_FLAT_TASK_GROUPS) + gl_LocalInvocationID.y) * MICRO_TRI_PER_TASK;
#else
  uint baseID = gl_WorkGroupID.x * MICRO_TRI_PER_TASK;
#endif

  processTasks(baseID);
}

#endif