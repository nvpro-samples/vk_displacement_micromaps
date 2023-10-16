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
#extension GL_EXT_shader_image_int64 : enable

#include "common.h"
#include "common_mesh.h"

////////////////////////////////////////////////////////////////

layout(scalar, binding = DRAWRAY_UBO_VIEW) uniform sceneBuffer {
  SceneData scene;
  SceneData sceneLast;
};
layout(scalar, binding = DRAWRAY_SSBO_STATS) coherent buffer statsBuffer {
  ShaderStats stats;
};

layout(binding=DRAWRAY_IMG_OUT, r64ui) uniform u64image2DArray imgOutBuffer;

layout(binding=DRAWRAY_ACC) uniform accelerationStructureEXT accScene;

////////////////////////////////////////////////////////////////

layout(location = 0) rayPayloadEXT uvec2 rayHit;

void main()
{ 
  ivec2 coord = ivec2(gl_LaunchIDEXT.xy);
  vec2  d = (vec2(gl_LaunchIDEXT.xy) + vec2(0.5)) / vec2(gl_LaunchSizeEXT.xy) * 2.0 - 1.0;
  
  vec3 origin    = (scene.viewMatrixI * vec4(0, 0, 0, 1)).xyz;
  vec3 target    = (scene.projMatrixI * vec4(d.x, d.y, 1, 1)).xyz;
  vec3 direction = (scene.viewMatrixI * vec4(normalize(target), 0)).xyz;

  float tMin     = scene.nearPlane;
  float tMax     = scene.farPlane;
  
  direction = normalize(direction);
  
  traceRayEXT(accScene, gl_RayFlagsCullBackFacingTrianglesEXT, 
    0xff,
    0, 1, // hit offset, hit stride
    0,    // miss offset
    origin.xyz, tMin, direction.xyz, tMax,
    0     // rayPayloadNV location qualifier
  );
  
  imageStore(imgOutBuffer, ivec3(coord,0), u64vec4(packUint2x32(rayHit),0,0,0)); 
}


