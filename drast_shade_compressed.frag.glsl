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

#include "common.h"
#include "common_mesh.h"
#include "common_barymap.h"
#include "common_micromesh_uncompressed.h"
#include "common_micromesh_compressed.h"

////////////////////////////////////////

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
layout(binding=DRAWCOMPRESSED_IMG_ATOMIC, r64ui) uniform u64image2DArray imgVisBuffer;

////////////////////////////////////////

#include "draw_shading.glsl"

////////////////////////////////////////

layout(location=0, index=0) out vec4 out_Color;

////////////////////////////////////////

void main()
{
  ivec2 coord      = ivec2(gl_FragCoord.xy);
  uint64_t pixel64 = imageLoad(imgVisBuffer, ivec3(coord,0)).x;
  uvec2    pixel   = unpackUint2x32(pixel64);
  
  uint clearDepth = 0xFFFFFFFFu;
  
  vec4 color = colorizePrimitive(int(pixel.x));
  
  out_Color    = pixel.y == clearDepth ? vec4(CLEAR_COLOR) : color;
  gl_FragDepth = pixel.y == clearDepth ? 1.0f : uintBitsToFloat(pixel.y);
}
