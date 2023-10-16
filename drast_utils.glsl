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

// basic software rasterization logic
// requires GL_EXT_shader_image_int64

void outputPixel(vec2 fragCoord, uint payload, float depth) 
{
  ivec2 coord = ivec2(fragCoord.xy);
  uvec2 pixel = uvec2(payload, floatBitsToUint(depth));

  imageAtomicMin(imgVisBuffer, ivec3(coord,0), packUint2x32(pixel));
#if ATOMIC_LAYERS > 1 && 0
  // for micro-meshes we may want to use two atomics to have more payload
  // e.g. avoid deocding for barycentrics, this acts as perf investigation proxy
  imageAtomicMin(imgVisBuffer, ivec3(coord,1), packUint2x32(pixel));
#endif
}

float edgeFunction(vec2 a, vec2 b, vec2 c, float winding) 
{ 
  return ((c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)) * winding;
} 

void rasterTriangle(vec2 pixel, uint basetri, uint microtri, uvec3 indices, RasterVertex a, RasterVertex b, RasterVertex c, float triArea, float winding)
{
  float baryA = edgeFunction(b.xy, c.xy, pixel, winding);
  float baryB = edgeFunction(c.xy, a.xy, pixel, winding);
  float baryC = edgeFunction(a.xy, b.xy, pixel, winding);

  if (baryA >= 0 && baryB >= 0 && baryC >= 0){
    baryA /= triArea;
    baryB /= triArea;
    baryC /= triArea;
    
    float depth = a.z * baryA + 
                  b.z * baryB + 
                  c.z * baryC;

    outputPixel(pixel, (basetri << 10) | (microtri), depth);
  }
}