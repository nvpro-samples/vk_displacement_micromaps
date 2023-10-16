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
uint getCullBits(vec4 hPos)
{
  uint cullBits = 0;
  cullBits |= hPos.x < -hPos.w ?  1 : 0;
  cullBits |= hPos.x >  hPos.w ?  2 : 0;
  cullBits |= hPos.y < -hPos.w ?  4 : 0;
  cullBits |= hPos.y >  hPos.w ?  8 : 0;
  cullBits |= hPos.z <  0      ? 16 : 0;
  cullBits |= hPos.z >  hPos.w ? 32 : 0;
  cullBits |= hPos.w <= 0      ? 64 : 0; 
  return cullBits;
}

vec2 getScreenPos(vec4 hPos)
{
  return vec2(((hPos.xy/hPos.w) * 0.5 + 0.5) * scene.viewportf);
}

//////////////////////////////////////////////////////////

#define RasterVertex vec4
#define RasterVertex_cullBits(r)  floatBitsToUint(r.w)

RasterVertex getRasterVertex(vec4 hPos)
{
  RasterVertex vtx;
  vtx.xy    = getScreenPos(hPos);
  vtx.z     = hPos.z/hPos.w;
  vtx.w     = uintBitsToFloat(getCullBits(hPos));
  
  return vtx;
}

void pixelBboxEpsilon(inout vec2 pixelMin, inout vec2 pixelMax)
{
  // apply some safety around the bbox to take into account fixed point rasterization
  // (our rasterization grid is 1/256)
  
  const float epsilon = (1.0 / 256);
  pixelMin -= epsilon;
  pixelMax += epsilon;
  pixelMin = round(pixelMin);
  pixelMax = round(pixelMax);
}

bool pixelBboxCull(vec2 pixelMin, vec2 pixelMax){
  // bbox culling
  bool cull = ( ( pixelMin.x == pixelMax.x) || ( pixelMin.y == pixelMax.y));
  return cull;
}

bool pixelViewportCull(vec2 pixelMin, vec2 pixelMax)
{
  return ((pixelMax.x < 0) || (pixelMin.x >= scene.viewportf.x) || (pixelMax.y < 0) || (pixelMin.y >= scene.viewportf.y));
}

bool testTriangle(vec2 a, vec2 b, vec2 c, float winding, bool frustum, out vec2 pixelMin, out vec2 pixelMax, out float triArea)
{
  // back face culling
  vec2 ab = b.xy - a.xy;
  vec2 ac = c.xy - a.xy;
  float cross_product = ab.y * ac.x - ab.x * ac.y;   
  
  triArea = cross_product * winding;
  
  if (cross_product * winding < 0) return false;

  // compute the min and max in each X and Y direction
  pixelMin = min(a,min(b,c));
  pixelMax = max(a,max(b,c));
  
  pixelBboxEpsilon(pixelMin, pixelMax);
  
  if (frustum && pixelViewportCull(pixelMin, pixelMax)) return false;
  
  if (pixelBboxCull(pixelMin, pixelMax)) return false;
  
  return true;
}

bool testTriangle(RasterVertex a, RasterVertex b, RasterVertex c, float winding, out vec2 pixelMin, out vec2 pixelMax, out float triArea)
{
  
  if ((RasterVertex_cullBits(a) & RasterVertex_cullBits(b) & RasterVertex_cullBits(c)) == 0 &&
      // don't attempt to to rasterize specially clipped triangles 
      (((RasterVertex_cullBits(a) | RasterVertex_cullBits(b) | RasterVertex_cullBits(c)) & (16 | 32 |64)) == 0))
  {
    return testTriangle(a.xy,b.xy,c.xy,winding, false, pixelMin, pixelMax, triArea);
  }
  return false;
}

bool testTriangle(RasterVertex a, RasterVertex b, RasterVertex c, float winding)
{
  if ((RasterVertex_cullBits(a) & RasterVertex_cullBits(b) & RasterVertex_cullBits(c)) == 0){
    vec2 pixelMin;
    vec2 pixelMax;
    float triArea;
    // trivially accept complex triangles, let hw culling take care of those
    return (((RasterVertex_cullBits(a) | RasterVertex_cullBits(b) | RasterVertex_cullBits(c)) & 64) != 0) 
           || testTriangle(a.xy,b.xy,c.xy,winding, false, pixelMin, pixelMax, triArea);
  }
  return false;
}

//////////////////////////////////////////////////////////

bool frustumCullSphere(vec3 center, float radius)
{
  [[unroll]]
  for (int n = 0; n < 6; n++)
  {
    if (dot(sceneLast.frustumPlanes[n], vec4(center,1)) < -radius)
      return true;
  }

  return false;
}

bool depthCullSphere(vec3 center, float radius)
{
#if SUPPORTS_HIZ && USE_OCCLUSION_CULLING
  
  // The occlusion culling done here is very basic.
  // We just test against last frame depth, using last
  // frame projection (we assume the object itself isn't moving).
  // 
  // We recommend a more sophisticated solution that also does something
  // more hierarchical rather than test every base triangle individually

  vec4 hPos2 = sceneLast.viewProjMatrix * vec4(center - sceneLast.viewDir.xyz * radius,1);
  float depth = hPos2.z / hPos2.w;
  
  if (hPos2.z < 0) return false;

  vec4 hPos = sceneLast.viewProjMatrix * vec4(center,1);
  vec2 pixelSize = radius * sceneLast.viewClipSize / hPos.w;
  
  const float c_epsilon    = 1.2e-07f;
  const float c_depthNudge = 2.0/float(1<<24);
  
  vec2 clipmin    = hPos.xy / hPos.w;
  vec2 clipmax    = clipmin;
  
  // add and subtract pixelSize of radius 
  clipmin -= pixelSize;
  clipmax += pixelSize; 
  
  clipmin.xy = clipmin.xy * 0.5 + 0.5;
  clipmax.xy = clipmax.xy * 0.5 + 0.5;
  
  clipmin.xy *= scene.hizSizeFactors.xy;
  clipmax.xy *= scene.hizSizeFactors.xy;
   
  clipmin.xy = min(clipmin.xy, scene.hizSizeFactors.zw);
  clipmax.xy = min(clipmax.xy, scene.hizSizeFactors.zw);
  
  vec2  size = (clipmax.xy - clipmin.xy);
  float maxsize = max(size.x, size.y) * scene.hizSizeMax;
  float miplevel = ceil(log2(maxsize));

  float hizDepth = textureLod(texHizFar,(clipmin.xy + clipmax.xy) * 0.5, miplevel).r;

  bool result = !(depth < hizDepth + c_depthNudge);
  
  return result;
#else
  return false;
#endif
}

uint computeSphereLod(vec3 center, float radius, uint subdiv)
{
  vec4 hPos      = sceneLast.viewProjMatrix * vec4(center,1);
  vec2 pixelSize = 1.0 * radius * scene.viewPixelSize / hPos.w;
  
  // heuristic value
  float areaPix = pixelSize.x * pixelSize.y * 0.333 * scene.lodScale * scene.lodScale;
  
  // Fit 1 triangle per N pixels, subdivision is powers of 4
  float pixelsPerTriangle = 1.0f;
  float subdivisionLod = (log(areaPix / pixelsPerTriangle) / log(4.0f));
  return clamp(uint(subdivisionLod), 0, subdiv);
}

uint computeTriangleLod(vec3 a, vec3 b, vec3 c, uint subdiv)
{
  // Project corners to screen space
  vec2 projected[3];
  projected[0] = getScreenPos(scene.viewProjMatrix * vec4(a,1));
  projected[1] = getScreenPos(scene.viewProjMatrix * vec4(b,1));
  projected[2] = getScreenPos(scene.viewProjMatrix * vec4(c,1));
  // Compute triangle area in pixels
  vec2 vecA = (projected[1] - projected[0]) * scene.lodScale;
  vec2 vecB = (projected[2] - projected[0]) * scene.lodScale;
  float areaPix = abs(vecA.x * vecB.y - vecA.y * vecB.x) * 0.5f;
  
  // Fit 1 triangle per N pixels, subdivision is powers of 4
  float pixelsPerTriangle = 1.0f;
  float subdivisionLod = (log(areaPix / pixelsPerTriangle) / log(4.0f));
  return clamp(uint(subdivisionLod), 0, subdiv);
}

//////////////////////////////////////////////////////////

bool cullSphere(vec3 center, float radius)
{
  bool valid = !frustumCullSphere(center, radius) && !depthCullSphere(center, radius);
  return valid;
}

bool cullAndLodSphere(vec3 center, float radius, uint subdiv, inout uint targetSubdiv)
{
  bool valid = !frustumCullSphere(center, radius) && !depthCullSphere(center, radius);
  if (valid) {
    targetSubdiv = computeSphereLod(center, radius, subdiv); 
  }
  return valid;
}

bool cullAndLodTriangle(vec3 a, vec3 b, vec3 c, vec3 center, float radius, uint subdiv, inout uint targetSubdiv)
{
  bool valid = !frustumCullSphere(center, radius) && !depthCullSphere(center, radius);
  if (valid) {
    targetSubdiv = computeTriangleLod(a,b,c, subdiv); 
  }
  return valid;
}

