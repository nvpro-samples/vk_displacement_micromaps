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

#define PI  3.14159265359f

// Returns a pseudorandom (R, G, B, 1) color per input.
vec4 colorizePrimitive(int primID)
{
  return vec4(unpackUnorm4x8(murmurHash(primID)).xyz, 1.0);
}

// Returns a measure of anisotropy for a triangle given its three vertices,
// from 0 (equilateral triangle) to 1 (degenerate triangle).
// This metric is (M-m)/(M+m), where M and m are the maximum and minimum angles
// (in [0, pi]), respectively.
//
// I chose this metric because it maps to the [0, 1] range pretty well.
// I considered a few other metrics, but they didn't work as well:
// - Looking at the longest and shortest edge lengths can produce different
// results for different degenerate triangles
// - Getting the eccentricity of the Steiner ellipse is complex and produces
// values that bunch near 1
// - 1 - 6/(3^1/4) sqrt(area)/perimeter is good, but doesn't go to 1 as quickly
// as I would expect for very anisotropic triangles.
float anisotropyMetric(vec3 v0, vec3 v1, vec3 v2)
{
  // Compute edge vectors between vertices. We'll catch NaNs at the end.
  const vec3 e01 = normalize(v0 - v1);
  const vec3 e02 = normalize(v0 - v2);
  const vec3 e12 = normalize(v1 - v2);
  // Get cosines of angles. This doesn't depend on the triangle's winding order.
  const float c0 = dot(e01, e02);   // <-e01, -e02>
  const float c1 = -dot(e01, e12);  // <e01, -e12>
  const float c2 = dot(e02, e12);

  const float bigM = acos(min(min(c0, c1), c2));
  const float ltlM = acos(max(max(c0, c1), c2));
  // This denominator should always be at least 2pi/3.
  const float e = (bigM - ltlM) / (bigM + ltlM);
  // Catch NaNs in case of undefined normalize behavior. If two vertices were
  // equal, the triangle's degenerate, so return 1.
  return isnan(e) ? 1.0 : e;
}

// Approximates the batlow color ramp from the scientific color ramps package.
// Input will be clamped to [0, 1]; output is sRGB.
vec3 batlow(float t)
{
  t             = clamp(t, 0.0f, 1.0f);
  const vec3 c5 = vec3(10.741, -0.934, -16.125);
  const vec3 c4 = vec3(-28.888, 2.021, 34.529);
  const vec3 c3 = vec3(24.263, -0.335, -20.561);
  const vec3 c2 = vec3(-6.069, -1.511, 2.47);
  const vec3 c1 = vec3(0.928, 1.455, 0.327);
  const vec3 c0 = vec3(0.007, 0.103, 0.341);

  const vec3 result = ((((c5 * t + c4) * t + c3) * t + c2) * t + c1) * t + c0;

  return min(result, vec3(1.0f));
}


vec3 hue2rgb(float hue)
{
  hue= fract(hue);
  return clamp(vec3(
    abs(hue*6.0-3.0)-1.0,
    2.0-abs(hue*6.0-2.0),
    2.0-abs(hue*6.0-4.0)
  ), vec3(0), vec3(1));
}

// Returns the color of the surface. When SURFACEVIS isn't equal to
// SURFACEVIS_SHADING, this should be written as the out color directly.
// v0, v1, and v2 should be the vertices of the triangle.
vec4 surfaceVisShading(
  #if SURFACEVIS == SURFACEVIS_ANISOTROPY
    vec3 v0,
    vec3 v1,
    vec3 v2,
  #endif
    int primID
)
{
  #if SURFACEVIS == SURFACEVIS_ANISOTROPY
    vec3 color = batlow(anisotropyMetric(v0, v1, v2));
  #elif(SURFACEVIS == SURFACEVIS_BASETRI) || DISABLE_SHADING_SPECIALS
    vec3 color = colorizePrimitive(primID).rgb;
  #elif SURFACEVIS == SURFACEVIS_SHADING
    vec3 color = vec3(0.8);
  #elif (SURFACEVIS == SURFACEVIS_MICROTRI) || (SURFACEVIS == SURFACEVIS_LOCALTRI)
    vec3 color = colorizePrimitive(primID).rgb;
  #elif (SURFACEVIS == SURFACEVIS_FORMAT)
    vec3 color = vec3(1);             // 11 bit / triangle
    switch(primID){
    case 1:
      color = vec3(0.5,1,0); break;   // 4 bit / triangle
    case 2:
      color = vec3(1,0.5,0); break;   // 1 bit / triangle
    }
  #elif (SURFACEVIS == SURFACEVIS_LODBIAS)
    //vec3 color = hue2rgb( 0.4 - (float(primID) / 4.5f) * 0.4);
    vec3 color = vec3(1,1,1);
    switch(primID){
    case 1:
      color = vec3(0,1,1); break;
    case 2:
      color = vec3(0,0.9,0); break;
    case 3:
      color = vec3(1,1,0); break;
    case 4:
      color = vec3(1,0,0); break;
    case 5:
      color = vec3(0.6,0,0); break;
    case 6:
      color = vec3(0.3,0,0); break;
    case 7:
      color = vec3(0.15,0,0); break;
    }
  #elif (SURFACEVIS == SURFACEVIS_BASESUBDIV) || (SURFACEVIS == SURFACEVIS_LODSUBDIV)
    vec3 color = vec3(1,1,1);
    switch(primID){
    case 0:
      color = vec3(0.8,0,0); break;
    case 1:
      color = vec3(1,0.6,0.1); break;
    case 2:
      color = vec3(1,1,0); break;
    case 3:
      color = vec3(0,1,0); break;
    case 4:
      color = vec3(0,0.5,1); break;
    case 5:
      color = vec3(0,0.1,1); break;
    case 6:
      color = vec3(0,0,0.5); break;
    case 7:
      color = vec3(0,0,0.15); break;
    }
  #elif(SURFACEVIS == SURFACEVIS_VALUERANGE)
    vec3 color = batlow(intBitsToFloat(primID));
  #endif

  // Apply red wireframe tinting so that we see this contrast even if we're using
  // unlit shading.
  #if USE_OVERLAY
    color *= vec3(0.9, 0.2, 0.2);
  #endif

  return vec4(color, 1.0);
}

vec4 shading(uint instanceID, int primID, vec3 wPos, vec2 tex, vec3 wNormal
  #if USE_TEXTURE_NORMALS
  , vec3 wTangent, vec3 wBitangent, uint normalMapID
  #endif
)
{
  vec4 color = vec4(0.8);
  
#if USE_TEXTURE_NORMALS
  uint normalMapID = nonuniformEXT(mesh.instances.d[instanceID].normalMapID);
  vec3 tnormal     = texture(tex2Ds[normalMapID], tex).rgb * 2 - 1;
  // FIXME proper Mikkt Space would do bitangent here, wBitangent = sign * cross(wNormal, wTangent);
  vec3 normal      = normalize(tnormal.x * (wTangent) + tnormal.y * (wBitangent) + tnormal.z * (wNormal));
#else
  vec3 normal      = normalize(wNormal.xyz);
#endif
  vec3 wEyePos  = vec3(scene.viewMatrixI[3].x,scene.viewMatrixI[3].y,scene.viewMatrixI[3].z);

  vec3 lightDir = normalize(scene.wLightPos.xyz - wPos.xyz);
  vec3 eyeDir   = normalize(wEyePos.xyz - wPos.xyz);
  vec3 reflDir  = normalize(-reflect(lightDir,normal));
  
  float lt  = abs(dot(normal,lightDir)) + pow(max(0,dot(reflDir, eyeDir)), 16) * 0.3;
  color = color * (lt);
  
  color += mix(vec4(0.1, 0.1, 0.4, 0), vec4(0.8, 0.6, 0.2, 0.0), dot(normal,scene.wUpDir.xyz) * 0.5 + 0.5) * 0.2;
  
  #ifndef DISABLE_SHADING_SPECIALS
  if(scene.reflection != 0)
  {
    float lineWidth = 1; //float(scene.supersample);
    float lineGaps  = 1.0 / 12.5; 
    float angle = asin(abs(dot(normal, reflDir))) * 180 / (PI);
    float h   = angle * lineGaps  - 0.5;
    float hf  = abs(fract (h)-0.5);
    float hd  = fwidth(h);
    
    float mi = max(0.0,lineWidth-1.0);
    float ma = max(1.0,lineWidth);

    
    vec4  band = min(vec4(1.0), unpackUnorm4x8( uint( (floor(abs(angle) * 2) / 90.0) * float(0xFFFFFF))) * 0.8 + 0.3);
    if ((scene.reflection & 2) != 0)
    {
      color  = mix(band, band * color, 0.8);
    }
    if ((scene.reflection & 1) != 0)
    {
      color *= clamp((hf-hd*mi)/(hd*(ma-mi)),max(0.0,1.0-lineWidth),1.0);
    }
  }
  #endif
  
  #if USE_OVERLAY
    color  *= vec4(1, 0.2, 0.2, 0);
  #elif USE_HIGHLIGHT
    if (primID == scene.highlightPrim) {
      color = mix(color,vec4(1,1,0,0), 0.75);
    }
  #endif
  
#if USE_TEXTURE_NORMALS && 0
  color = vec4(wTangent.xyz * 0.5 + 0.5, 1);
#endif

  return color;
}