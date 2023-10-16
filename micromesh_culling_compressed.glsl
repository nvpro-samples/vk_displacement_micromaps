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

#define MICRO_LOD_CLAMP   0

uint cullAndLodMicroBaseTriangle(uint microID, inout bool valid)
{
  MicromeshBaseTri microBaseTri  = microdata.basetriangles.d[microID];
  
  mat4 worldMatrix               = mesh.instances.d[push.instanceID].worldMatrix;
  
  uint baseSubdiv      = micromesh_getBaseSubdiv(microBaseTri);
  
#if USE_TRI_LOD
  uint tri         = microID + push.firstTriangle;;
  uvec3 triIndices = uvec3( mesh.indices.d[tri * 3 + 0],
                            mesh.indices.d[tri * 3 + 1],
                            mesh.indices.d[tri * 3 + 2]) + push.firstVertex;

  vec3 v0 = mesh.positions.d[triIndices.x].xyz;
  vec3 v1 = mesh.positions.d[triIndices.y].xyz;
  vec3 v2 = mesh.positions.d[triIndices.z].xyz;
  
  f16vec3 d0 = mesh.dispDirections.d[triIndices.x].xyz;
  f16vec3 d1 = mesh.dispDirections.d[triIndices.y].xyz;
  f16vec3 d2 = mesh.dispDirections.d[triIndices.z].xyz;

#if USE_DIRECTION_BOUNDS
  // fixme this logic is rather crude
  // to be more exact we would need to generate all prismoid
  // positions like the CPU code does

  boundsVec2 bounds0 = mesh.dispDirectionBounds.d[triIndices.x];
  boundsVec2 bounds1 = mesh.dispDirectionBounds.d[triIndices.y];
  boundsVec2 bounds2 = mesh.dispDirectionBounds.d[triIndices.z];
  
  v0 = v0 + d0 * bounds0.x;
  v1 = v1 + d1 * bounds1.x;
  v2 = v2 + d2 * bounds2.x;
  
  d0 = d0 * float16_t(bounds0.y);
  d1 = d1 * float16_t(bounds1.y);
  d2 = d2 * float16_t(bounds2.y);
  
  v0 = v0 + d0 * float16_t(0.5);
  v1 = v1 + d1 * float16_t(0.5);
  v2 = v2 + d2 * float16_t(0.5);
#endif
  
  // Generate vertices  
  vec3 va = (worldMatrix * vec4(v0,1)).xyz;
  vec3 vb = (worldMatrix * vec4(v1,1)).xyz;
  vec3 vc = (worldMatrix * vec4(v2,1)).xyz;
  
  float maxDirLength = max(max(length(d0),
                               length(d1)),
                               length(d2));
  
  int   cullDistance     = micromesh_getCullDist(microBaseTri);
  float maxDisplacement  = micromesh_getFloatDisplacement(cullDistance, push.scale_bias);
  
  uint  lodSubdiv    = baseSubdiv;
  vec3  sphereCenter = (va + vb + vc) / 3.0f;
  float sphereRadius = max(max(length(va - sphereCenter), length(vb - sphereCenter)), length(vc - sphereCenter));
        sphereRadius = sphereRadius + (maxDisplacement * maxDirLength) * length(mat3(worldMatrix) * vec3(1));
#else
  uint  lodSubdiv    = baseSubdiv;
  vec4  sphere       =  microdata.basespheres.d[microID];
  vec3  sphereCenter = (worldMatrix * vec4(sphere.xyz,1)).xyz;
  float sphereRadius = sphere.w * length(mat3(worldMatrix) * vec3(1));
#endif
  
  uint targetSubdiv = MICRO_BIN_INVALID_SUBDIV;
  
#if USE_TRI_LOD
  valid = valid && cullAndLodTriangle(va, vb, vc, sphereCenter, sphereRadius, lodSubdiv, targetSubdiv);
#else
  valid = valid && cullAndLodSphere(sphereCenter, sphereRadius, lodSubdiv, targetSubdiv);
#endif

  if (valid) {
    // subdivTarget is now relative to lodSubdiv, 
    // but we need it relative to baseSubdiv
    uint refSubdiv = baseSubdiv;
    uint refDelta  = refSubdiv;
    #if MICRO_LOD_CLAMP
      refDelta = min(refDelta, MICRO_LOD_CLAMP);
    #endif
    targetSubdiv = (refSubdiv - min(refDelta, lodSubdiv - targetSubdiv));
  }
  
  return targetSubdiv;
}