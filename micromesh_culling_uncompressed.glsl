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


//////////////////////////////////////////////////////////////////////////

uint cullAndLodBaseTriangle(uint triLocal, uint baseSubdiv, inout bool valid)
{
  mat4 worldMatrix         = mesh.instances.d[push.instanceID].worldMatrix;
  
  uint tri = triLocal + push.firstTriangle;
  
#if USE_TRI_LOD
  uvec3 triIndices = uvec3( mesh.indices.d[tri * 3 + 0],
                            mesh.indices.d[tri * 3 + 1],
                            mesh.indices.d[tri * 3 + 2]) + push.firstVertex;
  
  // Generate vertices  
  vec3 va = (worldMatrix * vec4(mesh.positions.d[triIndices.x].xyz,1)).xyz;
  vec3 vb = (worldMatrix * vec4(mesh.positions.d[triIndices.y].xyz,1)).xyz;
  vec3 vc = (worldMatrix * vec4(mesh.positions.d[triIndices.z].xyz,1)).xyz;
  
  float maxDisplacement = dbitsUnpack(microdata.triangleBitsMinMax.d[triLocal * 2 + 1]);
  maxDisplacement = maxDisplacement * push.scale_bias.x + push.scale_bias.y;
  
  uint  lodSubdiv    = baseSubdiv;
  vec3  sphereCenter = (va + vb + vc) / 3.0f;
  float sphereRadius = max(max(length(va - sphereCenter), length(vb - sphereCenter)), length(vc - sphereCenter));
        sphereRadius = sphereRadius + (maxDisplacement) * length(mat3(worldMatrix) * vec3(1));
#else
  uint  lodSubdiv    = baseSubdiv;
  vec4  sphere       = microdata.basespheres.d[triLocal];
  vec3  sphereCenter = (worldMatrix * vec4(sphere.xyz,1)).xyz;
  float sphereRadius = sphere.w * length(mat3(worldMatrix) * vec3(1));
#endif
  
  uint targetSubdiv = MICRO_BIN_INVALID_SUBDIV;

#if USE_TRI_LOD
  valid = valid && cullAndLodTriangle(va, vb, vc, sphereCenter, sphereRadius, lodSubdiv, targetSubdiv);
#else
  valid = valid && cullAndLodSphere(sphereCenter, sphereRadius, lodSubdiv, targetSubdiv);
#endif
  
  return targetSubdiv;
}