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

///////////////////////////////////////////////////////////////
// public api

// High-level api intended to be used by mesh or compute shaders.
// Different SubgroupMicromeshDecoder implementations are provided
// in their respective glsl files.
// - `micromesh_decoder_basetri.glsl`
// - `micromesh_decoder_microtri.glsl`
//
// Ideally mesh/compute shaders only use the functions provided
// here. None of the lower-level details.

// how many threads are used for each part, partSubdiv [0,3]
uint smicrodec_getThreadCount(uint partSubdiv);

// sets up the decoder state
// must be called in subgroup-uniform branch
//
// arguments can be divergent in subgroup, but must
// be uniform for threads with equal `cfg.packID`.
void smicrodec_subgroupInit(inout SubgroupMicromeshDecoder sdec,
                            MicroDecoderConfig            cfg,
                            MicromeshBaseTri              microBaseTri,
                            uint                          firstMicro,
                            uint                          firstTriangle,
                            uint                          firstData,
                            uint                          firstMipData);

uint smicrodec_getNumTriangles(inout SubgroupMicromeshDecoder sdec);
uint smicrodec_getNumVertices (inout SubgroupMicromeshDecoder sdec);
uint smicrodec_getMeshTriangle(inout SubgroupMicromeshDecoder sdec);
uint smicrodec_getBaseSubdiv  (inout SubgroupMicromeshDecoder sdec);
uint smicrodec_getMicroSubdiv (inout SubgroupMicromeshDecoder sdec);
uint smicrodec_getFormatIdx   (inout SubgroupMicromeshDecoder sdec);

// the decoding process of vertices and triangles may need multiple iterations 
// within the subgroup (known at compile-time)
uint smicrodec_getIterationCount();

// retrieve the triangle information for the part being decoded
struct MicroDecodedTriangle
{
  bool  valid;      // this thread has work to do
  uint  localIndex; // local index relative to current part
  uint  outIndex;   // output index across parts in all decoders within subgroup
  
  uvec3 indices;    // is adjusted for vertex output index
};

MicroDecodedTriangle smicrodec_getTriangle (inout SubgroupMicromeshDecoder sdec, uint iterationIndex);

// retrieve the vertex information for the part being decoded
// must be called in subgroup-uniform branch
struct MicroDecodedVertex
{
  bool  valid;      // this thread has work to do
  uint  localIndex; // local index relative to current part
  uint  outIndex;   // output index across parts in all decoders within subgroup
  
  ivec2 uv;           // relative to base triangle
#if MICRO_DECODER == MICRO_DECODER_MICROTRI_THREAD && MICRO_MTRI_USE_INTRINSIC
  uint  blasBasePrimitiveID; // base triangle within blas, used with intrinsic
#else
  int   displacement; // raw displacement
#endif
  vec3  bary;         // relative to base triangle
};

MicroDecodedVertex  smicrodec_subgroupGetVertex (inout SubgroupMicromeshDecoder sdec, uint iterationIndex);

vec4 smicrodec_getVertexPos(inout MicroDecodedVertex decoded,
    vec3 v0, vec3 v1, vec3 v2, f16vec3 d0, f16vec3 d1, f16vec3 d2,
    mat4 worldMatrix,
    uint instanceID,
    dispVec2 scale_bias)
{
#if MICRO_DECODER == MICRO_DECODER_MICROTRI_THREAD && MICRO_MTRI_USE_INTRINSIC
    // in this sample the blas is constructed with just one geometry
    int geometryIndex = 0;
    return worldMatrix * vec4(fetchMicroTriangleVertexPositionNV(sceneTlas, int(instanceID), geometryIndex, int(decoded.blasBasePrimitiveID), decoded.uv), 1.0);
#else
    float dispDistance = micromesh_getFloatDisplacement(decoded.displacement, scale_bias);
    
    // Compute interpolation
    vec3 dispDirection = getInterpolated(d0, d1, d2, decoded.bary);
  #if USE_NORMALIZED_DIR
    dispDirection = normalize(dispDirection);
  #endif

    vec3 pos  = getInterpolated(v0, v1, v2, decoded.bary) + vec3(dispDirection) * dispDistance;
    return worldMatrix * vec4(pos, 1.0);
#endif
}