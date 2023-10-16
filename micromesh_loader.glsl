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
 
/////////////////////////////////////////////////////

// Data loading was separated to ease integration with other renderers/apis

// Flat renderers need to pull per-mesh data from local pointers
// hence the ability to change the variable via macro here.
// Otherwise microdata is provided as a per-draw bound SSBO.
#ifndef mesh_microdata
#define mesh_microdata microdata
#endif

uint  microdata_loadDistance(uint idx)
{
  return mesh_microdata.distances.d[idx];
}

uvec2 microdata_loadDistance2(uint idx)
{
  uvec2s_in distances64 = uvec2s_in(mesh_microdata.distances);
  return distances64.d[idx];
}

uint  microdata_loadMipDistance(uint idx)
{
  return mesh_microdata.mipDistances.d[idx];
}

#ifndef decoder_microdata
#define decoder_microdata microdata
#endif

uint  microdata_loadFormatInfo(uint formatIdx, uint decodeSubdiv)
{
  return uint(decoder_microdata.formats.d[formatIdx].width_start[decodeSubdiv]);
}

#if MICRO_DECODER == MICRO_DECODER_BASETRI_MIP_SHUFFLE
MicromeshBTriVertex
#elif MICRO_DECODER == MICRO_DECODER_MICROTRI_THREAD
MicromeshMTriVertex
#else
MicromeshSTriVertex
#endif
microdata_loadMicromeshVertex(uint idx)
{
  return decoder_microdata.vertices.d[idx];
}
#if MICRO_DECODER == MICRO_DECODER_BASETRI_MIP_SHUFFLE
MicromeshBTriDescend
#elif MICRO_DECODER == MICRO_DECODER_MICROTRI_THREAD
MicromeshMTriDescend
#else
MicromeshSTriDescend
#endif
microdata_loadMicromeshDescend(uint idx)
{
  return decoder_microdata.descendInfos.d[idx];
}

uint microdata_loadTriangleIndices(uint idx)
{
  return decoder_microdata.triangleIndices.d[idx];
}