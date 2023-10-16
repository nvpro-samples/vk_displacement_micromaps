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

// Interface that describes key configuration of which
// portion of triangles and at what level of detail
// are to be decoded.

struct MicroDecoderConfig {
  // baseTriID or subTriID
  // depending on decoder type
  uint microID;
  
  // a micromesh may need multiple parts to be decoded, each part has a maximum of
  // MICRO_PART_MAX_PRIMITIVES triangles (64) and
  // MICRO_PART_MAX_VERTICES   vertices  (45)
  uint partID;
  // subdivision resolution of the part being decoded [0,3]
  uint partSubdiv; 
  
  // target subdivision [0, microSubdiv]
  uint targetSubdiv;
  
  // When multiple decoder states are packed within a subgroup
  // packID is the unique identifier for each such state.
  // Each pack will use packThreads many threads, and packThreadID
  // specifies which thread is calling the function.
  // packThreads * packID + packThreadID < SUBGROUP_SIZE
    
  uint packID;
  uint packThreads;
  uint packThreadID;
  
  // decoder could be invalid
  // if subgroup contains some that are not
  // supposed to participate
  bool valid;
};
