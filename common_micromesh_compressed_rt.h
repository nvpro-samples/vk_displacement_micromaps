/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

// this file is included by C++ and GLSL

#ifndef _COMMON_MICROMESH_COMPRESSED_RT_H_
#define _COMMON_MICROMESH_COMPRESSED_RT_H_

#include "common_micromesh.h"

struct MicromeshRtAttrTri
{
    uint32_t firstValue;
    uint32_t subdivLevel;
};

#ifndef __cplusplus
layout(buffer_reference, buffer_reference_align = 8, scalar) restrict readonly buffer MicromeshRtAttrTris_in
{
    MicromeshRtAttrTri d[];
};
#endif

struct MicromeshRtAttributes
{
    // subdivLevel,firstValue
    BUFFER_REF(MicromeshRtAttrTris_in)      attrTriangles;
    BUFFER_REF(uints_in)                    attrNormals;
};

struct MicromeshRtData 
{
    BUFFER_REF(uints_in)                    umajor2bmap[MICRO_MAX_LEVELS];
};

#endif