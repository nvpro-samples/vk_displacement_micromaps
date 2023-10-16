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

// this file is included by C++ and GLSL

#ifndef _COMMON_MICROMESH_H_
#define _COMMON_MICROMESH_H_

#define MICRO_GROUP_SIZE        SUBGROUP_SIZE
#define MICRO_TRI_PER_TASK      SUBGROUP_SIZE

// maximum subdiv overall
#define MICRO_MAX_SUBDIV            5
#define MICRO_MAX_LEVELS            (MICRO_MAX_SUBDIV+1)
#define MICRO_MAX_TRIANGLES         (1u << (MICRO_MAX_SUBDIV*2))

// following influence the workgroup size for the respective
// compute shaders used in compute rasterization.
//
// how many groups in the mesh shading phase
// can be changed via shader-reload
#define MICRO_FLAT_MESH_GROUPS        2

// how many groups in the task shading phase
// cannot be changed at runtime
#define MICRO_FLAT_TASK_GROUPS        2
#define MICRO_FLAT_SPLIT_TASK_GROUPS  1

#endif