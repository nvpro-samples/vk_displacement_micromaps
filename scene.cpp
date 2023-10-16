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

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <cctype>
#include <filesystem>
#include <nvh/misc.hpp>
#include <platform.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <thread>

#include "config.h"
#include "meshset_gltf.hpp"
#include "meshset_utils.hpp"
#include "parallel_work.hpp"
#include "scene.hpp"

namespace {
// Loads a mesh set from a file. When loading a glTF file, can load
// microdisplacement as well. If the filename is empty or loading fails,
// returns a struct will null elements.
bool loadMeshAndBarySet(MeshSet& meshSet, BaryAttributesSet* barySet, const char* filename)
{
  if(!filename)
    return false;

  size_t      len = strlen(filename);
  std::string ext = len > 3 ? filename + len - 4 : nullptr;
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
  // ext now contains the last four characters of the filename, in lower case.

  if(ext == "gltf" || ext == ".glb")
  {
    return loadGLTF(meshSet, barySet, filename);
  }
  else
  {
    return false;
  }
}
}  // namespace

namespace microdisp {
bool Scene::load(const char* filenameLo)
{
  {
    std::unique_ptr<MeshSet> loadedLo = std::make_unique<MeshSet>();

    if(loadMeshAndBarySet(*loadedLo, &barySet, filenameLo))
    {
      meshSetLo = std::move(loadedLo);
    }
    else
    {
      return false;
    }
  }

  // Set up lo-res mesh
  meshSetLoOrigInstanceCount        = meshSetLo->meshInstances.size();

  if(!meshSetLo->attributes.directionBounds.empty() && !meshSetLo->directionBoundsAreUniform)
  {
    hasDirectionBounds = true;
  }

  {
    if(!barySet.displacements.empty())
    {
      if(!finalizeLoadedBarySet())
      {
        LOGE("%s: Loading worked and at least one mesh had microdisplacement, but finalizeLoadedBaryMesh failed!", __FUNCTION__);
        return false;
      }
    }
  }

  finalizeMeshLo();

  if(!barySet.displacements.empty())
  {
    barySetShellVolume = barySet.computeShellVolume(*meshSetLo, true, true, g_numThreads);
    nvprintfLevel(LOGLEVEL_INFO, "total shell volume %f\n", barySetShellVolume);
  }


  return true;
}



void Scene::finalizeMeshLo()
{
  meshSetLo->setupProcessingGlobals(g_numThreads);

  // we build those on-demand later
  meshSetLo->clearDirectionBoundsGlobals();

  // for baking we want some reference matrix for each mesh
  // here we simply pick a meshInstance with a big bbox
  meshSetLo->setupLargestInstance();
  // for per-instance lod
  meshSetLo->setupEdgeLengths(g_numThreads);
}

bool Scene::finalizeLoadedBarySet()
{
  for(size_t d = 0; d < barySet.displacements.size(); d++)
  {
    BaryDisplacementAttribute& displacementAttr = barySet.displacements[d];

    if(displacementAttr.uncompressed)
    {
      hasUncompressedDisplacement = true;
    }
    else if(displacementAttr.compressed)
    {
      hasCompressedDisplacement = true;
    }
  }

  barySet.fillUniformDirectionBounds(*meshSetLo);
  barySet.updateStats();

  return true;
}
}  // namespace microdisp
