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

#include "renderer_vk.hpp"
#include <algorithm>
#include <assert.h>
#include <nvh/misc.hpp>
#include <nvh/alignment.hpp>
#include <nvmath/nvmath_glsltypes.h>

#include "micromesh_uncompressed_vk.hpp"

#include "common.h"
#include "common_mesh.h"
#include "common_barymap.h"
#include "common_micromesh_uncompressed.h"
#include "micromesh_binpack_flat_decl.h"


namespace microdisp {
//////////////////////////////////////////////////////////////////////////


class RendererUncompressedVK : public RendererVK
{
public:
  enum DisplacedRenderType
  {
    RENDER_UNCOMPRESSED_MS,
    RENDER_UNCOMPRESSED_SW_CS_FLAT,
  };

  bool m_isFlatSplit = false;

  class TypeUncompressed : public RendererVK::Type
  {
    bool isAvailable(const nvvk::Context* context) const override
    {
      return context->hasDeviceExtension(VK_NV_MESH_SHADER_EXTENSION_NAME);
    }
    const char* name() const override { return "uncompressed ms"; }
    RendererVK* create(ResourcesVK& resources) const override
    {
      RendererUncompressedVK* renderer = new RendererUncompressedVK(resources);
      renderer->m_renderType           = RENDER_UNCOMPRESSED_MS;
      return renderer;
    }
    uint32_t priority() const override { return 0; }
    bool     supportsCompressed() const override { return false; }
  };

  class TypeUncompressedSWCSflat : public RendererVK::Type
  {
    bool        isAvailable(const nvvk::Context* context) const override { return true; }
    const char* name() const override { return "uncompressed cs"; }
    RendererVK* create(ResourcesVK& resources) const override
    {
      RendererUncompressedVK* renderer = new RendererUncompressedVK(resources);
      renderer->m_renderType           = RENDER_UNCOMPRESSED_SW_CS_FLAT;
      return renderer;
    }
    uint32_t priority() const override { return 1; }
    bool     supportsCompressed() const override { return false; }
  };

  class TypeUncompressedSWCSflatSplit : public RendererVK::Type
  {
    bool        isAvailable(const nvvk::Context* context) const override { return true; }
    const char* name() const override { return "uncompressed split cs"; }
    RendererVK* create(ResourcesVK& resources) const override
    {
      RendererUncompressedVK* renderer = new RendererUncompressedVK(resources);
      renderer->m_renderType           = RENDER_UNCOMPRESSED_SW_CS_FLAT;
      renderer->m_isFlatSplit          = true;
      return renderer;
    }
    uint32_t priority() const override { return 1; }
    bool     supportsCompressed() const override { return false; }
  };

public:
  RendererUncompressedVK(ResourcesVK& resources)
      : RendererVK(resources)
  {
  }

  bool init(RenderList& list, const Config& config) override;

  void deinit() override;
  void draw(const FrameConfig& config, nvvk::ProfilerVK& profiler) override;

private:
  struct ShaderIDs
  {
    nvvk::ShaderModuleID meshDisplaced, taskDisplaced, fragmentDisplaced, meshDisplaced_overlay, fragmentDisplaced_overlay;
    nvvk::ShaderModuleID compIndirect;
    nvvk::ShaderModuleID swShadeVertex, swShadeFragment;
  };

  SceneVK*            m_scene = nullptr;
  Config              m_config;
  DisplacedRenderType m_renderType;

  ShaderIDs m_shaders;
  Setup     m_uncompressed;

  VkPipeline m_swBlitPipe = {nullptr};

  VkCommandBuffer m_cmdFlat                   = {nullptr};
  VkCommandBuffer m_cmdOpaque[NUM_CMD_TYPES]  = {0};
  VkCommandBuffer m_cmdOverlay[NUM_CMD_TYPES] = {0};

  struct FlatComputeBuffers
  {
    RBuffer indirect;
    RBuffer scratch;
    RBuffer pushDatas;
    RBuffer ubo;
  };

  FlatComputeBuffers         m_flatBuffers;
  MicromeshSetUncompressedVK m_micromeshSetUncVK;

  void initShaders();
  void deinitShaders();

  void initSetupUncompressed(const SceneVK* scene, const Config& config);
  void initFlatCompute(RenderList& list, const Config& config);

  const Setup& getDisplacedSetup() { return m_uncompressed; }

  VkPipeline getPipeline(const Setup& setup, bool overlay) const
  {
    if(overlay && setup.pipelineOverlay)
    {
      return setup.pipelineOverlay;
    }
    else
    {
      return setup.pipeline;
    }
  }

  bool isRenderTypeSW() const { return (m_renderType == RENDER_UNCOMPRESSED_SW_CS_FLAT); }

  bool isRenderTypeFlat() const { return (m_renderType == RENDER_UNCOMPRESSED_SW_CS_FLAT); }

  bool isRenderTypeCompute() const { return (m_renderType == RENDER_UNCOMPRESSED_SW_CS_FLAT); }

  std::string getShaderPrepend();

  DrawMicromeshUncPushData getDrawMicroUncPushData(const MeshSet&                    meshSet,
                                                   const BaryAttributesSet&          barySet,
                                                   const MicromeshSetUncompressedVK& micromeshSetVK,
                                                   size_t                            instanceIdx) const
  {
    const MeshInstance&                         meshInst         = meshSet.meshInstances[instanceIdx];
    const MeshInfo&                             mesh             = meshSet.meshInfos[meshInst.meshID];
    const MicromeshSetUncompressedVK::MeshData& meshData         = micromeshSetVK.meshDatas[meshInst.meshID];
    const BaryDisplacementAttribute&            baryDisplacement = barySet.displacements[mesh.displacementID];
    const bary::Group& baryGroup = baryDisplacement.uncompressed->groups[mesh.displacementGroup];

    DrawMicromeshUncPushData push;
    push.firstVertex   = mesh.firstVertex;
    push.firstTriangle = (mesh.firstPrimitive);
    push.triangleMax   = (mesh.numPrimitives) - 1;
    push.instanceID    = uint32_t(instanceIdx);
    push.scale_bias    = nvmath::vec2f(baryGroup.floatScale.r, baryGroup.floatBias.r);
    push.binding       = meshData.binding.addr;
    return push;
  }

  VkCommandBuffer generateCmdBufferFlatCompute(RenderList& list, const Config& config)
  {
    const Setup&    setup = getDisplacedSetup();
    VkCommandBuffer cmd   = m_res.createCmdBuffer(m_cmdPool, false, false, true, true);
    nvvk::DebugUtil(m_res.m_device).setObjectName(cmd, "renderer_vk_flat");

    {
      VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      memBarrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);
    }

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, setup.container.getPipeLayout(), 0, 1,
                            setup.container.at(0).getSets(0), 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, setup.pipelineFlat);
    vkCmdDispatchIndirect(cmd, m_flatBuffers.indirect.buffer, 0);

    vkEndCommandBuffer(cmd);

    return cmd;
  }

  VkCommandBuffer generateCmdBufferDisplaced(RenderList& list, const Config& config, bool overlay)
  {
    const SceneVK* scene        = list.m_scene;
    bool           isCompute    = isRenderTypeCompute();
    bool           useNormalMap = config.useNormalMap;

    const Setup&    setup = getDisplacedSetup();
    VkCommandBuffer cmd   = m_res.createCmdBuffer(m_cmdPool, false, false, true, isCompute);
    nvvk::DebugUtil(m_res.m_device).setObjectName(cmd, "renderer_di");

    nvmath::vec2f scale_bias = nvmath::vec2f(1, 0);

    if(!isCompute)
    {
      m_res.cmdDynamicState(cmd);
    }

    const MeshSet&                    meshSet   = *scene->meshSetLo;
    const MeshSetVK&                  meshSetVK = scene->meshSetLoVK;
    const BaryAttributesSet&          barySet   = scene->barySet;
    const MicromeshSetUncompressedVK& barySetVK = m_micromeshSetUncVK;

    VkPipelineBindPoint bindPoint = isCompute ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS;

    if(!isCompute && config.useNormalMap)
    {
      vkCmdBindDescriptorSets(cmd, bindPoint, setup.container.getPipeLayout(), 0, 1,
                              setup.container.at(DSET_TEXTURES).getSets(0), 0, nullptr);
    }

    if(!isCompute)
    {
      vkCmdBindIndexBuffer(cmd, meshSetVK.indices.info.buffer, 0, VK_INDEX_TYPE_UINT32);
    }

    uint32_t   lastMeshID   = MeshSetID::INVALID;
    VkPipeline lastPipeline = nullptr;
    size_t     numObjects   = meshSet.meshInstances.size();
    for(size_t i = 0; i < numObjects; i++)
    {
      size_t              instIdx = i;
      const MeshInstance& inst    = meshSet.meshInstances[instIdx];
      const MeshInfo&     info    = meshSet.meshInfos[inst.meshID];

      // Skip meshes without displacement.
      if(info.displacementID == MeshSetID::INVALID)
        continue;

      if(lastMeshID != inst.meshID)
      {
        vkCmdBindDescriptorSets(cmd, bindPoint, setup.container.getPipeLayout(), 0, 1,
                                setup.container.at(DSET_RENDERER).getSets(inst.meshID), 0, nullptr);

        lastMeshID = inst.meshID;
      }

      const BaryDisplacementAttribute& baryDisplacement = barySet.displacements[info.displacementID];
      // Skip this object if we can't render it with the current renderer.
      if(!baryDisplacement.uncompressed)
        continue;

      VkPipeline pipeline = getPipeline(setup, overlay);
      if(pipeline != lastPipeline)
      {
        vkCmdBindPipeline(cmd, bindPoint, pipeline);
        lastPipeline = pipeline;
      }

      {
        DrawMicromeshUncPushData push          = getDrawMicroUncPushData(meshSet, barySet, barySetVK, instIdx);
        uint                     triangleCount = push.triangleMax + 1;
        uint32_t                 groupCount    = (triangleCount + MICRO_TRI_PER_TASK - 1) / MICRO_TRI_PER_TASK;

        vkCmdPushConstants(cmd, setup.container.getPipeLayout(), setup.stageFlags, 0, sizeof(push), &push);

        if(m_renderType == RENDER_UNCOMPRESSED_SW_CS_FLAT)
        {
          uint32_t batchCount = m_isFlatSplit ? MICRO_FLAT_SPLIT_TASK_GROUPS : MICRO_FLAT_TASK_GROUPS;
          vkCmdDispatch(cmd, (groupCount + batchCount - 1) / batchCount, 1, 1);
        }
        else
        {
          vkCmdDrawMeshTasksNV(cmd, groupCount, 0);
        }
      }
    }

    if(isRenderTypeFlat())
    {
      VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      memBarrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      // simple launch to compute final indirect dispatch count
      vkCmdBindPipeline(cmd, bindPoint, setup.pipelineIndirect);
      vkCmdDispatch(cmd, 1, 1, 1);
    }

    vkEndCommandBuffer(cmd);

    return cmd;
  }
};


static RendererUncompressedVK::TypeUncompressed              s_type_bary_vk;
static RendererUncompressedVK::TypeUncompressedSWCSflat      s_type_bary2_vk;
static RendererUncompressedVK::TypeUncompressedSWCSflatSplit s_type_bary3_vk;

void RendererUncompressedVK::initShaders()
{
  std::string prepend = getShaderPrepend();

  initStandardShaders(prepend);
  if(m_renderType == RENDER_UNCOMPRESSED_MS)
  {
    std::string baseName = m_config.useLod ? "draw_uncompressed_lod." : "draw_uncompressed_basic.";

    m_shaders.taskDisplaced =
        m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_TASK_BIT_NV, baseName + "task.glsl", prepend);
    m_shaders.meshDisplaced =
        m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_MESH_BIT_NV, baseName + "mesh.glsl", prepend);
    m_shaders.meshDisplaced_overlay = m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_MESH_BIT_NV, baseName + "mesh.glsl",
                                                                               prepend + "#define USE_OVERLAY 1\n");

    m_shaders.fragmentDisplaced = m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, baseName + "frag.glsl",
                                                                           "#define USE_OVERLAY 0\n" + prepend);
    m_shaders.fragmentDisplaced_overlay = m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, baseName + "frag.glsl",
                                                                                   "#define USE_OVERLAY 1\n" + prepend);
  }
  else if(m_renderType == RENDER_UNCOMPRESSED_SW_CS_FLAT)
  {
    std::string baseNameTask = m_isFlatSplit ? "drast_uncompressed_lod_flatsplit_" : "drast_uncompressed_lod_flat_";
    std::string baseName     = "drast_uncompressed_lod_flat_";

    if(m_config.useLod)
    {
      prepend += "#define USE_LOD 1\n";
    }
    if(m_isFlatSplit)
    {
      // we know the mesh pass will not have subdiv < MICRO_BIN_SPLIT_SUBDIV
      // those are handled in task
      prepend += "#define MICROBINPACK_NO_SUBSPLIT 1\n";
    }

    m_shaders.meshDisplaced =
        m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, baseName + "mesh.comp.glsl", prepend);
    m_shaders.taskDisplaced =
        m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, baseNameTask + "task.comp.glsl", prepend);
    m_shaders.compIndirect = m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, baseNameTask + "task.comp.glsl",
                                                                      prepend + "#define IS_CONCAT_TASK 1\n");
  }
  if(isRenderTypeSW())
  {
    std::string baseName = "drast_shade_uncompressed";

    m_shaders.swShadeVertex =
        m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, baseName + ".vert.glsl", prepend);
    m_shaders.swShadeFragment =
        m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, baseName + ".frag.glsl", prepend);
  }
}


void RendererUncompressedVK::deinitShaders()
{
  deinitStandardShaders();
  m_res.m_shaderManager.destroyShaderModule(m_shaders.meshDisplaced);
  m_res.m_shaderManager.destroyShaderModule(m_shaders.meshDisplaced_overlay);
  m_res.m_shaderManager.destroyShaderModule(m_shaders.taskDisplaced);
  m_res.m_shaderManager.destroyShaderModule(m_shaders.fragmentDisplaced);
  m_res.m_shaderManager.destroyShaderModule(m_shaders.fragmentDisplaced_overlay);
  m_res.m_shaderManager.destroyShaderModule(m_shaders.compIndirect);
  m_res.m_shaderManager.destroyShaderModule(m_shaders.swShadeFragment);
  m_res.m_shaderManager.destroyShaderModule(m_shaders.swShadeVertex);
}

void RendererUncompressedVK::initFlatCompute(RenderList& list, const Config& config)
{
  const MeshSet& meshLo = *list.m_scene->meshSetLo;

  m_flatBuffers.indirect = m_res.createBuffer(sizeof(VkDispatchIndirectCommand),
                                              VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  // add a bit of safety margin at the end
  size_t flatBufferSize = sizeof(MicroBinPackFlat) * ((size_t(1) << config.maxVisibleBits) + 4096);
  m_flatBuffers.scratch = m_res.createBuffer(flatBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  LOGI("flat visible buffer size: %d MB\n", uint32_t(flatBufferSize / (1024 * 1024)));

  if(m_renderType == RENDER_UNCOMPRESSED_SW_CS_FLAT)
  {
    m_flatBuffers.pushDatas =
        m_res.createBuffer(sizeof(DrawMicromeshUncPushData) * meshLo.meshInstances.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_flatBuffers.ubo = m_res.createBuffer(sizeof(MicromeshUncScratchData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    {
      MicromeshUncScratchData scratchUbo;
      scratchUbo.atomicCounter     = m_flatBuffers.indirect.addr;
      scratchUbo.scratchData       = m_flatBuffers.scratch.addr;
      scratchUbo.instancePushDatas = m_flatBuffers.pushDatas.addr;
      scratchUbo.maxCount          = (1 << config.maxVisibleBits);
      scratchUbo.maxMask           = (1 << config.maxVisibleBits) - 1;

      m_res.simpleUploadBuffer(m_flatBuffers.ubo, &scratchUbo);
    }

    {
      const BaryAttributesSet& barySet = list.m_scene->barySet;

      std::vector<DrawMicromeshUncPushData> pushInstances(meshLo.meshInstances.size());
      for(size_t i = 0; i < meshLo.meshInstances.size(); i++)
      {
        DrawMicromeshUncPushData& push = pushInstances[i];
        // regular instances will never get rendered by flat renderer through this array,
        // so their push data can be kept undefined

        if(meshLo.meshInfos[meshLo.meshInstances[i].meshID].displacementID == MeshSetID::INVALID)
          continue;

        push = getDrawMicroUncPushData(meshLo, barySet, m_micromeshSetUncVK, i);
      }

      m_res.simpleUploadBuffer(m_flatBuffers.pushDatas, pushInstances.data());
    }
  }
}

std::string RendererUncompressedVK::getShaderPrepend()
{
  std::string prepend;

  // "if displacement edge flags exist and any mesh has minSubdivLevel != maxSubdivLevel"
  bool useNonUniform = (m_scene->meshSetLoVK.displacementEdgeFlags.buffer != VK_NULL_HANDLE);
  if(useNonUniform)
  {
    useNonUniform = false;
    for(const BaryDisplacementAttribute& baryDisplacement : m_scene->barySet.displacements)
    {
      if((baryDisplacement.uncompressed
          && baryDisplacement.uncompressed->minSubdivLevel != baryDisplacement.uncompressed->maxSubdivLevel)  //
         || (baryDisplacement.compressed && baryDisplacement.compressed->minSubdivLevel != baryDisplacement.compressed->maxSubdivLevel))
      {
        useNonUniform = true;
        break;
      }
    }
  }

  prepend += nvh::stringFormat("#define USE_NON_UNIFORM_SUBDIV %d\n", useNonUniform ? 1 : 0);
  prepend += nvh::stringFormat("#define USE_MICROVERTEX_NORMALS %d\n", m_config.useMicroVertexNormals ? 1 : 0);
  prepend += nvh::stringFormat("#define USE_TEXTURE_NORMALS %d\n", m_config.useNormalMap ? 1 : 0);
  prepend += nvh::stringFormat(
      "#define USE_DIRECTION_BOUNDS %d\n",
      !m_scene->meshSetLo->directionBoundsAreUniform && m_scene->meshSetLoVK.displacementDirectionBounds.buffer ? 1 : 0);

  {
    prepend += nvh::stringFormat("#define UNCOMPRESSED_UMAJOR %d\n",
                                 m_scene->barySet.uncompressedStats.valueOrder == bary::ValueLayout::eTriangleUmajor ? 1 : 0);

    int32_t bits = 16;

    switch(m_scene->barySet.uncompressedStats.valueFormat)
    {
      case bary::Format::eR8_unorm:
        bits = 8;
        break;
      case bary::Format::eR16_unorm:
        bits = 16;
        break;
      case bary::Format::eR32_sfloat:
        bits = 32;
        break;
      case bary::Format::eR11_unorm_pack16:
        bits = 11;
        break;
      case bary::Format::eR11_unorm_packed_align32:
        bits = -11;
        break;
    }

    prepend += nvh::stringFormat("#define UNCOMPRESSED_DISPLACEMENT_BITS %d\n", bits);
  }

#ifdef _DEBUG
  printf(prepend.c_str());
#endif

  return prepend;
}


bool RendererUncompressedVK::init(RenderList& list, const Config& config)
{
  m_config = config;
  m_scene  = list.m_scene;

  const SceneVK* scene = list.m_scene;

  if(!list.m_scene->hasUncompressedDisplacement)
  {
    return true;
  }

  if(isRenderTypeFlat())
  {
    if(list.m_scene->barySet.uncompressedStats.maxSubdivLevel > MAX_BASE_SUBDIV)
    {
      LOGE("flat renderers do not support maxSubdivLevel > %d\n", MAX_BASE_SUBDIV);
      return false;
    }
  }

  {
    m_micromeshSetUncVK.init(m_res, *scene->meshSetLo, scene->barySet, config.useMicroVertexNormals, config.numThreads);
  }

  if(isRenderTypeFlat())
  {
    initFlatCompute(list, config);
  }

  initShaders();

  if(!m_res.m_shaderManager.areShaderModulesValid())
  {
    return false;
  }

  initSetupStandard(scene, config);
  initSetupUncompressed(scene, config);

  {
    initCommandPool();

    for(uint32_t t = 0; t < NUM_CMD_TYPES; t++)
    {
      switch(t)
      {
        case CMD_TYPE_LO:
          m_cmdOpaque[t]  = generateCmdBufferLo(list, config, false, false);
          m_cmdOverlay[t] = generateCmdBufferLo(list, config, true, false);
          break;
        case CMD_TYPE_DISPLACED:
          m_cmdOpaque[t]  = generateCmdBufferDisplaced(list, config, false);
          m_cmdOverlay[t] = generateCmdBufferDisplaced(list, config, true);
          break;
        case CMD_TYPE_NONDISPLACED_LO:
          m_cmdOpaque[t]  = generateCmdBufferLo(list, config, false, true);
          m_cmdOverlay[t] = generateCmdBufferLo(list, config, true, true);
          break;
        case CMD_TYPE_SHELL:
          m_cmdOpaque[t]  = generateCmdBufferEmpty();
          m_cmdOverlay[t] = generateCmdBufferLo(list, config, true, false, true);
          break;
        default:
          m_cmdOpaque[t]  = generateCmdBufferEmpty();
          m_cmdOverlay[t] = generateCmdBufferEmpty();
          break;
      }
    }

    if(isRenderTypeFlat())
    {
      m_cmdFlat = generateCmdBufferFlatCompute(list, config);
    }
  }

  return true;
}

void RendererUncompressedVK::initSetupUncompressed(const SceneVK* scene, const Config& config)
{
  VkResult           result;
  VkShaderStageFlags stageFlags =
      isRenderTypeSW() ? (VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT) :
                         (VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_TASK_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT);
  m_uncompressed.stageFlags = stageFlags;
  m_uncompressed.container.init(m_res.m_device);
  auto& rendererSet = m_uncompressed.container.at(DSET_RENDERER);
  rendererSet.addBinding(DRAWUNCOMPRESSED_UBO_VIEW, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  rendererSet.addBinding(DRAWUNCOMPRESSED_SSBO_STATS, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageFlags);
  rendererSet.addBinding(DRAWUNCOMPRESSED_UBO_MESH, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  rendererSet.addBinding(DRAWUNCOMPRESSED_UBO_MAP, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  rendererSet.addBinding(DRAWUNCOMPRESSED_UBO_UNCOMPRESSED, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  if(isRenderTypeFlat())
  {
    rendererSet.addBinding(DRAWUNCOMPRESSED_UBO_SCRATCH, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  }
  if(m_config.useOcclusionHiz)
  {
    rendererSet.addBinding(DRAWUNCOMPRESSED_TEX_HIZ, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, stageFlags);
    rendererSet.setBindingFlags(DRAWUNCOMPRESSED_TEX_HIZ, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
  }
  if(isRenderTypeSW())
  {
    rendererSet.addBinding(DRAWUNCOMPRESSED_IMG_ATOMIC, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, stageFlags);
  }
  rendererSet.initLayout();
  rendererSet.initPool(uint32_t(m_micromeshSetUncVK.meshDatas.size()));

  initTextureSet(m_uncompressed, scene, config, VK_SHADER_STAGE_FRAGMENT_BIT);

  VkPushConstantRange pushRange;
  pushRange.offset     = 0;
  pushRange.size       = sizeof(DrawMicromeshUncPushData);
  pushRange.stageFlags = stageFlags;

  m_uncompressed.container.initPipeLayout(0, config.useNormalMap ? 2 : 1, 1, &pushRange);

  std::vector<VkWriteDescriptorSet> dsets;
  for(uint32_t i = 0; i < uint32_t(m_micromeshSetUncVK.meshDatas.size()); i++)
  {
    dsets.push_back(rendererSet.makeWrite(i, DRAWUNCOMPRESSED_UBO_VIEW, &m_res.m_common.view.info));
    dsets.push_back(rendererSet.makeWrite(i, DRAWUNCOMPRESSED_SSBO_STATS, &m_res.m_common.stats.info));
    dsets.push_back(rendererSet.makeWrite(i, DRAWUNCOMPRESSED_UBO_MESH, &scene->meshSetLoVK.binding.info));
    dsets.push_back(rendererSet.makeWrite(i, DRAWUNCOMPRESSED_UBO_MAP, &m_micromeshSetUncVK.baryMap.binding.info));
    dsets.push_back(
        rendererSet.makeWrite(i, DRAWUNCOMPRESSED_UBO_UNCOMPRESSED, &m_micromeshSetUncVK.meshDatas[i].binding.info));
    if(isRenderTypeFlat())
    {
      dsets.push_back(rendererSet.makeWrite(i, DRAWUNCOMPRESSED_UBO_SCRATCH, &m_flatBuffers.ubo.info));
    }
    if(m_config.useOcclusionHiz)
    {
      dsets.push_back(rendererSet.makeWrite(i, DRAWUNCOMPRESSED_TEX_HIZ, &m_res.m_hizUpdate.farImageInfo));
    }
    if(isRenderTypeSW())
    {
      static VkDescriptorImageInfo imgAtomic;
      imgAtomic.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      imgAtomic.imageView   = m_res.m_framebuffer.viewAtomic;
      imgAtomic.sampler     = VK_NULL_HANDLE;
      dsets.push_back(rendererSet.makeWrite(i, DRAWUNCOMPRESSED_IMG_ATOMIC, &imgAtomic));
    }
  }
  vkUpdateDescriptorSets(m_res.m_device, (uint32_t)dsets.size(), dsets.data(), 0, nullptr);

  nvvk::GraphicsPipelineState     state = m_res.m_basicGraphicsState;
  nvvk::GraphicsPipelineGenerator gen(state);
  gen.createInfo.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
  gen.setRenderPass(m_res.m_framebuffer.passPreserve);
  gen.setDevice(m_res.m_device);
  // pipelines
  gen.setLayout(m_uncompressed.container.getPipeLayout());

  if(isRenderTypeSW())
  {
    state.depthStencilState.depthCompareOp = VK_COMPARE_OP_ALWAYS;

    gen.clearShaders();
    gen.addShader(m_res.m_shaderManager.get(m_shaders.swShadeVertex), VK_SHADER_STAGE_VERTEX_BIT);
    gen.addShader(m_res.m_shaderManager.get(m_shaders.swShadeFragment), VK_SHADER_STAGE_FRAGMENT_BIT);
    m_swBlitPipe = gen.createPipeline();
  }

  if(m_renderType == RENDER_UNCOMPRESSED_MS)
  {
    state.depthStencilState.depthCompareOp           = VK_COMPARE_OP_LESS_OR_EQUAL;
    state.rasterizationState.cullMode                = VK_CULL_MODE_BACK_BIT;
    state.rasterizationState.depthBiasEnable         = VK_TRUE;
    state.rasterizationState.depthBiasConstantFactor = -1;
    state.rasterizationState.depthBiasSlopeFactor    = 1;

    gen.clearShaders();
    gen.addShader(m_res.m_shaderManager.get(m_shaders.taskDisplaced), VK_SHADER_STAGE_TASK_BIT_NV);
    gen.addShader(m_res.m_shaderManager.get(m_shaders.meshDisplaced), VK_SHADER_STAGE_MESH_BIT_NV);
    gen.addShader(m_res.m_shaderManager.get(m_shaders.fragmentDisplaced), VK_SHADER_STAGE_FRAGMENT_BIT);
    m_uncompressed.pipeline = gen.createPipeline();

    state.rasterizationState.depthBiasEnable = VK_FALSE;
    state.rasterizationState.polygonMode     = VK_POLYGON_MODE_LINE;
    state.rasterizationState.lineWidth       = 2.0f;
    state.depthStencilState.depthWriteEnable = VK_FALSE;
    gen.clearShaders();
    gen.addShader(m_res.m_shaderManager.get(m_shaders.taskDisplaced), VK_SHADER_STAGE_TASK_BIT_NV);
    gen.addShader(m_res.m_shaderManager.get(m_shaders.meshDisplaced_overlay), VK_SHADER_STAGE_MESH_BIT_NV);
    gen.addShader(m_res.m_shaderManager.get(m_shaders.fragmentDisplaced_overlay), VK_SHADER_STAGE_FRAGMENT_BIT);
    m_uncompressed.pipelineOverlay = gen.createPipeline();
  }
  else if(m_renderType == RENDER_UNCOMPRESSED_SW_CS_FLAT)
  {
    {
      VkComputePipelineCreateInfo createInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      createInfo.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
      createInfo.layout       = m_uncompressed.container.getPipeLayout();
      createInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      createInfo.stage.module = m_res.m_shaderManager.get(m_shaders.taskDisplaced);
      createInfo.stage.pName  = "main";
      createInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;

      result = vkCreateComputePipelines(m_res.m_device, NULL, 1, &createInfo, nullptr, &m_uncompressed.pipeline);
      assert(result == VK_SUCCESS);

      createInfo.stage.module = m_res.m_shaderManager.get(m_shaders.meshDisplaced);
      result = vkCreateComputePipelines(m_res.m_device, NULL, 1, &createInfo, nullptr, &m_uncompressed.pipelineFlat);
      assert(result == VK_SUCCESS);

      createInfo.stage.module = m_res.m_shaderManager.get(m_shaders.compIndirect);
      result = vkCreateComputePipelines(m_res.m_device, NULL, 1, &createInfo, nullptr, &m_uncompressed.pipelineIndirect);
      assert(result == VK_SUCCESS);
    }
  }
}

void RendererUncompressedVK::deinit()
{
  if(m_cmdPool)
  {
    vkFreeCommandBuffers(m_res.m_device, m_cmdPool, NV_ARRAY_SIZE(m_cmdOpaque), m_cmdOpaque);
    vkFreeCommandBuffers(m_res.m_device, m_cmdPool, NV_ARRAY_SIZE(m_cmdOverlay), m_cmdOverlay);
    vkDestroyCommandPool(m_res.m_device, m_cmdPool, nullptr);
  }


  destroyPipelines(m_standard);
  destroyPipelines(m_uncompressed);

  vkDestroyPipeline(m_res.m_device, m_swBlitPipe, nullptr);

  m_standard.container.deinit();
  m_uncompressed.container.deinit();

  m_res.m_allocator.destroy(m_flatBuffers.indirect);
  m_res.m_allocator.destroy(m_flatBuffers.ubo);
  m_res.m_allocator.destroy(m_flatBuffers.scratch);
  m_res.m_allocator.destroy(m_flatBuffers.pushDatas);

  deinitShaders();

  m_micromeshSetUncVK.deinit(m_res);
}

void RendererUncompressedVK::draw(const FrameConfig& config, nvvk::ProfilerVK& profiler)
{
  // generic state setup
  VkCommandBuffer primary = m_res.createTempCmdBuffer();
  auto            sec     = profiler.beginSection("Render", primary);

  vkCmdUpdateBuffer(primary, m_res.m_common.view.buffer, 0, sizeof(SceneData), (const uint32_t*)config.sceneUbo);
  vkCmdUpdateBuffer(primary, m_res.m_common.view.buffer, sizeof(SceneData), sizeof(SceneData), (const uint32_t*)config.sceneUboLast);
  vkCmdFillBuffer(primary, m_res.m_common.stats.buffer, 0, sizeof(ShaderStats), 0);

  if(!m_scene->hasUncompressedDisplacement)
  {
    m_res.cmdPipelineBarrier(primary);
    m_res.cmdBeginRenderPass(primary, true, false);
    vkCmdEndRenderPass(primary);
  }
  else
  {
    if(isRenderTypeFlat())
    {
      VkDispatchIndirectCommand indirect = {0, 1, 1};
      vkCmdUpdateBuffer(primary, m_flatBuffers.indirect.buffer, 0, sizeof(VkDispatchIndirectCommand), (const uint32_t*)&indirect);
    }

    ModelType opaqueType = config.opaque;

    if((isRenderTypeSW() && opaqueType == MODEL_DISPLACED))
    {
      VkClearColorValue cv;
      cv.uint32[0] = 0xFFFFFFFFu;
      cv.uint32[1] = 0xFFFFFFFFu;
      cv.uint32[2] = 0xFFFFFFFFu;
      cv.uint32[3] = 0xFFFFFFFFu;

      VkImageSubresourceRange range;
      range.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      range.baseArrayLayer = 0;
      range.baseMipLevel   = 0;
      range.levelCount     = VK_REMAINING_MIP_LEVELS;
      range.layerCount     = VK_REMAINING_ARRAY_LAYERS;
      vkCmdClearColorImage(primary, m_res.m_framebuffer.imgAtomic, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &range);
    }

    m_res.cmdPipelineBarrier(primary);

    if(opaqueType == MODEL_DISPLACED && isRenderTypeFlat())
    {
      auto secBuild = profiler.timeRecurring("Task", primary);

      vkCmdExecuteCommands(primary, 1, &m_cmdOpaque[CMD_TYPE_DISPLACED]);
    }

    // clear via pass
    {
      auto secDraw = profiler.timeRecurring("Draw", primary);

      if(isRenderTypeFlat() && opaqueType == MODEL_DISPLACED)
      {
        vkCmdExecuteCommands(primary, 1, &m_cmdFlat);
      }
      else
      {
        m_res.cmdBeginRenderPass(primary, true, true);

        // Render the mesh and its overlay.
        VkCommandBuffer cmds[4];
        uint32_t        cmdCount = 2;
        cmds[0]                  = m_cmdOpaque[getCmdType(opaqueType)];
        cmds[1]                  = m_cmdOverlay[getCmdType(config.overlay)];

        if(getCmdType(opaqueType) == CMD_TYPE_DISPLACED)
        {
          cmds[cmdCount++] = m_cmdOpaque[CMD_TYPE_NONDISPLACED_LO];
        }
        if(getCmdType(config.overlay) == CMD_TYPE_DISPLACED)
        {
          cmds[cmdCount++] = m_cmdOverlay[CMD_TYPE_NONDISPLACED_LO];
        }
        vkCmdExecuteCommands(primary, isRenderTypeSW() ? 1 : cmdCount, cmds);

        vkCmdEndRenderPass(primary);
      }

      if((isRenderTypeSW() && opaqueType == MODEL_DISPLACED))
      {
        auto secBlit = profiler.timeRecurring("Blit", primary);

        const Setup& setup = getDisplacedSetup();

        VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memBarrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
        memBarrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(primary, setup.stageFlags, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &memBarrier, 0,
                             nullptr, 0, nullptr);

        m_res.cmdBeginRenderPass(primary, false, false);

        m_res.cmdDynamicState(primary);
        vkCmdBindDescriptorSets(primary, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.container.getPipeLayout(), 0, 1,
                                setup.container.at(0).getSets(0), 0, nullptr);
        vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_GRAPHICS, m_swBlitPipe);
        vkCmdDraw(primary, 3, 1, 0, 0);

        vkCmdEndRenderPass(primary);

        {
          m_res.cmdBeginRenderPass(primary, false, true);

          VkCommandBuffer cmds[1];
          cmds[0] = m_cmdOpaque[CMD_TYPE_NONDISPLACED_LO];
          vkCmdExecuteCommands(primary, 1, cmds);

          vkCmdEndRenderPass(primary);
        }
      }
    }

    if(m_config.useOcclusionHiz && !config.cullFreeze)
    {
      auto secHiZ = profiler.timeRecurring("Hiz", primary);
      m_res.cmdBuildHiz(primary);
    }
  }

  profiler.endSection(sec, primary);
  m_res.cmdCopyStats(primary);
  vkEndCommandBuffer(primary);
  m_res.submissionEnqueue(primary);
}

}  // namespace microdisp
