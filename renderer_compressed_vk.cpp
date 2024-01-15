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

#include "micromesh_compressed_rt_vk.hpp"
#include "micromesh_decoder_basetri_vk.hpp"
#include "micromesh_decoder_microtri_vk.hpp"

#include "micromesh_compressed_vk.hpp"

#include "common.h"
#include "common_mesh.h"
#include "common_barymap.h"
#include "common_micromesh_compressed.h"
#include "micromesh_binpack_flat_decl.h"


namespace microdisp {
//////////////////////////////////////////////////////////////////////////


class RendererCompressedVK : public RendererVK
{
public:
  enum DisplacedRenderType
  {
    RENDER_COMPRESSED_MS,
    RENDER_COMPRESSED_SW_CS_FLAT,
  };

  bool m_isFlatSplit = false;

  class TypeCompressed : public RendererVK::Type
  {
    bool isAvailable(const nvvk::Context* context) const override
    {
      return context->hasDeviceExtension(VK_NV_MESH_SHADER_EXTENSION_NAME);
    }
    const char* name() const override { return "compressed ms"; }
    RendererVK* create(ResourcesVK& resources) const override
    {
      RendererCompressedVK* renderer = new RendererCompressedVK(resources);
      renderer->m_renderType         = RENDER_COMPRESSED_MS;
      return renderer;
    }
    uint32_t priority() const override { return 10; }
    bool     supportsCompressed() const override { return true; }
  };

  class TypeCompressedSWCSflat : public RendererVK::Type
  {
    bool        isAvailable(const nvvk::Context* context) const override { return true; }
    const char* name() const override { return "compressed cs"; }
    RendererVK* create(ResourcesVK& resources) const override
    {
      RendererCompressedVK* renderer = new RendererCompressedVK(resources);
      renderer->m_renderType         = RENDER_COMPRESSED_SW_CS_FLAT;
      return renderer;
    }
    uint32_t priority() const override { return 15; }
    bool     supportsCompressed() const override { return true; }
  };

  class TypeCompressedSWCSflatSplit : public RendererVK::Type
  {
    bool        isAvailable(const nvvk::Context* context) const override { return true; }
    const char* name() const override { return "compressed split cs"; }
    RendererVK* create(ResourcesVK& resources) const override
    {
      RendererCompressedVK* renderer = new RendererCompressedVK(resources);
      renderer->m_renderType         = RENDER_COMPRESSED_SW_CS_FLAT;
      renderer->m_isFlatSplit        = true;
      return renderer;
    }
    uint32_t priority() const override { return 15; }
    bool     supportsCompressed() const override { return true; }
  };


public:
  RendererCompressedVK(ResourcesVK& resources)
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

  Setup m_compressed;

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

  FlatComputeBuffers                m_flatBuffers;
  MicromeshSetCompressedVK          m_micromeshSetVK;
  MicromeshSetCompressedRayTracedVK m_micromeshSetRayVK;

  void initShaders();
  void deinitShaders();

  void initSetupCompressed(const SceneVK* scene, const Config& config);
  void initFlatCompute(RenderList& list, const Config& config);

  const Setup& getDisplacedSetup() { return m_compressed; }

  bool isRenderTypeSW() const { return (m_renderType == RENDER_COMPRESSED_SW_CS_FLAT); }

  bool isRenderTypeFlat() const { return (m_renderType == RENDER_COMPRESSED_SW_CS_FLAT); }

  bool isRenderTypeCompute() const { return (m_renderType == RENDER_COMPRESSED_SW_CS_FLAT); }

  std::string getShaderPrepend();


  DrawMicromeshPushData getDrawMicroPushData(const MeshSet&                  meshSet,
                                             const BaryAttributesSet&        barySet,
                                             const MicromeshSetCompressedVK& micromeshSetVK,
                                             size_t                          instanceIdx,
                                             bool                            useBaseMicroTriangles) const
  {
    const MeshInstance&                       meshInst         = meshSet.meshInstances[instanceIdx];
    const MeshInfo&                           mesh             = meshSet.meshInfos[meshInst.meshID];
    const MicromeshSetCompressedVK::MeshData& meshData         = micromeshSetVK.meshDatas[meshInst.meshID];
    const BaryDisplacementAttribute&          baryDisplacement = barySet.displacements[mesh.displacementID];
    const bary::Group&                        baryGroup = baryDisplacement.compressed->groups[mesh.displacementGroup];

    DrawMicromeshPushData push;
    push.firstTriangle = mesh.firstPrimitive;
    push.firstVertex   = mesh.firstVertex;
    push.instanceID    = uint32_t(instanceIdx);
    push.microMax      = meshData.microTriangleCount - 1;
    push.scale_bias    = {float16_t(baryGroup.floatScale.r), float16_t(baryGroup.floatBias.r)};
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
    const SceneVK* scene                 = list.m_scene;
    bool           isCompute             = isRenderTypeCompute();
    bool           useNormalMap          = config.useNormalMap;
    bool           useBaseMicroTriangles = m_micromeshSetVK.hasBaseTriangles;

    const Setup&    setup = getDisplacedSetup();
    VkCommandBuffer cmd   = m_res.createCmdBuffer(m_cmdPool, false, false, true, isCompute);
    nvvk::DebugUtil(m_res.m_device).setObjectName(cmd, "renderer_di");

    glm::vec2 scale_bias = glm::vec2(1, 0);

    if(!isCompute)
    {
      m_res.cmdDynamicState(cmd);
    }

    const MeshSet&           meshSet   = *scene->meshSetLo;
    const MeshSetVK&         meshSetVK = scene->meshSetLoVK;
    const BaryAttributesSet& barySet   = scene->barySet;

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
      if(!baryDisplacement.compressed)
        continue;

      VkPipeline pipeline = getPipeline(setup, overlay);
      if(pipeline != lastPipeline)
      {
        vkCmdBindPipeline(cmd, bindPoint, pipeline);
        lastPipeline = pipeline;
      }

      {
        DrawMicromeshPushData push = getDrawMicroPushData(meshSet, barySet, m_micromeshSetVK, instIdx, useBaseMicroTriangles);
        uint32_t micromeshCount = push.microMax + 1;
        uint32_t groupCount     = (micromeshCount + MICRO_TRI_PER_TASK - 1) / MICRO_TRI_PER_TASK;

        vkCmdPushConstants(cmd, setup.container.getPipeLayout(), setup.stageFlags, 0, sizeof(push), &push);

        if(m_renderType == RENDER_COMPRESSED_SW_CS_FLAT)
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


static RendererCompressedVK::TypeCompressed              s_type_umesh_vk;
static RendererCompressedVK::TypeCompressedSWCSflat      s_type_umesh_swcs2_vk;
static RendererCompressedVK::TypeCompressedSWCSflatSplit s_type_umesh_swcs3_vk;

void RendererCompressedVK::initShaders()
{
  std::string prepend = getShaderPrepend();

  initStandardShaders(prepend);

  if(m_renderType == RENDER_COMPRESSED_MS)
  {
    std::string baseName = m_config.useLod ? "draw_compressed_lod." : "draw_compressed_basic.";

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
  else if(m_renderType == RENDER_COMPRESSED_SW_CS_FLAT)
  {
    std::string baseNameTask = m_isFlatSplit ? "drast_compressed_lod_flatsplit_" : "drast_compressed_lod_flat_";
    std::string baseName     = "drast_compressed_lod_flat_";

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
    std::string baseName = "drast_shade_compressed";

    m_shaders.swShadeVertex =
        m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, baseName + ".vert.glsl", prepend);
    m_shaders.swShadeFragment =
        m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, baseName + ".frag.glsl", prepend);
  }
}


void RendererCompressedVK::deinitShaders()
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

void RendererCompressedVK::initFlatCompute(RenderList& list, const Config& config)
{
  const MeshSet& meshLo = *list.m_scene->meshSetLo;

  m_flatBuffers.indirect = m_res.createBuffer(sizeof(VkDispatchIndirectCommand),
                                              VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  // add a bit of safety margin at the end
  size_t flatBufferSize = sizeof(MicroBinPackFlat) * ((size_t(1) << config.maxVisibleBits) + 4096);
  m_flatBuffers.scratch = m_res.createBuffer(flatBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  LOGI("flat visible buffer size: %d MB\n", uint32_t(flatBufferSize / (1024 * 1024)));

  if(m_renderType == RENDER_COMPRESSED_SW_CS_FLAT)
  {
    bool useMicroBaseTriangles = m_micromeshSetVK.hasBaseTriangles;

    m_flatBuffers.pushDatas =
        m_res.createBuffer(sizeof(DrawMicromeshPushData) * meshLo.meshInstances.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_flatBuffers.ubo = m_res.createBuffer(sizeof(MicromeshScratchData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    {
      MicromeshScratchData scratchUbo;
      scratchUbo.atomicCounter     = m_flatBuffers.indirect.addr;
      scratchUbo.scratchData       = m_flatBuffers.scratch.addr;
      scratchUbo.instancePushDatas = m_flatBuffers.pushDatas.addr;
      scratchUbo.maxCount          = (1 << config.maxVisibleBits);
      scratchUbo.maxMask           = (1 << config.maxVisibleBits) - 1;

      m_res.simpleUploadBuffer(m_flatBuffers.ubo, &scratchUbo);
    }

    {
      const BaryAttributesSet& barySet = list.m_scene->barySet;

      std::vector<DrawMicromeshPushData> pushInstances(meshLo.meshInstances.size());
      for(size_t i = 0; i < meshLo.meshInstances.size(); i++)
      {
        DrawMicromeshPushData& push = pushInstances[i];

        // regular instances will never get rendered by flat renderer through this array,
        // so their push data can be kept undefined

        if(meshLo.meshInfos[meshLo.meshInstances[i].meshID].displacementID == MeshSetID::INVALID)
          continue;

        push = getDrawMicroPushData(meshLo, barySet, m_micromeshSetVK, i, useMicroBaseTriangles);
      }

      m_res.simpleUploadBuffer(m_flatBuffers.pushDatas, pushInstances.data());
    }
  }
}

std::string RendererCompressedVK::getShaderPrepend()
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
    switch(m_config.decoderType)
    {
      case RendererVK::DECODER_BASETRI_MIP:
        prepend += nvh::stringFormat("#define MICRO_DECODER MICRO_DECODER_BASETRI_MIP_SHUFFLE\n");
        break;
      case RendererVK::DECODER_MICROTRI:
        prepend += nvh::stringFormat("#define MICRO_MTRI_USE_INTRINSIC 0\n");
        prepend += nvh::stringFormat("#define MICRO_DECODER MICRO_DECODER_MICROTRI_THREAD\n");
        break;
      case RendererVK::DECODER_MICROTRI_INTRINSIC:
        prepend += nvh::stringFormat("#define MICRO_MTRI_USE_INTRINSIC 1\n");
        prepend += nvh::stringFormat("#define MICRO_DECODER MICRO_DECODER_MICROTRI_THREAD\n");
        break;
      default:
        assert(0);
        break;
    }

    // always must enable the basic format for decoder logic
    uint32_t formatBits = 1 << MICRO_FORMAT_64T_512B;

    if(m_micromeshSetVK.usedFormats[uint32_t(bary::BlockFormatDispC1::eR11_unorm_lvl3_pack512)])
    {
      formatBits |= 1 << MICRO_FORMAT_64T_512B;
    }
    if(m_micromeshSetVK.usedFormats[uint32_t(bary::BlockFormatDispC1::eR11_unorm_lvl4_pack1024)])
    {
      formatBits |= 1 << MICRO_FORMAT_256T_1024B;
    }
    if(m_micromeshSetVK.usedFormats[uint32_t(bary::BlockFormatDispC1::eR11_unorm_lvl5_pack1024)])
    {
      formatBits |= 1 << MICRO_FORMAT_1024T_1024B;
    }
    prepend += nvh::stringFormat("#define MICRO_SUPPORTED_FORMAT_BITS %d\n", formatBits);
    prepend += nvh::stringFormat("#define SHADING_UMAJOR %d\n",
                                 m_scene->barySet.shadingStats.valueOrder == bary::ValueLayout::eTriangleUmajor ? 1 : 0);
  }

#ifdef _DEBUG
  printf(prepend.c_str());
#endif

  return prepend;
}


bool RendererCompressedVK::init(RenderList& list, const Config& config)
{
  m_config = config;
  m_scene  = list.m_scene;

  const SceneVK* scene = list.m_scene;

  if(!list.m_scene->hasCompressedDisplacement)
  {
    return true;
  }

  if(isRenderTypeFlat())
  {
    if(list.m_scene->barySet.compressedStats.maxSubdivLevel > MAX_BASE_SUBDIV)
    {
      LOGE("flat renderers do not support maxSubdivLevel > %d\n", MAX_BASE_SUBDIV);
      return false;
    }
  }

  {
    bool useFallback           = true;
    bool useMicroBaseTriangles = false;

    if(config.decoderType == RendererVK::DECODER_MICROTRI_INTRINSIC
       && m_res.m_context->hasDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME))
    {
      // create VkMicromaps for displacement
      m_micromeshSetRayVK.init(m_res, *scene->meshSetLo, scene->barySet, config.numThreads);
      // build raytracing scene with micromaps
      list.m_scene->meshSetLoVK.deinitRayTracing(m_res);
      list.m_scene->meshSetLoVK.initRayTracingGeometry(m_res, *scene->meshSetLo, &m_micromeshSetRayVK);
      list.m_scene->meshSetLoVK.initRayTracingScene(m_res, *scene->meshSetLo, &m_micromeshSetRayVK);

      MicromeshMicroTriangleDecoderVK setup(m_micromeshSetVK);
      if(setup.init(m_res, *scene->meshSetLo, scene->barySet, config.useMicroVertexNormals, true, config.numThreads))
      {
        useMicroBaseTriangles = true;
        m_config.decoderType  = config.decoderType;
        useFallback           = false;
      }
    }
    else if(config.decoderType == RendererVK::DECODER_BASETRI_MIP)
    {
      MicromeshBaseTriangleDecoderVK setup(m_micromeshSetVK);
      if(setup.init(m_res, *scene->meshSetLo, scene->barySet, config.useMicroVertexNormals, config.numThreads))
      {
        useMicroBaseTriangles = true;
        m_config.decoderType  = config.decoderType;
        useFallback           = false;
      }
    }

    if(useFallback || config.decoderType == RendererVK::DECODER_MICROTRI)
    {
      MicromeshMicroTriangleDecoderVK setup(m_micromeshSetVK);
      if(setup.init(m_res, *scene->meshSetLo, scene->barySet, config.useMicroVertexNormals, false, config.numThreads))
      {
        useMicroBaseTriangles = true;
        m_config.decoderType  = RendererVK::DECODER_MICROTRI;
        useFallback           = false;
      }
    }
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
  initSetupCompressed(scene, config);
  initCommandPool();

  {
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

void RendererCompressedVK::initSetupCompressed(const SceneVK* scene, const Config& config)
{
  VkResult           result;
  VkShaderStageFlags stageFlags =
      isRenderTypeSW() ? (VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT) :
                         (VK_SHADER_STAGE_MESH_BIT_NV | VK_SHADER_STAGE_TASK_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT);

  m_compressed.stageFlags = stageFlags;
  m_compressed.container.init(m_res.m_device);
  auto& rendererSet = m_compressed.container.at(DSET_RENDERER);
  rendererSet.addBinding(DRAWCOMPRESSED_UBO_VIEW, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  rendererSet.addBinding(DRAWCOMPRESSED_SSBO_STATS, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageFlags);
  rendererSet.addBinding(DRAWCOMPRESSED_UBO_MESH, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  rendererSet.addBinding(DRAWCOMPRESSED_UBO_COMPRESSED, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  if(m_renderType == RENDER_COMPRESSED_SW_CS_FLAT)
  {
    rendererSet.addBinding(DRAWCOMPRESSED_UBO_SCRATCH, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  }
  if(m_config.useOcclusionHiz)
  {
    rendererSet.addBinding(DRAWCOMPRESSED_TEX_HIZ, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, stageFlags);
    rendererSet.setBindingFlags(DRAWCOMPRESSED_TEX_HIZ, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
  }
  if(isRenderTypeSW())
  {
    rendererSet.addBinding(DRAWCOMPRESSED_IMG_ATOMIC, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, stageFlags);
  }
  if(m_config.decoderType == RendererCompressedVK::DECODER_MICROTRI_INTRINSIC)
  {
    rendererSet.addBinding(DRAWCOMPRESSED_ACC, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, stageFlags);
  }
  rendererSet.initLayout();
  rendererSet.initPool(uint32_t(m_micromeshSetVK.meshDatas.size()));

  initTextureSet(m_compressed, scene, config, VK_SHADER_STAGE_FRAGMENT_BIT);

  VkPushConstantRange pushRange;
  pushRange.offset     = 0;
  pushRange.size       = sizeof(DrawMicromeshPushData);
  pushRange.stageFlags = stageFlags;

  m_compressed.container.initPipeLayout(0, config.useNormalMap ? 2 : 1, 1, &pushRange);

  std::vector<VkWriteDescriptorSet> dsets;
  for(uint32_t i = 0; i < uint32_t(m_micromeshSetVK.meshDatas.size()); i++)
  {
    dsets.push_back(rendererSet.makeWrite(i, DRAWCOMPRESSED_UBO_VIEW, &m_res.m_common.view.info));
    dsets.push_back(rendererSet.makeWrite(i, DRAWCOMPRESSED_SSBO_STATS, &m_res.m_common.stats.info));
    dsets.push_back(rendererSet.makeWrite(i, DRAWCOMPRESSED_UBO_MESH, &scene->meshSetLoVK.binding.info));
    dsets.push_back(rendererSet.makeWrite(i, DRAWCOMPRESSED_UBO_COMPRESSED, &m_micromeshSetVK.meshDatas[i].binding.info));
    if(isRenderTypeFlat())
    {
      dsets.push_back(rendererSet.makeWrite(i, DRAWCOMPRESSED_UBO_SCRATCH, &m_flatBuffers.ubo.info));
    }
    if(m_config.useOcclusionHiz)
    {
      dsets.push_back(rendererSet.makeWrite(i, DRAWCOMPRESSED_TEX_HIZ, &m_res.m_hizUpdate.farImageInfo));
    }
    if(isRenderTypeSW())
    {
      static VkDescriptorImageInfo imgAtomic;
      imgAtomic.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      imgAtomic.imageView   = m_res.m_framebuffer.viewAtomic;
      imgAtomic.sampler     = VK_NULL_HANDLE;
      dsets.push_back(rendererSet.makeWrite(i, DRAWCOMPRESSED_IMG_ATOMIC, &imgAtomic));
    }
    if(m_config.decoderType == RendererCompressedVK::DECODER_MICROTRI_INTRINSIC)
    {
      dsets.push_back(rendererSet.makeWrite(i, DRAWCOMPRESSED_ACC, &scene->meshSetLoVK.sceneTlasInfo));
    }
  }
  vkUpdateDescriptorSets(m_res.m_device, (uint32_t)dsets.size(), dsets.data(), 0, nullptr);

  nvvk::GraphicsPipelineState     state = m_res.m_basicGraphicsState;
  nvvk::GraphicsPipelineGenerator gen(state);
  gen.createInfo.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
  gen.setRenderPass(m_res.m_framebuffer.passPreserve);
  gen.setDevice(m_res.m_device);
  // pipelines
  gen.setLayout(m_compressed.container.getPipeLayout());

  if(isRenderTypeSW())
  {
    state.depthStencilState.depthCompareOp = VK_COMPARE_OP_ALWAYS;

    gen.clearShaders();
    gen.addShader(m_res.m_shaderManager.get(m_shaders.swShadeVertex), VK_SHADER_STAGE_VERTEX_BIT);
    gen.addShader(m_res.m_shaderManager.get(m_shaders.swShadeFragment), VK_SHADER_STAGE_FRAGMENT_BIT);
    m_swBlitPipe = gen.createPipeline();
  }

  if(m_renderType == RENDER_COMPRESSED_MS)
  {
    state.depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS;
    state.rasterizationState.cullMode      = VK_CULL_MODE_BACK_BIT;

    state.depthStencilState.depthWriteEnable         = VK_TRUE;
    state.depthStencilState.depthCompareOp           = VK_COMPARE_OP_LESS_OR_EQUAL;
    state.rasterizationState.depthBiasEnable         = VK_TRUE;
    state.rasterizationState.depthBiasConstantFactor = -1;
    state.rasterizationState.depthBiasSlopeFactor    = 1;
    state.rasterizationState.polygonMode             = VK_POLYGON_MODE_FILL;
    state.rasterizationState.lineWidth               = 1.0f;

    gen.clearShaders();
    gen.addShader(m_res.m_shaderManager.get(m_shaders.taskDisplaced), VK_SHADER_STAGE_TASK_BIT_NV);
    gen.addShader(m_res.m_shaderManager.get(m_shaders.meshDisplaced), VK_SHADER_STAGE_MESH_BIT_NV);
    gen.addShader(m_res.m_shaderManager.get(m_shaders.fragmentDisplaced), VK_SHADER_STAGE_FRAGMENT_BIT);
    m_compressed.pipeline = gen.createPipeline();

    state.rasterizationState.depthBiasEnable = VK_FALSE;
    state.rasterizationState.polygonMode     = VK_POLYGON_MODE_LINE;
    state.rasterizationState.lineWidth       = 2.0f;
    state.depthStencilState.depthWriteEnable = VK_FALSE;
    gen.clearShaders();
    gen.addShader(m_res.m_shaderManager.get(m_shaders.taskDisplaced), VK_SHADER_STAGE_TASK_BIT_NV);
    gen.addShader(m_res.m_shaderManager.get(m_shaders.meshDisplaced_overlay), VK_SHADER_STAGE_MESH_BIT_NV);
    gen.addShader(m_res.m_shaderManager.get(m_shaders.fragmentDisplaced_overlay), VK_SHADER_STAGE_FRAGMENT_BIT);
    m_compressed.pipelineOverlay = gen.createPipeline();
  }
  else if(m_renderType == RENDER_COMPRESSED_SW_CS_FLAT)
  {
    VkComputePipelineCreateInfo createInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    createInfo.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    createInfo.layout       = m_compressed.container.getPipeLayout();
    createInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    createInfo.stage.module = m_res.m_shaderManager.get(m_shaders.taskDisplaced);
    createInfo.stage.pName  = "main";
    createInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;

    result = vkCreateComputePipelines(m_res.m_device, NULL, 1, &createInfo, nullptr, &m_compressed.pipeline);
    assert(result == VK_SUCCESS);

    createInfo.stage.module = m_res.m_shaderManager.get(m_shaders.meshDisplaced);
    result = vkCreateComputePipelines(m_res.m_device, NULL, 1, &createInfo, nullptr, &m_compressed.pipelineFlat);
    assert(result == VK_SUCCESS);

    createInfo.stage.module = m_res.m_shaderManager.get(m_shaders.compIndirect);
    result = vkCreateComputePipelines(m_res.m_device, NULL, 1, &createInfo, nullptr, &m_compressed.pipelineIndirect);
    assert(result == VK_SUCCESS);
  }
}


void RendererCompressedVK::deinit()
{
  if(m_cmdPool)
  {
    vkFreeCommandBuffers(m_res.m_device, m_cmdPool, NV_ARRAY_SIZE(m_cmdOpaque), m_cmdOpaque);
    vkFreeCommandBuffers(m_res.m_device, m_cmdPool, NV_ARRAY_SIZE(m_cmdOverlay), m_cmdOverlay);
    vkDestroyCommandPool(m_res.m_device, m_cmdPool, nullptr);
  }

  destroyPipelines(m_standard);
  destroyPipelines(m_compressed);

  vkDestroyPipeline(m_res.m_device, m_swBlitPipe, nullptr);

  m_standard.container.deinit();
  m_compressed.container.deinit();

  m_res.m_allocator.destroy(m_flatBuffers.indirect);
  m_res.m_allocator.destroy(m_flatBuffers.ubo);
  m_res.m_allocator.destroy(m_flatBuffers.scratch);
  m_res.m_allocator.destroy(m_flatBuffers.pushDatas);

  deinitShaders();

  m_micromeshSetRayVK.deinit(m_res);
  m_micromeshSetVK.deinit(m_res);
}

void RendererCompressedVK::draw(const FrameConfig& config, nvvk::ProfilerVK& profiler)
{
  // generic state setup
  VkCommandBuffer primary = m_res.createTempCmdBuffer();
  auto            sec     = profiler.beginSection("Render", primary);

  vkCmdUpdateBuffer(primary, m_res.m_common.view.buffer, 0, sizeof(SceneData), (const uint32_t*)config.sceneUbo);
  vkCmdUpdateBuffer(primary, m_res.m_common.view.buffer, sizeof(SceneData), sizeof(SceneData), (const uint32_t*)config.sceneUboLast);
  vkCmdFillBuffer(primary, m_res.m_common.stats.buffer, 0, sizeof(ShaderStats), 0);

  if(!m_scene->hasCompressedDisplacement)
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

    if(isRenderTypeSW() && opaqueType == MODEL_DISPLACED)
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

      if(isRenderTypeSW() && opaqueType == MODEL_DISPLACED)
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
