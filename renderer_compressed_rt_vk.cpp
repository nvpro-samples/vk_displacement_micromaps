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

#include "common.h"
#include "common_mesh.h"
#include "common_barymap.h"
#include "common_micromesh_compressed.h"
#include "micromesh_binpack_flat_decl.h"


namespace microdisp {
//////////////////////////////////////////////////////////////////////////


class RendererCompressedRayVK : public RendererVK
{
public:
  enum DisplacedRenderType
  {
    RENDER_COMPRESSED_RAY,
  };

  class TypeCompressedRay : public RendererVK::Type
  {
    bool isAvailable(const nvvk::Context* context) const override
    {
      // we either rely on the real extension
      // or if g_enableMicromeshRTExtensions is false enable this renderer always (it won't do the real work but is useful dummy)
      return context->hasDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME) || !g_enableMicromeshRTExtensions;
    }
    const char* name() const override { return "compressed ray"; }
    RendererVK* create(ResourcesVK& resources) const override
    {
      RendererCompressedRayVK* renderer = new RendererCompressedRayVK(resources);
      renderer->m_renderType            = RENDER_COMPRESSED_RAY;
      return renderer;
    }
    uint32_t priority() const override { return 20; }
    bool supportsCompressed() const override { return true; }
  };


public:
  RendererCompressedRayVK(ResourcesVK& resources)
      : RendererVK(resources)
  {
  }

  bool init(RenderList& list, const Config& config) override;

  void deinit() override;
  void draw(const FrameConfig& config, nvvk::ProfilerVK& profiler) override;

private:
  struct ShaderIDs
  {
    nvvk::ShaderModuleID vertex, fragment, fragment_overlay;
    nvvk::ShaderModuleID vertexRay, fragmentRay;

    nvvk::ShaderModuleID rgen;
    nvvk::ShaderModuleID rhit;
    nvvk::ShaderModuleID rhitMicro;
    nvvk::ShaderModuleID rmiss;
  };

  SceneVK*            m_scene = nullptr;
  Config              m_config;
  DisplacedRenderType m_renderType;

  ShaderIDs m_shaders;
  Setup     m_compressedRay;

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rayProperties;

  struct RaySBT
  {
    VkStridedDeviceAddressRegionKHR gen;
    VkStridedDeviceAddressRegionKHR miss;
    VkStridedDeviceAddressRegionKHR hit;
    VkStridedDeviceAddressRegionKHR callable;

    RBuffer buffer;
  };

  RaySBT m_raySBT;

  VkPipeline m_swBlitPipe = {nullptr};

  VkCommandBuffer m_cmdOpaque[NUM_CMD_TYPES]  = {0};
  VkCommandBuffer m_cmdOverlay[NUM_CMD_TYPES] = {0};

  MicromeshSetCompressedRayTracedVK m_micromeshSetRayVK;

  void initShaders();
  void deinitShaders();

  void initSetupCompressedRay(const SceneVK* scene, const Config& config);

  const Setup& getDisplacedSetup() { return m_compressedRay; }

  std::string getShaderPrepend();

  VkCommandBuffer generateCmdBufferDisplaced(RenderList& list, const Config& config, bool overlay)
  {
    const SceneVK* scene        = list.m_scene;
    bool           useNormalMap = config.useNormalMap;

    const Setup&    setup = getDisplacedSetup();
    VkCommandBuffer cmd   = m_res.createCmdBuffer(m_cmdPool, false, false, true, true);
    nvvk::DebugUtil(m_res.m_device).setObjectName(cmd, "renderer_di");

    glm::vec2 scale_bias = glm::vec2(1, 0);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, setup.container.getPipeLayout(), 0, 1,
                            setup.container.at(DSET_RENDERER).getSets(0), 0, nullptr);

    if(config.useNormalMap)
    {
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, setup.container.getPipeLayout(), 0, 1,
                              setup.container.at(DSET_TEXTURES).getSets(0), 0, nullptr);
    }

    VkPipeline lastPipeline = nullptr;
    {
      VkPipeline pipeline = getPipeline(setup, overlay);
      if(pipeline != lastPipeline)
      {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
        lastPipeline = pipeline;
      }

      vkCmdTraceRaysKHR(cmd, &m_raySBT.gen, &m_raySBT.miss, &m_raySBT.hit, &m_raySBT.callable,
                        m_res.m_framebuffer.renderWidth, m_res.m_framebuffer.renderHeight, 1);
    }

    vkEndCommandBuffer(cmd);
    return cmd;
  }
};


static RendererCompressedRayVK::TypeCompressedRay s_type_umesh_ray_vk;

void RendererCompressedRayVK::initShaders()
{
  std::string prepend = getShaderPrepend();

  initStandardShaders(prepend);

  // shaders
  m_shaders.vertex = m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "draw_standard.vert.glsl", prepend);
  m_shaders.fragment = m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "draw_standard.frag.glsl",
                                                                "#define USE_OVERLAY 0\n" + prepend);
  m_shaders.fragment_overlay = m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "draw_standard.frag.glsl",
                                                                        "#define USE_OVERLAY 1\n" + prepend);

  if(m_renderType == RENDER_COMPRESSED_RAY)
  {


    m_shaders.vertexRay = m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "dray_blit.vert.glsl");
    m_shaders.fragmentRay = m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "dray_blit.frag.glsl");

    m_shaders.rgen  = m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_RAYGEN_BIT_KHR, "dray_trace.rgen.glsl");
    m_shaders.rmiss = m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_MISS_BIT_KHR, "dray_trace.rmiss.glsl");

    m_shaders.rhit = m_res.m_shaderManager.createShaderModule(
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "dray_trace.rchit.glsl",
        "#define USE_OVERLAY 0\n#define USE_DEPTHONLY 0\n#define SUPPORTS_MICROMESH_RT 0\n" + prepend);

    std::string microPrepend =
        nvh::stringFormat("#define SUPPORTS_MICROMESH_RT %d\n",
                          m_res.m_context->hasDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME) ? 1 : 0);
    m_shaders.rhitMicro =
        m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "dray_trace.rchit.glsl",
                                                 "#define USE_OVERLAY 0\n#define USE_DEPTHONLY 0\n" + prepend + microPrepend);
  }
}


void RendererCompressedRayVK::deinitShaders()
{
  deinitStandardShaders();

  m_res.m_shaderManager.destroyShaderModule(m_shaders.vertex);
  m_res.m_shaderManager.destroyShaderModule(m_shaders.fragment);
  m_res.m_shaderManager.destroyShaderModule(m_shaders.fragment_overlay);
  m_res.m_shaderManager.destroyShaderModule(m_shaders.vertexRay);
  m_res.m_shaderManager.destroyShaderModule(m_shaders.fragmentRay);
  m_res.m_shaderManager.destroyShaderModule(m_shaders.rgen);
  m_res.m_shaderManager.destroyShaderModule(m_shaders.rmiss);
  m_res.m_shaderManager.destroyShaderModule(m_shaders.rhit);
  m_res.m_shaderManager.destroyShaderModule(m_shaders.rhitMicro);
}


std::string RendererCompressedRayVK::getShaderPrepend()
{
  std::string prepend;

  prepend += nvh::stringFormat("#define USE_MICROVERTEX_NORMALS %d\n", m_config.useMicroVertexNormals ? 1 : 0);
  prepend += nvh::stringFormat("#define USE_TEXTURE_NORMALS %d\n", m_config.useNormalMap ? 1 : 0);
  prepend += nvh::stringFormat("#define UNCOMPRESSED_UMAJOR %d\n",
                               m_scene->barySet.shadingStats.valueOrder == bary::ValueLayout::eTriangleUmajor ? 1 : 0);

#ifdef _DEBUG
  printf(prepend.c_str());
#endif

  return prepend;
}


bool RendererCompressedRayVK::init(RenderList& list, const Config& config)
{
  m_config = config;
  m_scene  = list.m_scene;

  const SceneVK* scene = list.m_scene;

  if(!list.m_scene->hasCompressedDisplacement)
  {
    return true;
  }

  {
    m_micromeshSetRayVK.init(m_res, *scene->meshSetLo, scene->barySet, config.numThreads);
    list.m_scene->meshSetLoVK.deinitRayTracing(m_res);
    list.m_scene->meshSetLoVK.initRayTracingGeometry(m_res, *scene->meshSetLo, &m_micromeshSetRayVK);
    list.m_scene->meshSetLoVK.initRayTracingScene(m_res, *scene->meshSetLo, &m_micromeshSetRayVK);

    if(config.useMicroVertexNormals)
    {
      m_micromeshSetRayVK.initAttributeNormals(m_res, *scene->meshSetLo, scene->barySet, config.numThreads);
    }
  }

  initShaders();

  if(!m_res.m_shaderManager.areShaderModulesValid())
  {
    return false;
  }

  initSetupStandard(scene, config);
  initSetupCompressedRay(scene, config);
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
  }

  return true;
}

void RendererCompressedRayVK::initSetupCompressedRay(const SceneVK* scene, const Config& config)
{
  VkResult result;
  {
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    m_rayProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    prop2.pNext     = &m_rayProperties;
    vkGetPhysicalDeviceProperties2(m_res.m_physical, &prop2);
  }

  VkShaderStageFlags stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  m_compressedRay.stageFlags = stageFlags;
  m_compressedRay.container.init(m_res.m_device);
  auto& rendererSet = m_compressedRay.container.at(DSET_RENDERER);
  rendererSet.addBinding(DRAWRAY_UBO_VIEW, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  rendererSet.addBinding(DRAWRAY_SSBO_STATS, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageFlags);
  rendererSet.addBinding(DRAWRAY_IMG_OUT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, stageFlags);
  rendererSet.addBinding(DRAWRAY_UBO_MESH, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  rendererSet.addBinding(DRAWRAY_ACC, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, stageFlags);
  if(config.useMicroVertexNormals)
  {
    rendererSet.addBinding(DRAWRAY_SSBO_RTINSTANCES, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageFlags);
    rendererSet.addBinding(DRAWRAY_SSBO_RTDATA, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageFlags);
  }
  rendererSet.initLayout();
  rendererSet.initPool(1);

  initTextureSet(m_compressedRay, scene, config, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);

  m_compressedRay.container.initPipeLayout(0, config.useNormalMap ? 2 : 1, 0, nullptr, 0);

  std::vector<VkWriteDescriptorSet> dsets;
  {
    dsets.push_back(rendererSet.makeWrite(0, DRAWRAY_UBO_VIEW, &m_res.m_common.view.info));
    dsets.push_back(rendererSet.makeWrite(0, DRAWRAY_SSBO_STATS, &m_res.m_common.stats.info));

    dsets.push_back(rendererSet.makeWrite(0, DRAWRAY_UBO_MESH, &scene->meshSetLoVK.binding.info));
    dsets.push_back(rendererSet.makeWrite(0, DRAWRAY_ACC, &scene->meshSetLoVK.sceneTlasInfo));

    {
      static VkDescriptorImageInfo imgAtomic;
      imgAtomic.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      imgAtomic.imageView   = m_res.m_framebuffer.viewAtomic;
      imgAtomic.sampler     = VK_NULL_HANDLE;
      dsets.push_back(rendererSet.makeWrite(0, DRAWRAY_IMG_OUT, &imgAtomic));
    }

    if(config.useMicroVertexNormals)
    {
      dsets.push_back(rendererSet.makeWrite(0, DRAWRAY_SSBO_RTINSTANCES, &m_micromeshSetRayVK.instanceAttributes.info));
      dsets.push_back(rendererSet.makeWrite(0, DRAWRAY_SSBO_RTDATA, &m_micromeshSetRayVK.binding.info));
    }
  }
  vkUpdateDescriptorSets(m_res.m_device, (uint32_t)dsets.size(), dsets.data(), 0, nullptr);

  {
    nvvk::GraphicsPipelineState     state = m_res.m_basicGraphicsState;
    nvvk::GraphicsPipelineGenerator gen(state);
    gen.createInfo.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    gen.setRenderPass(m_res.m_framebuffer.passPreserve);
    gen.setDevice(m_res.m_device);
    // pipelines
    gen.setLayout(m_compressedRay.container.getPipeLayout());

    state.depthStencilState.depthWriteEnable = VK_TRUE;
    state.depthStencilState.depthCompareOp   = VK_COMPARE_OP_ALWAYS;
    state.rasterizationState.cullMode        = VK_CULL_MODE_NONE;

    gen.clearShaders();
    gen.addShader(m_res.m_shaderManager.get(m_shaders.vertexRay), VK_SHADER_STAGE_VERTEX_BIT);
    gen.addShader(m_res.m_shaderManager.get(m_shaders.fragmentRay), VK_SHADER_STAGE_FRAGMENT_BIT);
    m_swBlitPipe = gen.createPipeline();
  }

  {
    VkPipelineShaderStageCreateInfo stages[SBT_ENTRIES];
    stages[SBT_ENTRY_RGEN]        = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[SBT_ENTRY_RGEN].pName  = "main";
    stages[SBT_ENTRY_RGEN].stage  = VK_SHADER_STAGE_RAYGEN_BIT_NV;
    stages[SBT_ENTRY_RGEN].module = m_res.m_shaderManager.get(m_shaders.rgen);

    stages[SBT_ENTRY_RMISS]        = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[SBT_ENTRY_RMISS].pName  = "main";
    stages[SBT_ENTRY_RMISS].stage  = VK_SHADER_STAGE_MISS_BIT_NV;
    stages[SBT_ENTRY_RMISS].module = m_res.m_shaderManager.get(m_shaders.rmiss);

    stages[SBT_ENTRY_RHIT]        = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[SBT_ENTRY_RHIT].pName  = "main";
    stages[SBT_ENTRY_RHIT].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
    stages[SBT_ENTRY_RHIT].module = m_res.m_shaderManager.get(m_shaders.rhit);

    stages[SBT_ENTRY_RHIT_MICRO]        = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[SBT_ENTRY_RHIT_MICRO].pName  = "main";
    stages[SBT_ENTRY_RHIT_MICRO].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
    stages[SBT_ENTRY_RHIT_MICRO].module = m_res.m_shaderManager.get(m_shaders.rhitMicro);

    // in this sample we have as many shadergroups as we have shaders

    VkRayTracingShaderGroupCreateInfoKHR sgroups[SBT_ENTRIES];
    memset(sgroups, 0, sizeof(sgroups));
    for(int i = 0; i < SBT_ENTRIES; i++)
    {
      sgroups[i].sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
      sgroups[i].anyHitShader       = VK_SHADER_UNUSED_KHR;
      sgroups[i].closestHitShader   = VK_SHADER_UNUSED_KHR;
      sgroups[i].intersectionShader = VK_SHADER_UNUSED_KHR;
      sgroups[i].generalShader      = VK_SHADER_UNUSED_KHR;
    }

    sgroups[SBT_ENTRY_RGEN].type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    sgroups[SBT_ENTRY_RGEN].generalShader = SBT_ENTRY_RGEN;

    sgroups[SBT_ENTRY_RMISS].type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    sgroups[SBT_ENTRY_RMISS].generalShader = SBT_ENTRY_RMISS;

    sgroups[SBT_ENTRY_RHIT].type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    sgroups[SBT_ENTRY_RHIT].closestHitShader = SBT_ENTRY_RHIT;

    sgroups[SBT_ENTRY_RHIT_MICRO].type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    sgroups[SBT_ENTRY_RHIT_MICRO].closestHitShader = SBT_ENTRY_RHIT_MICRO;

    VkRayTracingPipelineCreateInfoKHR traceInfo = {VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    if(m_renderType == RENDER_COMPRESSED_RAY && g_enableMicromeshRTExtensions)
    {
      traceInfo.flags = VK_PIPELINE_CREATE_RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV;
    }

    traceInfo.layout                       = m_compressedRay.container.getPipeLayout();
    traceInfo.stageCount                   = NV_ARRAY_SIZE(stages);
    traceInfo.maxPipelineRayRecursionDepth = 0;
    traceInfo.pStages                      = stages;
    traceInfo.groupCount                   = NV_ARRAY_SIZE(sgroups);
    traceInfo.pGroups                      = sgroups;
    result = vkCreateRayTracingPipelinesKHR(m_res.m_device, nullptr, nullptr, 1, &traceInfo, nullptr, &m_compressedRay.pipeline);
    assert(result == VK_SUCCESS);
  }

  {
    RaySBT&    sbt        = m_raySBT;
    VkPipeline pipeline   = m_compressedRay.pipeline;
    uint32_t   handleSize = m_rayProperties.shaderGroupHandleSize;

    // The SBT (buffer) need to have starting groups to be aligned and handles in the group to be aligned.
    uint32_t handleSizeAligned = nvh::align_up(handleSize, m_rayProperties.shaderGroupHandleAlignment);


    sbt.gen.stride      = nvh::align_up(handleSizeAligned, m_rayProperties.shaderGroupBaseAlignment);
    sbt.gen.size        = sbt.gen.stride;
    sbt.miss.stride     = handleSizeAligned;
    sbt.miss.size       = nvh::align_up(handleSizeAligned, m_rayProperties.shaderGroupBaseAlignment);
    sbt.hit.stride      = handleSizeAligned;
    sbt.hit.size        = nvh::align_up(handleSizeAligned * 2, m_rayProperties.shaderGroupBaseAlignment);
    sbt.callable.size   = 0;
    sbt.callable.stride = handleSizeAligned;

    VkDeviceSize         dataSize = SBT_ENTRIES * m_rayProperties.shaderGroupHandleSize;
    std::vector<uint8_t> handles(dataSize);
    vkGetRayTracingShaderGroupHandlesKHR(m_res.m_device, pipeline, 0, SBT_ENTRIES, dataSize, handles.data());

    VkDeviceSize sbtSize = sbt.gen.size + sbt.miss.size + sbt.hit.size;

    sbt.buffer                 = m_res.createBuffer(sbtSize, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR);
    sbt.gen.deviceAddress      = sbt.buffer.addr;
    sbt.miss.deviceAddress     = sbt.buffer.addr + sbt.gen.size;
    sbt.hit.deviceAddress      = sbt.buffer.addr + sbt.gen.size + sbt.miss.size;
    sbt.callable.deviceAddress = sbt.buffer.addr + +sbt.gen.size + sbt.miss.size + sbt.hit.size;

    std::vector<uint8_t> sbtData(sbtSize, 0);
    memcpy(sbtData.data() + (sbt.gen.deviceAddress - sbt.buffer.addr), handles.data() + (SBT_ENTRY_RGEN * handleSize), handleSize);
    memcpy(sbtData.data() + (sbt.miss.deviceAddress - sbt.buffer.addr), handles.data() + (SBT_ENTRY_RMISS * handleSize), handleSize);
    memcpy(sbtData.data() + (sbt.hit.deviceAddress - sbt.buffer.addr), handles.data() + (SBT_ENTRY_RHIT * handleSize), handleSize);
    memcpy(sbtData.data() + (sbt.hit.deviceAddress - sbt.buffer.addr) + handleSizeAligned,
           handles.data() + (SBT_ENTRY_RHIT_MICRO * handleSize), handleSize);
    m_res.simpleUploadBuffer(sbt.buffer, sbtData.data());
  }
}

void RendererCompressedRayVK::deinit()
{
  if(m_cmdPool)
  {
    vkFreeCommandBuffers(m_res.m_device, m_cmdPool, NV_ARRAY_SIZE(m_cmdOpaque), m_cmdOpaque);
    vkFreeCommandBuffers(m_res.m_device, m_cmdPool, NV_ARRAY_SIZE(m_cmdOverlay), m_cmdOverlay);
    vkDestroyCommandPool(m_res.m_device, m_cmdPool, nullptr);
  }

  destroyPipelines(m_standard);
  destroyPipelines(m_compressedRay);

  vkDestroyPipeline(m_res.m_device, m_swBlitPipe, nullptr);

  m_res.destroy(m_raySBT.buffer);

  m_standard.container.deinit();
  m_compressedRay.container.deinit();


  deinitShaders();

  m_scene->meshSetLoVK.deinitRayTracing(m_res);
  m_micromeshSetRayVK.deinit(m_res);
}

void RendererCompressedRayVK::draw(const FrameConfig& config, nvvk::ProfilerVK& profiler)
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
  else if(config.opaque == MODEL_DISPLACED)
  {
    ModelType opaqueType = config.opaque;
    bool      isRay      = true;

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

    // clear via pass
    {
      auto secDraw = profiler.timeRecurring("Draw", primary);

      vkCmdExecuteCommands(primary, 1, &m_cmdOpaque[CMD_TYPE_DISPLACED]);

      {
        auto secBlit = profiler.timeRecurring("Blit", primary);

        const Setup& setup = m_compressedRay;

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
      }
    }
  }
  else
  {
    auto secDraw = profiler.timeRecurring("Draw", primary);

    m_res.cmdPipelineBarrier(primary);

    m_res.cmdBeginRenderPass(primary, true, true);

    // Render the mesh and its overlay.
    VkCommandBuffer cmds[4];
    uint32_t        cmdCount = 2;
    cmds[0]                  = m_cmdOpaque[getCmdType(config.opaque)];
    cmds[1]                  = m_cmdOverlay[getCmdType(config.overlay)];

    if(getCmdType(config.opaque) == CMD_TYPE_DISPLACED)
    {
      cmds[cmdCount++] = m_cmdOpaque[CMD_TYPE_NONDISPLACED_LO];
    }
    if(getCmdType(config.overlay) == CMD_TYPE_DISPLACED)
    {
      cmds[cmdCount++] = m_cmdOverlay[CMD_TYPE_NONDISPLACED_LO];
    }
    vkCmdExecuteCommands(primary, cmdCount, cmds);

    vkCmdEndRenderPass(primary);
  }

  profiler.endSection(sec, primary);
  m_res.cmdCopyStats(primary);
  vkEndCommandBuffer(primary);
  m_res.submissionEnqueue(primary);
}

}  // namespace microdisp
