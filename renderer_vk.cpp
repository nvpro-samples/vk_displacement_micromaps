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

using namespace glm;
#include "common.h"
#include "common_mesh.h"

#pragma pack(1)


namespace microdisp {

//////////////////////////////////////////////////////////////////////////

void RenderList::setup(SceneVK* scene, ShaderStats* stats, const Config& config)
{
  m_scene  = scene;
  m_config = config;
  m_stats  = stats;

  memset(m_stats, 0, sizeof(ShaderStats));
}

//////////////////////////////////////////////////////////////////////////

void RendererVK::initTextureSet(Setup& setup, const SceneVK* scene, const Config& config, VkShaderStageFlags shaderStageFlags)
{
  if(!config.useNormalMap)
    return;

  auto& textureSet = setup.container.at(DSET_TEXTURES);

  textureSet.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        uint32_t(scene->meshSetLoVK.materialTextures.size()), shaderStageFlags);
  textureSet.initLayout();
  textureSet.initPool(1);

  std::vector<VkWriteDescriptorSet> dsets;
  for(uint32_t i = 0; i < uint32_t(scene->meshSetLoVK.materialTextures.size()); i++)
  {
    dsets.push_back(textureSet.makeWrite(0, 0, &scene->meshSetLoVK.materialTextures[i].descriptor, i));
  }
  vkUpdateDescriptorSets(m_res.m_device, (uint32_t)dsets.size(), dsets.data(), 0, nullptr);
}

void RendererVK::initCommandPool()
{
  VkCommandPoolCreateInfo cmdPoolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cmdPoolInfo.queueFamilyIndex        = 0;
  VkResult result                     = vkCreateCommandPool(m_res.m_device, &cmdPoolInfo, nullptr, &m_cmdPool);
  assert(result == VK_SUCCESS);
}

void RendererVK::destroyPipelines(Setup& setup)
{
  vkDestroyPipeline(m_res.m_device, setup.pipeline, nullptr);
  vkDestroyPipeline(m_res.m_device, setup.pipelineOverlay, nullptr);
  vkDestroyPipeline(m_res.m_device, setup.pipelineFlat, nullptr);
  vkDestroyPipeline(m_res.m_device, setup.pipelineIndirect, nullptr);
}

VkCommandBuffer RendererVK::generateCmdBufferEmpty()
{
  VkCommandBuffer cmd = m_res.createCmdBuffer(m_cmdPool, false, false, true, false);
  nvvk::DebugUtil(m_res.m_device).setObjectName(cmd, "renderer_empty");
  vkEndCommandBuffer(cmd);
  return cmd;
}

VkCommandBuffer RendererVK::generateCmdBufferLo(RenderList& list, const Config& config, bool overlay, bool onlyNonDisplaced, bool shell)
{
  const SceneVK* scene        = list.m_scene;
  bool           useNormalMap = config.useNormalMap;

  const Setup&    setup = m_standard;
  VkCommandBuffer cmd   = m_res.createCmdBuffer(m_cmdPool, false, false, true, false);
  nvvk::DebugUtil(m_res.m_device).setObjectName(cmd, "renderer_lo");

  m_res.cmdDynamicState(cmd);

  const MeshSet&           meshSet   = *scene->meshSetLo;
  const MeshSetVK&         meshSetVK = scene->meshSetLoVK;
  const BaryAttributesSet& barySet   = scene->barySet;

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.container.getPipeLayout(), 0, 1,
                          setup.container.at(DSET_RENDERER).getSets(0), 0, nullptr);

  if(config.useNormalMap)
  {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, setup.container.getPipeLayout(), 0, 1,
                            setup.container.at(DSET_TEXTURES).getSets(0), 0, nullptr);
  }

  vkCmdBindIndexBuffer(cmd, meshSetVK.indices.info.buffer, 0, VK_INDEX_TYPE_UINT32);

  VkPipeline lastPipeline = nullptr;
  size_t     numObjects   = meshSet.meshInstances.size();
  for(size_t i = 0; i < numObjects; i++)
  {
    size_t              instIdx = (i);
    const MeshInstance& inst    = meshSet.meshInstances[instIdx];
    const MeshInfo&     info    = meshSet.meshInfos[inst.meshID];

    if(onlyNonDisplaced && info.displacementID != MeshSetID::INVALID)
      continue;

    if(shell && info.displacementID == MeshSetID::INVALID)
      continue;

    VkPipeline pipeline = getPipeline(setup, overlay, shell);
    if(pipeline != lastPipeline)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      lastPipeline = pipeline;
    }

    {
      DrawPushData push;
      push.firstIndex  = info.firstIndex;
      push.firstVertex = info.firstVertex;
      push.triangleMax = (info.numPrimitives) - 1;
      push.instanceID  = uint32_t(instIdx);
      push.shellDir    = 0;

      if(shell)
      {
        const BaryDisplacementAttribute& baryDisplacement = barySet.displacements[info.displacementID];
        const bary::Group&               group            = baryDisplacement.compressed ?
                                                                baryDisplacement.compressed->groups[info.displacementGroup] :
                                                                baryDisplacement.uncompressed->groups[info.displacementGroup];

        push.shellMin = group.floatBias.r;
        push.shellMax = group.floatBias.r + group.floatScale.r;
      }

      vkCmdPushConstants(cmd, setup.container.getPipeLayout(), setup.stageFlags, 0, sizeof(push), &push);
      vkCmdDrawIndexed(cmd, info.numIndices, shell ? 2 : 1, info.firstIndex, 0, 0);

      if(shell)
      {
        push.shellDir = 1;
        vkCmdPushConstants(cmd, setup.container.getPipeLayout(), setup.stageFlags, 0, sizeof(push), &push);
        vkCmdDraw(cmd, info.numVertices * 3, 1, 0, 0);
      }
    }
  }

  vkEndCommandBuffer(cmd);

  return cmd;
}

void RendererVK::initStandardShaders(std::string prepend)
{
  // shaders
  m_standardShaders.vertex =
      m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "draw_standard.vert.glsl", prepend);
  m_standardShaders.fragment = m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "draw_standard.frag.glsl",
                                                                        "#define USE_OVERLAY 0\n" + prepend);
  m_standardShaders.fragment_overlay = m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "draw_standard.frag.glsl",
                                                                                "#define USE_OVERLAY 1\n" + prepend);

  m_standardShaders.shellVertex =
      m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "draw_shell.vert.glsl", prepend);
  m_standardShaders.shellFragment = m_res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "draw_shell.frag.glsl");
}

void RendererVK::deinitStandardShaders()
{
  m_res.m_shaderManager.destroyShaderModule(m_standardShaders.vertex);
  m_res.m_shaderManager.destroyShaderModule(m_standardShaders.fragment);
  m_res.m_shaderManager.destroyShaderModule(m_standardShaders.fragment_overlay);

  m_res.m_shaderManager.destroyShaderModule(m_standardShaders.shellFragment);
  m_res.m_shaderManager.destroyShaderModule(m_standardShaders.shellVertex);
}

void RendererVK::initSetupStandard(const SceneVK* scene, const Config& config)
{
  VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  m_standard.stageFlags         = stageFlags;
  m_standard.container.init(m_res.m_device);
  auto& rendererSet = m_standard.container.at(DSET_RENDERER);
  rendererSet.addBinding(DRAWSTD_UBO_VIEW, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  rendererSet.addBinding(DRAWSTD_SSBO_STATS, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageFlags);
  rendererSet.addBinding(DRAWSTD_UBO_MESH, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  rendererSet.initLayout();
  rendererSet.initPool(1);

  initTextureSet(m_standard, scene, config, VK_SHADER_STAGE_FRAGMENT_BIT);

  VkPushConstantRange pushRange;
  pushRange.offset     = 0;
  pushRange.size       = sizeof(DrawPushData);
  pushRange.stageFlags = stageFlags;

  m_standard.container.initPipeLayout(0, config.useNormalMap ? 2 : 1, 1, &pushRange);

  std::vector<VkWriteDescriptorSet> dsets;
  {
    dsets.push_back(rendererSet.makeWrite(0, DRAWSTD_UBO_VIEW, &m_res.m_common.view.info));
    dsets.push_back(rendererSet.makeWrite(0, DRAWSTD_SSBO_STATS, &m_res.m_common.stats.info));
    dsets.push_back(rendererSet.makeWrite(0, DRAWSTD_UBO_MESH, &scene->meshSetLoVK.binding.info));
  }
  vkUpdateDescriptorSets(m_res.m_device, (uint32_t)dsets.size(), dsets.data(), 0, nullptr);

  nvvk::GraphicsPipelineState     state = m_res.m_basicGraphicsState;
  nvvk::GraphicsPipelineGenerator gen(state);
  gen.createInfo.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
  gen.setRenderPass(m_res.m_framebuffer.passPreserve);
  gen.setDevice(m_res.m_device);
  // pipelines
  gen.setLayout(m_standard.container.getPipeLayout());

  {
    state.depthStencilState.depthCompareOp           = VK_COMPARE_OP_LESS_OR_EQUAL;
    state.rasterizationState.cullMode                = VK_CULL_MODE_BACK_BIT;
    state.rasterizationState.depthBiasEnable         = VK_TRUE;
    state.rasterizationState.polygonMode             = VK_POLYGON_MODE_FILL;
    state.rasterizationState.depthBiasConstantFactor = -1;
    state.rasterizationState.depthBiasSlopeFactor    = 1;

    gen.clearShaders();
    gen.addShader(m_res.m_shaderManager.get(m_standardShaders.vertex), VK_SHADER_STAGE_VERTEX_BIT);
    gen.addShader(m_res.m_shaderManager.get(m_standardShaders.fragment), VK_SHADER_STAGE_FRAGMENT_BIT);
    m_standard.pipeline = gen.createPipeline();

    state.rasterizationState.depthBiasEnable = VK_FALSE;
    state.rasterizationState.polygonMode     = VK_POLYGON_MODE_LINE;
    state.rasterizationState.lineWidth       = 2.0f;
    state.depthStencilState.depthWriteEnable = VK_FALSE;
    gen.clearShaders();
    gen.addShader(m_res.m_shaderManager.get(m_standardShaders.vertex), VK_SHADER_STAGE_VERTEX_BIT);
    gen.addShader(m_res.m_shaderManager.get(m_standardShaders.fragment_overlay), VK_SHADER_STAGE_FRAGMENT_BIT);
    m_standard.pipelineOverlay = gen.createPipeline();

    state.rasterizationState.cullMode = 0;
    gen.clearShaders();
    gen.addShader(m_res.m_shaderManager.get(m_standardShaders.shellVertex), VK_SHADER_STAGE_VERTEX_BIT);
    gen.addShader(m_res.m_shaderManager.get(m_standardShaders.shellFragment), VK_SHADER_STAGE_FRAGMENT_BIT);
    m_standard.pipelineFlat = gen.createPipeline();
  }
}

}  // namespace microdisp
