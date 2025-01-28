/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "resources_vk.hpp"
#include "renderer_vk.hpp"
#include <inttypes.h>
#include <nvpsystem.hpp>
#include <nvvk/debug_util_vk.hpp>
#include <nvvk/error_vk.hpp>
#include <nvvk/images_vk.hpp>
#include <nvvk/renderpasses_vk.hpp>
#include <nvvk/samplers_vk.hpp>

#include <imgui/backends/imgui_vk_extra.h>
#include <algorithm>

#include "common.h"

namespace microdisp {


static const VkFormat s_colorFormat = VK_FORMAT_R16G16B16A16_UNORM;

/////////////////////////////////////////////////////////////////////////////////

void ResourcesVK::submissionExecute(VkFence fence, bool useImageReadWait, bool useImageWriteSignals)
{
  if (m_swapChain)
  {
    if(useImageReadWait && m_submissionWaitForRead)
    {
      VkSemaphore semRead = m_swapChain->getActiveReadSemaphore();
      if(semRead)
      {
        m_submission.enqueueWait(semRead, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
      }
      m_submissionWaitForRead = false;
    }

    if(useImageWriteSignals)
    {
      VkSemaphore semWritten = m_swapChain->getActiveWrittenSemaphore();
      if(semWritten)
      {
        m_submission.enqueueSignal(semWritten);
      }
    }
  }

  m_submission.execute(fence);
}

void ResourcesVK::tempSyncSubmit(VkCommandBuffer cmd, bool reset)
{
  vkEndCommandBuffer(cmd);
  submissionEnqueue(cmd);
  submissionExecute();
  if(reset)
  {
    tempResetResources();
  }
  else
  {
    synchronize("sync tempSyncSubmit");
  }
}

void ResourcesVK::beginFrame()
{
  assert(m_swapChain);

  assert(!m_withinFrame);
  m_withinFrame           = true;
  m_submissionWaitForRead = true;
  m_ringFences.setCycleAndWait(m_frame);
  m_ringCmdPool.setCycle(m_ringFences.getCycleIndex());
}

void ResourcesVK::cmdHBAO(VkCommandBuffer cmd, const FrameConfig& config, nvvk::ProfilerVK& profiler)
{
  assert(m_swapChain);

  nvvk::ProfilerVK::Section sec(profiler, "Hbao", cmd);

  bool    useResolved = m_framebuffer.useResolved && !m_hbaoFullRes;
  VkImage colorImage  = useResolved ? m_framebuffer.imgColorResolved : m_framebuffer.imgColor;

  VkImageLayout colorLayout = useResolved ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  VkAccessFlags colorAccess = useResolved ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  // transition depth to read optimal, color to general
  cmdImageTransition(cmd, colorImage, VK_IMAGE_ASPECT_COLOR_BIT, colorAccess,
                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, colorLayout, VK_IMAGE_LAYOUT_GENERAL);
  cmdImageTransition(cmd, m_framebuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  m_hbaoPass.cmdCompute(cmd, m_hbaoFrame, m_hbaoSettings);

  cmdImageTransition(cmd, colorImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_SHADER_WRITE_BIT, colorAccess,
                     VK_IMAGE_LAYOUT_GENERAL, colorLayout);
  cmdImageTransition(cmd, m_framebuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                     VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
}

VkRenderPass ResourcesVK::s_passUI = VK_NULL_HANDLE;
void         ResourcesVK::initImGui(const nvvk::Context& context)
{
  assert(!s_passUI);

  // Create the ui render pass
  VkAttachmentDescription attachments[1] = {};
  attachments[0].format                  = s_colorFormat;
  attachments[0].samples                 = VK_SAMPLE_COUNT_1_BIT;
  attachments[0].loadOp                  = VK_ATTACHMENT_LOAD_OP_LOAD;
  attachments[0].storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].initialLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  attachments[0].finalLayout             = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;  // for blit operation
  attachments[0].flags                   = 0;

  VkSubpassDescription subpass       = {};
  subpass.pipelineBindPoint          = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.inputAttachmentCount       = 0;
  VkAttachmentReference colorRefs[1] = {{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
  subpass.colorAttachmentCount       = NV_ARRAY_SIZE(colorRefs);
  subpass.pColorAttachments          = colorRefs;
  subpass.pDepthStencilAttachment    = nullptr;
  VkRenderPassCreateInfo rpInfo      = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  rpInfo.attachmentCount             = NV_ARRAY_SIZE(attachments);
  rpInfo.pAttachments                = attachments;
  rpInfo.subpassCount                = 1;
  rpInfo.pSubpasses                  = &subpass;
  rpInfo.dependencyCount             = 0;

  VkResult result = vkCreateRenderPass(context.m_device, &rpInfo, NULL, &s_passUI);
  assert(result == VK_SUCCESS);

  ImGui::InitVK(context.m_device, context.m_physicalDevice, context.m_queueGCT, context.m_queueGCT.familyIndex, s_passUI);
}
void ResourcesVK::deinitImGui(const nvvk::Context& context)
{
  vkDestroyRenderPass(context.m_device, s_passUI, nullptr);
  ImGui::ShutdownVK();
}

void ResourcesVK::blitFrame(const FrameConfig& config, nvvk::ProfilerVK& profiler)
{
  assert(m_swapChain);

  VkCommandBuffer cmd = createTempCmdBuffer();

  auto sec = profiler.beginSection("BltUI", cmd);

  VkImage imageBlitRead = m_framebuffer.imgColor;

  if(m_hbaoActive && m_hbaoFullRes)
  {
    cmdHBAO(cmd, config, profiler);
  }

  if(m_framebuffer.useResolved)
  {
    cmdImageTransition(cmd, m_framebuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // blit to resolved
    VkImageBlit region               = {0};
    region.dstOffsets[1].x           = config.winWidth;
    region.dstOffsets[1].y           = config.winHeight;
    region.dstOffsets[1].z           = 1;
    region.srcOffsets[1].x           = m_framebuffer.renderWidth;
    region.srcOffsets[1].y           = m_framebuffer.renderHeight;
    region.srcOffsets[1].z           = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;

    imageBlitRead = m_framebuffer.imgColorResolved;

    vkCmdBlitImage(cmd, m_framebuffer.imgColor, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, imageBlitRead,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);
  }

  if(m_hbaoActive && !m_hbaoFullRes)
  {
    cmdHBAO(cmd, config, profiler);
  }

  if(config.imguiDrawData)
  {
    if(imageBlitRead != m_framebuffer.imgColor)
    {
      cmdImageTransition(cmd, imageBlitRead, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }

    VkRenderPassBeginInfo renderPassBeginInfo    = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    renderPassBeginInfo.renderPass               = s_passUI;
    renderPassBeginInfo.framebuffer              = m_framebuffer.fboUI;
    renderPassBeginInfo.renderArea.offset.x      = 0;
    renderPassBeginInfo.renderArea.offset.y      = 0;
    renderPassBeginInfo.renderArea.extent.width  = config.winWidth;
    renderPassBeginInfo.renderArea.extent.height = config.winHeight;
    renderPassBeginInfo.clearValueCount          = 0;
    renderPassBeginInfo.pClearValues             = nullptr;

    vkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdSetViewport(cmd, 0, 1, &m_framebuffer.viewportUI);
    vkCmdSetScissor(cmd, 0, 1, &m_framebuffer.scissorUI);

    ImGui_ImplVulkan_RenderDrawData(config.imguiDrawData, cmd);

    vkCmdEndRenderPass(cmd);

    // turns imageBlitRead to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
  }
  else
  {
    if(m_framebuffer.useResolved)
    {
      cmdImageTransition(cmd, m_framebuffer.imgColorResolved, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    }
    else
    {
      cmdImageTransition(cmd, m_framebuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                         VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    }
  }

  {
    // blit to vk backbuffer
    VkImageBlit region               = {0};
    region.dstOffsets[1].x           = config.winWidth;
    region.dstOffsets[1].y           = config.winHeight;
    region.dstOffsets[1].z           = 1;
    region.srcOffsets[1].x           = config.winWidth;
    region.srcOffsets[1].y           = config.winHeight;
    region.srcOffsets[1].z           = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;

    cmdImageTransition(cmd, m_swapChain->getActiveImage(), VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    vkCmdBlitImage(cmd, imageBlitRead, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_swapChain->getActiveImage(),
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_NEAREST);

    cmdImageTransition(cmd, m_swapChain->getActiveImage(), VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
  }

  if(m_framebuffer.useResolved)
  {
    cmdImageTransition(cmd, m_framebuffer.imgColorResolved, VK_IMAGE_ASPECT_COLOR_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                       VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  }

  profiler.endSection(sec, cmd);

  vkEndCommandBuffer(cmd);
  submissionEnqueue(cmd);
}

void ResourcesVK::blankFrame()
{
  assert(m_swapChain);

  // generic state setup
  VkCommandBuffer primary = createTempCmdBuffer();
  cmdPipelineBarrier(primary);
  cmdBeginRenderPass(primary, true, false);
  vkCmdEndRenderPass(primary);
  vkEndCommandBuffer(primary);
  submissionEnqueue(primary);
}

void ResourcesVK::endFrame()
{
  assert(m_swapChain);

  submissionExecute(m_ringFences.getFence(), true, true);
  assert(m_withinFrame);
  m_withinFrame = false;
}

void ResourcesVK::cmdCopyStats(VkCommandBuffer cmd) const
{
  VkBufferCopy region;
  region.size      = sizeof(ShaderStats);
  region.srcOffset = 0;
  region.dstOffset = m_ringFences.getCycleIndex() * sizeof(ShaderStats);
  vkCmdCopyBuffer(cmd, m_common.stats.buffer, m_common.statsRead.buffer, 1, &region);
}

void ResourcesVK::cmdBuildHiz(VkCommandBuffer cmd) const
{
  assert(m_swapChain);

  cmdImageTransition(cmd, m_framebuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  m_hiz.cmdUpdateHiz(cmd, m_hizUpdate, (uint32_t)0);

  cmdImageTransition(cmd, m_framebuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                          VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
}

void ResourcesVK::getStats(ShaderStats& stats)
{

  const ShaderStats* pStats = (const ShaderStats*)m_allocator.map(m_common.statsRead);
  stats                     = pStats[m_ringFences.getCycleIndex()];
  m_allocator.unmap(m_common.statsRead);
}

bool ResourcesVK::init(nvvk::Context* context, nvvk::SwapChain* swapChain, const std::vector<std::string>& shaderSearchPaths)
{
  m_fboChangeID  = 0;
  m_pipeChangeID = 0;

  {
    m_context     = context;
    m_queue       = context->m_queueGCT;
    m_queueFamily = context->m_queueGCT.familyIndex;
    m_swapChain   = swapChain;
  }

  m_physical = m_context->m_physicalDevice;
  m_device   = m_context->m_device;

  nvvk::DebugUtil debugUtil(m_device);

  if(true)
  {
    m_supportedShaderPipelineStages |= VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV | VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV;
  }
  if(true)
  {
    m_supportedShaderPipelineStages |= VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_NV;
  }

  // submission queue
  m_submission.init(m_queue);

  // fences
  m_ringFences.init(m_device);

  // temp cmd pool
  m_ringCmdPool.init(m_device, m_queueFamily, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);

  m_memAllocator.init(m_device, m_physical);
  m_memAllocator.setAllocateFlags(VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, true);
  m_allocator.init(m_device, m_physical, &m_memAllocator);

  {
    // common
    m_common.view = createBuffer(sizeof(SceneData) * 2, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    DEBUGUTIL_SET_NAME(m_common.view.buffer);

    m_common.stats = createBuffer(sizeof(ShaderStats), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    DEBUGUTIL_SET_NAME(m_common.stats.buffer);

    m_common.statsRead = createBuffer(sizeof(ShaderStats) * nvvk::DEFAULT_RING_SIZE, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    DEBUGUTIL_SET_NAME(m_common.statsRead.buffer);
  }

  m_shaderManager.init(m_device, 1, 2);
  m_shaderManager.m_filetype        = nvh::ShaderFileManager::FILETYPE_GLSL;
  m_shaderManager.m_keepModuleSPIRV = true;
  for (auto it : shaderSearchPaths){
    m_shaderManager.addDirectory(it);
  }
  {
    VkQueryPoolCreateInfo queryPoolInfo = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    queryPoolInfo.queryType             = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolInfo.queryCount            = 2;
    vkCreateQueryPool(m_device, &queryPoolInfo, nullptr, &m_stopWatchQueryPool);

    uint32_t validBits  = m_context->m_physicalInfo.queueProperties[m_queueFamily].timestampValidBits;
    m_stopWatchTempMask = validBits == 64 ? uint64_t(-1) : ((uint64_t(1) << validBits) - uint64_t(1));
  }


  if (swapChain)
  {
    // Create the render passes
    {
      m_framebuffer.passClear    = createPass(true);
      m_framebuffer.passPreserve = createPass(false);
    }

    {
      m_basicGraphicsState = nvvk::GraphicsPipelineState();

      m_basicGraphicsState.inputAssemblyState.topology  = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      m_basicGraphicsState.rasterizationState.cullMode  = (VK_CULL_MODE_NONE);  //VK_CULL_MODE_BACK_BIT
      m_basicGraphicsState.rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
      m_basicGraphicsState.rasterizationState.lineWidth = float(m_framebuffer.supersample);

      m_basicGraphicsState.depthStencilState.depthTestEnable       = VK_TRUE;
      m_basicGraphicsState.depthStencilState.depthWriteEnable      = VK_TRUE;
      m_basicGraphicsState.depthStencilState.depthCompareOp        = VK_COMPARE_OP_LESS;
      m_basicGraphicsState.depthStencilState.depthBoundsTestEnable = VK_FALSE;
      m_basicGraphicsState.depthStencilState.stencilTestEnable     = VK_FALSE;
      m_basicGraphicsState.depthStencilState.minDepthBounds        = 0.0f;
      m_basicGraphicsState.depthStencilState.maxDepthBounds        = 1.0f;

      m_basicGraphicsState.multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    }

    {
      HbaoPass::Config config;
      config.maxFrames    = 1;
      config.targetFormat = s_colorFormat;

      m_hbaoPass.init(m_device, &m_allocator, &m_shaderManager, config);
    }
    {
      NVHizVK::Config config;
      config.msaaSamples          = 0;
      config.reversedZ            = false;
      config.supportsMinmaxFilter = m_context->hasDeviceExtension(VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME);
      config.supportsSubGroupShuffle = (m_context->m_physicalInfo.properties11.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT)
                                       && m_context->m_physicalInfo.properties11.subgroupSupportedOperations
                                              & (VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT);
      m_hiz.init(m_device, config, 1);

      for(uint32_t i = 0; i < NVHizVK::SHADER_COUNT; i++)
      {
        m_hizShaders[i] = m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT,
                                                                      "nvhiz-update.comp.glsl", m_hiz.getShaderDefines(i));
      }
    }
  }  

  return true;
}

void ResourcesVK::updatedShaders()
{
  if (m_swapChain)
  {
    m_hbaoPass.reloadShaders();

    {
      VkShaderModule shaders[NVHizVK::SHADER_COUNT];
      for(uint32_t i = 0; i < NVHizVK::SHADER_COUNT; i++)
      {
        shaders[i] = m_shaderManager.get(m_hizShaders[i]);
      }
      m_hiz.initPipelines(shaders);
    }
  }
}

void ResourcesVK::deinit()
{
  synchronize("sync deinit");

  {
    destroy(m_common.view);
    destroy(m_common.stats);
    destroy(m_common.statsRead);
  }

  if (m_swapChain)
  {
    m_hbaoPass.deinitFrame(m_hbaoFrame);
    m_hbaoPass.deinit();

    vkDestroyRenderPass(m_device, m_framebuffer.passClear, nullptr);
    vkDestroyRenderPass(m_device, m_framebuffer.passPreserve, nullptr);

    for(uint32_t i = 0; i < NVHizVK::SHADER_COUNT; i++)
    {
      m_shaderManager.destroyShaderModule(m_hizShaders[i]);
    }

    deinitFramebuffer();
    m_hiz.deinit();
  }

  m_ringFences.deinit();
  m_ringCmdPool.deinit();
  m_allocator.deinit();
  m_memAllocator.deinit();
  m_shaderManager.deinit();
  vkDestroyQueryPool(m_device, m_stopWatchQueryPool, nullptr);
}


VkRenderPass ResourcesVK::createPass(bool clear)
{
  VkResult result;

  VkAttachmentLoadOp loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;

  VkSampleCountFlagBits samplesUsed = VK_SAMPLE_COUNT_1_BIT;

  // Create the render pass
  VkAttachmentDescription attachments[2] = {};
  attachments[0].format                  = s_colorFormat;
  attachments[0].samples                 = samplesUsed;
  attachments[0].loadOp                  = loadOp;
  attachments[0].storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].initialLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  attachments[0].finalLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  attachments[0].flags                   = 0;

  VkFormat depthStencilFormat = nvvk::findDepthStencilFormat(m_physical);

  attachments[1].format              = depthStencilFormat;
  attachments[1].samples             = samplesUsed;
  attachments[1].loadOp              = loadOp;
  attachments[1].storeOp             = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[1].stencilLoadOp       = loadOp;
  attachments[1].stencilStoreOp      = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[1].initialLayout       = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  attachments[1].finalLayout         = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  attachments[1].flags               = 0;
  VkSubpassDescription subpass       = {};
  subpass.pipelineBindPoint          = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.inputAttachmentCount       = 0;
  VkAttachmentReference colorRefs[1] = {{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
  subpass.colorAttachmentCount       = NV_ARRAY_SIZE(colorRefs);
  subpass.pColorAttachments          = colorRefs;
  VkAttachmentReference depthRefs[1] = {{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}};
  subpass.pDepthStencilAttachment    = depthRefs;
  VkRenderPassCreateInfo rpInfo      = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  rpInfo.attachmentCount             = NV_ARRAY_SIZE(attachments);
  rpInfo.pAttachments                = attachments;
  rpInfo.subpassCount                = 1;
  rpInfo.pSubpasses                  = &subpass;
  rpInfo.dependencyCount             = 0;

  VkRenderPass rp;
  result = vkCreateRenderPass(m_device, &rpInfo, nullptr, &rp);
  assert(result == VK_SUCCESS);
  return rp;
}

bool ResourcesVK::initFramebuffer(int winWidth, int winHeight, int supersample, bool vsync)
{
  VkResult result;

  m_fboChangeID++;

  if(m_framebuffer.imgColor != 0)
  {
    deinitFramebuffer();
  }

  nvvk::DebugUtil debugUtil(m_device);

  m_framebuffer.memAllocator.init(m_device, m_physical);

  bool oldResolved = m_framebuffer.supersample > 1;

  m_framebuffer.renderWidth  = winWidth * supersample;
  m_framebuffer.renderHeight = winHeight * supersample;
  m_framebuffer.supersample  = supersample;
  m_framebuffer.vsync        = vsync;

  LOGI("framebuffer: %d x %d\n", m_framebuffer.renderWidth, m_framebuffer.renderHeight);

  m_framebuffer.useResolved = supersample > 1;

  VkSampleCountFlagBits samplesUsed = VK_SAMPLE_COUNT_1_BIT;
  {
    // color
    VkImageCreateInfo cbImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    cbImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    cbImageInfo.format            = s_colorFormat;
    cbImageInfo.extent.width      = m_framebuffer.renderWidth;
    cbImageInfo.extent.height     = m_framebuffer.renderHeight;
    cbImageInfo.extent.depth      = 1;
    cbImageInfo.mipLevels         = 1;
    cbImageInfo.arrayLayers       = 1;
    cbImageInfo.samples           = samplesUsed;
    cbImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    cbImageInfo.flags             = 0;
    cbImageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    cbImageInfo.usage             = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                        | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;


    nvvk::AllocationID allocId;
    m_framebuffer.imgColor = m_framebuffer.memAllocator.createImage(cbImageInfo, allocId, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    DEBUGUTIL_SET_NAME(m_framebuffer.imgColor);


    VkImageCreateInfo atomicInfo = cbImageInfo;
    atomicInfo.format            = VK_FORMAT_R64_UINT;
    atomicInfo.arrayLayers       = ATOMIC_LAYERS;
    atomicInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    m_framebuffer.imgAtomic = m_framebuffer.memAllocator.createImage(atomicInfo, allocId, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    DEBUGUTIL_SET_NAME(m_framebuffer.imgAtomic);
  }

  // depth stencil
  VkFormat depthStencilFormat = nvvk::findDepthStencilFormat(m_physical);

  {
    VkImageCreateInfo dsImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    dsImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    dsImageInfo.format            = depthStencilFormat;
    dsImageInfo.extent.width      = m_framebuffer.renderWidth;
    dsImageInfo.extent.height     = m_framebuffer.renderHeight;
    dsImageInfo.extent.depth      = 1;
    dsImageInfo.mipLevels         = 1;
    dsImageInfo.arrayLayers       = 1;
    dsImageInfo.samples           = samplesUsed;
    dsImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    dsImageInfo.usage             = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    dsImageInfo.flags             = 0;
    dsImageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;


    nvvk::AllocationID allocId;
    m_framebuffer.imgDepthStencil =
        m_framebuffer.memAllocator.createImage(dsImageInfo, allocId, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    DEBUGUTIL_SET_NAME(m_framebuffer.imgDepthStencil);
  }

  {
    m_hiz.setupUpdateInfos(m_hizUpdate, m_framebuffer.renderWidth, m_framebuffer.renderHeight, depthStencilFormat,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    // hiz
    VkImageCreateInfo hizImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    hizImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    hizImageInfo.format            = m_hizUpdate.farInfo.format;
    hizImageInfo.extent.width      = m_hizUpdate.farInfo.width;
    hizImageInfo.extent.height     = m_hizUpdate.farInfo.height;
    hizImageInfo.mipLevels         = m_hizUpdate.farInfo.mipLevels;
    hizImageInfo.extent.depth      = 1;
    hizImageInfo.arrayLayers       = 1;
    hizImageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
    hizImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    hizImageInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    hizImageInfo.flags             = 0;
    hizImageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;

    {
      nvvk::AllocationID allocId;
      m_framebuffer.imgHizFar = m_framebuffer.memAllocator.createImage(hizImageInfo, allocId, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      DEBUGUTIL_SET_NAME(m_framebuffer.imgHizFar);
    }

    m_hizUpdate.sourceImage = m_framebuffer.imgDepthStencil;
    m_hizUpdate.farImage    = m_framebuffer.imgHizFar;
    m_hizUpdate.nearImage   = VK_NULL_HANDLE;
  }

  if(m_framebuffer.useResolved)
  {
    // resolve image
    VkImageCreateInfo resImageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    resImageInfo.imageType         = VK_IMAGE_TYPE_2D;
    resImageInfo.format            = s_colorFormat;
    resImageInfo.extent.width      = winWidth;
    resImageInfo.extent.height     = winHeight;
    resImageInfo.extent.depth      = 1;
    resImageInfo.mipLevels         = 1;
    resImageInfo.arrayLayers       = 1;
    resImageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
    resImageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
    resImageInfo.usage             = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                         | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    resImageInfo.flags         = 0;
    resImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    {
      nvvk::AllocationID allocId;
      m_framebuffer.imgColorResolved =
          m_framebuffer.memAllocator.createImage(resImageInfo, allocId, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      DEBUGUTIL_SET_NAME(m_framebuffer.imgColorResolved);
    }
  }


  VkDeviceSize allocatedSize;
  VkDeviceSize usedSize;
  float        util = m_framebuffer.memAllocator.getUtilization(allocatedSize, usedSize);

  LOGI("framebuffer: memory used %d KB, alloc %d KB (%.2f)\n", usedSize / 1024, allocatedSize / 1024, util);

  // views after allocation handling
  {
    VkImageViewCreateInfo cbImageViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    cbImageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    cbImageViewInfo.format                          = s_colorFormat;
    cbImageViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
    cbImageViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
    cbImageViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
    cbImageViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
    cbImageViewInfo.flags                           = 0;
    cbImageViewInfo.subresourceRange.levelCount     = 1;
    cbImageViewInfo.subresourceRange.baseMipLevel   = 0;
    cbImageViewInfo.subresourceRange.layerCount     = 1;
    cbImageViewInfo.subresourceRange.baseArrayLayer = 0;
    cbImageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;

    cbImageViewInfo.image = m_framebuffer.imgColor;
    result                = vkCreateImageView(m_device, &cbImageViewInfo, nullptr, &m_framebuffer.viewColor);
    assert(result == VK_SUCCESS);
    DEBUGUTIL_SET_NAME(m_framebuffer.viewColor);


    if(m_framebuffer.useResolved)
    {
      cbImageViewInfo.image = m_framebuffer.imgColorResolved;
      result                = vkCreateImageView(m_device, &cbImageViewInfo, nullptr, &m_framebuffer.viewColorResolved);
      assert(result == VK_SUCCESS);
      DEBUGUTIL_SET_NAME(m_framebuffer.viewColorResolved);
    }
  }
  {
    VkImageViewCreateInfo dsImageViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    dsImageViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    dsImageViewInfo.format                          = depthStencilFormat;
    dsImageViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
    dsImageViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
    dsImageViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
    dsImageViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
    dsImageViewInfo.flags                           = 0;
    dsImageViewInfo.subresourceRange.levelCount     = 1;
    dsImageViewInfo.subresourceRange.baseMipLevel   = 0;
    dsImageViewInfo.subresourceRange.layerCount     = 1;
    dsImageViewInfo.subresourceRange.baseArrayLayer = 0;
    dsImageViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_DEPTH_BIT;

    dsImageViewInfo.image = m_framebuffer.imgDepthStencil;
    result                = vkCreateImageView(m_device, &dsImageViewInfo, nullptr, &m_framebuffer.viewDepthStencil);
    assert(result == VK_SUCCESS);
    DEBUGUTIL_SET_NAME(m_framebuffer.viewDepthStencil);

    dsImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    result = vkCreateImageView(m_device, &dsImageViewInfo, nullptr, &m_framebuffer.viewDepth);
    assert(result == VK_SUCCESS);
    DEBUGUTIL_SET_NAME(m_framebuffer.viewDepth);
  }

  {
    VkImageViewCreateInfo atomicViewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    atomicViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    atomicViewInfo.format                          = VK_FORMAT_R64_UINT;
    atomicViewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
    atomicViewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
    atomicViewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
    atomicViewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
    atomicViewInfo.flags                           = 0;
    atomicViewInfo.subresourceRange.levelCount     = 1;
    atomicViewInfo.subresourceRange.baseMipLevel   = 0;
    atomicViewInfo.subresourceRange.layerCount     = 1;
    atomicViewInfo.subresourceRange.baseArrayLayer = 0;
    atomicViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;

    atomicViewInfo.image = m_framebuffer.imgAtomic;
    result               = vkCreateImageView(m_device, &atomicViewInfo, nullptr, &m_framebuffer.viewAtomic);
    assert(result == VK_SUCCESS);
    DEBUGUTIL_SET_NAME(m_framebuffer.viewAtomic);
  }

  m_hiz.initUpdateViews(m_hizUpdate);
  m_hiz.updateDescriptorSet(m_hizUpdate, 0);

  // initial resource transitions
  {
    VkCommandBuffer cmd = createTempCmdBuffer();
    debugUtil.setObjectName(cmd, "framebufferCmd");

    m_swapChain->cmdUpdateBarriers(cmd);

    cmdImageTransition(cmd, m_framebuffer.imgAtomic, VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_ACCESS_SHADER_WRITE_BIT,
                       VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    cmdImageTransition(cmd, m_framebuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_ACCESS_TRANSFER_READ_BIT,
                       VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    cmdImageTransition(cmd, m_framebuffer.imgDepthStencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0,
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    cmdImageTransition(cmd, m_framebuffer.imgHizFar, VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_ACCESS_SHADER_WRITE_BIT,
                       VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    if(m_framebuffer.useResolved)
    {
      cmdImageTransition(cmd, m_framebuffer.imgColorResolved, VK_IMAGE_ASPECT_COLOR_BIT, 0,
                         VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    }

    {
      HbaoPass::FrameConfig config;
      config.blend                   = true;
      config.sourceHeightScale       = m_hbaoFullRes ? 1 : supersample;
      config.sourceWidthScale        = m_hbaoFullRes ? 1 : supersample;
      config.targetWidth             = m_hbaoFullRes ? m_framebuffer.renderWidth : winWidth;
      config.targetHeight            = m_hbaoFullRes ? m_framebuffer.renderHeight : winHeight;
      config.sourceDepth.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      config.sourceDepth.imageView   = m_framebuffer.viewDepth;
      config.sourceDepth.sampler     = VK_NULL_HANDLE;
      config.targetColor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      config.targetColor.imageView =
          m_framebuffer.useResolved && !m_hbaoFullRes ? m_framebuffer.viewColorResolved : m_framebuffer.viewColor;
      config.targetColor.sampler = VK_NULL_HANDLE;

      m_hbaoPass.initFrame(m_hbaoFrame, config, cmd);
    }

    vkEndCommandBuffer(cmd);

    submissionEnqueue(cmd);
    submissionExecute();
    synchronize("sync initFramebuffer");
    tempResetResources();
  }

  {
    // Create framebuffers
    VkImageView bindInfos[2];
    bindInfos[0] = m_framebuffer.viewColor;
    bindInfos[1] = m_framebuffer.viewDepthStencil;

    VkFramebuffer           fb;
    VkFramebufferCreateInfo fbInfo = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbInfo.attachmentCount         = NV_ARRAY_SIZE(bindInfos);
    fbInfo.pAttachments            = bindInfos;
    fbInfo.width                   = m_framebuffer.renderWidth;
    fbInfo.height                  = m_framebuffer.renderHeight;
    fbInfo.layers                  = 1;

    fbInfo.renderPass = m_framebuffer.passClear;
    result            = vkCreateFramebuffer(m_device, &fbInfo, nullptr, &fb);
    assert(result == VK_SUCCESS);
    m_framebuffer.fboScene = fb;
  }


  // ui related
  {
    VkImageView uiTarget = m_framebuffer.useResolved ? m_framebuffer.viewColorResolved : m_framebuffer.viewColor;

    // Create framebuffers
    VkImageView bindInfos[1];
    bindInfos[0] = uiTarget;

    VkFramebuffer           fb;
    VkFramebufferCreateInfo fbInfo = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbInfo.attachmentCount         = NV_ARRAY_SIZE(bindInfos);
    fbInfo.pAttachments            = bindInfos;
    fbInfo.width                   = winWidth;
    fbInfo.height                  = winHeight;
    fbInfo.layers                  = 1;

    fbInfo.renderPass = s_passUI;
    result            = vkCreateFramebuffer(m_device, &fbInfo, nullptr, &fb);
    assert(result == VK_SUCCESS);
    m_framebuffer.fboUI = fb;
  }

  {
    VkViewport vp;
    VkRect2D   sc;
    vp.x        = 0;
    vp.y        = 0;
    vp.width    = float(m_framebuffer.renderWidth);
    vp.height   = float(m_framebuffer.renderHeight);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;

    sc.offset.x      = 0;
    sc.offset.y      = 0;
    sc.extent.width  = m_framebuffer.renderWidth;
    sc.extent.height = m_framebuffer.renderHeight;

    m_framebuffer.viewport = vp;
    m_framebuffer.scissor  = sc;

    vp.width         = float(winWidth);
    vp.height        = float(winHeight);
    sc.extent.width  = winWidth;
    sc.extent.height = winHeight;

    m_framebuffer.viewportUI = vp;
    m_framebuffer.scissorUI  = sc;
  }


  return true;
}

void ResourcesVK::deinitFramebuffer()
{
  synchronize("sync deinitFramebuffer");

  vkDestroyImageView(m_device, m_framebuffer.viewColor, nullptr);
  vkDestroyImageView(m_device, m_framebuffer.viewDepthStencil, nullptr);
  vkDestroyImageView(m_device, m_framebuffer.viewDepth, nullptr);
  vkDestroyImageView(m_device, m_framebuffer.viewAtomic, nullptr);
  m_framebuffer.viewColor        = VK_NULL_HANDLE;
  m_framebuffer.viewDepthStencil = VK_NULL_HANDLE;
  m_framebuffer.viewDepth        = VK_NULL_HANDLE;
  m_framebuffer.viewAtomic       = VK_NULL_HANDLE;

  m_hiz.deinitUpdateViews(m_hizUpdate);

  vkDestroyImage(m_device, m_framebuffer.imgColor, nullptr);
  vkDestroyImage(m_device, m_framebuffer.imgDepthStencil, nullptr);
  vkDestroyImage(m_device, m_framebuffer.imgAtomic, nullptr);
  vkDestroyImage(m_device, m_framebuffer.imgHizFar, nullptr);
  m_framebuffer.imgColor        = VK_NULL_HANDLE;
  m_framebuffer.imgDepthStencil = VK_NULL_HANDLE;
  m_framebuffer.imgAtomic       = VK_NULL_HANDLE;
  m_framebuffer.imgHizFar       = VK_NULL_HANDLE;

  if(m_framebuffer.imgColorResolved)
  {
    vkDestroyImageView(m_device, m_framebuffer.viewColorResolved, nullptr);
    m_framebuffer.viewColorResolved = VK_NULL_HANDLE;

    vkDestroyImage(m_device, m_framebuffer.imgColorResolved, nullptr);
    m_framebuffer.imgColorResolved = VK_NULL_HANDLE;
  }

  m_framebuffer.memAllocator.freeAll();
  m_framebuffer.memAllocator.deinit();

  vkDestroyFramebuffer(m_device, m_framebuffer.fboScene, nullptr);
  m_framebuffer.fboScene = VK_NULL_HANDLE;

  vkDestroyFramebuffer(m_device, m_framebuffer.fboUI, nullptr);
  m_framebuffer.fboUI = VK_NULL_HANDLE;
}

void ResourcesVK::cmdDynamicState(VkCommandBuffer cmd) const
{
  vkCmdSetViewport(cmd, 0, 1, &m_framebuffer.viewport);
  vkCmdSetScissor(cmd, 0, 1, &m_framebuffer.scissor);
}

void ResourcesVK::cmdBeginRenderPass(VkCommandBuffer cmd, bool clear, bool hasSecondary) const
{
  VkRenderPassBeginInfo renderPassBeginInfo    = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  renderPassBeginInfo.renderPass               = clear ? m_framebuffer.passClear : m_framebuffer.passPreserve;
  renderPassBeginInfo.framebuffer              = m_framebuffer.fboScene;
  renderPassBeginInfo.renderArea.offset.x      = 0;
  renderPassBeginInfo.renderArea.offset.y      = 0;
  renderPassBeginInfo.renderArea.extent.width  = m_framebuffer.renderWidth;
  renderPassBeginInfo.renderArea.extent.height = m_framebuffer.renderHeight;
  renderPassBeginInfo.clearValueCount          = 2;

  glm::vec4 bgColor(0.1, 0.13, 0.15, 0);

  VkClearValue clearValues[2];
  clearValues[0].color.float32[0]     = bgColor.x;
  clearValues[0].color.float32[1]     = bgColor.y;
  clearValues[0].color.float32[2]     = bgColor.z;
  clearValues[0].color.float32[3]     = bgColor.w;
  clearValues[1].depthStencil.depth   = 1.0f;
  clearValues[1].depthStencil.stencil = 0;
  renderPassBeginInfo.pClearValues    = clearValues;
  vkCmdBeginRenderPass(cmd, &renderPassBeginInfo,
                       hasSecondary ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS : VK_SUBPASS_CONTENTS_INLINE);
}

void ResourcesVK::cmdPipelineBarrier(VkCommandBuffer cmd) const
{
  // transfers
  {
    VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memBarrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
    memBarrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_FALSE, 1, &memBarrier, 0, nullptr, 0, nullptr);
  }
#
  // color transition
  {
    VkImageSubresourceRange colorRange;
    memset(&colorRange, 0, sizeof(colorRange));
    colorRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    colorRange.baseMipLevel   = 0;
    colorRange.levelCount     = VK_REMAINING_MIP_LEVELS;
    colorRange.baseArrayLayer = 0;
    colorRange.layerCount     = 1;

    VkImageMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    memBarrier.srcAccessMask        = VK_ACCESS_TRANSFER_READ_BIT;
    memBarrier.dstAccessMask        = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    memBarrier.oldLayout            = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    memBarrier.newLayout            = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    memBarrier.image                = m_framebuffer.imgColor;
    memBarrier.subresourceRange     = colorRange;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_FALSE,
                         0, nullptr, 0, nullptr, 1, &memBarrier);
  }

  // Prepare the depth+stencil for reading.

  {
    VkImageSubresourceRange depthStencilRange;
    memset(&depthStencilRange, 0, sizeof(depthStencilRange));
    depthStencilRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    depthStencilRange.baseMipLevel   = 0;
    depthStencilRange.levelCount     = VK_REMAINING_MIP_LEVELS;
    depthStencilRange.baseArrayLayer = 0;
    depthStencilRange.layerCount     = 1;

    VkImageMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    memBarrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    memBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    memBarrier.oldLayout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    memBarrier.newLayout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    memBarrier.image         = m_framebuffer.imgDepthStencil;
    memBarrier.subresourceRange = depthStencilRange;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                         VK_FALSE, 0, nullptr, 0, nullptr, 1, &memBarrier);
  }
}


void ResourcesVK::cmdImageTransition(VkCommandBuffer    cmd,
                                   VkImage            img,
                                   VkImageAspectFlags aspects,
                                   VkAccessFlags      src,
                                   VkAccessFlags      dst,
                                   VkImageLayout      oldLayout,
                                   VkImageLayout      newLayout) const
{
  VkPipelineStageFlags srcPipe = nvvk::makeAccessMaskPipelineStageFlags(src, m_supportedShaderPipelineStages);
  VkPipelineStageFlags dstPipe = nvvk::makeAccessMaskPipelineStageFlags(dst, m_supportedShaderPipelineStages);

  VkImageSubresourceRange range;
  memset(&range, 0, sizeof(range));
  range.aspectMask     = aspects;
  range.baseMipLevel   = 0;
  range.levelCount     = VK_REMAINING_MIP_LEVELS;
  range.baseArrayLayer = 0;
  range.layerCount     = VK_REMAINING_ARRAY_LAYERS;

  VkImageMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  memBarrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  memBarrier.dstAccessMask        = dst;
  memBarrier.srcAccessMask        = src;
  memBarrier.oldLayout            = oldLayout;
  memBarrier.newLayout            = newLayout;
  memBarrier.image                = img;
  memBarrier.subresourceRange     = range;

  vkCmdPipelineBarrier(cmd, srcPipe, dstPipe, VK_FALSE, 0, nullptr, 0, nullptr, 1, &memBarrier);
}

VkCommandBuffer ResourcesVK::createCmdBuffer(VkCommandPool pool, bool singleshot, bool primary, bool secondaryInClear, bool isCompute) const
{
  VkResult result;
  bool     secondary = !primary;

  // Create the command buffer.
  VkCommandBufferAllocateInfo cmdInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cmdInfo.commandPool                 = pool;
  cmdInfo.level                       = primary ? VK_COMMAND_BUFFER_LEVEL_PRIMARY : VK_COMMAND_BUFFER_LEVEL_SECONDARY;
  cmdInfo.commandBufferCount          = 1;
  VkCommandBuffer cmd;
  result = vkAllocateCommandBuffers(m_device, &cmdInfo, &cmd);
  assert(result == VK_SUCCESS);

  cmdBegin(cmd, singleshot, primary, secondaryInClear, isCompute);

  return cmd;
}

VkCommandBuffer ResourcesVK::createTempCmdBuffer()
{
  VkCommandBuffer cmd =
      m_ringCmdPool.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

  nvvk::DebugUtil(m_device).setObjectName(cmd, "tempCmdBuffer");

  return cmd;
}

RBuffer ResourcesVK::createBuffer(VkDeviceSize size, VkBufferUsageFlags flags, VkMemoryPropertyFlags memFlags)
{
  RBuffer entry = {nullptr};

  if(size)
  {
    ((nvvk::Buffer&)entry) =
        m_allocator.createBuffer(size, flags | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, memFlags);
    entry.info.buffer = entry.buffer;
    entry.info.offset = 0;
    entry.info.range  = size;
    entry.addr        = nvvk::getBufferDeviceAddress(m_device, entry.buffer);
  }

  return entry;
}

void ResourcesVK::cmdBegin(VkCommandBuffer cmd, bool singleshot, bool primary, bool secondaryInClear, bool isCompute) const
{
  VkResult result;
  bool     secondary = !primary;

  VkCommandBufferInheritanceInfo inheritInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
  if(secondary && !isCompute)
  {
    inheritInfo.renderPass  = secondaryInClear ? m_framebuffer.passClear : m_framebuffer.passPreserve;
    inheritInfo.framebuffer = m_framebuffer.fboScene;
  }


  VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  // the sample is resubmitting re-use commandbuffers to the queue while they may still be executed by GPU
  // we only use fences to prevent deleting commandbuffers that are still in flight
  beginInfo.flags = singleshot ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  // the sample's secondary buffers always are called within passes as they contain drawcalls
  beginInfo.flags |= secondary && !isCompute ? VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT : 0;
  beginInfo.pInheritanceInfo = &inheritInfo;

  result = vkBeginCommandBuffer(cmd, &beginInfo);
  assert(result == VK_SUCCESS);
}

void ResourcesVK::simpleUploadBuffer(const RBuffer& dst, const void* src)
{
  if(src && dst.info.range)
  {
    VkCommandBuffer cmd = createTempCmdBuffer();
    m_allocator.getStaging()->cmdToBuffer(cmd, dst.buffer, 0, dst.info.range, src);
    tempSyncSubmit(cmd);
  }
}

void ResourcesVK::simpleDownloadBuffer(void* dst, const RBuffer& src)
{
  if(dst && src.info.range)
  {
    VkCommandBuffer cmd     = createTempCmdBuffer();
    const void*     mapped  = m_allocator.getStaging()->cmdFromBuffer(cmd, src.buffer, 0, src.info.range);
    // Ensure writes to the buffer we're mapping are accessible by the host
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask   = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    tempSyncSubmit(cmd, false);
    memcpy(dst, mapped, src.info.range);
    tempResetResources();
  }
}

void ResourcesVK::simpleDownloadBuffer(std::function<void(const void*, VkDeviceSize size)> dstFn, const RBuffer& src)
{
  if(dstFn && src.info.range)
  {
    VkCommandBuffer cmd     = createTempCmdBuffer();
    const void*     mapped  = m_allocator.getStaging()->cmdFromBuffer(cmd, src.buffer, 0, src.info.range);
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask   = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    tempSyncSubmit(cmd, false);
    dstFn(mapped, src.info.range);
    tempResetResources();
  }
}

void ResourcesVK::simpleDownloadTexture(void* dst, size_t dstSize, const RTextureRW& src)
{
  if(dst && dstSize)
  {
    VkCommandBuffer          cmd         = createTempCmdBuffer();
    VkOffset3D               offset      = {0, 0, 0};
    VkExtent3D               extent      = src.extent;
    VkImageSubresourceLayers subresource = {0};
    subresource.aspectMask               = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource.mipLevel                 = 0;
    subresource.layerCount               = 1;
    subresource.baseArrayLayer           = 0;
    const void* mapped =
        m_allocator.getStaging()->cmdFromImage(cmd, src.image, offset, extent, subresource, dstSize, VK_IMAGE_LAYOUT_GENERAL);
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask   = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    tempSyncSubmit(cmd, false);
    memcpy(dst, mapped, dstSize);
    tempResetResources();
  }
}

void ResourcesVK::simpleDownloadTexture(std::function<void(const void*)> dstFn, size_t dstSize, const RTextureRW& src)
{
  VkCommandBuffer          cmd         = createTempCmdBuffer();
  VkOffset3D               offset      = {0, 0, 0};
  VkExtent3D               extent      = src.extent;
  VkImageSubresourceLayers subresource = {0};
  subresource.aspectMask               = VK_IMAGE_ASPECT_COLOR_BIT;
  subresource.mipLevel                 = 0;
  subresource.layerCount               = 1;
  subresource.baseArrayLayer           = 0;
  const void* mapped =
      m_allocator.getStaging()->cmdFromImage(cmd, src.image, offset, extent, subresource, dstSize, VK_IMAGE_LAYOUT_GENERAL);
  VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask   = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  tempSyncSubmit(cmd, false);
  dstFn(mapped);
  tempResetResources();
}

void ResourcesVK::tempResetResources()
{
  synchronize("sync resetTempResources");
  m_ringFences.reset();
  m_ringCmdPool.reset();
  m_allocator.releaseStaging();
}

void ResourcesVK::tempStopWatch(VkCommandBuffer cmd, bool begin) const
{
  if(begin)
  {
    vkCmdResetQueryPool(cmd, m_stopWatchQueryPool, 0, 2);
  }
  vkCmdWriteTimestamp(cmd, begin ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      m_stopWatchQueryPool, begin ? 0 : 1);
}

double ResourcesVK::getStopWatchResult() const
{
  uint64_t times[2];
  VkResult result = vkGetQueryPoolResults(m_device, m_stopWatchQueryPool, 0, 2, sizeof(uint64_t) * 2, times,
                                          sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
  if(result == VK_SUCCESS)
  {
    float    frequency = m_context->m_physicalInfo.properties10.limits.timestampPeriod;
    uint64_t mask      = m_stopWatchTempMask;
    return (double((times[1] & mask) - (times[0] & mask)) * double(frequency)) / double(1000);
  }

  return 0;
}

void ResourcesVK::synchronize(const char* debugMsg)
{
  VkResult result = vkDeviceWaitIdle(m_device);
  nvvk::checkResult(result, debugMsg ? debugMsg : "Resources::synchronize");
}


}  // namespace microdisp
