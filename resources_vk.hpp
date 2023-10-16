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

#pragma once

#include <algorithm>
#include <platform.h>
#include <nvmath/nvmath_glsltypes.h>
#include <nvh/nvprint.hpp>
#include <nvvk/buffers_vk.hpp>
#include <nvvk/commands_vk.hpp>
#include <nvvk/context_vk.hpp>
#include <nvvk/debug_util_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/memorymanagement_vk.hpp>
#include <nvvk/pipeline_vk.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/shadermodulemanager_vk.hpp>
#include <nvvk/swapchain_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>
#include "resources_base_vk.hpp"
#include "hbao_pass.hpp"
#include "nvhiz_vk.hpp"
#include "config.h"

#include <functional>

struct ImDrawData;


// allows to use mesh renderers without hw support, simply falls back to regular
#define USE_MESH_FAKE_TEST 0

namespace microdisp {

inline size_t alignedSize(size_t sz, size_t align)
{
  return ((sz + align - 1) / align) * align;
}

struct SceneData;
struct ShaderStats;

struct FrameConfig
{
  SceneData*  sceneUbo     = nullptr;
  SceneData*  sceneUboLast = nullptr;
  ModelType   opaque       = MODEL_DISPLACED;
  ModelType   overlay      = NUM_MODELTYPES;
  bool        cullFreeze   = false;
  int         winWidth;
  int         winHeight;
  ImDrawData* imguiDrawData      = nullptr;
  bool        updateDisplacement = false;
};

#define DEBUGUTIL_SET_NAME(var) debugUtil.setObjectName(var, #var)

class ResourcesVK
{
public:

  // must be static because we are changing resource object during ui events
  // while imgui resources must remain unchanged over app's lifetime
  static VkRenderPass s_passUI;
  static void         initImGui(const nvvk::Context& context);
  static void         deinitImGui(const nvvk::Context& context);

  struct FrameBuffer
  {
    int  renderWidth  = 0;
    int  renderHeight = 0;
    int  supersample  = 0;
    bool useResolved  = false;
    bool vsync        = false;

    VkViewport viewport;
    VkViewport viewportUI;
    VkRect2D   scissor;
    VkRect2D   scissorUI;

    VkRenderPass passClear    = VK_NULL_HANDLE;
    VkRenderPass passPreserve = VK_NULL_HANDLE;

    VkFramebuffer fboScene = VK_NULL_HANDLE;
    VkFramebuffer fboUI    = VK_NULL_HANDLE;

    VkImage imgColor         = VK_NULL_HANDLE;
    VkImage imgColorResolved = VK_NULL_HANDLE;
    VkImage imgDepthStencil  = VK_NULL_HANDLE;
    VkImage imgAtomic        = VK_NULL_HANDLE;
    VkImage imgHizFar        = VK_NULL_HANDLE;

    VkImageView viewColor         = VK_NULL_HANDLE;
    VkImageView viewColorResolved = VK_NULL_HANDLE;
    VkImageView viewDepthStencil  = VK_NULL_HANDLE;
    VkImageView viewDepth         = VK_NULL_HANDLE;
    VkImageView viewAtomic        = VK_NULL_HANDLE;

    nvvk::DeviceMemoryAllocator memAllocator;
  };

  struct CommonResources
  {
    RBuffer view;
    RBuffer stats;
    RBuffer statsRead;
  };

  VkPipelineStageFlags m_supportedShaderPipelineStages =
      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT
      | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT
      | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

  const nvvk::SwapChain* m_swapChain;
  const nvvk::Context*   m_context;

  VkDevice         m_device = VK_NULL_HANDLE;
  VkPhysicalDevice m_physical;
  VkQueue          m_queue;
  uint32_t         m_queueFamily;

  nvvk::DeviceMemoryAllocator m_memAllocator;
  nvvk::ResourceAllocator     m_allocator;

  nvvk::RingFences      m_ringFences;
  nvvk::RingCommandPool m_ringCmdPool;

  bool                  m_submissionWaitForRead;
  nvvk::BatchSubmission m_submission;

  nvvk::ShaderModuleManager m_shaderManager;

  CommonResources m_common;

  VkQueryPool m_stopWatchQueryPool = VK_NULL_HANDLE;
  uint64_t    m_stopWatchTempMask;

  // these are only operational if m_swapChain is valid
  // which tells us whether we actually want to render anything,
  // otherwise we are in pure baking/computation mode
  uint32_t                    m_frame       = 0;
  bool                        m_withinFrame = false;
  size_t                      m_pipeChangeID;
  size_t                      m_fboChangeID;
  FrameBuffer                 m_framebuffer;
  NVHizVK                     m_hiz;
  NVHizVK::Update             m_hizUpdate;
  nvvk::ShaderModuleID        m_hizShaders[NVHizVK::SHADER_COUNT];
  bool                        m_hbaoActive  = true;
  bool                        m_hbaoFullRes = false;
  HbaoPass                    m_hbaoPass;
  HbaoPass::Frame             m_hbaoFrame;
  HbaoPass::Settings          m_hbaoSettings;
  nvvk::GraphicsPipelineState m_basicGraphicsState;


  // ShaderOptimizationWAR workaround for a bug in spirv-opt triggered by shaderc
  // and causes a compilation failure with shaderc_optimization_level_performance
  class ShaderOptimizationWAR
  {
  private:
    ResourcesVK& m_res;

  public:
    ShaderOptimizationWAR(ResourcesVK& res)
        : m_res(res)
    {
#if 1
      m_res.m_shaderManager.setOptimizationLevel(shaderc_optimization_level_zero);
#endif
    }
    ~ShaderOptimizationWAR() { m_res.m_shaderManager.setOptimizationLevel(shaderc_optimization_level_performance); }
  };

  // if swapChain is nullptr we assume offline mode where no rendering is done
  // this will disable init/deinitFramebuffer, hiz, hbao, begin/blit/blank/endFrame

  bool init(nvvk::Context* context, nvvk::SwapChain* swapChain, const std::vector<std::string>& shaderSearchPaths);
  void deinit();

  void updatedShaders();

  void synchronize(const char* debugMsg = nullptr);

  /////////////////////////////////////////
  // these are only valid if swapChain was non-nullptr

  bool initFramebuffer(int width, int height, int supersample, bool vsync);
  void deinitFramebuffer();

  void beginFrame();
  void blitFrame(const FrameConfig& config, nvvk::ProfilerVK& profiler);
  void blankFrame();
  void endFrame();

  void cmdBuildHiz(VkCommandBuffer cmd) const;
  void cmdHBAO(VkCommandBuffer cmd, const FrameConfig& config, nvvk::ProfilerVK& profiler);

  /////////////////////////////////////////

  void cmdCopyStats(VkCommandBuffer cmd) const;
  void getStats(ShaderStats& stats);

  /////////////////////////////////////////////////

  VkRenderPass createPass(bool clear);

  VkCommandBuffer createCmdBuffer(VkCommandPool pool, bool singleshot, bool primary, bool secondaryInClear, bool isCompute = false) const;
  VkCommandBuffer createTempCmdBuffer();

  RBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags flags, VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  template <typename T>
  RBuffer createBufferT(const std::vector<T>& vec, VkBufferUsageFlags flags, VkCommandBuffer cmd = VK_NULL_HANDLE)
  {
    RBuffer entry = createBuffer(vec.size() * sizeof(T), flags);
    if(cmd)
    {
      m_allocator.getStaging()->cmdToBuffer(cmd, entry.buffer, entry.info.offset, entry.info.range, vec.data());
    }

    return entry;
  }
  template <typename T>
  RBuffer createBufferT(const T& obj, VkBufferUsageFlags flags, VkCommandBuffer cmd = VK_NULL_HANDLE)
  {
    RBuffer entry = createBuffer(sizeof(T), flags);
    if(cmd)
    {
      m_allocator.getStaging()->cmdToBuffer(cmd, entry.buffer, entry.info.offset, entry.info.range, &obj);
    }

    return entry;
  }

  template <typename T>
  RBuffer createBufferT(const T* obj, size_t count, VkBufferUsageFlags flags, VkCommandBuffer cmd = VK_NULL_HANDLE)
  {
    RBuffer entry = createBuffer(sizeof(T) * count, flags);
    if(cmd)
    {
      m_allocator.getStaging()->cmdToBuffer(cmd, entry.buffer, entry.info.offset, entry.info.range, obj);
    }

    return entry;
  }

  void destroy(RBuffer& obj)
  {
    m_allocator.destroy(obj);
    obj.info = {nullptr};
    obj.addr = 0;
  }

  void destroy(RTextureR& obj) { m_allocator.destroy(obj); }

  void destroy(RTextureRW& obj)
  {
    m_allocator.destroy(obj);
    obj.descriptor      = {nullptr};
    obj.descriptorImage = {nullptr};
  }

  void simpleUploadBuffer(const RBuffer& dst, const void* src);
  void simpleDownloadBuffer(void* dst, const RBuffer& src);
  void simpleDownloadBuffer(std::function<void(const void*, VkDeviceSize size)> dstFn, const RBuffer& src);
  void simpleDownloadTexture(void* dst, size_t dstSize, const RTextureRW& src);
  void simpleDownloadTexture(std::function<void(const void*)> dstFn, size_t dstSize, const RTextureRW& src);

  // submit for batched execution
  void submissionEnqueue(VkCommandBuffer cmdbuffer) { m_submission.enqueue(cmdbuffer); }
  void submissionEnqueue(uint32_t num, const VkCommandBuffer* cmdbuffers) { m_submission.enqueue(num, cmdbuffers); }
  // perform queue submit
  void submissionExecute(VkFence fence = VK_NULL_HANDLE, bool useImageReadWait = false, bool useImageWriteSignals = false);

  void tempSyncSubmit(VkCommandBuffer cmd, bool reset = true);
  void tempResetResources();

  void   tempStopWatch(VkCommandBuffer cmd, bool begin) const;
  double getStopWatchResult() const;

  // misc utilities

  void cmdBeginRenderPass(VkCommandBuffer cmd, bool clear, bool hasSecondary = false) const;
  void cmdPipelineBarrier(VkCommandBuffer cmd) const;
  void cmdDynamicState(VkCommandBuffer cmd) const;
  void cmdImageTransition(VkCommandBuffer    cmd,
                          VkImage            img,
                          VkImageAspectFlags aspects,
                          VkAccessFlags      src,
                          VkAccessFlags      dst,
                          VkImageLayout      oldLayout,
                          VkImageLayout      newLayout) const;
  void cmdBegin(VkCommandBuffer cmd, bool singleshot, bool primary, bool secondaryInClear, bool isCompute = false) const;
};

}  // namespace microdisp
