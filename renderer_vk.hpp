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

#ifndef RENDERER_H__
#define RENDERER_H__

#include "scene_vk.hpp"
#include "resources_vk.hpp"
#include <nvvk/profiler_vk.hpp>

namespace microdisp {

struct ShaderStats;

class RenderList
{
public:
  struct Config
  {
  };


  void setup(SceneVK* scene, ShaderStats* stats, const Config& config);

  Config               m_config;
  SceneVK* NV_RESTRICT m_scene = nullptr;
  ShaderStats*         m_stats = nullptr;
};

class RendererVK
{
public:
  enum DecoderType
  {
    DECODER_BASETRI_MIP,
    DECODER_MICROTRI,
    DECODER_MICROTRI_INTRINSIC,
  };

  struct Config
  {
    bool        useNormalMap          = false;
    bool        useMicroVertexNormals = false;
    bool        useLod                = false;
    bool        useOcclusionHiz       = false;
    DecoderType decoderType           = DECODER_BASETRI_MIP;
    uint32_t    numThreads            = 0;
    uint32_t    maxVisibleBits        = 20;
  };

  class Type
  {
  public:
    Type() { getRegistry().push_back(this); }

  public:
    virtual bool        isAvailable(const nvvk::Context* context) const = 0;
    virtual const char* name() const                                    = 0;
    virtual RendererVK* create(ResourcesVK& resources) const            = 0;
    virtual bool        supportsCompressed() const                      = 0;
    virtual uint32_t    priority() const { return 0xFF; }
  };

  typedef std::vector<Type*> Registry;

  static Registry& getRegistry()
  {
    static Registry s_registry;
    return s_registry;
  }

  RendererVK(ResourcesVK& resources)
      : m_res(resources)
  {
  }

  ResourcesVK& m_res;

  virtual bool init(RenderList& list, const Config& config)                = 0;
  virtual void deinit()                                                    = 0;
  virtual void draw(const FrameConfig& config, nvvk::ProfilerVK& profiler) = 0;

  virtual ~RendererVK() {}

  //////////////////////////////////////////////////////////////////////////

  enum CmdType
  {
    CMD_TYPE_LO,
    CMD_TYPE_DISPLACED,
    CMD_TYPE_NONDISPLACED_LO,
    CMD_TYPE_SHELL,
    CMD_TYPE_EMPTY,
    NUM_CMD_TYPES,
  };

  struct StandardShaderIDs
  {
    nvvk::ShaderModuleID vertex, fragment, fragment_overlay;
    nvvk::ShaderModuleID shellVertex, shellFragment;
  };

  struct Setup
  {
    VkPipeline                       pipeline         = {nullptr};
    VkPipeline                       pipelineOverlay  = {nullptr};
    VkPipeline                       pipelineFlat     = {nullptr};
    VkPipeline                       pipelineIndirect = {nullptr};
    nvvk::TDescriptorSetContainer<2> container;
    VkShaderStageFlags               stageFlags;
  };

  Setup             m_standard;
  StandardShaderIDs m_standardShaders;
  VkCommandPool     m_cmdPool = VK_NULL_HANDLE;

  CmdType getCmdType(ModelType modelType) const
  {
    switch(modelType)
    {
      case MODEL_LO:
        return CMD_TYPE_LO;
      case MODEL_DISPLACED:
        return CMD_TYPE_DISPLACED;
      case MODEL_SHELL:
        return CMD_TYPE_SHELL;
      default:
        return CMD_TYPE_EMPTY;
    }
  }

  VkPipeline getPipeline(const Setup& setup, bool overlay = false, bool shell = false) const
  {
    if(shell && setup.pipelineFlat)
    {
      return setup.pipelineFlat;
    }
    else if(overlay && setup.pipelineOverlay)
    {
      return setup.pipelineOverlay;
    }
    else
    {
      return setup.pipeline;
    }
  }

  void initStandardShaders(std::string prepend);
  void deinitStandardShaders();
  void initSetupStandard(const SceneVK* scene, const Config& config);
  void initTextureSet(Setup& setup, const SceneVK* scene, const Config& config, VkShaderStageFlags shaderStageFlags);
  void initCommandPool();
  void destroyPipelines(Setup& setup);

  VkCommandBuffer generateCmdBufferEmpty();
  VkCommandBuffer generateCmdBufferLo(RenderList& list, const Config& config, bool overlay, bool onlyNonDisplaced, bool shell = false);
};
}  // namespace microdisp

#endif
