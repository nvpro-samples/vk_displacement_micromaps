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

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp>

#undef GLFW_INCLUDE_VULKAN
#include <imgui/imgui_helper.h>

#include <nvh/cameracontrol.hpp>
#include <nvh/fileoperations.hpp>
#include <nvh/misc.hpp>
#include <nvvk/appwindowprofiler_vk.hpp>
#include "vk_nv_micromesh_prototypes.h"

#include "renderer_vk.hpp"
#include "scene_vk.hpp"

#include "common.h"
#include "common_barymap.h"
#include "common_micromesh_compressed.h"

bool     g_enableMicromeshRTExtensions = true;
bool     g_verbose                     = false;
uint32_t g_numThreads                  = 0;

static_assert(MAX_BASE_SUBDIV < MAX_BARYMAP_LEVELS, "MAX_BARYMAP_LEVELS must allow for MAX_BASE_SUBDIV");

namespace microdisp {
int const SAMPLE_SIZE_WIDTH(1024);
int const SAMPLE_SIZE_HEIGHT(1024);

void setupContextRequirements(nvvk::ContextCreateInfo& contextInfo)
{
  static VkPhysicalDeviceMeshShaderFeaturesNV meshFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV};
  static VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR execPropertiesFeatures = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
  static VkPhysicalDeviceFragmentShaderBarycentricFeaturesNV baryFeatures = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_NV};
  static VkPhysicalDeviceShaderClockFeaturesKHR clockFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};

  static VkPhysicalDeviceAccelerationStructureFeaturesKHR accFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  static VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  static VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayPipelineFeatures = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  static VkPhysicalDeviceShaderFloat16Int8FeaturesKHR f16i8Features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR};
  static VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT imageAtom64Features = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT};
  static VkPhysicalDeviceShaderAtomicFloatFeaturesEXT floatFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};

  static VkPhysicalDeviceOpacityMicromapFeaturesEXT mmOpacityFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT};
  static VkPhysicalDeviceDisplacementMicromapFeaturesNV mmDisplacementFeatures = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_FEATURES_NV};

  contextInfo.apiMajor = 1;
  contextInfo.apiMinor = 3;

  contextInfo.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

#if defined(_DEBUG) && 0
  // enable debugPrintf
  contextInfo.addDeviceExtension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME, false);
  static VkValidationFeaturesEXT      validationInfo    = {VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
  static VkValidationFeatureEnableEXT enabledFeatures[] = {VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};
  validationInfo.enabledValidationFeatureCount          = NV_ARRAY_SIZE(enabledFeatures);
  validationInfo.pEnabledValidationFeatures             = enabledFeatures;
  contextInfo.instanceCreateInfoExt                     = &validationInfo;
#ifdef _WIN32
  _putenv_s("DEBUG_PRINTF_TO_STDOUT", "1");
#else
  putenv("DEBUG_PRINTF_TO_STDOUT=1");
#endif
#endif  // _DEBUG

  contextInfo.addDeviceExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, false, &clockFeatures);
  contextInfo.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, false);

  contextInfo.addDeviceExtension(VK_NV_MESH_SHADER_EXTENSION_NAME, false, &meshFeatures);
  contextInfo.addDeviceExtension(VK_NV_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, false, &baryFeatures);

  contextInfo.addDeviceExtension(VK_NV_FILL_RECTANGLE_EXTENSION_NAME, false);

  contextInfo.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, false);

  contextInfo.addDeviceExtension(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME, false);
  contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, false);
  contextInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accFeatures);
  contextInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayQueryFeatures);
  contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rayPipelineFeatures);

  contextInfo.addDeviceExtension(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME, false, &execPropertiesFeatures);

  contextInfo.addDeviceExtension(VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME, false);
  contextInfo.addDeviceExtension(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME, false, &imageAtom64Features);

  contextInfo.addDeviceExtension(VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME, false);
  contextInfo.addDeviceExtension(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, false, &floatFeatures);

  contextInfo.addDeviceExtension(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME, true, &mmOpacityFeatures);
  contextInfo.addDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME, true, &mmDisplacementFeatures);
}

class Frustum
{
public:
  enum
  {
    PLANE_NEAR,
    PLANE_FAR,
    PLANE_LEFT,
    PLANE_RIGHT,
    PLANE_TOP,
    PLANE_BOTTOM,
    NUM_PLANES
  };

  static inline void init(glm::vec4 planes[NUM_PLANES], const glm::mat4& viewProj)
  {
    const float* clip = glm::value_ptr(viewProj);

    planes[PLANE_RIGHT][0] = clip[3] - clip[0];
    planes[PLANE_RIGHT][1] = clip[7] - clip[4];
    planes[PLANE_RIGHT][2] = clip[11] - clip[8];
    planes[PLANE_RIGHT][3] = clip[15] - clip[12];

    planes[PLANE_LEFT][0] = clip[3] + clip[0];
    planes[PLANE_LEFT][1] = clip[7] + clip[4];
    planes[PLANE_LEFT][2] = clip[11] + clip[8];
    planes[PLANE_LEFT][3] = clip[15] + clip[12];

    planes[PLANE_BOTTOM][0] = clip[3] + clip[1];
    planes[PLANE_BOTTOM][1] = clip[7] + clip[5];
    planes[PLANE_BOTTOM][2] = clip[11] + clip[9];
    planes[PLANE_BOTTOM][3] = clip[15] + clip[13];

    planes[PLANE_TOP][0] = clip[3] - clip[1];
    planes[PLANE_TOP][1] = clip[7] - clip[5];
    planes[PLANE_TOP][2] = clip[11] - clip[9];
    planes[PLANE_TOP][3] = clip[15] - clip[13];

    planes[PLANE_FAR][0] = clip[3] - clip[2];
    planes[PLANE_FAR][1] = clip[7] - clip[6];
    planes[PLANE_FAR][2] = clip[11] - clip[10];
    planes[PLANE_FAR][3] = clip[15] - clip[14];

    planes[PLANE_NEAR][0] = clip[3] + clip[2];
    planes[PLANE_NEAR][1] = clip[7] + clip[6];
    planes[PLANE_NEAR][2] = clip[11] + clip[10];
    planes[PLANE_NEAR][3] = clip[15] + clip[14];

    for(int i = 0; i < NUM_PLANES; i++)
    {
      float length    = sqrtf(planes[i][0] * planes[i][0] + planes[i][1] * planes[i][1] + planes[i][2] * planes[i][2]);
      float magnitude = 1.0f / length;

      for(int n = 0; n < 4; n++)
      {
        planes[i][n] *= magnitude;
      }
    }
  }
};

// used for loading viewpoint files and material filter files
class SimpleParameterFile
{
public:
  // loads a text file and stores the tokens in a vector per line and
  // a vector of lines
  // everything after a # gets ignored
  SimpleParameterFile(std::string fileName)
  {
    std::ifstream f;
    f.open(fileName);
    if(!f)
      return;

    std::string lineOfFile;
    while(getline(f, lineOfFile))
    {
      if(lineOfFile.length() == 0)
        continue;

      ParameterLine pLine;

      std::stringstream ss(lineOfFile);
      std::string       token;

      while(getline(ss, token, ' '))
      {
        if(token.length() == 0)
          continue;
        if(token[0] == '#')
        {
          // ignore rest of this line
          break;
        }
        Parameter p;
        p.strValue = token;
        pLine.parameter.push_back(p);
      }

      if(pLine.parameter.size() > 0)
      {
        line.push_back(pLine);
      }
    }
    f.close();
  }

  struct Parameter
  {
    // the string of the token
    std::string strValue;

    // returns true on success of storing the token as an int to toFill
    bool toInt(int& toFill)
    {
      bool success = true;
      try
      {
        toFill = std::stoi(strValue);
      }
      catch(...)
      {
        success = false;
      }
      return success;
    }

    // returns true on success of storing the token as a float to toFill
    bool toFloat(float& toFill)
    {
      bool success = true;
      try
      {
        toFill = std::stof(strValue);
      }
      catch(...)
      {
        success = false;
      }
      return success;
    }
  };

  struct ParameterLine
  {
    std::vector<Parameter> parameter;
  };

  std::vector<ParameterLine> line;
};

class Sample : public nvvk::AppWindowProfilerVK

{
  enum LodType
  {
    LOD_PRECOMPUTED_SPHERE,
    LOD_DYNAMIC_TRIANGLE,
  };

  enum NormalType
  {
    NORMAL_FACET,
    NORMAL_VERTEX,
    NORMAL_TEXTURE,
    NORMAL_MICROVERTEX,
  };

  enum GuiEnums
  {
    GUI_VIEWPOINT,
    GUI_RENDERER,
    GUI_SUPERSAMPLE,
    GUI_SURFACEVIS,
    GUI_LODTYPE,
    GUI_DECODERTYPE,
    GUI_LAYOUT,
    GUI_FORMAT,
    GUI_MODEL,
    GUI_MODEL_OVERLAY,
    GUI_NORMALS,
  };

public:
  struct Tweak
  {
    // shader / shading related
    bool                    useStats             = false;
    bool                    showReflectionLine   = false;
    bool                    showReflectionBand   = false;
    bool                    hbaoFullRes          = false;
    float                   hbaoRadius           = 0.05f;
    bool                    colorize             = false;
    bool                    fp16displacementMath = false;
    NormalType              normalType           = NORMAL_FACET;
    float                   lodAreaScale         = 1.0f;
    LodType                 lodType              = LOD_PRECOMPUTED_SPHERE;
    bool                    useLod               = false;
    bool                    usePrimitiveCulling  = false;
    float                   displacementScale    = 1.0f;
    int                     surfaceVisualization = SURFACEVIS_SHADING;
    bool                    useOcclusionCulling  = false;
    RendererVK::DecoderType decoderType          = RendererVK::DECODER_BASETRI_MIP;

    // render / scene setup
    int      renderer            = 0;
    int      viewPoint           = 0;
    int      supersample         = 2;
    float    fov                 = 45.0f;
    uint32_t objectFrom          = 0;
    uint32_t objectNum           = ~0u;
    uint32_t gridCopies          = 1;
    uint32_t gridAxis            = 5;
    float    gridSpacing         = 1.05f;
    float    renderScale         = 1.0f;
    float    renderBias          = 0.0f;
    float    rotateModelSpeed    = 0.0f;
    vec2     rotateModelDistance = vec2(0.4f, 0.1f);
    uint32_t maxVisibleBits      = 20;

    static constexpr float minFov = 1.0f, maxFov = 130.0f;
  };

  struct ViewPoint
  {
    std::string name;
    glm::mat4   mat;
    float       sceneScale;
    float       fov;
  };

  bool m_useUI = true;
#ifdef _DEBUG
  bool m_advancedUI = true;
#else
  bool m_advancedUI = false;
#endif
  ImGuiH::Registry m_ui;
  double           m_uiTime = 0;

  Tweak  m_tweak;
  Tweak  m_lastTweak;
  bool   m_lastVsync;
  size_t m_lastFbo = 0;

  SceneVK                   m_scene;
  std::string               m_rendererName;
  std::vector<unsigned int> m_renderersSorted;
  uint32_t                  m_rendererType;
  std::string               m_rendererShaderPrepend;
  std::string               m_rendererLastShaderPrepend;

  RendererVK* NV_RESTRICT m_renderer = nullptr;
  ResourcesVK             m_resources;
  RenderList              m_renderList;
  FrameConfig             m_frameConfig;
  SceneData               m_sceneUbo;
  SceneData               m_sceneUboLast;

  std::string            m_viewpointFilename;
  std::vector<ViewPoint> m_viewPoints;

  std::string m_modelFilenameLo;

  glm::vec3 m_modelUpVector = glm::vec3(0, 1, 0);

  int                m_frames        = 0;
  double             m_lastFrameTime = 0;
  double             m_statsCpuTime  = 0;
  double             m_statsGpuTime  = 0;
  double             m_statsRayTime  = 0;
  double             m_statsRstTime  = 0;
  double             m_statsRdrTime  = 0;
  double             m_statsTskTime  = 0;
  double             m_statsDrwTime  = 0;
  ShaderStats        m_stats;
  nvh::CameraControl m_control;
  bool               m_cameraParseSuccess = true;

  bool setRendererFromName(const std::string& name);

  bool initScene(const char* filenameLo);
  bool initFramebuffers(int width, int height);
  void initRenderer(int type);

  void updateGrid(bool gpu = true)
  {
    m_scene.updateGrid(m_resources, m_tweak.gridCopies, m_tweak.gridAxis, m_tweak.gridSpacing, gpu);
  }
  void updateLow() { m_scene.updateLow(m_resources); }

  bool initCore();
  void postInitScene();

  void deinitRenderer();

  void saveViewpoint();
  void loadViewpoints();
  void setViewpoint();

  void setupConfigParameters();

  std::string getShaderPrepend();

  template <typename T>
  bool tweakChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_tweak);
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_lastTweak) + offset, sizeof(T)) != 0;
  }

  template <typename T>
  bool tweakChangedNonZero(const T& val) const
  {
    size_t   offset  = size_t(&val) - size_t(&m_tweak);
    const T* lastVal = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(&m_lastTweak) + offset);
    bool     state   = (val != 0) != (*lastVal != 0);
    return state;
  }

  template <typename T>
  bool tweakChangedPositive(const T& val) const
  {
    size_t   offset  = size_t(&val) - size_t(&m_tweak);
    const T* lastVal = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(&m_lastTweak) + offset);
    bool     state   = (val >= 0) != (*lastVal >= 0);
    return state;
  }

  Sample()
      : AppWindowProfilerVK(false)
  {
    setupConfigParameters();
    setupContextRequirements(m_contextInfo);

    // we need to ignore errors regarding storageInputOutput16
    // due to an ovesight in the spec task shaders output and mesh shader input would have to adhere to this feature,
    // but they are treated different compared to vertex shader input/outputs.
    m_context.ignoreDebugMessage(0x6e224e9);
    m_context.ignoreDebugMessage(0x715035dd);

#if defined(NDEBUG)
    setVsync(false);
#endif
  }

public:
  void processUI(int width, int height, double time);

  bool validateConfig() override;

  bool begin() override;

  void think(double time) override;
  void resize(int width, int height) override;

  void postBenchmarkAdvance() override;

  void end() override;

  // return true to prevent m_window updates
  bool mouse_pos(int x, int y) override
  {
    if(!m_useUI)
      return false;

    return ImGuiH::mouse_pos(x, y);
  }
  bool mouse_button(int button, int action) override
  {
    if(!m_useUI)
      return false;

    return ImGuiH::mouse_button(button, action);
  }
  bool mouse_wheel(int wheel) override
  {
    if(!m_useUI)
      return false;

    return ImGuiH::mouse_wheel(wheel);
  }
  bool key_char(int key) override
  {
    if(!m_useUI)
      return false;

    return ImGuiH::key_char(key);
  }
  bool key_button(int button, int action, int mods) override
  {
    if(!m_useUI)
      return false;

    return ImGuiH::key_button(button, action, mods);
  }
};


std::string Sample::getShaderPrepend()
{
  std::string prepend;
  prepend += nvh::stringFormat("#define USE_PRIMITIVE_CULLING %d\n", m_tweak.usePrimitiveCulling ? 1 : 0);
  prepend += nvh::stringFormat("#define USE_OCCLUSION_CULLING %d\n", m_tweak.useOcclusionCulling ? 1 : 0);
  prepend += nvh::stringFormat("#define USE_TRI_LOD %d\n", m_tweak.lodType == LOD_DYNAMIC_TRIANGLE ? 1 : 0);
  prepend += nvh::stringFormat("#define USE_STATS %d\n", m_tweak.useStats ? 1 : 0);
  prepend += nvh::stringFormat("#define USE_FACET_SHADING %d\n", m_tweak.normalType == NORMAL_FACET ? 1 : 0);
  prepend += nvh::stringFormat("#define USE_FP16_DISPLACEMENT_MATH %d\n", m_tweak.fp16displacementMath ? 1 : 0);
  prepend += nvh::stringFormat("#define SURFACEVIS %d\n", m_tweak.surfaceVisualization);

  LOGI("new shader setup\n");
#ifdef _DEBUG
  printf(prepend.c_str());
#endif

  return prepend;
}

bool Sample::initScene(const char* filenameLo)
{
  m_scene = SceneVK();

  if(!filenameLo)
  {
    // Nothing to load
    return true;
  }
  bool status = m_scene.load(filenameLo);
  if(status)
  {
    if(m_scene.meshSetLo)
    {
      LOGI("lo-res mesh: %s\n", filenameLo);
      LOGI("vertices:   %9d\n", uint32_t(m_scene.meshSetLo->attributes.positions.size()));
      LOGI("primitives: %9d\n", uint32_t(m_scene.meshSetLo->indices.size() / 3));
      LOGI("materials:  %9d\n", int32_t(m_scene.meshSetLo->materials.size()));
      LOGI("instances:  %9d\n", int32_t(m_scene.meshSetLo->meshInstances.size()));
      LOGI("bboxdim: %f, %f, %f\n", m_scene.meshSetLo->bbox.diagonal().x, m_scene.meshSetLo->bbox.diagonal().y,
           m_scene.meshSetLo->bbox.diagonal().z);
    }
    LOGI("\n");
  }
  else
  {
    LOGE("\ncould not load model (%s)\n", filenameLo);
  }

  return status;
}

bool Sample::initFramebuffers(int width, int height)
{
  return m_resources.initFramebuffer(width, height, m_tweak.supersample, getVsync());
}

void Sample::postInitScene()
{
  m_scene.init(m_resources);


  updateGrid(false);
  updateLow();

  m_sceneUbo                 = {};
  m_frameConfig.sceneUbo     = &m_sceneUbo;
  m_frameConfig.sceneUboLast = &m_sceneUboLast;

  m_control.m_sceneUp = m_modelUpVector;
  // Handle the case where we have no hi-res mesh properly
  MeshBBox bbox              = m_scene.meshSetLo->bbox;
  m_control.m_sceneOrbit     = glm::vec3((bbox.maxs + bbox.mins)) * 0.5f;
  m_control.m_sceneDimension = glm::length((bbox.maxs - bbox.mins));
  m_control.m_viewMatrix = glm::lookAt(m_control.m_sceneOrbit - (-glm::vec3(1, 1, 1) * m_control.m_sceneDimension * 0.5f),
                                       m_control.m_sceneOrbit, m_modelUpVector);

  m_sceneUbo.wLightPos = glm::vec4((bbox.maxs + bbox.mins) * 0.5f + m_control.m_sceneDimension, 1.0);

  loadViewpoints();

  if(m_useUI)
  {
    m_ui.enumReset(GUI_VIEWPOINT);
    for(auto it = m_viewPoints.begin(); it != m_viewPoints.end(); it++)
    {
      m_ui.enumAdd(GUI_VIEWPOINT, int(it - m_viewPoints.begin()), it->name.c_str());
    }
    if(m_viewPoints.empty())
    {
      m_ui.enumAdd(GUI_VIEWPOINT, 0, "default");
    }
    m_ui.enumReset(GUI_NORMALS);
    m_ui.enumAdd(GUI_NORMALS, NORMAL_FACET, "facet");
    m_ui.enumAdd(GUI_NORMALS, NORMAL_VERTEX, "base-vertex");
    if(m_scene.meshSetLo->textures.size() > 1)
    {
      m_ui.enumAdd(GUI_NORMALS, NORMAL_TEXTURE, "texture");
    }
    if(m_scene.barySet.displacements.size() && m_scene.barySet.shadings.size() == m_scene.barySet.displacements.size())
    {
      m_ui.enumAdd(GUI_NORMALS, NORMAL_MICROVERTEX, "micro-vertex");
    }

    m_ui.enumReset(GUI_DECODERTYPE);
    if(m_scene.barySet.supportsCompressedMips())
    {
      m_ui.enumAdd(GUI_DECODERTYPE, RendererVK::DECODER_BASETRI_MIP, "base w. mip");
    }

    m_ui.enumAdd(GUI_DECODERTYPE, RendererVK::DECODER_MICROTRI, "micro");
    if(m_context.hasDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME))
    {
      m_ui.enumAdd(GUI_DECODERTYPE, RendererVK::DECODER_MICROTRI_INTRINSIC, "micro (intrinsic)");
    }

    {
      const RendererVK::Registry& registry = RendererVK::getRegistry();
      m_ui.enumReset(GUI_RENDERER);
      for(size_t i = 0; i < m_renderersSorted.size(); i++)
      {
        auto rendererType = registry[m_renderersSorted[i]];
        if(rendererType->supportsCompressed() == m_scene.hasCompressedDisplacement)
        {
          m_ui.enumAdd(GUI_RENDERER, int(i), registry[m_renderersSorted[i]]->name());
        }
      }
    }
  }

  if(!m_scene.barySet.supportsCompressedMips() && m_tweak.decoderType == RendererVK::DECODER_BASETRI_MIP)
  {
    // if not supported but set, then revert to MICROTRI decoder, preferably the intrinsic version
    m_tweak.decoderType = m_context.hasDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME) ?
                              RendererVK::DECODER_MICROTRI_INTRINSIC :
                              RendererVK::DECODER_MICROTRI;
  }

  if(!m_context.hasDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME) && m_tweak.decoderType == RendererVK::DECODER_MICROTRI_INTRINSIC)
  {
    // if intrinsics not supported use fallback
    m_tweak.decoderType = RendererVK::DECODER_MICROTRI;
  }

  setViewpoint();

  if(m_scene.hasCompressedDisplacement)
  {
    setRendererFromName("compressed ms");
  }
  else if(m_scene.hasUncompressedDisplacement)
  {
    setRendererFromName("uncompressed ms");
  }
}

void Sample::deinitRenderer()
{
  if(m_renderer)
  {
    m_resources.synchronize("sync deinitRenderer");
    m_renderer->deinit();
    delete m_renderer;
    m_renderer = nullptr;
  }
}

void Sample::initRenderer(int typesort)
{
  if(!(m_scene.meshSetLo))
    return;

  int type       = m_renderersSorted[typesort % m_renderersSorted.size()];
  m_rendererType = type;

  deinitRenderer();

  {
    RenderList::Config config;
    m_renderList.setup(&m_scene, &m_stats, config);
  }
  {
    RendererVK::Config config;
    config.useLod                = m_tweak.useLod;
    config.numThreads            = g_numThreads;
    config.useOcclusionHiz       = m_tweak.useOcclusionCulling;
    config.decoderType           = m_tweak.decoderType;
    config.maxVisibleBits        = m_tweak.maxVisibleBits;
    config.useNormalMap          = m_tweak.normalType == NORMAL_TEXTURE;
    config.useMicroVertexNormals = m_tweak.normalType == NORMAL_MICROVERTEX;

    LOGI("renderer: %s\n", RendererVK::getRegistry()[type]->name());
    m_renderer = RendererVK::getRegistry()[type]->create(m_resources);
    if(!m_renderer->init(m_renderList, config))
    {
      LOGE("renderer init failed\n");
      exit(-1);
    }
  }
}

void Sample::loadViewpoints()
{
  m_viewPoints.clear();

  if(m_viewpointFilename.empty())
    return;

  SimpleParameterFile vpParameters(m_viewpointFilename);
  for(SimpleParameterFile::ParameterLine line : vpParameters.line)
  {
    // name + 16 for the matrix + optional scale + optional fov
    if(line.parameter.size() >= 17 && line.parameter.size() <= 19)
    {
      bool      lineIsOK = true;
      ViewPoint vp;
      vp.name = line.parameter[0].strValue;

      // read matrix
      float* mat_array = glm::value_ptr(vp.mat);
      for(auto i = 0; i < 16; ++i)
      {
        bool valueIsFloat = line.parameter[1 + i].toFloat(mat_array[i]);

        lineIsOK = lineIsOK & valueIsFloat;
      }

      // optionally scene scale
      if(line.parameter.size() == 18)
      {
        lineIsOK = lineIsOK & line.parameter[17].toFloat(vp.sceneScale);
      }
      else
      {
        vp.sceneScale = 1.0f;
      }

      // optionally real scale
      if(line.parameter.size() == 19)
      {
        lineIsOK = lineIsOK & line.parameter[18].toFloat(vp.fov);
      }
      else
      {
        vp.fov = 0;
      }

      // only save if all parameters were read correctly
      if(lineIsOK)
      {
        m_viewPoints.push_back(vp);
      }
    }
  }
}

void Sample::setViewpoint()
{
  if(m_viewPoints.empty())
  {
    m_tweak.viewPoint = 0;
    return;
  }

  m_tweak.viewPoint = std::min(std::max(m_tweak.viewPoint, 0), int(m_viewPoints.size() - 1));

  m_control.m_viewMatrix = m_viewPoints[m_tweak.viewPoint].mat;
  if(m_viewPoints[m_tweak.viewPoint].fov)
  {
    m_tweak.fov = m_viewPoints[m_tweak.viewPoint].fov;
  }
}

bool Sample::initCore()
{
  m_context.ignoreDebugMessage(0xa7bb8db6);  // not a bug: complains about StorageInputOutput16
  //m_context.ignoreDebugMessage(0x6bbb14);    // not a bug: complains about InconsistentSpirv during optimization
  //m_context.ignoreDebugMessage(0x23e43bb7);  // not a bug: complains about InputNotProduced for pervertexNV variables - see https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/3194

  LOGI("num threads: %u\n", g_numThreads);

  std::vector<std::string> shaderSearchPaths;
  std::string              path = NVPSystem::exePath();
  shaderSearchPaths.push_back(NVPSystem::exePath());
  shaderSearchPaths.push_back(std::string("GLSL_" PROJECT_NAME));
  shaderSearchPaths.push_back(NVPSystem::exePath() + std::string("GLSL_" PROJECT_NAME));
  shaderSearchPaths.push_back(NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY));

  m_resources.m_hbaoFullRes = m_tweak.hbaoFullRes;
  bool validated            = initScene(m_modelFilenameLo.empty() ? nullptr : m_modelFilenameLo.c_str());
  validated                 = validated && m_resources.init(&m_context, &m_swapChain, shaderSearchPaths);
  validated                 = validated
              && m_resources.initFramebuffer(m_windowState.m_swapSize[0], m_windowState.m_swapSize[1],
                                             m_tweak.supersample, getVsync());

  m_resources.m_shaderManager.m_prepend = getShaderPrepend();

  if(!validated)
  {
    return false;
  }
  if(m_scene.meshSetLo)
  {
    postInitScene();
    // postInitScene may change some flags
    m_resources.m_shaderManager.m_prepend = getShaderPrepend();
  }

  return true;
}

bool Sample::begin()
{
  m_profilerPrint = false;
  m_timeInTitle   = true;
  m_renderer      = nullptr;

  if(m_context.hasDeviceExtension(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME))
  {
    load_VK_EXT_opacity_micromap_prototypes(m_context.m_device, vkGetDeviceProcAddr);
  }

  // ImGUI must come first
  ImGuiH::Init(m_windowState.m_winSize[0], m_windowState.m_winSize[1], this, ImGuiH::FONT_MONOSPACED_SCALED);
  ResourcesVK::initImGui(m_context);

  const RendererVK::Registry& registry = RendererVK::getRegistry();
  {
    // setup renderer list
    for(size_t i = 0; i < registry.size(); i++)
    {
      if(registry[i]->isAvailable(&m_context))
      {
        uint sortkey = uint(i);
        sortkey |= registry[i]->priority() << 16;
        m_renderersSorted.push_back(sortkey);
      }
    }

    if(m_renderersSorted.empty())
    {
      LOGE("No renderers available\n");
      return false;
    }

    std::sort(m_renderersSorted.begin(), m_renderersSorted.end());

    for(size_t i = 0; i < m_renderersSorted.size(); i++)
    {
      m_renderersSorted[i] &= 0xFFFF;
      LOGI("renderer %d: %s\n", uint32_t(i), registry[m_renderersSorted[i]]->name());
    }
  }

  bool validated = initCore();
  if(!validated)
  {
    return false;
  }

  if(!setRendererFromName(m_rendererName))
  {
    return false;
  }

  // setup UI
  if(m_useUI)
  {
    auto& imgui_io = ImGui::GetIO();

    m_ui.enumAdd(GUI_MODEL, MODEL_LO, "Base");
    m_ui.enumAdd(GUI_MODEL, MODEL_DISPLACED, "Displaced");
    m_ui.enumAdd(GUI_MODEL, NUM_MODELTYPES, "None");

    m_ui.enumAdd(GUI_MODEL_OVERLAY, MODEL_LO, "Base");
    m_ui.enumAdd(GUI_MODEL_OVERLAY, MODEL_DISPLACED, "Displaced");
    m_ui.enumAdd(GUI_MODEL_OVERLAY, MODEL_SHELL, "Shell");
    m_ui.enumAdd(GUI_MODEL_OVERLAY, NUM_MODELTYPES, "None");

    m_ui.enumAdd(GUI_LAYOUT, (int32_t)bary::ValueLayout::eTriangleUmajor, "U-MAJOR");
    m_ui.enumAdd(GUI_LAYOUT, (int32_t)bary::ValueLayout::eTriangleBirdCurve, "BIRD_CURVE");

    m_ui.enumAdd(GUI_FORMAT, (int32_t)(bary::Format::eR8_unorm), "8_UNORM");
    m_ui.enumAdd(GUI_FORMAT, (int32_t)(bary::Format::eR16_unorm), "16_UNORM");
    m_ui.enumAdd(GUI_FORMAT, (int32_t)(bary::Format::eR11_unorm_pack16), "11_UNORM_PACK16");
    m_ui.enumAdd(GUI_FORMAT, (int32_t)(bary::Format::eR32_sfloat), "32_SFLOAT");

    m_ui.enumAdd(GUI_SURFACEVIS, SURFACEVIS_SHADING, "Default Shading");
    m_ui.enumAdd(GUI_SURFACEVIS, SURFACEVIS_ANISOTROPY, "Anisotropy");
    m_ui.enumAdd(GUI_SURFACEVIS, SURFACEVIS_BASETRI, "Base Triangle index");
    m_ui.enumAdd(GUI_SURFACEVIS, SURFACEVIS_MICROTRI, "Global microtriangle index (raster)");
    m_ui.enumAdd(GUI_SURFACEVIS, SURFACEVIS_LOCALTRI, "Local microtriangle index (raster)");
    m_ui.enumAdd(GUI_SURFACEVIS, SURFACEVIS_FORMAT, "Encoding Format (raster compressed)");
    m_ui.enumAdd(GUI_SURFACEVIS, SURFACEVIS_VALUERANGE, "Value Range (raster)");
    m_ui.enumAdd(GUI_SURFACEVIS, SURFACEVIS_BASESUBDIV, "Base Subdiv (raster)");
    m_ui.enumAdd(GUI_SURFACEVIS, SURFACEVIS_LODBIAS, "Dynamic LoD Bias (raster)");
    m_ui.enumAdd(GUI_SURFACEVIS, SURFACEVIS_LODSUBDIV, "Dynamic LoD Base Subdiv (raster)");

    m_ui.enumAdd(GUI_LODTYPE, LOD_PRECOMPUTED_SPHERE, "precomp. sphere");
    m_ui.enumAdd(GUI_LODTYPE, LOD_DYNAMIC_TRIANGLE, "dynamic triangle");
  }

  m_resources.updatedShaders();
  initRenderer(m_tweak.renderer);

  m_lastTweak = m_tweak;
  m_lastVsync = getVsync();
  m_lastFbo   = m_resources.m_fboChangeID;

  return validated;
}


void Sample::end()
{
#if !_DEBUG
  exit(0);
#endif
  if(!m_resources.m_device)
    return;

  deinitRenderer();
  m_scene.deinit(m_resources);
  m_resources.deinit();
  ResourcesVK::deinitImGui(m_context);
}

void Sample::processUI(int width, int height, double time)
{
  // Update imgui configuration
  auto& imgui_io       = ImGui::GetIO();
  imgui_io.DeltaTime   = static_cast<float>(time - m_uiTime);
  imgui_io.DisplaySize = ImVec2(static_cast<float>(width), static_cast<float>(height));

  m_uiTime = time;

  ImGui::NewFrame();
  ImGui::SetNextWindowPos(ImVec2(5, 5), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(ImGuiH::dpiScaled(310), SAMPLE_SIZE_HEIGHT - 16), ImGuiCond_FirstUseEver);


  ImVec4 advancedColor = {70.0f / 255.0f, 58.0f / 255.0f, 89.0f / 255.0f, 0.8f};

  if(ImGui::Begin("NVIDIA " PROJECT_NAME))
  {
    ImGui::PushItemWidth(ImGuiH::dpiScaled(130));
    ImGui::Checkbox("enable advanced UI options", &m_advancedUI);

    bool earlyOut = !(m_scene.meshSetLo);

    if(ImGui::CollapsingHeader("LOAD", ImGuiTreeNodeFlags_DefaultOpen))
    {
      bool doLoadFiles = false;

      if(ImGui::Button("DISPLACED MODEL"))
      {
        std::string fileNameLo = NVPWindow::openFileDialog("Pick lo-res model with displacement (mandatory)",
                                                           "Supported (glTF 2.0)|*.gltf;*.glb;*.csf;"
                                                           "|All|*.*");

        if(!fileNameLo.empty())
        {
          m_modelFilenameLo   = fileNameLo;
          m_viewpointFilename = std::string();
          doLoadFiles         = true;
        }
      }
      ImGui::SameLine();
      if(ImGui::Button("CFG FILE"))
      {
        std::string newFileName = openFileDialog("Config File", "cfg|*.cfg");
        if(!newFileName.empty())
        {
          m_modelFilenameLo   = std::string();
          m_viewpointFilename = std::string();
          parseConfigFile(newFileName.c_str());
          if(!m_modelFilenameLo.empty())
          {
            doLoadFiles = true;
          }
        }
      }

      if(doLoadFiles)
      {
        m_resources.synchronize("open file");
        m_scene.deinit(m_resources);

        m_scene = SceneVK();

        if(!initScene(m_modelFilenameLo.c_str()))
        {
          exit(-1);
        }
        postInitScene();
        // postInitScene may change some flags
        m_resources.m_shaderManager.m_prepend = getShaderPrepend();
        initRenderer(m_tweak.renderer);
        m_lastTweak = m_tweak;
      }
    }
    if(earlyOut)
    {
      ImGui::End();
      return;
    }

    if(ImGui::CollapsingHeader("View", ImGuiTreeNodeFlags_DefaultOpen))
    {
      ImGui::PushItemWidth(ImGuiH::dpiScaled(170));
      m_ui.enumCombobox(GUI_RENDERER, "renderer", &m_tweak.renderer);
      m_ui.enumCombobox(GUI_VIEWPOINT, "viewpoint", &m_tweak.viewPoint);
      m_ui.enumCombobox(GUI_MODEL, "opaque", &m_frameConfig.opaque);
      m_ui.enumCombobox(GUI_MODEL_OVERLAY, "overlay", &m_frameConfig.overlay);
      ImGui::PopItemWidth();

      if(ImGui::Button("Copy camera to clipboard"))
      {
        glm::mat4 viewI = glm::inverse(m_control.m_viewMatrix);
        glm::vec3 eye   = glm::vec3(viewI[3]);  // position of eye in the world
        glm::vec3 dir   = glm::vec3(viewI[2]);
        glm::vec3 up    = glm::vec3(viewI[1]);

        glm::vec3 ctr = eye - dir * m_control.m_sceneDimension * 0.5f;

        std::string clip = nvh::stringFormat("{%f, %f, %f}, {%f, %f, %f}, {%f, %f, %f}, %f", eye.x, eye.y, eye.z, ctr.x,
                                             ctr.y, ctr.z, up.x, up.y, up.z, m_tweak.fov);
        ImGui::SetClipboardText(clip.c_str());
      }
      if(ImGui::Button("Paste camera from clipboard"))
      {
        const char* text = ImGui::GetClipboardText();

        float val[10];
        int   result = text ? sscanf(text, "{%f, %f, %f}, {%f, %f, %f}, {%f, %f, %f}, %f", &val[0], &val[1], &val[2],
                                     &val[3], &val[4], &val[5], &val[6], &val[7], &val[8], &val[9]) :
                              0;
        m_cameraParseSuccess = result >= 9;
        if(m_cameraParseSuccess)
        {
          glm::vec3 eye{val[0], val[1], val[2]};
          glm::vec3 ctr{val[3], val[4], val[5]};
          glm::vec3 up{val[6], val[7], val[8]};

          m_control.m_viewMatrix = glm::lookAt(eye, ctr, up);
        }
        if(result == 10)
        {
          m_tweak.fov = glm::clamp<float>(val[9], m_tweak.minFov, m_tweak.maxFov);
        }
      }
      ImGuiH::tooltip(
          "Copy/Paste camera to/from clipboard as\n"
          "{eye.x, eye.y, eye.z}, {center.x, center.y, center.z}, {up.x, up.y, up.z}, FOV\n"
          "FOV in degrees, optional");

      if(!m_cameraParseSuccess)
      {
        m_cameraParseSuccess |= ImGui::Button("Dismiss###dismissFailedToParseCamera");
        ImGui::SameLine();
        ImGui::Text("Failed to parse clipboard");
      }
    }

    if(ImGui::CollapsingHeader("Shading", ImGuiTreeNodeFlags_DefaultOpen))
    {
      ImGui::PushItemWidth(ImGuiH::dpiScaled(170));
      m_ui.enumCombobox(GUI_NORMALS, "shading normals", (int*)&m_tweak.normalType);
      m_ui.enumCombobox(GUI_SURFACEVIS, "visualization mode", &m_tweak.surfaceVisualization);
      ImGui::PopItemWidth();
      ImGui::Checkbox("screen-space ambient occlusion (hbao)", &m_resources.m_hbaoActive);
    }

    if(ImGui::CollapsingHeader("Mesh Infos", ImGuiTreeNodeFlags_DefaultOpen))
    {
      bool isDisplaced = m_frameConfig.opaque == MODEL_DISPLACED;

      ImGui::Text("       : b  m  k");
      ImGui::Text("Lo tris: %10d %s", uint32_t(m_scene.meshSetLo->indices.size() / 3), !isDisplaced ? "active" : "");
      const baryutils::BaryStats& stats =
          m_scene.hasUncompressedDisplacement ? m_scene.barySet.uncompressedStats : m_scene.barySet.compressedStats;
      ImGui::Text("Di tris: %10d %s", stats.microTriangles, isDisplaced ? "active" : "");
      ImGui::Text("Di umsh: %10d", uint32_t(stats.mapTriangles));
      ImGui::Text("Di shell volume: %f", m_scene.barySetShellVolume);
    }

    if(m_scene.hasCompressedDisplacement && ImGui::CollapsingHeader("Compressed Details"))
    {
      ImGui::Text("encoding: %s", bary::baryFormatGetName(m_scene.barySet.compressedStats.valueFormat));
      ImGui::Text("subdiv:   %d - %d", m_scene.barySet.compressedStats.minSubdivLevel, m_scene.barySet.compressedStats.maxSubdivLevel);
      const baryutils::BaryStats& stats = m_scene.barySet.compressedStats;
      ImGui::Text("uncompr.KB: %8d (16-bit)", uint32_t(stats.microVertices * sizeof(uint16_t)) / 1024);
      ImGui::Text("compr.  KB: %8d (%3d %%)", stats.dataByteSize / 1024,
                  uint32_t(double(stats.dataByteSize) * 100.0 / double(stats.microVertices * sizeof(uint16_t))));
      ImGui::Text("mip     KB: %8d", m_scene.barySet.compressedMipByteSize / 1024);
      ImGui::NewLine();

      ImGui::Text("64   tri  64 bytes");
      ImGui::SameLine();
      ImGui::Text("(%3d %%)", uint32_t(double(stats.blocksPerFormat[uint32_t(bary::BlockFormatDispC1::eR11_unorm_lvl3_pack512)])
                                       * 100.0 / double(stats.blocks)));

      ImGui::Text("256  tri 128 bytes");
      ImGui::SameLine();
      ImGui::Text("(%3d %%)", uint32_t(double(stats.blocksPerFormat[uint32_t(bary::BlockFormatDispC1::eR11_unorm_lvl4_pack1024)])
                                       * 100.0 / double(stats.blocks)));

      ImGui::Text("1024 tri 128 bytes");
      ImGui::SameLine();
      ImGui::Text("(%3d %%)", uint32_t(double(stats.blocksPerFormat[uint32_t(bary::BlockFormatDispC1::eR11_unorm_lvl5_pack1024)])
                                       * 100.0 / double(stats.blocks)));

      ImGui::Text("<= %4d subtri: %3d %%", 1,
                  uint32_t(double(stats.blocksPerTriangleHisto[0]) * 100.0 / double(stats.mapTriangles)));
      for(uint32_t i = 1; i < 8; i += 2)
      {
        ImGui::Text("<= %4d subtri: %3d %%", 1 << (i + 1),
                    uint32_t(double(stats.blocksPerTriangleHisto[i] + stats.blocksPerTriangleHisto[i + 1]) * 100.0
                             / double(stats.mapTriangles)));
      }
    }

    if(m_scene.hasUncompressedDisplacement && ImGui::CollapsingHeader("Uncompressed Details"))
    {
      ImGui::Text("encoding: %s", bary::baryFormatGetName(m_scene.barySet.uncompressedStats.valueFormat));
      ImGui::Text("subdiv:   %d - %d", m_scene.barySet.uncompressedStats.minSubdivLevel,
                  m_scene.barySet.uncompressedStats.maxSubdivLevel);
      ImGui::Text("KB:       %d", m_scene.barySet.uncompressedStats.dataByteSize / 1024);

      bool doPrint = true;
      for(size_t d = 0; d < m_scene.barySet.displacements.size(); d++)
      {
        const BaryDisplacementAttribute& baryDisplacement = m_scene.barySet.displacements[d];
        if(baryDisplacement.uncompressed && doPrint)
        {
          const baryutils::BaryBasicData& uncompressed = *baryDisplacement.uncompressed;
          ImGui::Text("scale:  %f group[0]", uncompressed.groups[0].floatScale.r);
          ImGui::Text("bias:   %f group[0]", uncompressed.groups[0].floatBias.r);
          doPrint = false;
        }
      }
    }

    if(ImGui::CollapsingHeader("Render Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
      ImGui::PushItemWidth(ImGuiH::dpiScaled(170));
      ImGui::Checkbox("use dynamic lod", &m_tweak.useLod);
      ImGui::InputFloat("lod area scale", &m_tweak.lodAreaScale, 0.025f, 0.25f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
      ImGuiH::InputFloatClamped("fov", &m_tweak.fov, m_tweak.minFov, m_tweak.maxFov, 10, 20, "%.1f", ImGuiInputTextFlags_EnterReturnsTrue);
      m_ui.enumCombobox(GUI_DECODERTYPE, "decoder type", (int*)&m_tweak.decoderType);
      m_ui.enumCombobox(GUI_LODTYPE, "lod type", (int*)&m_tweak.lodType);
      ImGui::Checkbox("use primitive culling", &m_tweak.usePrimitiveCulling);
      ImGui::Checkbox("use occlusion culling (simple, needs lod)", &m_tweak.useOcclusionCulling);
      ImGui::PopItemWidth();
    }

    if(ImGui::CollapsingHeader("Performance Timings"))
    {
      int avg = 50;

      if(m_lastFrameTime == 0)
      {
        m_lastFrameTime = time;
        m_frames        = -1;
      }

      if(m_frames > 4)
      {
        double curavg = (time - m_lastFrameTime) / m_frames;
        if(curavg > 1.0 / 30.0)
        {
          avg = 10;
        }
      }

      if(m_profiler.getTotalFrames() % avg == avg - 1)
      {
        double dummy;
        m_statsTskTime = 0;
        m_statsDrwTime = 0;
        m_profiler.getAveragedValues("Render", m_statsCpuTime, m_statsGpuTime);
        m_profiler.getAveragedValues("Task", dummy, m_statsTskTime);
        m_profiler.getAveragedValues("Draw", dummy, m_statsDrwTime);
        m_lastFrameTime = time;
        m_frames        = -1;
      }

      m_frames++;

      float gpuTimeF = float(m_statsGpuTime);
      float tskTimeF = float(m_statsTskTime);
      float drwTimeF = float(m_statsDrwTime);
      ImGui::Text("Total Render GPU [ms]: %2.3f", gpuTimeF / 1000.0f);
      ImGui::Text("   Task Pass GPU [ms]: %2.3f", tskTimeF / 1000.0f);
      ImGui::Text("   Draw Pass GPU [ms]: %2.3f", drwTimeF / 1000.0f);
    }

    ImGui::PushStyleColor(ImGuiCol_Header, advancedColor);

    if(m_advancedUI && ImGui::CollapsingHeader("Render Advanced"))
    {
      ImGui::InputFloat("render disp scale", &m_tweak.renderScale, 0.01f, 0.1f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
      ImGui::InputFloat("render disp bias", &m_tweak.renderBias, 0.01f, 0.1f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
      ImGui::Checkbox("freeze cull/lod view", &m_frameConfig.cullFreeze);
      ImGui::Checkbox("reflection contours", &m_tweak.showReflectionLine);
      ImGui::Checkbox("reflection band", &m_tweak.showReflectionBand);
      ImGuiH::InputIntClamped("grid copies", &m_tweak.gridCopies, 1, 0xFFFF, 4, 16, ImGuiInputTextFlags_EnterReturnsTrue);
      ImGuiH::InputIntClamped("grid axis bits", (int*)&m_tweak.gridAxis, 1, 63, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue);
      ImGuiH::InputFloatClamped("grid spacing", &m_tweak.gridSpacing, -FLT_MAX, 100, 0.1f, 0.1f, "%.2f",
                                ImGuiInputTextFlags_EnterReturnsTrue);
      ImGui::InputFloat("rotate speed", &m_tweak.rotateModelSpeed, 0.25f, 0.25f, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue);
      ImGui::InputFloat("rotate dist bias", &m_tweak.rotateModelDistance.x, 0.1f, 0.25f, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue);
      ImGui::InputFloat("rotate dist scale", &m_tweak.rotateModelDistance.y, 0.1f, 0.25f, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue);
      ImGuiH::InputIntClamped("flat max visible mshlts\n(in number of bits)", (int*)&m_tweak.maxVisibleBits, 16, 24, 1,
                              1, ImGuiInputTextFlags_EnterReturnsTrue);
      ImGui::InputFloat("mouse zoom sense", &m_control.m_senseZoom, 0.0001f, 0.001f, "%.5f", ImGuiInputTextFlags_EnterReturnsTrue);
      ImGui::InputFloat("mouse rotate sense", &m_control.m_senseRotate, 0.0001f, 0.001f, "%.5f", ImGuiInputTextFlags_EnterReturnsTrue);
    }

    if(m_advancedUI && ImGui::CollapsingHeader("HBAO Settings"))
    {
      ImGui::Checkbox("screen-space ambient occlusion", &m_resources.m_hbaoActive);
      ImGui::Checkbox("full resolution", &m_tweak.hbaoFullRes);
      ImGui::InputFloat("radius", &m_tweak.hbaoRadius, 0.01f);
      ImGui::InputFloat("blur sharpness", &m_resources.m_hbaoSettings.blurSharpness, 1.0f);
      ImGui::InputFloat("intensity", &m_resources.m_hbaoSettings.intensity, 0.1f);
      ImGui::InputFloat("bias", &m_resources.m_hbaoSettings.bias, 0.01f);
    }

    ShaderStats stats;
    if(m_advancedUI && ImGui::CollapsingHeader("Generate Stats"))
    {
      m_tweak.useStats = 1;
      m_resources.getStats(stats);
      ImGui::Text("       : b  m  k");
      ImGui::Text("tris   : %10u", stats.triangles);
      ImGui::Text("mshlts : %10u", stats.meshlets);
      ImGui::Text("clocks : %10u", stats.clocksAvg ? uint32_t(uint64_t(stats.clocksSum) / uint64_t(stats.clocksAvg)) : 0);
    }
    else
    {
      m_tweak.useStats = 0;
    }
    if(m_advancedUI && ImGui::CollapsingHeader("Debug Shader Values"))
    {
      ImGui::InputInt("dbgInt", (int*)&m_sceneUbo.dbgUint);
      ImGui::InputFloat("dbgFloat", &m_sceneUbo.dbgFloat, 0.1f, 1.0f, "%.3f");

      if(!m_tweak.useStats)
      {
        m_resources.getStats(stats);
      }
      ImGui::Text(" clocks : %10u", stats.clocksAvg ? uint32_t(uint64_t(stats.clocksSum) / uint64_t(stats.clocksAvg)) : 0);
      ImGui::Text(" debugI :  %10d", stats.debugI);
      ImGui::Text(" debugUI:  %10u", stats.debugUI);
      ImGui::Text(" debugF :   %f", stats.debugF);
      if(m_hadProfilerPrint)
      {
        printf("lanes \n");
      }
      for(uint32_t i = 0; i < 64; i++)
      {
        ImGui::Text("%2d: %8u %8u %8u %8u", i, stats.debugA[i], stats.debugB[i], stats.debugC[i], stats.debugD[i]);
        if(m_hadProfilerPrint)
        {
          printf("%8u %8u %8u %8u\n", stats.debugA[i], stats.debugB[i], stats.debugC[i], stats.debugD[i]);
        }
      }
      if(m_hadProfilerPrint)
      {
        printf("\n");
      }
    }
    ImGui::PopStyleColor();
  }
  ImGui::End();
}

void alignViewToUpVector(glm::mat4& view, const glm::vec3& upVector)
{
  glm::vec3 eyepos = glm::vec3(glm::inverse(view)[3]);
  // align matrix to upVector
  glm::vec4 xorig = glm::row(view, 0);
  glm::vec4 yorig = glm::row(view, 1);
  glm::vec4 zorig = glm::row(view, 2);

  glm::vec3 z = glm::vec3(zorig);
  z -= dot(z, upVector) * upVector;
  z = glm::normalize(z);
  glm::vec3 x = glm::cross(upVector, z);
  glm::vec3 y = glm::cross(z, x);
  x = glm::normalize(x);
  y = glm::normalize(y);

  glm::row(view, 1) = glm::vec4(x, -glm::dot(x, eyepos));
  glm::row(view, 2) = glm::vec4(x, -glm::dot(y, eyepos));
  glm::row(view, 3) = glm::vec4(x, -glm::dot(z, eyepos));
  //view.set_row(0, glm::vec4(x, -glm::dot(x, eyepos)));
  //view.set_row(1, glm::vec4(y, -glm::dot(y, eyepos)));
  //view.set_row(2, glm::vec4(z, -glm::dot(z, eyepos)));
}

void Sample::think(double time)
{
  int width  = m_windowState.m_swapSize[0];
  int height = m_windowState.m_swapSize[1];

  if(m_useUI)
  {
    processUI(width, height, time);
  }

  m_control.processActions({m_windowState.m_winSize[0], m_windowState.m_winSize[1]},
                           glm::vec2(m_windowState.m_mouseCurrent[0], m_windowState.m_mouseCurrent[1]),
                           m_windowState.m_mouseButtonFlags, m_windowState.m_mouseWheel);

  if(m_tweak.rotateModelSpeed)
  {
    float t = float(time * m_tweak.rotateModelSpeed);
    //t = float(m_resources->m_frame) / float(120.0f);
    // mat4  rotator          = glm::rotation_mat4_y(t * 0.5f);
    glm::quat rotator      = glm::angleAxis(t, m_modelUpVector);
    vec3      dir          = m_modelUpVector * (sinf(t * 0.6f) * 0.5f + 0.5f) + (vec3(1, 1, 1) - m_modelUpVector);
    dir                    = rotator * -dir; //glm::rotate_by(-dir, rotator);
    float distance         = m_tweak.rotateModelDistance.x + sinf(t * 0.7f) * m_tweak.rotateModelDistance.y;
    m_control.m_viewMatrix = glm::lookAt(m_control.m_sceneOrbit - (dir * m_control.m_sceneDimension * distance),
                                         m_control.m_sceneOrbit, m_modelUpVector);
  }

  bool shaderChanged = false;
  if(m_windowState.onPress(KEY_R))
  {
    shaderChanged = true;
  }
  else if(m_windowState.onPress(KEY_T))
  {
    m_profilerPrint = !m_profilerPrint;
  }
  else if(m_windowState.onPress(KEY_C))
  {
    saveViewpoint();
  }
  else if(m_windowState.onPress(KEY_A))
  {
    alignViewToUpVector(m_control.m_viewMatrix, m_modelUpVector);
  }

  if(tweakChanged(m_tweak.supersample) || getVsync() != m_lastVsync || tweakChanged(m_tweak.hbaoFullRes))
  {
    m_lastVsync               = getVsync();
    m_resources.m_hbaoFullRes = m_tweak.hbaoFullRes;
    m_resources.initFramebuffer(width, height, m_tweak.supersample, getVsync());
  }

  bool shaderRenderChanged = false;

  if(tweakChanged(m_tweak.useStats) || tweakChanged(m_tweak.lodType) || tweakChanged(m_tweak.normalType)
     || tweakChanged(m_tweak.fp16displacementMath) || tweakChanged(m_tweak.decoderType) || tweakChanged(m_tweak.surfaceVisualization)
     || tweakChanged(m_tweak.usePrimitiveCulling) || tweakChanged(m_tweak.useOcclusionCulling))
  {
    shaderRenderChanged                   = true;
    m_resources.m_shaderManager.m_prepend = getShaderPrepend();
  }

  bool sceneChanged = false;
  if(tweakChanged(m_tweak.gridCopies) || tweakChanged(m_tweak.gridAxis) || tweakChanged(m_tweak.gridSpacing))
  {
    m_resources.synchronize("sync grid changed");
    updateGrid(true);
    sceneChanged = true;
  }

  bool rendererChanged = false;
  if(sceneChanged || shaderChanged || shaderRenderChanged || tweakChanged(m_tweak.renderer)
     || tweakChanged(m_tweak.objectFrom) || tweakChanged(m_tweak.objectNum) || m_lastFbo != m_resources.m_fboChangeID
     || tweakChanged(m_tweak.normalType) || tweakChanged(m_tweak.useLod) || tweakChanged(m_tweak.maxVisibleBits))
  {
    rendererChanged = true;
    m_resources.synchronize("sync renderer changed");
    m_resources.updatedShaders();
    initRenderer(m_tweak.renderer);
  }

  if(m_tweak.viewPoint != m_lastTweak.viewPoint)
  {
    setViewpoint();
  }

  m_resources.beginFrame();

  m_frameConfig.winWidth  = width;
  m_frameConfig.winHeight = height;

  {
    ModelType  modelOpaque = m_frameConfig.opaque;
    SceneData& sceneUbo    = m_sceneUbo;

    if(!m_frameConfig.cullFreeze)
    {
      m_sceneUboLast = sceneUbo;
    }

    sceneUbo.lodScale   = m_tweak.lodAreaScale;
    sceneUbo.disp_bias  = m_tweak.renderBias;
    sceneUbo.disp_scale = m_tweak.renderScale;

    uint32_t renderWidth   = width * m_tweak.supersample;
    uint32_t renderHeight  = height * m_tweak.supersample;
    sceneUbo.reflection    = (m_tweak.showReflectionLine ? 1 : 0) | (m_tweak.showReflectionBand ? 2 : 0);
    sceneUbo.highlightPrim = -1;
    sceneUbo.time          = float(time);
    sceneUbo.frame         = m_resources.m_frame;

    sceneUbo.viewport    = ivec2(renderWidth, renderHeight);
    sceneUbo.viewportf   = vec2(renderWidth, renderHeight);
    sceneUbo.supersample = m_tweak.supersample;
    sceneUbo.nearPlane   = m_control.m_sceneDimension * 0.001f;
    sceneUbo.farPlane    = m_control.m_sceneDimension * 100.0f;
    sceneUbo.wUpDir      = glm::vec4(m_modelUpVector, 0.0f);

    if(m_scene.meshSetLo)
    {
      sceneUbo.wBboxMin = glm::vec4(m_scene.meshSetLo->bbox.mins, 0.0f);
      sceneUbo.wBboxMax = glm::vec4(m_scene.meshSetLo->bbox.maxs, 0.0f);
    }

    glm::mat4 projection = glm::perspectiveRH_ZO(glm::radians(m_tweak.fov), float(width) / float(height), sceneUbo.nearPlane, sceneUbo.farPlane);
    projection[1][1] *= -1;

    glm::mat4 view  = m_control.m_viewMatrix;
    glm::mat4 viewI = glm::inverse(view);

    sceneUbo.viewProjMatrix  = projection * view;
    sceneUbo.viewProjMatrixI = glm::inverse(sceneUbo.viewProjMatrix);
    sceneUbo.viewMatrix      = view;
    sceneUbo.viewMatrixI     = viewI;
    sceneUbo.projMatrix      = projection;
    sceneUbo.projMatrixI     = glm::inverse(projection);

    Frustum::init(sceneUbo.frustumPlanes, sceneUbo.viewProjMatrix);

    glm::vec4 hPos   = projection * glm::vec4(1.0f, 1.0f, -sceneUbo.farPlane, 1.0f);
    glm::vec2 hCoord = glm::vec2(hPos.x / hPos.w, hPos.y / hPos.w);
    glm::vec2 dim    = glm::abs(hCoord);

    // helper to quickly get footprint of a point at a given distance
    //
    // __.__hPos (far plane is width x height)
    // \ | /
    //  \|/
    //   x camera
    //
    // here: viewPixelSize / point.w = size of point in pixels
    // * 0.5f because renderWidth/renderHeight represents [-1,1] but we need half of frustum
    sceneUbo.viewPixelSize = dim * (glm::vec2(float(renderWidth), float(renderHeight)) * 0.5f) * sceneUbo.farPlane;
    // here: viewClipSize / point.w = size of point in clip-space units
    // no extra scale as half clip space is 1.0 in extent
    sceneUbo.viewClipSize = dim * sceneUbo.farPlane;

    sceneUbo.viewPos = sceneUbo.viewMatrixI[3];  // position of eye in the world
    sceneUbo.viewDir = -viewI[2];

    sceneUbo.viewPlane   = sceneUbo.viewDir;
    sceneUbo.viewPlane.w = -glm::dot(glm::vec3(sceneUbo.viewPos), glm::vec3(sceneUbo.viewDir));

    sceneUbo.wLightPos   = sceneUbo.viewMatrixI[3];  // place light at position of eye in the world
    sceneUbo.wLightPos.w = 1.0;

    glm::vec3 viewDir = glm::vec3(-viewI[2]);
    glm::vec3 sideDir = glm::vec3(viewI[0]);
    glm::vec3 upDir   = glm::vec3(viewI[1]);
    //sceneUbo.wLightPos += (sideDir + upDir - viewDir * 0.5f) * m_control.m_sceneDimension * 0.25f;

    {
      // hiz setup
      sceneUbo.hizSizeMax = (float)std::max(m_resources.m_hizUpdate.farInfo.width, m_resources.m_hizUpdate.farInfo.height);
      m_resources.m_hizUpdate.farInfo.getShaderFactors(glm::value_ptr(sceneUbo.hizSizeFactors));
      m_resources.m_hizUpdate.nearInfo.getShaderFactors(glm::value_ptr(sceneUbo.nearSizeFactors));
    }
    {
      // hbao setup
      auto& hbaoView                    = m_resources.m_hbaoSettings.view;
      hbaoView.farPlane                 = sceneUbo.farPlane;
      hbaoView.nearPlane                = sceneUbo.nearPlane;
      hbaoView.isOrtho                  = false;
      hbaoView.projectionMatrix         = projection;
      m_resources.m_hbaoSettings.radius = m_control.m_sceneDimension * m_tweak.hbaoRadius;

      glm::vec4 hi = sceneUbo.projMatrixI * glm::vec4(1, 1, -0.9, 1);
      hi /= hi.w;
      float tanx           = hi.x / fabsf(hi.z);
      float tany           = hi.y / fabsf(hi.z);
      hbaoView.halfFovyTan = tany;
    }
  }

  {
    if(m_renderer)
    {
      m_renderer->draw(m_frameConfig, m_profilerVK);
    }
    else
    {
      m_resources.blankFrame();
    }
  }

  {
    if(m_useUI)
    {
      ImGui::Render();
      m_frameConfig.imguiDrawData = ImGui::GetDrawData();
    }
    else
    {
      m_frameConfig.imguiDrawData = nullptr;
    }

    m_resources.blitFrame(m_frameConfig, m_profilerVK);
  }

  m_resources.endFrame();
  m_resources.m_frame++;

  if(m_useUI)
  {
    ImGui::EndFrame();
  }

  m_lastTweak = m_tweak;
  m_lastFbo   = m_resources.m_fboChangeID;
}

void Sample::resize(int width, int height)
{
  initFramebuffers(width, height);
}

void Sample::postBenchmarkAdvance()
{
  if(!setRendererFromName(m_rendererName))
  {
    exit(-1);
  }
}

void Sample::saveViewpoint()
{
  int idx = int(m_viewPoints.size());

  ViewPoint vp;
  vp.mat        = m_control.m_viewMatrix;
  vp.sceneScale = 1.0;
  vp.fov        = m_tweak.fov;
  vp.name       = nvh::stringFormat("ViewPoint%d", idx);
  m_ui.enumAdd(GUI_VIEWPOINT, idx, vp.name.c_str());

  m_viewPoints.push_back(vp);
  m_tweak.viewPoint = idx;

  if(m_viewpointFilename.empty())
    return;

  std::ofstream f;
  f.open(m_viewpointFilename, std::ios_base::app | std::ios_base::out);
  if(f)
  {
    f << vp.name << " ";
    for(auto i = 0; i < 16; ++i)
    {
      f << glm::value_ptr(vp.mat)[i] << " ";
    }
    f << vp.sceneScale << " ";
    f << vp.fov;
    f << "\n";
    f.close();
  }

  LOGI("viewpoint file updated: %s\n", m_viewpointFilename.c_str());
}

void Sample::setupConfigParameters()
{
  // generic / app
  m_parameterList.add("verbose|may trigger additional prints", &g_verbose);
  m_parameterList.add("device|vulkan device index", &m_contextInfo.compatibleDeviceIndex);
  m_parameterList.add("noui|skip ui", &m_useUI, false);
  m_parameterList.add("threads|number of cpu threads for various processing tasks", &g_numThreads);
  m_parameterList.add("advancedui|state of showing advanced ui 0/1", &m_advancedUI);

  // input files
  m_parameterList.addFilename("lo|lo poly source mesh for baking/rendering displacements", &m_modelFilenameLo);
  m_parameterList.addFilename(".gltf", &m_modelFilenameLo);

  // scene / view
  m_parameterList.addFilename("viewpoints|viewpoint file", &m_viewpointFilename);
  m_parameterList.add("viewpoint|starting viewpoint index", &m_tweak.viewPoint);
  m_parameterList.add("fov|field of view", &m_tweak.fov);
  m_parameterList.add("upvector|scene up vector (shading/camera orientation)", &m_modelUpVector.x, nullptr, 3);
  m_parameterList.add("mouseorbit|uses orbit mode for mouse control (default true), otherwise fly mode", &m_control.m_useOrbit);
  m_parameterList.add("mousesensepan|mouse sensitivity for pan", &m_control.m_sensePan);
  m_parameterList.add("mousesenserotate|mouse sensitivity for rotation", &m_control.m_senseRotate);
  m_parameterList.add("mousesensezoom|mouse sensitivity for zoom", &m_control.m_senseZoom);
  m_parameterList.add("mousesensewzoom|mouse sensitivity for wheel zoom", &m_control.m_senseWheelZoom);
  m_parameterList.add("gridcopies|create this many instances of the scene along a grid", &m_tweak.gridCopies);
  m_parameterList.add("gridaxis|which axis (1 bit per x,z,y) are enabled to distribute instances", &m_tweak.gridAxis);
  m_parameterList.add("gridspacing|relative scale of bbox used to shift instances on grid", &m_tweak.gridSpacing);


  // various rendering
  m_parameterList.add("renderer|renderer index as in ui list/print", &m_tweak.renderer);
  m_parameterList.add("renderernamed|renderer string as in ui list/print", &m_rendererName);
  m_parameterList.add("lodenable|enable dynamic lod in renderers that support it", &m_tweak.useLod);
  m_parameterList.add("lodareascale|scale projected area prior lod level decision", &m_tweak.lodAreaScale);
  m_parameterList.add("lodtype|0: pre-computed sphere, 1: dynamic triangle, 2: instance sphere", (int*)&m_tweak.lodType);
  m_parameterList.add("normaltype|0: facet, 1: base vertex, 2: texture, 3: micro vertex", (int*)&m_tweak.normalType);
  m_parameterList.add("renderscale|additional displacement scale", &m_tweak.renderScale);
  m_parameterList.add("renderbias|additional displacement bias", &m_tweak.renderBias);
  m_parameterList.add("surfacevis|selects a surface debug visualization mode", &m_tweak.surfaceVisualization);
  m_parameterList.add("primitiveculling|enable per-triangle culling", &m_tweak.usePrimitiveCulling);
  m_parameterList.add("occlusionculling|enable basic occlusion culling", &m_tweak.useOcclusionCulling);
  m_parameterList.add("rotatemodelspeed|automatic rotation speed", &m_tweak.rotateModelSpeed);
  m_parameterList.add("rotatemodeldistance|automatic rotation distance (bias,scale)", &m_tweak.rotateModelDistance.x, nullptr, 2);
  m_parameterList.add("decodertype|switch decoder type", (int*)&m_tweak.decoderType);
  m_parameterList.add("hbao|use hbao", &m_resources.m_hbaoActive);
  m_parameterList.add("maxvisiblebits|how many max visible items (1<<bits)", &m_tweak.maxVisibleBits);
  m_parameterList.add("viewa|which mesh to view in slot a: 0: lo, 1: displaced", (int*)&m_frameConfig.opaque);

  m_parameterList.add("micromeshrtextensions| 0/1 to enable/disable rt extensions", &g_enableMicromeshRTExtensions);
}

bool Sample::validateConfig()
{
  if(g_enableMicromeshRTExtensions)
  {
#if defined(_DEBUG)
    // with the real extension enabled some errors may still be triggered in validation
    // mostly spir-v related
    // you may want to disable the validation here
    m_contextInfo.removeInstanceLayer("VK_LAYER_KHRONOS_validation");
#endif
  }
  else
  {
    // ensure extension is removed
    m_contextInfo.removeDeviceExtension(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME);
    m_contextInfo.removeDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME);
  }


  if(m_modelFilenameLo.empty())
  {
    std::vector<std::string> directories;
    directories.push_back(NVPSystem::exePath() + "/umesh_Murex_Romosus");
    directories.push_back(NVPSystem::exePath() + "/media/umesh_Murex_Romosus");
    directories.push_back(NVPSystem::exePath() + std::string(PROJECT_DOWNLOAD_RELDIRECTORY) + "/umesh_Murex_Romosus");
    m_modelFilenameLo = nvh::findFile(std::string("umesh_Murex_Romosus_compressed.gltf"), directories);
  }

  return true;
}

bool Sample::setRendererFromName(const std::string& name)
{
  if(!name.empty() && name != "_")
  {
    const RendererVK::Registry registry = RendererVK::getRegistry();
    for(size_t i = 0; i < m_renderersSorted.size(); i++)
    {
      if(strcmp(name.c_str(), registry[m_renderersSorted[i]]->name()) == 0)
      {
        m_tweak.renderer = int(i);
        return true;
      }
    }
    LOGE("Renderer %s not found\n", name.c_str());
    return false;
  }

  return true;
}

}  // namespace microdisp

using namespace microdisp;


#include <omp.h>
#include <thread>

int main(int argc, const char** argv)
{
  NVPSystem sys(PROJECT_NAME);

  g_numThreads = std::thread::hardware_concurrency();

  Sample sample;
  sample.setVsync(true);

  return sample.run(PROJECT_NAME, argc, argv, SAMPLE_SIZE_WIDTH, SAMPLE_SIZE_HEIGHT);
}
