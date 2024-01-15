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

#include <cgltf.h>
#include <baryutils/baryutils.h>
#include "meshset_gltf.hpp"
#include "config.h"
#include "meshset.hpp"
#include "parallel_work.hpp"
#include <iostream>
#include <memory>
#include <nvh/nvprint.hpp>
#include <nvh/filemapping.hpp>
#include <unordered_map>
#include "dlib_url.h"
#include "glm/gtc/type_ptr.hpp"

//////////////////////////////////////////////////////////////////////////

struct FileMappingList
{
  struct Entry
  {
    nvh::FileReadMapping mapping;
    int64_t              refCount = 1;
  };
  std::unordered_map<std::string, Entry>       m_nameToMapping;
  std::unordered_map<const void*, std::string> m_dataToName;
#ifdef _DEBUG
  int64_t m_openBias = 0;
#endif

  bool open(const char* path, size_t* size, void** data)
  {
#ifdef _DEBUG
    m_openBias++;
#endif

    std::string pathStr(path);

    auto it = m_nameToMapping.find(pathStr);
    if(it != m_nameToMapping.end())
    {
      *data = const_cast<void*>(it->second.mapping.data());
      *size = it->second.mapping.size();
      it->second.refCount++;
      return true;
    }

    Entry entry;
    if(entry.mapping.open(path))
    {
      const void* mappingData = entry.mapping.data();
      *data                   = const_cast<void*>(mappingData);
      *size                   = entry.mapping.size();
      m_dataToName.insert({mappingData, pathStr});
      m_nameToMapping.insert({pathStr, std::move(entry)});
      return true;
    }

    return false;
  }

  void close(void* data)
  {
#ifdef _DEBUG
    m_openBias--;
#endif
    auto itName = m_dataToName.find(data);
    if(itName != m_dataToName.end())
    {
      auto itMapping = m_nameToMapping.find(itName->second);
      if(itMapping != m_nameToMapping.end())
      {
        itMapping->second.refCount--;

        if(!itMapping->second.refCount)
        {
          m_nameToMapping.erase(itMapping);
          m_dataToName.erase(itName);
        }
      }
    }
  }

  ~FileMappingList()
  {
#ifdef _DEBUG
    assert(m_openBias == 0 && "open/close bias wrong");
#endif
    assert(m_nameToMapping.empty() && m_dataToName.empty() && "not all opened files were closed");
  }
};

static cgltf_result cgltf_read(const struct cgltf_memory_options* memory_options,
                               const struct cgltf_file_options*   file_options,
                               const char*                        path,
                               cgltf_size*                        size,
                               void**                             data)
{
  FileMappingList* mappings = (FileMappingList*)file_options->user_data;
  if(mappings->open(path, size, data))
  {
    return cgltf_result_success;
  }

  return cgltf_result_io_error;
}

static void cgltf_release(const struct cgltf_memory_options* memory_options, const struct cgltf_file_options* file_options, void* data)
{
  FileMappingList* mappings = (FileMappingList*)file_options->user_data;
  mappings->close(data);
}

static bary::Result baryutils_read(const baryutils::BaryMemoryApi* memory_options,
                                   const baryutils::BaryFileApi*   file_options,
                                   const char*                     path,
                                   size_t*                         size,
                                   void**                          data)
{
  FileMappingList* mappings = (FileMappingList*)file_options->userData;
  if(mappings->open(path, size, data))
  {
    return bary::Result::eSuccess;
  }

  return bary::Result::eErrorIO;
}

static void baryutils_release(const baryutils::BaryMemoryApi* memory_options, const baryutils::BaryFileApi* file_options, void* data)
{
  FileMappingList* mappings = (FileMappingList*)file_options->userData;
  mappings->close(data);
}

//////////////////////////////////////////////////////////////////////////

// Given the path to the glTF file and a relative filesystem URI, concatenates
// the two. The resulting string must be freed with delete[].
static char* getGLTFURIPath(const char* gltfPath, const char* uri)
{
  if(gltfPath == nullptr || uri == nullptr)
  {
    return nullptr;
  }

  // This would be shorter using std::filesystem, but doing this manually reduces dependence on C++.

  // Find the position of the last slash in gltfPath
  const char* fwdSlashPos = strrchr(gltfPath, '/');
  const char* bwdSlashPos = strrchr(gltfPath, '\\');
  const char* slashPos    = (fwdSlashPos > bwdSlashPos ? fwdSlashPos : bwdSlashPos);
  if(slashPos == nullptr)
  {
    // glTF path contained no slashes, so can't find the parent directory.
    return nullptr;
  }
  // Get its index from the start of the string
  size_t slashIndex = slashPos - gltfPath;

  // Concatenate the two, including the slash
  size_t uriLen = strlen(uri);
  char*  result = new char[slashIndex + 1 + uriLen + 1];  // Include slash and null
  memcpy(result, gltfPath, slashIndex + 1);
  memcpy(result + slashIndex + 1, uri, uriLen);
  result[slashIndex + 1 + uriLen] = '\0';
  return result;
}

static std::string getGLTFURIPathStr(const char* gltfPath, const char* uri)
{
  char* result = getGLTFURIPath(gltfPath, uri);
  if(result == nullptr)
    return "";
  std::string resultStr(result);
  delete[] result;
  return resultStr;
}

enum BaryFileType
{
  BARY_FILE_DISPLACEMENT,
  BARY_FILE_ATTRIBUTE,
};

// use undefined for displacement
uint32_t loadGLTFBary(FileMappingList&         mappingList,
                      const cgltf_nv_micromap* baryMicromap,
                      const char*              gltfPath,
                      BaryAttributesSet&       barySet,
                      baryutils::BaryFile&     bfile,
                      BaryFileType             btype)
{
  // Note that at the moment we use the raw bary APIs here, as
  // the output objects can't be constructed from streams yet.
  if(baryMicromap == nullptr)
  {
    return MeshSetID::INVALID;
  }
  const char* uri = baryMicromap->uri;
  if(uri == nullptr)
  {
    LOGE("loadGLTFDisplacement: The displacement map array couldn't be loaded, because the URI was nullptr.\n");
    return MeshSetID::INVALID;
  }
  if(strncmp(uri, "data:", 5) == 0)
  {
    LOGE(
        "loadGLTFDisplacement: The displacement map array couldn't be loaded, because it was a data URI, which isn't "
        "supported yet.\n");
    return MeshSetID::INVALID;  // TODO: Support data URIs
  }
  if(strstr(uri, "://") != nullptr)  // Must not be a network resource
  {
    LOGE(
        "loadGLTFDisplacement: The displacement map array couldn't be loaded, because it was a network resource, which "
        "isn't supported yet.\n");
    return MeshSetID::INVALID;
  }
  if(gltfPath == nullptr)  // Path to the glTF file must be specified
  {
    LOGE(
        "loadGLTFDisplacement: The displacement map array couldn't be loaded, because the gltfPath argument was a "
        "nullptr. This is an error in the code itself.\n");
    return MeshSetID::INVALID;
  }

  std::string uriDec = dlib::urldecode(uri);

  const std::string baryFilePathStr  = getGLTFURIPathStr(gltfPath, uriDec.c_str());
  const char*       baryFilePathCStr = baryFilePathStr.c_str();

  baryutils::BaryFileOpenOptions openOptions = {0};
  openOptions.fileApi.userData               = &mappingList;
  openOptions.fileApi.read                   = baryutils_read;
  openOptions.fileApi.release                = baryutils_release;

  if(btype == BARY_FILE_DISPLACEMENT)
    return barySet.loadDisplacement(baryFilePathCStr, bfile, &openOptions);
  else
    return barySet.loadAttribute(baryFilePathCStr, bfile, &openOptions);
}

//////////////////////////////////////////////////////////////////////////

using cgltfPrimToIdMap       = std::unordered_map<cgltf_primitive*, size_t>;
using cgltMicromapToPrimsMap = std::unordered_map<cgltf_nv_micromap*, std::vector<cgltf_primitive*>>;

// legacy
static_assert(sizeof(cgltf_image) == sizeof(cgltf_nv_micromap), "cgltf_image cgltf_nv_micromap mismatch");
static_assert(offsetof(cgltf_image, name) == offsetof(cgltf_image, name), "cgltf_image cgltf_nv_micromap mismatch");
static_assert(offsetof(cgltf_image, uri) == offsetof(cgltf_nv_micromap, uri), "cgltf_image cgltf_nv_micromap mismatch");
static_assert(offsetof(cgltf_image, mime_type) == offsetof(cgltf_nv_micromap, mime_type), "cgltf_image cgltf_nv_micromap mismatch");
static_assert(offsetof(cgltf_image, buffer_view) == offsetof(cgltf_nv_micromap, buffer_view), "cgltf_image cgltf_nv_micromap mismatch");

static const uint8_t* cgltfBufferView_getData(const cgltf_buffer_view* view)
{
  if(view->data)
    return (const uint8_t*)view->data;

  if(!view->buffer->data)
    return NULL;

  const uint8_t* result = (const uint8_t*)view->buffer->data;
  result += view->offset;
  return result;
}

// Creates an glm::vec3 from the first three elements of a float array.
glm::vec3 floatArrayToVec3f(const float* arr)
{
  return glm::vec3(arr[0], arr[1], arr[2]);
}

// Traverses the glTF node and any of its children, adding a MeshInstance to
// the meshSet for each referenced glTF primitive.
void addMeshInstancesFromNode(MeshSet&            meshSet,
                              const cgltf_node*   node,
                              cgltfPrimToIdMap&   gltfPrimToMeshMap,
                              const glm::mat4 parentObjToWorldTransform = glm::mat4(1))
{
  if(node == nullptr)
    return;

  // Compute this node's object-to-world transform.
  // See https://github.com/KhronosGroup/glTF-Tutorials/blob/master/gltfTutorial/gltfTutorial_004_ScenesNodes.md .
  // Note that this depends on glm::mat4 being column-major.
  // The documentation above also means that vectors are multiplied on the right.
  glm::mat4 localNodeTransform(1);
  cgltf_node_transform_local(node, glm::value_ptr(localNodeTransform));
  const glm::mat4 nodeObjToWorldTransform = parentObjToWorldTransform * localNodeTransform;

  // If this node has a mesh, add instances for its primitives.
  if(node->mesh != nullptr)
  {
    const cgltf_mesh& mesh          = *(node->mesh);
    const size_t      numPrimitives = mesh.primitives_count;
    for(size_t primIdx = 0; primIdx < numPrimitives; primIdx++)
    {
      cgltf_primitive* prim = &(mesh.primitives[primIdx]);
      const auto       it   = gltfPrimToMeshMap.find(prim);
      if(it != gltfPrimToMeshMap.end())
      {
        MeshInstance instance;
        instance.meshID = static_cast<uint32_t>(it->second);
        instance.bbox   = meshSet.meshInfos[instance.meshID].bbox.transformed(nodeObjToWorldTransform);
        instance.xform  = nodeObjToWorldTransform;
        meshSet.meshInstances.push_back(instance);
        meshSet.bbox.merge(instance.bbox);
      }
    }
  }

  // Recurse over any children of this node.
  const size_t numChildren = node->children_count;
  for(size_t childIdx = 0; childIdx < numChildren; childIdx++)
  {
    addMeshInstancesFromNode(meshSet, node->children[childIdx], gltfPrimToMeshMap, nodeObjToWorldTransform);
  }
}

std::string getImageFilename(const char* gltfPath, const cgltf_image* image)
{
  // Note that at the moment we use the raw bary APIs here, as
  // the output objects can't be constructed from streams yet.
  if(image == nullptr)
  {
    return {};
  }
  const char* uri = image->uri;
  if(uri == nullptr)
  {
    LOGE("The normal map couldn't be loaded, because the URI was nullptr.\n");
    return {};
  }
  if(strncmp(uri, "data:", 5) == 0)
  {
    LOGE(
        "The normal map couldn't be loaded, because it was a data URI, which isn't "
        "supported yet.\n");
    return {};
  }
  if(strstr(uri, "://") != nullptr)  // Must not be a network resource
  {
    LOGE(
        "The normal map couldn't be loaded, because it was a network resource, which "
        "isn't supported yet.\n");
    return {};
  }

  std::string uriDec = dlib::urldecode(uri);

  return getGLTFURIPathStr(gltfPath, uriDec.c_str());
}


// Defines a unique_ptr that can be used for cgltf_data objects.
// Freeing a unique_cgltf_ptr calls cgltf_free, instead of delete.
// This can be constructed using unique_cgltf_ptr foo(..., &cgltf_free).
using unique_cgltf_ptr = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;

bool loadGLTF(MeshSet& meshSet, BaryAttributesSet* barySet, const char* filename)
{
  meshSet = MeshSet();

  if(barySet)
  {
    *barySet = BaryAttributesSet();
  }

  // Parse the glTF file using cgltf
  cgltf_options options = {};

  FileMappingList mappings;
  options.file.read      = cgltf_read;
  options.file.release   = cgltf_release;
  options.file.user_data = &mappings;

  cgltf_result     cgltfResult;
  unique_cgltf_ptr data = unique_cgltf_ptr(nullptr, &cgltf_free);
  {
    // We have this local pointer followed by an ownership transfer here
    // because cgltf_parse_file takes a pointer to a pointer to cgltf_data.
    cgltf_data* rawData = nullptr;
    cgltfResult         = cgltf_parse_file(&options, filename, &rawData);
    data                = unique_cgltf_ptr(rawData, &cgltf_free);
  }
  // Check for errors; special message for legacy files
  if(cgltfResult == cgltf_result_legacy_gltf)
  {
    LOGE(
        "loadGLTF: This glTF file is an unsupported legacy file - probably glTF 1.0, while cgltf only supports glTF "
        "2.0 files. Please load a glTF 2.0 file instead.\n");
    return false;
  }
  else if((cgltfResult != cgltf_result_success) || (data == nullptr))
  {
    LOGE("loadGLTF: cgltf_parse_file failed. Is this a valid glTF file? (cgltf result: %d)\n", cgltfResult);
    return false;
  }

  // Perform additional validation.
  cgltfResult = cgltf_validate(data.get());
  if(cgltfResult != cgltf_result_success)
  {
    LOGE(
        "loadGLTF: The glTF file could be parsed, but cgltf_validate failed. Consider using the glTF Validator at "
        "https://github.khronos.org/glTF-Validator/ to see if the non-displacement parts of the glTF file are correct. "
        "(cgltf result: %d)\n",
        cgltfResult);
    return false;
  }


  cgltMicromapToPrimsMap baryDisplacementMicromapsToPrimitivesMap;
  cgltMicromapToPrimsMap baryNormalMicromapsToPrimitivesMap;
  baryDisplacementMicromapsToPrimitivesMap.reserve(std::max(data->images_count, data->nv_micromaps_count));
  baryNormalMicromapsToPrimitivesMap.reserve(std::max(data->images_count, data->nv_micromaps_count));

  // For now, also tell cgltf to go ahead and load all buffers.
  cgltfResult = cgltf_load_buffers(&options, data.get(), filename);
  if(cgltfResult != cgltf_result_success)
  {
    LOGE(
        "loadGLTF: The glTF file was valid, but cgltf_load_buffers failed. Are the glTF file's referenced file paths "
        "valid? (cgltf result: %d)\n",
        cgltfResult);
    return false;
  }

  // For now, let's not support instancing (which would be the
  // EXT_mesh_gpu_instancing glTF extension), so each glTF primitive
  // (section of a mesh with a material) will have its own MeshInstance and
  // Mesh.
  //
  // First, load all the materials.
  meshSet.materials.resize(data->materials_count);
  // We'll need an unordered map to reassociate material pointers with indices.
  std::unordered_map<cgltf_material*, uint32_t> matPtrToIdx;
  std::unordered_map<cgltf_texture*, uint32_t>  texPtrToIdx;

  meshSet.textures.push_back({"defaultNormalMap"});

  for(size_t n = 0; n < meshSet.materials.size(); n++)
  {
    MeshMaterial&        material     = meshSet.materials[n];
    const cgltf_material gltfMaterial = data->materials[n];

    if(gltfMaterial.name != nullptr)
    {
      material.name = std::string(gltfMaterial.name);
    }

    material.diffuse = glm::vec3(1.0f, 1.0f, 1.0f);
    if(gltfMaterial.has_pbr_metallic_roughness)
    {
      material.diffuse = floatArrayToVec3f(gltfMaterial.pbr_metallic_roughness.base_color_factor);
    }
    else if(gltfMaterial.has_pbr_specular_glossiness)
    {
      material.diffuse = floatArrayToVec3f(gltfMaterial.pbr_specular_glossiness.diffuse_factor);
    }

    if(gltfMaterial.normal_texture.texture)
    {
      auto it = texPtrToIdx.find(gltfMaterial.normal_texture.texture);
      if(it != texPtrToIdx.end())
      {
        material.normalMapTextureID = it->second;
      }
      else
      {
        std::string normalMapFilename = getImageFilename(filename, gltfMaterial.normal_texture.texture->image);
        if(!normalMapFilename.empty())
        {
          material.normalMapTextureID                      = static_cast<uint32_t>(meshSet.textures.size());
          texPtrToIdx[gltfMaterial.normal_texture.texture] = material.normalMapTextureID;
          meshSet.textures.push_back({normalMapFilename});
        }
        else
        {
          material.normalMapTextureID = 0;
        }
      }
    }
    else
    {
      material.normalMapTextureID = 0;
    }

    matPtrToIdx[&data->materials[n]] = static_cast<uint32_t>(n);
  }

  // Iterate over meshes, and over each of their primitives.
  // For each primitive, we'll append its data into the MeshSet's
  // VertexAttributes and indices. This will wind up giving some redundancy,
  // but is mostly necessary (without a much more advanced approach) since glTF
  // doesn't distinguish in its low-level types what buffers are intended to be
  // used for.

  // This maps the location of each glTF primitive to the Mesh object it represents.
  cgltfPrimToIdMap gltfPrimToMeshMap;

  MeshAttributes& vtxAttribs = meshSet.attributes;
  for(size_t meshIdx = 0; meshIdx < data->meshes_count; meshIdx++)
  {
    const cgltf_mesh gltfMesh = data->meshes[meshIdx];

    for(size_t primIdx = 0; primIdx < gltfMesh.primitives_count; primIdx++)
    {
      cgltf_primitive* gltfPrim          = &gltfMesh.primitives[primIdx];
      size_t           beforeNumVertices = vtxAttribs.positions.size();
      MeshInfo         mesh{};

      if(gltfPrim->type != cgltf_primitive_type_triangles)
      {
        LOGW("loadGLTF: Mesh %zu, primitive %zu is not 'triangles'! Loading this primitive will be skipped.\n", meshIdx, primIdx);
        continue;
      }

      // If the mesh has no attributes, there's nothing we can do
      if(gltfPrim->attributes_count == 0)
      {
        LOGW("loadGLTF: Mesh %zu, primitive %zu had no attributes! Loading this primitive will be skipped.\n", meshIdx, primIdx);
        continue;
      }
      // Check to ensure that all attribute accessors have the same count.
      size_t primNumVertices = -1;  // Sentinel value
      for(size_t attribIdx = 0; attribIdx < gltfPrim->attributes_count; attribIdx++)
      {
        const size_t thisCount = gltfPrim->attributes[attribIdx].data->count;
        if(primNumVertices == -1)
        {
          primNumVertices = thisCount;
        }
        if(primNumVertices != thisCount)
        {
          LOGE(
              "loadGLTF: The attributes for mesh %zu, primitive %zu had different numbers of elements! This glTF file "
              "is invalid.\n",
              meshIdx, primIdx);
          return {};
        }
      }

      if(primNumVertices == -1)
      {
        assert(!"Something went wrong internally - all attributes had count -1!");
      }

      // The new size of the read attribute buffers after appending this primitive's vertices.
      const size_t newNumVertices = beforeNumVertices + primNumVertices;

      bool hasPositions = false;
      bool hasTangents  = false;
      bool hasNormals   = false;

      const cgltf_accessor* normalAccessor = nullptr;

      // Iterate over attributes again and write their data, resizing vectors as needed.
      for(size_t attribIdx = 0; attribIdx < gltfPrim->attributes_count; attribIdx++)
      {
        const cgltf_attribute gltfAttrib = gltfPrim->attributes[attribIdx];
        const cgltf_accessor* accessor   = gltfAttrib.data;

        // TODO: Can we assume alignment in order to make these a single read_float call?
        if(strcmp(gltfAttrib.name, "POSITION") == 0)
        {
          hasPositions = true;
          vtxAttribs.positions.resize(newNumVertices);
          for(size_t i = 0; i < primNumVertices; i++)
          {
            cgltf_accessor_read_float(accessor, i, &vtxAttribs.positions[beforeNumVertices + i].x, 3);
            // While we're here, also calculate the object-space bounding box.
            mesh.bbox.merge(vtxAttribs.positions[beforeNumVertices + i]);
          }
        }
        else if(strcmp(gltfAttrib.name, "NORMAL") == 0)
        {
          hasNormals = true;
          vtxAttribs.normals.resize(newNumVertices);
          for(size_t i = 0; i < primNumVertices; i++)
          {
            cgltf_accessor_read_float(accessor, i, &vtxAttribs.normals[beforeNumVertices + i].x, 3);
          }
          normalAccessor = accessor;
        }
        else if(strcmp(gltfAttrib.name, "TANGENT") == 0)
        {
          hasTangents = true;

          // Because glTF tangents are vec4s, we must handle this as a special case.
          vtxAttribs.tangents.resize(newNumVertices);
          vtxAttribs.bitangents.resize(newNumVertices);
          for(size_t i = 0; i < primNumVertices; i++)
          {
            glm::vec4 gltfTangent;
            cgltf_accessor_read_float(accessor, i, &gltfTangent.x, 4);
            vtxAttribs.tangents[beforeNumVertices + i] = glm::vec3(gltfTangent.x, gltfTangent.y, gltfTangent.z);
            // store handedness for now, then later compute bitangents
            vtxAttribs.bitangents[beforeNumVertices + i] = glm::vec3(gltfTangent.w < 0.0f ? -1.0f : 1.0f);
          }
        }
        else if(strcmp(gltfAttrib.name, "TEXCOORD_0") == 0 || strcmp(gltfAttrib.name, "TEXCOORD") == 0)
        {
          vtxAttribs.texcoords0.resize(newNumVertices);
          for(size_t i = 0; i < primNumVertices; i++)
          {
            cgltf_accessor_read_float(accessor, i, &vtxAttribs.texcoords0[beforeNumVertices + i].x, 2);
          }
        }
      }

      if(!hasPositions)
      {
        LOGE("%s: Mesh %zu, primitive %zu had no positions attribute! This glTF file cannot be loaded.\n", __FUNCTION__,
             meshIdx, primIdx);
        return {};
      }

      if(hasTangents && hasNormals)
      {
        for(size_t i = 0; i < primNumVertices; i++)
        {
          // bitangents store handedness at this point
          vtxAttribs.bitangents[beforeNumVertices + i] =
              glm::normalize(glm::cross(vtxAttribs.normals[beforeNumVertices + i], vtxAttribs.tangents[beforeNumVertices + i]))
              * vtxAttribs.bitangents[beforeNumVertices + i].x;
        }
      }

      // Now read indices:
      const size_t beforeNumIndices = meshSet.indices.size();
      const size_t primNumIndices   = gltfPrim->indices->count;
      const size_t newNumIndices    = beforeNumIndices + primNumIndices;
      meshSet.indices.resize(newNumIndices);
      // TODO: Can we assume close packing here?
      for(size_t i = 0; i < primNumIndices; i++)
      {
        uint32_t localIndex = 0;
        cgltf_accessor_read_uint(gltfPrim->indices, i, &localIndex, 1);
        meshSet.indices[beforeNumIndices + i] = static_cast<uint32_t>(localIndex);
      }

      // Now set up the Mesh object:
      mesh.materialID     = matPtrToIdx[gltfPrim->material];
      mesh.firstVertex    = static_cast<uint32_t>(beforeNumVertices);
      mesh.numVertices    = static_cast<uint32_t>(primNumVertices);
      mesh.firstIndex     = static_cast<uint32_t>(beforeNumIndices);
      mesh.numIndices     = static_cast<uint32_t>(primNumIndices);
      mesh.firstPrimitive = mesh.firstIndex / 3;
      mesh.numPrimitives  = mesh.numIndices / 3;
      // Bounding box was set above
      gltfPrimToMeshMap[gltfPrim] = meshSet.meshInfos.size();


      // for lopoly always fill in vtxAttribs.directions and bounds regardless of the complexity of their usage
      if(barySet)
      {
        vtxAttribs.directions.resize(newNumVertices);
        vtxAttribs.directionBounds.resize(newNumVertices);
      }
      if(barySet && gltfPrim->has_nv_displacement_micromap)
      {
        const cgltf_nv_displacement_micromap& gltfDispPrim    = gltfPrim->nv_displacement_micromap;
        const cgltf_nv_micromap_tooling&      gltfToolingPrim = gltfPrim->nv_micromap_tooling;

        if(gltfDispPrim.map_indices != nullptr)
        {
          LOGE("%s: The displacement map_indices for mesh %zu, primitive %zu are unsupported by this app.\n",
               __FUNCTION__, meshIdx, primIdx);
          return {};
        }
        if(gltfDispPrim.map_offset != 0)
        {
          LOGE("%s: The displacement map_offset for mesh %zu, primitive %zu is non-zero and therefore unsupported by this app.\n",
               __FUNCTION__, meshIdx, primIdx);
          return {};
        }

        cgltf_nv_micromap* micromap = gltfDispPrim.micromap ? gltfDispPrim.micromap : ((cgltf_nv_micromap*)gltfDispPrim.image);
        if(micromap)
        {
          // register this primitive for the micromap
          auto it = baryDisplacementMicromapsToPrimitivesMap.find((cgltf_nv_micromap*)micromap);
          if(it == baryDisplacementMicromapsToPrimitivesMap.end())
          {
            baryDisplacementMicromapsToPrimitivesMap.insert({micromap, {gltfPrim}});
          }
          else
          {
            it->second.push_back(gltfPrim);
          }
        }

        // handle immediate accessors here

        // Add directions as an attribute:
        cgltf_accessor* directions = gltfDispPrim.directions;

        if(directions != nullptr && directions != normalAccessor)
        {
          // directions may be stored as vec3 (fp32) or vec4 (fp32 or fp16)
          // but we always just care about xyz
          cgltf_size components = cgltf_num_components(directions->type);

          for(size_t i = 0; i < primNumVertices; i++)
          {
            glm::vec4 temp;
            cgltf_accessor_read_float(directions, i, &temp.x, components);
            vtxAttribs.directions[beforeNumVertices + i] = {temp.x, temp.y, temp.z};
          }
          mesh.directionsUseMeshNormals    = false;
          meshSet.directionsUseMeshNormals = false;
        }
        else
        {
          // Copy from normals
          memcpy(&vtxAttribs.directions[beforeNumVertices], &vtxAttribs.normals[beforeNumVertices], 3 * sizeof(float) * primNumVertices);
          mesh.directionsUseMeshNormals = true;
        }

        cgltf_accessor* primitive_flags = gltfDispPrim.primitive_flags;

        if(primitive_flags != nullptr)
        {
          meshSet.decimateEdgeFlags.resize(newNumIndices / 3, 0);

          if(primitive_flags->count != primNumIndices / 3)
          {
            LOGE(
                "loadGLTF: The displacement topologyFlags for mesh %zu, primitive %zu had different numbers of "
                "elements! This glTF file is invalid.\n",
                meshIdx, primIdx);
            return {};
          }

          if(primitive_flags->component_type != cgltf_component_type_r_8u)
          {
            LOGE(
                "loadGLTF: The displacement topologyFlags for mesh %zu, primitive %zu are not u8. "
                "This glTF file is "
                "invalid.\n",
                meshIdx, primIdx);
            return {};
          }

          uint8_t*       decimateEdgeFlags = &meshSet.decimateEdgeFlags[mesh.firstPrimitive];
          const uint8_t* gltfEdgeFlags = cgltfBufferView_getData(primitive_flags->buffer_view) + primitive_flags->offset;

          for(size_t i = 0; i < primitive_flags->count; i++)
          {
            decimateEdgeFlags[i] = *(gltfEdgeFlags + (i * primitive_flags->stride));
          }
        }

        cgltf_accessor* direction_bounds = gltfDispPrim.direction_bounds;

        if(direction_bounds != nullptr)
        {
          meshSet.directionBoundsAreUniform = false;
          mesh.directionBoundsAreUniform    = false;
          for(size_t i = 0; i < primNumVertices; i++)
          {
            cgltf_accessor_read_float(direction_bounds, i, &vtxAttribs.directionBounds[beforeNumVertices + i].x, 2);
          }
        }
      }
      else if(barySet)
      {
        // Copy default properties to prevent renderers from breaking
        if(hasNormals)
        {
          memcpy(&vtxAttribs.directions[beforeNumVertices], &vtxAttribs.normals[beforeNumVertices], 3 * sizeof(float) * primNumVertices);
          mesh.directionsUseMeshNormals = true;
        }
        else
        {
          // Wow, we didn't have anything! We might still be able to continue.
          LOGW("%s: Mesh %zu, primitive %zu had no normals or directions!\n", __FUNCTION__, meshIdx, primIdx);
        }
      }

      if(barySet && gltfPrim->has_nv_attribute_micromap && !gltfPrim->has_nv_displacement_micromap)
      {
        LOGW("%s: Mesh %zu, primitive %zu has nv_attribute_micromap but no nv_displacement_micromap (unsupported by this app)!\n",
             __FUNCTION__, meshIdx, primIdx);
      }
      else if(barySet && gltfPrim->has_nv_attribute_micromap && gltfPrim->has_nv_attribute_micromap)
      {
        const cgltf_nv_attribute_micromap& gltfAttrPrim = gltfPrim->nv_attribute_micromap;

        if(gltfAttrPrim.map_indices != nullptr)
        {
          LOGE("%s: The attribute micromap map_indices for mesh %zu, primitive %zu are unsupported by this app.\n",
               __FUNCTION__, meshIdx, primIdx);
          return {};
        }
        if(gltfAttrPrim.map_offset != 0)
        {
          LOGE("%s: The attribute micromap map_offset for mesh %zu, primitive %zu is non-zero and therefore unsupported by this app.\n",
               __FUNCTION__, meshIdx, primIdx);
          return {};
        }


        if(gltfAttrPrim.attributes_count != 1 || gltfAttrPrim.attributes[0].type != cgltf_attribute_type_normal)
        {
          LOGE("%s: The attribute micromap for mesh %zu, primitive %zu does not match NORMAL alone (unsupported by this app.)\n",
               __FUNCTION__, meshIdx, primIdx);
          return {};
        }

        cgltf_nv_micromap* micromap = gltfAttrPrim.attributes[0].micromap ?
                                          gltfAttrPrim.attributes[0].micromap :
                                          ((cgltf_nv_micromap*)gltfAttrPrim.attributes[0].image);
        if(micromap)
        {
          // register this primitive for the micromap
          auto it = baryNormalMicromapsToPrimitivesMap.find(micromap);
          if(it == baryNormalMicromapsToPrimitivesMap.end())
          {
            baryNormalMicromapsToPrimitivesMap.insert({micromap, {gltfPrim}});
          }
          else
          {
            it->second.push_back(gltfPrim);
          }
        }

        // we don't care about loading topologyFlags as if they exist, they must match
        // displacement, which was handled before this (and we ignore this extension if displacement doesn't exist)
      }

      meshSet.meshInfos.push_back(mesh);
    }
  }

  // if decimateEdgeFlags exist, must ensure they exist for all triangles
  // as shaders and later processing will the assume they exist for all.
  // default to 0
  if(meshSet.decimateEdgeFlags.size())
  {
    meshSet.decimateEdgeFlags.resize(meshSet.indices.size() / 3, 0);
  }

  if(barySet)
  {
    bary::Format      valueFormat = bary::Format::eUndefined;
    bary::ValueLayout valueLayout = bary::ValueLayout::eUndefined;

    for(const auto& itImage : baryDisplacementMicromapsToPrimitivesMap)
    {
      baryutils::BaryFile bfile;
      uint32_t baryImageId = loadGLTFBary(mappings, itImage.first, filename, *barySet, bfile, BARY_FILE_DISPLACEMENT);
      if(baryImageId == MeshSetID::INVALID)
      {
        LOGE("Could not load required bary file\n");
        return {};
      }

      const bary::BasicView& basic = bfile.m_content.basic;

      if(valueLayout == bary::ValueLayout::eUndefined)
      {
        valueFormat = basic.valuesInfo->valueFormat;
        valueLayout = basic.valuesInfo->valueLayout;
      }

      if(basic.valuesInfo->valueFormat != valueFormat || basic.valuesInfo->valueLayout != valueLayout)
      {
        LOGE("All bary files must have equal valueFormat, valueOrder and splitOrder\n");
        return {};
      }

      for(cgltf_primitive* gltfPrim : itImage.second)
      {
        const cgltf_nv_displacement_micromap& gltfDispPrim = gltfPrim->nv_displacement_micromap;

        size_t            meshIdx   = gltfPrimToMeshMap[gltfPrim];
        MeshInfo&         mesh      = meshSet.meshInfos[meshIdx];
        const bary::Group baryGroup = basic.groups[gltfDispPrim.group_index];

        mesh.displacementGroup     = static_cast<uint32_t>(gltfDispPrim.group_index);
        mesh.displacementMapOffset = static_cast<uint32_t>(gltfDispPrim.map_offset);
        mesh.displacementID        = baryImageId;


        if(baryGroup.triangleCount != mesh.numPrimitives)
        {
          LOGE("Bary file's BaryGroup::numPrimitives must match mesh::primitive triangle count\n");
          return {};
        }

        if(gltfDispPrim.map_indices == nullptr && bfile.hasProperty(bary::StandardPropertyType::eMeshTriangleMappings))
        {
          LOGE("Bary file bary::StandardPropertyType::eMeshTriangleMappings is unsupported\n");
          return {};
        }

        // look for data in bary file if accessor is null
        if(!gltfDispPrim.direction_bounds && bfile.hasProperty(bary::StandardPropertyType::eMeshDisplacementDirectionBounds))
        {
          if(bfile.m_content.mesh.meshDisplacementDirectionBoundsInfo->elementFormat != bary::Format::eRG32_sfloat)
          {
            LOGE("Bary file directionBoundsFormat unsupported");
            return {};
          }

          uint64_t             arrayCount = bfile.m_content.mesh.meshDisplacementDirectionBoundsInfo->elementCount;
          const glm::vec2* arrayData =
              reinterpret_cast<const glm::vec2*>(bfile.m_content.mesh.meshDisplacementDirectionBounds);
          assert((gltfDispPrim.direction_bounds_offset + mesh.numVertices) <= arrayCount);

          memcpy(&vtxAttribs.directionBounds[mesh.firstVertex], arrayData + gltfDispPrim.direction_bounds_offset,
                 sizeof(glm::vec2) * mesh.numVertices);

          mesh.directionBoundsAreUniform    = false;
          meshSet.directionBoundsAreUniform = false;
        }

        if(!gltfDispPrim.directions && bfile.hasProperty(bary::StandardPropertyType::eMeshDisplacementDirections))
        {
          if(bfile.m_content.mesh.meshDisplacementDirectionsInfo->elementFormat != bary::Format::eRGB32_sfloat)
          {
            LOGE("Bary file uses unsupported directionFormat");
            return {};
          }

          uint64_t             arrayCount = bfile.m_content.mesh.meshDisplacementDirectionsInfo->elementCount;
          const glm::vec3* arrayData = reinterpret_cast<const glm::vec3*>(bfile.m_content.mesh.meshDisplacementDirections);
          assert((gltfDispPrim.directions_offset + mesh.numVertices) <= arrayCount);

          memcpy(&vtxAttribs.directions[mesh.firstVertex], arrayData + gltfDispPrim.directions_offset,
                 sizeof(glm::vec3) * mesh.numVertices);

          mesh.directionsUseMeshNormals    = false;
          meshSet.directionsUseMeshNormals = false;
        }

        if(!gltfDispPrim.primitive_flags && bfile.hasProperty(bary::StandardPropertyType::eMeshTriangleFlags))
        {
          if(bfile.m_content.mesh.meshTriangleFlagsInfo->elementFormat != bary::Format::eR8_uint)
          {
            LOGE("Bary file is using unspported meshPrimitiveFlagFormat");
            return {};
          }

          uint64_t       arrayCount = bfile.m_content.mesh.meshTriangleFlagsInfo->elementCount;
          const uint8_t* arrayData  = reinterpret_cast<const uint8_t*>(bfile.m_content.mesh.meshTriangleFlags);
          assert((gltfDispPrim.primitive_flags_offset + (mesh.numPrimitives)) <= arrayCount);

          if(meshSet.decimateEdgeFlags.size() < (mesh.firstIndex + mesh.numIndices) / 3)
          {
            meshSet.decimateEdgeFlags.resize((mesh.firstIndex + mesh.numIndices) / 3);
          }

          memcpy(&meshSet.decimateEdgeFlags[mesh.firstPrimitive], arrayData + gltfDispPrim.primitive_flags_offset,
                 sizeof(uint8_t) * (mesh.numPrimitives));
        }
      }
    }

    valueLayout = bary::ValueLayout::eUndefined;

    // must process after displacement
    for(const auto& itImage : baryNormalMicromapsToPrimitivesMap)
    {
      baryutils::BaryFile bfile;
      uint32_t baryImageId = loadGLTFBary(mappings, itImage.first, filename, *barySet, bfile, BARY_FILE_ATTRIBUTE);
      if(baryImageId == MeshSetID::INVALID)
      {
        LOGE("Could not load required bary file\n");
        return {};
      }

      BaryShadingAttribute* shadingAttribute = barySet->getShading(baryImageId);

      if(valueLayout == bary::ValueLayout::eUndefined)
      {
        valueLayout = bfile.m_content.basic.valuesInfo->valueLayout;
      }

      if(bfile.m_content.basic.valuesInfo->valueLayout != valueLayout)
      {
        LOGE("All bary files must have equal valueFormat, valueOrder and splitOrder\n");
        return {};
      }

      if(bfile.m_content.basic.valuesInfo->valueFormat != bary::Format::eRG16_snorm)
      {
        LOGE("All normal attribute bary files must use eRG16_snorm (octant encoding)\n");
        return {};
      }

      for(cgltf_primitive* gltfPrim : itImage.second)
      {
        const cgltf_nv_attribute_micromap& gltfBaryAttrPrim = gltfPrim->nv_attribute_micromap;

        size_t            meshIdx   = gltfPrimToMeshMap[gltfPrim];
        MeshInfo&         mesh      = meshSet.meshInfos[meshIdx];
        const bary::Group baryGroup = bfile.m_content.basic.groups[gltfBaryAttrPrim.group_index];

        if(static_cast<uint32_t>(gltfBaryAttrPrim.group_index) != mesh.displacementGroup
           || static_cast<uint32_t>(gltfBaryAttrPrim.map_offset) != mesh.displacementMapOffset)
        {
          LOGE("All normal attribute bary files must use same groupOffset mapOffset matching displacement\n");
          return {};
        }

        if(shadingAttribute->displacementID == MeshSetID::INVALID)
        {
          shadingAttribute->displacementID = mesh.displacementID;
        }

        if(mesh.displacementID == MeshSetID::INVALID || shadingAttribute->displacementID != mesh.displacementID)
        {
          LOGE("All normal attribute bary files must have unique pairing with another displacement file\n");
          return {};
        }

        shadingAttribute->attributeFlags |= SHADING_ATTRIBUTE_NORMAL_BIT;
        mesh.baryNormalID = baryImageId;

        if(baryGroup.triangleCount != mesh.numPrimitives)
        {
          LOGE("Bary file's BaryGroup::triangleCount must match mesh::primitive triangle count\n");
          return {};
        }

        if(gltfBaryAttrPrim.map_indices == nullptr && bfile.hasProperty(bary::StandardPropertyType::eMeshTriangleMappings))
        {
          LOGE("Bary file bary::StandardPropertyType::eMeshTriangleMappings is unsupported\n");
          return {};
        }

        // we don't care about loading topologyFlags as if they exist, they must match
        // displacement, which was handled before this (and we ignore this extension if displacement doesn't exist)
      }

      const BaryDisplacementAttribute& displacementAttr = barySet->displacements[shadingAttribute->displacementID];
      if(!(displacementAttr.compressed || displacementAttr.uncompressed))
      {
        LOGE("Bary file displacement must exist for shading attribute\n");
        return {};
      }

      // let's test a few random primitives for equal subdivLevel
      size_t                basePrimSize      = shadingAttribute->attribute->triangles.size();
      const bary::Triangle* displacementPrims = displacementAttr.uncompressed ?
                                                    displacementAttr.uncompressed->triangles.data() :
                                                    displacementAttr.compressed->triangles.data();
      const bary::Triangle* shadingPrims      = shadingAttribute->attribute->triangles.data();

      for(uint32_t i = 0; i < 100; i++)
      {
        size_t test = (size_t(rand()) + size_t(rand()) * RAND_MAX) % basePrimSize;

        if(shadingPrims[test].subdivLevel != displacementPrims[test].subdivLevel)
        {
          LOGE("Bary file shading attribute subdivLevel mismatches displacement\n");
          return {};
        }
      }
    }
  }

  // Extend attribute vectors so that each non-empty vector has the same length
  // as every other non-empty vector. Different meshes can have different
  // sets of attributes, resulting in a situation that looks like this:
  // positions  AAAAAAABBBBBBBB
  // texcoord0  AAAAAAA
  // directions -------BBBBBBBB
  // In this case, we want to extend texcoord0. This makes assumptions later on
  // ("every attribute has the same length or 0") hold.
  {
    const size_t maxLength = vtxAttribs.positions.size();  // Every prim has positions, so it's the longest
    vtxAttribs.normals.resize(vtxAttribs.normals.empty() ? 0 : maxLength);
    vtxAttribs.texcoords0.resize(vtxAttribs.texcoords0.empty() ? 0 : maxLength);
    vtxAttribs.tangents.resize(vtxAttribs.tangents.empty() ? 0 : maxLength);
    vtxAttribs.bitangents.resize(vtxAttribs.bitangents.empty() ? 0 : maxLength);
    vtxAttribs.directions.resize(vtxAttribs.directions.empty() ? 0 : maxLength);
    vtxAttribs.directionBounds.resize(vtxAttribs.directionBounds.empty() ? 0 : maxLength);
  }


  // Traverse the node hierarchy and create MeshInstances.
  // If we have an initial scene, use that; otherwise, use scene 0 if it exists.
  // If no scenes exist, traverse over all root nodes (nodes with no parent).
  if(data->scenes_count > 0)
  {
    const cgltf_scene scene = (data->scene != nullptr) ? (*(data->scene)) : (data->scenes[0]);
    for(size_t nodeIdx = 0; nodeIdx < scene.nodes_count; nodeIdx++)
    {
      addMeshInstancesFromNode(meshSet, scene.nodes[nodeIdx], gltfPrimToMeshMap);
    }
  }
  else
  {
    for(size_t nodeIdx = 0; nodeIdx < data->nodes_count; nodeIdx++)
    {
      if(data->nodes[nodeIdx].parent == nullptr)
      {
        addMeshInstancesFromNode(meshSet, &(data->nodes[nodeIdx]), gltfPrimToMeshMap);
      }
    }
  }

  // and I believe we're now done!
  return true;
}
