cmake_minimum_required(VERSION 3.5)
get_filename_component(PROJNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
Project(${PROJNAME})

Message(STATUS "-------------------------------")
Message(STATUS "Processing Project ${PROJNAME}:")

#####################################################################################
# Add the custom branch of cgltf as a target before nvpro_core, so that nvpro_core
# and this project use the custom branch.
if(NOT TARGET cgltf)
  add_library(cgltf INTERFACE)
  target_include_directories(cgltf INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/external/cgltf")
endif()

#####################################################################################
# Set the default for enabling beta extensions before nvpro_core. Beta
# extensions can conflict if the Vulkan SDK and nvpro_core generated
# nvvk/extensions_vk.hpp versions are different.
option(VK_ENABLE_BETA_EXTENSIONS "Enable beta extensions provided by the Vulkan SDK" OFF)

#####################################################################################
# look for nvpro_core 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
#

SET(MODEL_DOWNLOAD_${PROJNAME} ON CACHE BOOL "CHECK to download the 3d models needed for this sample" )

if(NOT BASE_DIRECTORY)
  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../.. 
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
endif()
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/utilities.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()
    
_add_project_definitions(${PROJNAME})

#####################################################################################
# additions from packages needed for this sample
# add refs  in LIBRARIES_OPTIMIZED
# add refs  in LIBRARIES_DEBUG
# add files in PACKAGE_SOURCE_FILES

_add_package_VulkanSDK()
_add_package_ShaderC()
_add_package_IMGUI()

if(SUPPORT_AFTERMATH)
  _add_package_NsightAftermath()
endif()

#####################################################################################
_find_package_OpenMP()
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#####################################################################################
# process the rest of some cmake code that needs to be done *after* the packages add
_add_nvpro_core_lib()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

string(TOLOWER "${CMAKE_CXX_COMPILER_ID}" COMPILER_ID)

if (COMPILER_ID STREQUAL "gnu" OR COMPILER_ID STREQUAL "clang" OR COMPILER_ID STREQUAL "intel")

    # Turn on all warnings
    if(CMAKE_COMPILER_IS_ICC)
        add_definitions(-w3)       
    else()
        add_definitions(-Wall)
    endif()
    
    # TODO: add these per target using target_compile_definitions, not globally
    #add_definitions(-Wextra -Werror)    

elseif (MSVC)

    # Turn on all warnings
    add_definitions(/W3 /WX)

     # Make sure WinDef.h does not define min and max macros which
     # will conflict with std::min() and std::max().
     add_definitions(/DNOMINMAX)
     
     # Eliminate deprecation warnings for the older, less secure
     # CRT functions
     add_definitions(/D_CRT_SECURE_NO_WARNINGS)
    
endif()

#####################################################################################
download_files(FILENAMES umesh_Murex_Romosus.zip EXTRACT)

#####################################################################################
# Source files for this project
#

file(GLOB SOURCE_FILES *.cpp *.hpp *.inl *.h *.c)
file(GLOB GLSL_FILES *.glsl)

#####################################################################################
# Executable
#
if(NOT WIN32)
  add_definitions(-fpermissive)
endif()

add_executable(${PROJNAME} ${SOURCE_FILES} ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES} ${GLSL_FILES} ${MESHOPTIMIZER_SOURCE_FILES})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJNAME})


#####################################################################################
# common source code needed for this sample
#
source_group(common FILES 
  ${COMMON_SOURCE_FILES}
  ${PACKAGE_SOURCE_FILES}
)
source_group(shaders FILES 
  ${GLSL_FILES}
)
source_group("Source Files" FILES ${SOURCE_FILES})
source_group(meshoptimizer FILES ${MESHOPTIMIZER_SOURCE_FILES})

if(UNIX)
  set(UNIXLINKLIBS dl pthread)
else()
  set(UNIXLINKLIBS)
endif()

#####################################################################################

set(BARY_CORE_BUILD_UTILS ON CACHE BOOL "Build the utilities for bary_core")
if (NOT TARGET bary_core)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/bary_core ${CMAKE_CURRENT_SOURCE_DIR}/external/bary_core)
endif()


#####################################################################################
# Linkage
#

target_link_libraries(${PROJNAME} ${PLATFORM_LIBRARIES} nvpro_core bary_core bary_utils)

foreach(DEBUGLIB ${LIBRARIES_DEBUG})
  target_link_libraries(${PROJNAME} debug ${DEBUGLIB})
endforeach(DEBUGLIB)

foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
  target_link_libraries(${PROJNAME} optimized ${RELEASELIB})
endforeach(RELEASELIB)

#####################################################################################
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#

_finalize_target( ${PROJNAME} )

LIST(APPEND GLSL_FILES "config.h")
LIST(APPEND GLSL_FILES "common.h")
LIST(APPEND GLSL_FILES "common_barymap.h")
LIST(APPEND GLSL_FILES "common_mesh.h")
LIST(APPEND GLSL_FILES "common_micromesh_uncompressed.h")
LIST(APPEND GLSL_FILES "common_micromesh_compressed.h")
LIST(APPEND GLSL_FILES "common_micromesh_compressed_rt.h")
LIST(APPEND GLSL_FILES "hbao.h")
LIST(APPEND GLSL_FILES "octant_encoding.h")
LIST(APPEND GLSL_FILES "micromesh_binpack_flat_decl.h")
install(FILES ${GLSL_FILES} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/GLSL_${PROJNAME}")
install(FILES ${GLSL_FILES} CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug/GLSL_${PROJNAME}")
