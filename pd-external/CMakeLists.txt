cmake_minimum_required(VERSION 3.0)

if(WIN32)
  set(EXTENSION "~.dll")
endif()

if(APPLE)
  set(EXTENSION "~.pd_darwin")
endif()

if(UNIX AND NOT APPLE)
  set(EXTENSION "~.pd_linux")
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL Debug)
  add_definitions(-DDEBUG)
endif()

project(pd-external)

find_package(Torch REQUIRED)

set(SRC vae_sampler~.cpp)
add_library(vae_sampler SHARED ${SRC})
set_target_properties(vae_sampler PROPERTIES SUFFIX ${EXTENSION} PREFIX "")
set_target_properties(vae_sampler PROPERTIES CMAKE_CXX_STANDARD 11)
target_link_libraries(vae_sampler "${TORCH_LIBRARIES}")

if(APPLE)
  set_target_properties(vae_sampler PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  add_custom_command(TARGET vae_sampler POST_BUILD COMMAND install_name_tool -add_rpath @loader_path/libtorch/lib vae_sampler~.pd_darwin)
endif()
