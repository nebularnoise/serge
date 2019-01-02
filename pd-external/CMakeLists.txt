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

project(pd-external)

set(SRC vae_sampler~.c)

add_library(vae_sampler SHARED ${SRC})
set_target_properties(vae_sampler PROPERTIES SUFFIX ${EXTENSION} PREFIX "")
