cmake_minimum_required(VERSION 3.0)

project(pd-external)

set(SRC vae_sampler~.c)

add_library(vae_sampler SHARED ${SRC})
set_target_properties(vae_sampler PROPERTIES SUFFIX "~.pd_linux" PREFIX "")
