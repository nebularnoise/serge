#AUTHOR: ANTOINE CAILLON
cmake_minimum_required(VERSION 3.0)

project(PYGLIM)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ENABLE_FLOAT ON CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
add_subdirectory(fftw/)

add_library(glim SHARED src/griffin_lim_generic.cpp)
target_link_libraries(glim PUBLIC fftw3f)
target_include_directories(glim PRIVATE fftw/api)
target_compile_definitions(glim PRIVATE FP_SINGLE_PRECISION)
