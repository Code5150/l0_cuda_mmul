cmake_minimum_required(VERSION 3.19.2)
project(cuda_mmul CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

find_package(CUDAToolkit)
find_package(OpenMP REQUIRED)

add_library(cpu_mmul STATIC cpu_mmul.h cpu_mmul.cpp)
target_link_libraries(cpu_mmul PUBLIC OpenMP::OpenMP_CXX)
target_precompile_headers(cpu_mmul PUBLIC [["omp.h"]])

add_executable(cuda_mmul main.cu)
target_link_directories(cuda_mmul PUBLIC ${CUDAToolkit_LIBRARY_DIR})

set_target_properties(cuda_mmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuda_mmul PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_precompile_headers(cuda_mmul PUBLIC <cassert> <cstdio> [["cuda_runtime.h"]] [["curand.h"]] [["cublas.h"]] [["cublas_v2.h"]] <iostream> [["cpu_mmul.h"]])
target_link_libraries(cuda_mmul PUBLIC cublas)
target_link_libraries(cuda_mmul PUBLIC curand)
target_link_libraries(cuda_mmul PRIVATE cpu_mmul)