cmake_minimum_required(VERSION 3.20)
project(cuda_mmul CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

#find_package(CUDA REQUIRED)
find_package(CUDAToolkit)
add_executable(cuda_mmul main.cu)
set(PATH_TO_CUDA_LIBS "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\lib\\x64")
target_link_directories(cuda_mmul PUBLIC ${PATH_TO_CUDA_LIBS})

set_target_properties(cuda_mmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuda_mmul PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_precompile_headers(cuda_mmul PUBLIC <cassert> <cstdio> [["cuda_runtime.h"]] [["curand.h"]] [["cublas.h"]] [["cublas_v2.h"]] <iostream>)
target_link_libraries(cuda_mmul PUBLIC cublas)
target_link_libraries(cuda_mmul PUBLIC curand)