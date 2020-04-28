cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)

add_library(algo_BellmanFord
        BellmanFord.h
        BellmanFord.cpp
        BellmanFord_impl.cpp)

target_link_libraries(algo_BellmanFord
        core_Graph
        core_GraphUtil
        core_MessageSet)

if (CUDA_FOUND)
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; ${ARCH_FLAGS}; -std=c++14;)

    cuda_add_library(algo_BellmanFordGPU_kernel STATIC
            kernel_src/BellmanFordGPU_kernel.h
            kernel_src/BellmanFordGPU_kernel.cu)

    add_library(algo_BellmanFordGPU
            BellmanFordGPU.h
            BellmanFordGPU.cpp
            BellmanFordGPU_impl.cpp)

    target_link_libraries(algo_BellmanFordGPU
            algo_BellmanFord
            algo_BellmanFordGPU_kernel)

endif (CUDA_FOUND)


set(OpenCL_FOUND TRUE)

if (OpenCL_FOUND)

    set(OpenCL_INCLUDE_DIR "/usr/local/cuda-10.0/include/CL")
    set(OpenCL_LIBRARY "/usr/local/cuda-10.0/lib64/libOpenCL.so")
    # find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIR})
    #  link_directories(${OpenCL_LIBRARY})

    configure_file(kernel_src/BellmanFordCL_kernel.cl ${CMAKE_CURRENT_BINARY_DIR}/BellmanFordCL_kernel.cl COPYONLY)
    add_library(algo_BellmanFordCL
            BellmanFordCL.h
            BellmanFordCL.cpp
            BellmanFordCL_impl.cpp)
    target_link_libraries(
            algo_BellmanFordCL
            algo_BellmanFord
            ${OpenCL_LIBRARY}
    )


endif (OpenCL_FOUND)