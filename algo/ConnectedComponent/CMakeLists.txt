cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)

add_library(algo_ConnectedComponent
        ConnectedComponent.h
        ConnectedComponent.cpp
        ConnectedComponent_impl.cpp)

target_link_libraries(algo_ConnectedComponent
        core_Graph
        core_GraphUtil
        core_MessageSet)

if(CUDA_FOUND)
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; ${ARCH_FLAGS}; -std=c++14;)

    cuda_add_library(algo_ConnectedComponentGPU_kernel STATIC
            kernel_src/ConnectedComponentGPU_kernel.h
            kernel_src/ConnectedComponentGPU_kernel.cu)

    add_library(algo_ConnectedComponentGPU
            ConnectedComponentGPU.h
            ConnectedComponentGPU.cpp
            ConnectedComponentGPU_impl.cpp)

    target_link_libraries(algo_ConnectedComponentGPU
            algo_ConnectedComponent
            algo_ConnectedComponentGPU_kernel)

endif(CUDA_FOUND)


set(OpenCL_FOUND TRUE)

if (OpenCL_FOUND)

    set(OpenCL_INCLUDE_DIR "/usr/local/cuda-10.0/include")
    set(OpenCL_LIBRARY "/usr/local/cuda-10.0/lib64/libOpenCL.so")
    # find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIR})
    #  link_directories(${OpenCL_LIBRARY})

    configure_file(kernel_src/ConnectedComponentCL_kernel.cl ${CMAKE_CURRENT_BINARY_DIR}/ConnectedComponentCL_kernel.cl COPYONLY)
    add_library(algo_ConnectedComponentCL
        ConnectedComponentCL.h      
        ConnectedComponentCL.cpp
        ConnectedComponentCL_impl.cpp)
    target_link_libraries(
            algo_ConnectedComponentCL
            algo_ConnectedComponent
            ${OpenCL_LIBRARY}
    )

endif (OpenCL_FOUND)