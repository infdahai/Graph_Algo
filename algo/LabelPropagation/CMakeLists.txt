cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)

add_library(algo_LabelPropagation
        LabelPropagation.h
        LabelPropagation.cpp
        LabelPropagation_impl.cpp)

target_link_libraries(algo_LabelPropagation
        core_Graph
        core_GraphUtil
        core_MessageSet)

if(CUDA_FOUND)
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; ${ARCH_FLAGS}; -std=c++14;)

    cuda_add_library(algo_LabelPropagationGPU_kernel STATIC
            kernel_src/LabelPropagationGPU_kernel.h
            kernel_src/LabelPropagationGPU_kernel.cu)

    add_library(algo_LabelPropagationGPU
            LabelPropagationGPU.h
            LabelPropagationGPU.cpp
            LabelPropagationGPU_impl.cpp)

    target_link_libraries(algo_LabelPropagationGPU
            algo_LabelPropagation
            algo_LabelPropagationGPU_kernel)

endif(CUDA_FOUND)


set(OpenCL_FOUND TRUE)

if (OpenCL_FOUND)

    set(OpenCL_INCLUDE_DIR "/usr/local/cuda-10.0/include")
    set(OpenCL_LIBRARY "/usr/local/cuda-10.0/lib64/libOpenCL.so")
    # find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIR})
    #  link_directories(${OpenCL_LIBRARY})

    configure_file(kernel_src/LabelPropagationCL_kernel.cl ${CMAKE_CURRENT_BINARY_DIR}/LabelPropagationCL_kernel.cl COPYONLY)
    add_library(algo_LabelPropagationCL
        LabelPropagationCL.h      
        LabelPropagationCL.cpp
        LabelPropagationCL_impl.cpp)
    target_link_libraries(
            algo_LabelPropagationCL
            algo_LabelPropagation
            ${OpenCL_LIBRARY}
    )

endif (OpenCL_FOUND)