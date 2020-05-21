cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)

if(UNIX)
    add_library(srv_UNIX_shm
            UNIX_shm.h
            UNIX_shm.cpp)

    add_library(srv_UNIX_msg
            UNIX_msg.h
            UNIX_msg.cpp)

    set(OpenCL_FOUND TRUE)
    set(OpenCL_INCLUDE_DIR "/usr/local/cuda-10.0/include")
    set(OpenCL_LIBRARY "/usr/local/cuda-10.0/lib64/libOpenCL.so")
    include_directories(${OpenCL_INCLUDE_DIR})

    if(CUDA_FOUND)
        include_directories("${CUDA_INCLUDE_DIRS}")
                        
      

   #     configure_file(kernel_src/ConnectedComponentCL_kernel.cl ${CMAKE_CURRENT_BINARY_DIR}/ConnectedComponentCL_kernel.cl COPYONLY)

        add_library(srv_UtilServer
                UtilServer.h
                UtilServer.cpp
                impl/UtilServer_impl.cpp
                impl/UtilServer_implGPU.cpp
                impl/UtilServer_implCL.cpp)

        target_link_libraries(srv_UtilServer
                srv_UNIX_msg
                srv_UNIX_shm
                core_Graph
                core_GraphUtil
                algo_PageRankGPU_kernel
                algo_ConnectedComponentGPU_kernel
                algo_BellmanFordGPU_kernel
                algo_LabelPropagationGPU_kernel
                ${OpenCL_LIBRARY})
    else()
        add_library(srv_UtilServer
                UtilServer.h
                UtilServer.cpp
                impl/UtilServer_impl.cpp
                impl/UtilServer_implCL.cpp)

        target_link_libraries(srv_UtilServer
                srv_UNIX_msg
                srv_UNIX_shm
                core_Graph
                core_GraphUtil
                ${OpenCL_LIBRARY})
    endif(CUDA_FOUND)

    add_library(srv_UtilClient
            UtilClient.h
            UtilClient.cpp
            impl/UtilClient_impl.cpp)

    target_link_libraries(srv_UtilClient
            srv_UNIX_msg
            srv_UNIX_shm
            core_Graph
            core_GraphUtil)
ENDIF(UNIX)