cmake_minimum_required(VERSION 3.9)
project(Graph_Algo)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)

add_executable(core_GraphUtilTest
        GraphUtilTest.cpp)

target_link_libraries(core_GraphUtilTest
        core_Graph
        core_GraphUtil
        algo_BellmanFord)

add_executable(algo_BellmanFordTest
        BellmanFordTest.cpp)

target_link_libraries(algo_BellmanFordTest
        algo_BellmanFord
        core_Graph
        core_MessageSet
        core_GraphUtil)
                
        
#//////////////////

set(OpenCL_INCLUDE_DIR "/usr/local/cuda-10.0/include/CL")
set(OpenCL_LIBRARY "/usr/local/cuda-10.0/lib64/libOpenCL.so")
include_directories(${OpenCL_INCLUDE_DIR})
#link_directories(${OpenCL_LIBRARY})
add_executable(algo_BellmanFordCLTest
        BellmanFordCL_Test.cpp)

target_link_libraries(algo_BellmanFordCLTest
        algo_BellmanFordCL
        core_Graph
        core_MessageSet
        core_GraphUtil)

#//////////////////


add_executable(algo_LabelPropagationTest
		LabelPropagationTest.cpp)

target_link_libraries(algo_LabelPropagationTest
        algo_LabelPropagation
        core_Graph
        core_MessageSet
        core_GraphUtil)

add_executable(algo_ConnectedComponentTest
        ConnectedComponentTest.cpp)

target_link_libraries(algo_ConnectedComponentTest
        algo_ConnectedComponent
        core_Graph
        core_MessageSet
        core_GraphUtil)

add_executable(core_GraphAccessTest
        GraphAccessTest.cpp)

target_link_libraries(core_GraphAccessTest
        core_Graph
        core_MessageSet
        core_GraphUtil)

if(CUDA_FOUND)
    add_executable(algo_BellmanFordGPUTest
            BellmanFordGPUTest.cpp)

    target_link_libraries(algo_BellmanFordGPUTest
            algo_BellmanFordGPU
            core_Graph
            core_MessageSet
            core_GraphUtil)
endif(CUDA_FOUND)

if(UNIX)
    add_executable(srv_Test_srv
            SRVTest_srv.cpp)

    target_link_libraries(srv_Test_srv
            srv_UNIX_shm
            srv_UNIX_msg
            srv_UtilServer
            srv_UtilClient)

    add_executable(srv_Test_cli
            SRVTest_cli.cpp)

    target_link_libraries(srv_Test_cli
            srv_UNIX_shm
            srv_UNIX_msg
            srv_UtilServer
            srv_UtilClient)

    IF(CUDA_FOUND)
        add_executable(srv_UtilServerTest_BellmanFordGPU
                UtilServerTest_BellmanFordGPU.cpp)

        target_link_libraries(srv_UtilServerTest_BellmanFordGPU
                srv_UNIX_shm
                srv_UNIX_msg
                srv_UtilServer
                algo_BellmanFord
                algo_LabelPropagation
                algo_BellmanFordGPU)
    ENDIF(CUDA_FOUND)

    add_executable(srv_UtilServerTest_BellmanFord
            UtilServerTest_BellmanFord.cpp)

    target_link_libraries(srv_UtilServerTest_BellmanFord
            srv_UNIX_shm
            srv_UNIX_msg
            srv_UtilServer
            algo_BellmanFord
            algo_LabelPropagation)

    add_executable(srv_UtilServerTest_LabelPropagation
            UtilServerTest_LabelPropagation.cpp)

    target_link_libraries(srv_UtilServerTest_LabelPropagation
            srv_UNIX_shm
            srv_UNIX_msg
            srv_UtilServer
            algo_BellmanFord
            algo_LabelPropagation)

    add_executable(srv_UtilClientTest
            UtilClientTest.cpp)

    target_link_libraries(srv_UtilClientTest
            pthread
            srv_UNIX_shm
            srv_UNIX_msg
            srv_UtilClient)
ENDIF(UNIX)