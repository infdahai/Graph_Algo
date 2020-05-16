

#pragma once

#ifndef GRAPH_ALGO_BELLMANFORD_CL_H
#define GRAPH_ALGO_BELLMANFORD_CL_H

#include "BellmanFord.h"
#include "../../include/GPUconfig.h"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
//#include <pthread.h>
#include <cfloat>
#include <ctime>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else

#include <CL/cl.h>

#endif

template<typename VertexValueType, typename MessageValueType>
class BellmanFordCL : public BellmanFord<VertexValueType, MessageValueType> {
public:
    BellmanFordCL();

    void Init(int vCount, int eCount, int numOfInitV) override;

    void
    GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList) override;

    void Deploy(int vCount, int eCount, int numOfInitV) override;

    void Free() override;

    int MSGApply1(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice,
                  const MessageSet<MessageValueType> &mSet);


    int
    MSGGenMerge_CL1(Graph<VertexValueType> &g, std::vector<int> &initVSet, std::set<int> &activeVertice,
                    MessageSet<MessageValueType> &mSet);

    //    int MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues,MessageValueType *mValues) override;
    //   int MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, MessageValueType *mValues) override;

    //  cl_device_id getMaxFlopsDev(cl_context);
    void loadAndBuildProgram(cl_context, const char *);

    void ApplyStep(Graph<VertexValueType> &g, std::vector<int> &initVSet, std::set<int> &activeVertices);

    void ApplyD_CL(Graph<VertexValueType> &g, std::vector<int> &initVList, int partitionCount);

    void Buffer_alloc1(Vertex *vSet, Edge *eSet, int numOfInitV, VertexValueType *vValues,
                       MessageValueType *mValues, int vcount, int ecount, int flag);

    void Free_little();

protected:
    int vertexLimit;
    int mPerMSGSet;
    int ePerEdgeSet;

    cl_mem hostVSet;
    cl_mem hostESet;
    //   cl_mem host_initVSet;
    //  cl_int host_avCount;
    cl_mem hostVValues;
    cl_mem hostMValues;

    cl_mem vSet;
    cl_mem eSet;
    //  cl_mem initVSet;
    cl_mem vValues;
    cl_mem mValues;
    //   cl_int avCount;
    cl_kernel MSGApply_array_kernel;
    cl_kernel MSGGenMerge_array_CL_kernel;
    cl_kernel MSGInitial_array_kernel_1;
    cl_kernel MSGInitial_array_kernel_2;

    typedef struct CL_device {
        cl_context context;
        cl_device_id device;
        int numResults;

        CL_device() {
            context = 0;
            device = 0;
            numResults = 0;
        }
    } CL_DEVICE;
    bool MutliGPU_isOrNot = false;
    bool GPU_isOrNot = true;
    cl_uint device_count;
    CL_DEVICE *cl_device_array;
    cl_device_id *devices;

    cl_platform_id platform = 0;
    cl_uint numPlatforms;
    cl_platform_id *platformIds;

    cl_context cpu_contxt, gpu_context = NULL;
    cl_program program;
    cl_command_queue comman_queue;
    cl_kernel kernel;
    cl_int errNum;
    cl_event readDone;
    //  cl_event copyDone;
    size_t local_work_size;
    size_t global_work_size;
    // int numOfInitV;
    // cl_mem vertexArrayDevice;
    // cl_mem edgeArrayDevice;
    // cl_mem weightArrayDevice;
    // cl_mem maskArrayDevice;
    // cl_mem costArrayDevice;
    // cl_mem updatingCostArrayDevice;

private:
};

void displayPlatformInfo(
        cl_platform_id id,
        cl_platform_info name,
        std::string str);

template<typename T>
void displayDeviceInfo(
        cl_device_id id,
        cl_device_info name,
        std::string str);

void checkErrorFileLine(int errNum, int expected, const char *file, const int lineNumber);

cl_device_id getMaxFlopsDev(cl_context cxGPUContext);

int logError(int, const std::string &);

int roundWorkSize(int, int);

#endif //GRAPH_ALGO_BELLMANFORD_CL_H