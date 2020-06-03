
#pragma once

#ifndef GRAPH_ALGO_CONNECTEDCOMPONENT_CL_H
#define GRAPH_ALGO_CONNECTEDCOMPONENT_CL_H

#include "ConnectedComponent.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else

#include <CL/cl.h>

#endif

template <typename VertexValueType, typename MessageValueType>
class ConnectedComponentCL : public ConnectedComponent<VertexValueType, MessageValueType>
{
public:
    ConnectedComponentCL();

    int MSGApply_CL(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice,
                    const MessageSet<MessageValueType> &mSet);

    int MSGGenMerge_CL(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice,
                       MessageSet<MessageValueType> &mSet);

    int MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, MessageValueType *mValues) override;
    int MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, MessageValueType *mValues) override;

    void Init(int vCount, int eCount, int numOfInitV) override;

    void
    GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList) override;

    void Deploy(int vCount, int eCount, int numOfInitV) override;

    void Free() override;

    void loadAndBuildProgram(cl_context, const char *);

    void Buffer_alloc(Vertex *vSet, Edge *eSet, int numOfInitV, VertexValueType *vValues,
                      MessageValueType *mValues, int vcount, int ecount, int flag);

    void ApplyStep(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices);

    void ApplyD_CL(Graph<VertexValueType> &g, const std::vector<int> &initVList, int partitionCount);

    void Free_little();

    void checkErrorFileLine(int errNum, int expected, const char *file, const int lineNumber);

    int roundWorkSize(int, int);

    cl_device_id getMaxFlopsDev(cl_context cxGPUContext);

protected:
    cl_context cpu_contxt, gpu_context = NULL;
    cl_program program;
    cl_command_queue comman_queue;
    cl_int errNum;
    cl_event readDone;
    //  cl_event copyDone;
    size_t local_work_size;
    size_t global_work_size;

    cl_mem hostVSet;
    cl_mem hostESet;
    cl_mem hostVValues;
    cl_mem hostMValues;

    cl_mem vSet;
    cl_mem eSet;
    cl_mem vValues;
    cl_mem mValues;
    int numOfInitV;

    cl_device_id *devices;

    cl_kernel MSGApply_array_kernel;
    cl_kernel MSGGenMerge_array_CL_kernel;
    cl_kernel MSGInitial_array_kernel_1;
    cl_kernel MSGInitial_array_kernel_2;

private:
};

#endif