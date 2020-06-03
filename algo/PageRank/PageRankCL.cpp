#include "PageRankCL.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cfloat>
#include <ctime>
#include <chrono>
#include <algorithm>

#define Check_Err(m, n) checkErrorFileLine(m, n, __FILE__, __LINE__)
#define GetMaxPerDev(clcontext) getMaxFlopsDev(clcontext)

template <typename VertexValueType, typename MessageValueType>
int PageRankCL<VertexValueType, MessageValueType>::roundWorkSize(int group_size, int total_size)
{
    int rem = total_size % group_size;
    if (rem == 0)
    {
        return total_size;
    }
    else
    {
        return total_size + group_size - rem;
    }
}

template <typename VertexValueType, typename MessageValueType>
void PageRankCL<VertexValueType, MessageValueType>::
    loadAndBuildProgram(cl_context context, const char *file_name)
{
    std::ifstream kernelFile(file_name, std::ios::in);
    Check_Err(kernelFile.is_open(), true);

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    // std::cout<<"\n\n"<<srcStdStr<<"\n"<<std::endl;
    const char *sourceFile = srcStdStr.c_str();
    Check_Err((sourceFile != NULL), true);

    // std::cout << "\tcontext:" << context << "\tfile_name" << file_name << std::endl;
    this->program = clCreateProgramWithSource(context, 1, (const char **)&sourceFile, NULL, &errNum);
    //this->program =  clCreateProgramWithBinary(context, 1, (const char **)&sourceFile, NULL, &errNum);
    Check_Err(errNum, CL_SUCCESS);
    errNum = clBuildProgram(this->program, 0, NULL, NULL, NULL, NULL);
    char clBuildLog[10240];
    if (errNum != CL_SUCCESS)
    {
        clGetProgramBuildInfo(this->program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(clBuildLog),
                              clBuildLog, NULL);
        std::cerr << clBuildLog << std::endl;
        Check_Err(errNum, CL_SUCCESS);
    }
    else
    {
        clGetProgramBuildInfo(this->program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(clBuildLog),
                              clBuildLog, NULL);
        printf("Kernel Build Success\n%s\n", clBuildLog);
    }
}

template <typename VertexValueType, typename MessageValueType>
PageRankCL<VertexValueType, MessageValueType>::PageRankCL()
{
    // PageRank<VertexValueType, MessageValueType>::PageRank();
    this->resetProb = 0.15;
    this->deltaThreshold = 0.01;
}

template <typename VertexValueType, typename MessageValueType>
void PageRankCL<VertexValueType, MessageValueType>::Buffer_alloc(Vertex *vSet1, Edge *eSet1,
                                                                 int numOfInitV,
                                                                 VertexValueType *vValues1,
                                                                 MessageValueType *mValues1, int vcount,
                                                                 int ecount, int flag)
{

    if (flag == 0)
    {
        hostVSet = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(Vertex) * vcount, vSet1, &errNum);
        Check_Err(errNum, CL_SUCCESS);
        hostESet = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(Edge) * ecount, eSet1, &errNum);
        Check_Err(errNum, CL_SUCCESS);
        hostVValues = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                     sizeof(VertexValueType) * vcount, vValues1, &errNum);
        Check_Err(errNum, CL_SUCCESS);
        hostMValues = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                     sizeof(MessageValueType) * vcount, mValues1, &errNum);
        Check_Err(errNum, CL_SUCCESS);

        vSet = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(Vertex) * vcount, NULL, &errNum);
        //vSet = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(Vertex) * global_work_size, NULL);
        Check_Err(errNum, CL_SUCCESS);
        eSet = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, sizeof(Edge) * ecount, NULL, &errNum);
        Check_Err(errNum, CL_SUCCESS);
        vValues = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(VertexValueType) * vcount, NULL,
                                 &errNum);
        Check_Err(errNum, CL_SUCCESS);
        mValues = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(MessageValueType) * vcount, NULL,
                                 &errNum);
        Check_Err(errNum, CL_SUCCESS);

        errNum = clEnqueueCopyBuffer(comman_queue, hostVSet, vSet, 0, 0,
                                     sizeof(Vertex) * vcount, 0, NULL, NULL);
        Check_Err(errNum, CL_SUCCESS);

        errNum = clEnqueueCopyBuffer(comman_queue, hostESet, eSet, 0, 0,
                                     sizeof(Edge) * ecount, 0, NULL, NULL);
        Check_Err(errNum, CL_SUCCESS);

        errNum = clEnqueueCopyBuffer(comman_queue, hostVValues, vValues, 0, 0,
                                     sizeof(VertexValueType) * vcount, 0, NULL, NULL);
        Check_Err(errNum, CL_SUCCESS);

        errNum = clEnqueueCopyBuffer(comman_queue, hostMValues, mValues, 0, 0,
                                     sizeof(MessageValueType) * vcount, 0, NULL, NULL);
        Check_Err(errNum, CL_SUCCESS);
    }
    if (flag == 1)
    {
        cl_mem hostMValues_temp = clCreateBuffer(this->gpu_context,
                                                 CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                                 sizeof(MessageValueType) * vcount,
                                                 mValues1, &errNum);
        Check_Err(errNum, CL_SUCCESS);
        errNum = clEnqueueCopyBuffer(comman_queue, hostMValues_temp, mValues, 0, 0,
                                     sizeof(MessageValueType) * vcount, 0, NULL, NULL);
        Check_Err(errNum, CL_SUCCESS);
        clReleaseMemObject(hostMValues_temp);
    }
}

template <typename VertexValueType, typename MessageValueType>
void PageRankCL<VertexValueType, MessageValueType>::Init(int vCount, int eCount, int numOfInitV)
{
    PageRank<VertexValueType, MessageValueType>::Init(vCount, eCount, numOfInitV);

    cl_platform_id *platformIds;
    cl_uint numPlatforms, device_count;
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    std::cout << "\tNumber of OpenCL Platforms: \t" << numPlatforms << std::endl;
    platformIds = (cl_platform_id *)alloca(sizeof(cl_platform_id) * numPlatforms);
    errNum = clGetPlatformIDs(numPlatforms, platformIds, NULL);
    errNum = clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, NULL, &device_count);
    devices = (cl_device_id *)alloca(sizeof(cl_device_id) * device_count);
    errNum = clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, device_count, devices, NULL);
    cl_context_properties context_properties[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIds[0],
        (cl_context_properties)NULL};

    gpu_context = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
    cl_device_id device_id = GetMaxPerDev(gpu_context);
    this->comman_queue = clCreateCommandQueue(gpu_context, device_id, 0, &errNum);

    loadAndBuildProgram(gpu_context, "../algo/PageRank/PageRankCL_kernel.cl");
    //loadAndBuildProgram(gpu_context, "../algo/PageRank/PageRankCL_kernel.out");
    size_t max_workgroup_size;
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup_size, NULL);
    std::cout << "max_group_size : " << max_workgroup_size << std::endl;
    local_work_size = max_workgroup_size;
    global_work_size = roundWorkSize(local_work_size, vCount);
}

template <typename VertexValueType, typename MessageValueType>
void PageRankCL<VertexValueType, MessageValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices,
                                                              const std::vector<int> &initVList)
{
    PageRank<VertexValueType, MessageValueType>::GraphInit(g, activeVertices, initVList);
}

template <typename VertexValueType, typename MessageValueType>
void PageRankCL<VertexValueType, MessageValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{
    PageRank<VertexValueType, MessageValueType>::Deploy(vCount, eCount, numOfInitV);
}

template <typename VertexValueType, typename MessageValueType>
void PageRankCL<VertexValueType, MessageValueType>::Free()
{
    PageRank<VertexValueType, MessageValueType>::Free();
    errNum = CL_SUCCESS;

    errNum |= clReleaseEvent(readDone);

    errNum |= clReleaseKernel(MSGApply_array_kernel);

    errNum |= clReleaseKernel(MSGGenMerge_array_CL_kernel);

    errNum |= clReleaseKernel(MSGInitial_array_kernel_1);

    errNum |= clReleaseKernel(MSGInitial_array_kernel_2);

    errNum |= clReleaseCommandQueue(comman_queue);

    errNum |= clReleaseProgram(program);

    Check_Err(errNum, CL_SUCCESS);
}

template <typename VertexValueType, typename MessageValueType>
int PageRankCL<VertexValueType, MessageValueType>::MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, MessageValueType *mValues)
{
    //array form computation
    // Buffer_alloc(g.vList.data(), g.eList.data(), this->numOfInitV, g.verticesValue.data(), mValues, g.vCount,
    //             g.eCount, 1);
    Buffer_alloc(vSet, eSet, this->numOfInitV, vValues, mValues, vCount, eCount, 1);
    MSGInitial_array_kernel_2 = clCreateKernel(program, "MSGInitial_array_kernel_2", &errNum);
    Check_Err(errNum, CL_SUCCESS);
    errNum |= clSetKernelArg(MSGInitial_array_kernel_2, 0, sizeof(cl_mem), &this->vSet);
    errNum |= clSetKernelArg(MSGInitial_array_kernel_2, 1, sizeof(int), &g.vCount);
    Check_Err(errNum, CL_SUCCESS);
    errNum = clEnqueueNDRangeKernel(comman_queue, MSGInitial_array_kernel_2,
                                    1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    Check_Err(errNum, CL_SUCCESS);

    MSGApply_array_kernel = clCreateKernel(program, "MSGApply_array_CL", &errNum);
    Check_Err(errNum, CL_SUCCESS);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 0, sizeof(cl_mem), &this->vSet);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 1, sizeof(cl_mem), &this->eSet);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 2, sizeof(cl_mem), &this->vValues);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 3, sizeof(cl_mem), &this->mValues);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 4, sizeof(int), &g.vCount);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 5, sizeof(int), &g.eCount);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 6, sizeof(int), &this->numOfInitV);
    Check_Err(errNum, CL_SUCCESS);
    errNum = clEnqueueNDRangeKernel(comman_queue, MSGApply_array_kernel,
                                    1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    Check_Err(errNum, CL_SUCCESS);

    errNum = clEnqueueReadBuffer(comman_queue, this->vValues, CL_FALSE, 0,
                                 sizeof(VertexValueType) * g.vCount,
                                 vValues, 0, NULL, &readDone);
    Check_Err(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
    errNum = clEnqueueReadBuffer(comman_queue, this->vSet, CL_FALSE, 0,
                                 sizeof(Vertex) * vCount,
                                 vSet, 0, NULL, &readDone);
    Check_Err(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
}

template <typename VertexValueType, typename MessageValueType>
int PageRankCL<VertexValueType, MessageValueType>::MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, MessageValueType *mValues)
{
    //MSGGenMerge_array(g.vCount, g.eCount, &g.vList[0], &g.eList[0], 0, &initVSet[0], &g.verticesValue[0], mValues);

    Vertex *vSet = const_cast<Vertex *> vSet;
    Edge *eSet = const_cast<Edge *> eSet;
    Buffer_alloc(vSet, eSet, this->numOfInitV, vValues, mValues, vCount, eCount, 0);
    // Buffer_alloc(g.vList.data(), g.eList.data(), this->numOfInitV, g.verticesValue.data(), mValues, g.vCount,
    //             g.eCount, 0);
    MSGInitial_array_kernel_1 = clCreateKernel(program, "MSGInitial_array_kernel_1", &errNum);
    errNum |= clSetKernelArg(MSGInitial_array_kernel_1, 0, sizeof(cl_mem), &this->mValues);
    errNum |= clSetKernelArg(MSGInitial_array_kernel_1, 1, sizeof(int), &vCount);
    Check_Err(errNum, CL_SUCCESS);
    errNum = clEnqueueNDRangeKernel(comman_queue, MSGInitial_array_kernel_1,
                                    1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    Check_Err(errNum, CL_SUCCESS);

    MSGGenMerge_array_CL_kernel = clCreateKernel(program, "MSGGenMerge_array_CL", &errNum);
    Check_Err(errNum, CL_SUCCESS);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 0, sizeof(cl_mem), &this->vSet);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 1, sizeof(cl_mem), &this->eSet);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 2, sizeof(cl_mem), &this->vValues);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 3, sizeof(cl_mem), &this->mValues);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 4, sizeof(int), &vCount);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 5, sizeof(int), &eCount);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 6, sizeof(int), &this->numOfInitV);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 7, sizeof(int), &this->deltaThreshold);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 8, sizeof(int), &this->resetProb);
    Check_Err(errNum, CL_SUCCESS);

    errNum = clEnqueueNDRangeKernel(comman_queue, MSGGenMerge_array_CL_kernel,
                                    1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    Check_Err(errNum, CL_SUCCESS);
    errNum = clEnqueueReadBuffer(comman_queue, this->mValues, CL_FALSE, 0,
                                 sizeof(MessageValueType) * g.vCount,
                                 mValues, 0, NULL, &readDone);
    clWaitForEvents(1, &readDone);
    Check_Err(errNum, CL_SUCCESS);
}

template <typename VertexValueType, typename MessageValueType>
int PageRankCL<VertexValueType, MessageValueType>::MSGApply_CL(Graph<VertexValueType> &g, const std::vector<int> &initVSet,
                                                               std::set<int> &activeVertice,
                                                               const MessageSet<MessageValueType> &mSet)
{
    //Availability check
    if (g.eCount <= 0 || g.vCount <= 0)
        return 0;

    auto msgSize = mSet.mSet.size();

    //mValues init
    MessageValueType *mValues = new MessageValueType[msgSize];

    for (int i = 0; i < msgSize; i++)
    {
        mValues[i] = mSet.mSet.at(i).value;
    }

    int avCount;
    bool flag_apply = false;
    if (flag_apply)
    {
        avCount = MSGApply_array(g.vCount, msgSize, &g.vList[0], 0, &initVSet[0], &g.verticesValue[0], mValues);
    }
    else
    {
        //array form computation
        Buffer_alloc(g.vList.data(), g.eList.data(), this->numOfInitV, g.verticesValue.data(), mValues, g.vCount,
                     g.eCount, 1);
        MSGInitial_array_kernel_2 = clCreateKernel(program, "MSGInitial_array_kernel_2", &errNum);
        Check_Err(errNum, CL_SUCCESS);
        errNum |= clSetKernelArg(MSGInitial_array_kernel_2, 0, sizeof(cl_mem), &this->vSet);
        errNum |= clSetKernelArg(MSGInitial_array_kernel_2, 1, sizeof(int), &g.vCount);
        Check_Err(errNum, CL_SUCCESS);
        errNum = clEnqueueNDRangeKernel(comman_queue, MSGInitial_array_kernel_2,
                                        1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        Check_Err(errNum, CL_SUCCESS);

        MSGApply_array_kernel = clCreateKernel(program, "MSGApply_array_CL", &errNum);
        Check_Err(errNum, CL_SUCCESS);
        errNum |= clSetKernelArg(MSGApply_array_kernel, 0, sizeof(cl_mem), &this->vSet);
        errNum |= clSetKernelArg(MSGApply_array_kernel, 1, sizeof(cl_mem), &this->eSet);
        errNum |= clSetKernelArg(MSGApply_array_kernel, 2, sizeof(cl_mem), &this->vValues);
        errNum |= clSetKernelArg(MSGApply_array_kernel, 3, sizeof(cl_mem), &this->mValues);
        errNum |= clSetKernelArg(MSGApply_array_kernel, 4, sizeof(int), &g.vCount);
        errNum |= clSetKernelArg(MSGApply_array_kernel, 5, sizeof(int), &g.eCount);
        errNum |= clSetKernelArg(MSGApply_array_kernel, 6, sizeof(int), &this->numOfInitV);
        Check_Err(errNum, CL_SUCCESS);
        errNum = clEnqueueNDRangeKernel(comman_queue, MSGApply_array_kernel,
                                        1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        Check_Err(errNum, CL_SUCCESS);

        errNum = clEnqueueReadBuffer(comman_queue, this->vValues, CL_FALSE, 0,
                                     sizeof(VertexValueType) * g.vCount,
                                     g.verticesValue.data(), 0, NULL, &readDone);
        Check_Err(errNum, CL_SUCCESS);
        clWaitForEvents(1, &readDone);
        errNum = clEnqueueReadBuffer(comman_queue, this->vSet, CL_FALSE, 0,
                                     sizeof(Vertex) * g.vCount,
                                     g.vList.data(), 0, NULL, &readDone);
        Check_Err(errNum, CL_SUCCESS);
        clWaitForEvents(1, &readDone);

        avCount = 10;
    }

    //test
    //    std::cout << "=============apply info================" << std::endl;
    //    for(int i = 0; i < g.vCount; i++)
    //    {
    //        std::cout << g.vList[i].isActive << " " << g.verticesValue[i].first << " " << g.verticesValue[i].second << std::endl;
    //    }
    //    std::cout << "================end====================" << std::endl;

    //test
    //    std::cout << avCount << std::endl;

    delete[] mValues;

    return avCount;
}

template <typename VertexValueType, typename MessageValueType>
int PageRankCL<VertexValueType, MessageValueType>::MSGGenMerge_CL(Graph<VertexValueType> &g,
                                                                  const std::vector<int> &initVSet,
                                                                  std::set<int> &activeVertice,
                                                                  MessageSet<MessageValueType> &mSet)
{
    //Availability check
    if (g.eCount <= 0 || g.vCount <= 0)
        return 0;

    //mValues init
    MessageValueType *mValues = new MessageValueType[g.vCount];
    int msgCnt;

    bool flag_merge = false;
    if (flag_merge)
    {
        msgCnt = MSGGenMerge_array(g.vCount, g.eCount, &g.vList[0], &g.eList[0], 0, &initVSet[0], &g.verticesValue[0], mValues);
    }
    else
    {
        Buffer_alloc(g.vList.data(), g.eList.data(), this->numOfInitV, g.verticesValue.data(), mValues, g.vCount,
                     g.eCount, 0);
        MSGInitial_array_kernel_1 = clCreateKernel(program, "MSGInitial_array_kernel_1", &errNum);
        errNum |= clSetKernelArg(MSGInitial_array_kernel_1, 0, sizeof(cl_mem), &this->mValues);
        errNum |= clSetKernelArg(MSGInitial_array_kernel_1, 1, sizeof(int), &g.vCount);
        Check_Err(errNum, CL_SUCCESS);
        errNum = clEnqueueNDRangeKernel(comman_queue, MSGInitial_array_kernel_1,
                                        1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        Check_Err(errNum, CL_SUCCESS);

        MSGGenMerge_array_CL_kernel = clCreateKernel(program, "MSGGenMerge_array_CL", &errNum);
        Check_Err(errNum, CL_SUCCESS);
        errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 0, sizeof(cl_mem), &this->vSet);
        errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 1, sizeof(cl_mem), &this->eSet);
        errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 2, sizeof(cl_mem), &this->vValues);
        errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 3, sizeof(cl_mem), &this->mValues);
        errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 4, sizeof(int), &g.vCount);
        errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 5, sizeof(int), &g.eCount);
        errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 6, sizeof(int), &this->numOfInitV);
        errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 7, sizeof(int), &this->deltaThreshold);
        errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 8, sizeof(int), &this->resetProb);
        Check_Err(errNum, CL_SUCCESS);

        errNum = clEnqueueNDRangeKernel(comman_queue, MSGGenMerge_array_CL_kernel,
                                        1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
        Check_Err(errNum, CL_SUCCESS);
        errNum = clEnqueueReadBuffer(comman_queue, this->mValues, CL_FALSE, 0,
                                     sizeof(MessageValueType) * g.vCount,
                                     mValues, 0, NULL, &readDone);
        clWaitForEvents(1, &readDone);
        Check_Err(errNum, CL_SUCCESS);

        msgCnt = g.vCount;
    }

    //Generate merged msgs directly
    mSet.mSet.clear();
    mSet.mSet.reserve(msgCnt);

    for (int i = 0; i < msgCnt; i++)
    {
        mSet.insertMsg(Message<MessageValueType>(0, mValues[i].destVId, mValues[i]));
    }

    //test
    //    std::cout << "=============msg info============" << std::endl;
    //    for(int i = 0; i < msgCnt; i++)
    //    {
    //        std::cout << mValues[i].destVId << " " << mValues[i].rank << std::endl;
    //    }
    //    std::cout << "===============end===============" << std::endl;

    delete[] mValues;

    return msgCnt;
}

template <typename VertexValueType, typename MessageValueType>
void PageRankCL<VertexValueType, MessageValueType>::ApplyStep(Graph<VertexValueType> &g, const std::vector<int> &initVSet,
                                                              std::set<int> &activeVertices)
{
    MessageSet<MessageValueType> mMergedSet = MessageSet<MessageValueType>();

    mMergedSet.mSet.clear();

    auto start = std::chrono::system_clock::now();
    MSGGenMerge_CL(g, initVSet, activeVertices, mMergedSet);
    auto mergeEnd = std::chrono::system_clock::now();

    MSGApply_CL(g, initVSet, activeVertices, mMergedSet);
    auto applyEnd = std::chrono::system_clock::now();
}

template <typename VertexValueType, typename MessageValueType>
void PageRankCL<VertexValueType, MessageValueType>::ApplyD_CL(Graph<VertexValueType> &g, const std::vector<int> &initVList,
                                                              int partitionCount)
{
    //Init the Graph
    std::set<int> activeVertice = std::set<int>();

    std::vector<std::set<int>> AVSet = std::vector<std::set<int>>();
    for (int i = 0; i < partitionCount; i++)
        AVSet.push_back(std::set<int>());
    std::vector<MessageSet<MessageValueType>> mGenSetSet = std::vector<MessageSet<MessageValueType>>();
    for (int i = 0; i < partitionCount; i++)
        mGenSetSet.push_back(MessageSet<MessageValueType>());
    std::vector<MessageSet<MessageValueType>> mMergedSetSet = std::vector<MessageSet<MessageValueType>>();
    for (int i = 0; i < partitionCount; i++)
        mMergedSetSet.push_back(MessageSet<MessageValueType>());

    Init(g.vCount, g.eCount, initVList.size());

    GraphInit(g, activeVertice, initVList);

    Deploy(g.vCount, g.eCount, initVList.size());

    int iterCount = 0;

    bool isActive = true;

    while (isActive)
    {
        isActive = false;

        std::cout << "iterCount: " << iterCount << std::endl;
        auto start = std::chrono::system_clock::now();
        auto subGraphSet = this->DivideGraphByEdge(g, partitionCount);
        auto divideGraphFinish = std::chrono::system_clock::now();

        for (int i = 0; i < partitionCount; i++)
            ApplyStep(subGraphSet.at(i), initVList, AVSet.at(i));

        activeVertice.clear();

        auto mergeGraphStart = std::chrono::system_clock::now();
        MergeGraph(g, subGraphSet, activeVertice, AVSet, initVList);
        iterCount++;
        auto end = std::chrono::system_clock::now();

        for (int i = 0; i < g.vCount; i++)
        {
            if (g.vList.at(i).isActive)
            {
                isActive = true;
                break;
            }
        }

        //test
        //        for(int i = 0; i < g.vCount; i++)
        //        {
        //            //std::cout << "outdegree: " << g.vList.at(i).outDegree << std::endl;
        //            std::cout << i << " " << g.verticesValue.at(i).first << " " << g.verticesValue.at(i).second << std::endl;
        //        }

        //time test
        std::cout << "merge time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - mergeGraphStart).count() << std::endl;
    }

    for (int i = 0; i < g.vCount; i++)
    {
        std::cout << i << " " << g.verticesValue.at(i).first << " " << g.verticesValue.at(i).second << std::endl;
    }

    Free();
}

template <typename VertexValueType, typename MessageValueType>
void PageRankCL<VertexValueType, MessageValueType>::Free_little()
{
    errNum = CL_SUCCESS;

    errNum |= clReleaseMemObject(hostESet);

    errNum |= clReleaseMemObject(hostMValues);

    errNum |= clReleaseMemObject(hostVSet);

    errNum |= clReleaseMemObject(hostVValues);

    errNum |= clReleaseMemObject(vSet);

    errNum |= clReleaseMemObject(eSet);

    errNum |= clReleaseMemObject(mValues);

    errNum |= clReleaseMemObject(vValues);

    Check_Err(errNum, CL_SUCCESS);
}

template <typename VertexValueType, typename MessageValueType>
void PageRankCL<VertexValueType, MessageValueType>::MergeGraph(Graph<VertexValueType> &g,
                                                               const std::vector<Graph<VertexValueType>> &subGSet,
                                                               std::set<int> &activeVertices,
                                                               const std::vector<std::set<int>> &activeVerticeSet,
                                                               const std::vector<int> &initVList)
{
    PageRank<VertexValueType, MessageValueType>::MergeGraph(g, subGSet, activeVertices, activeVerticeSet, initVList);
}

template <typename VertexValueType, typename MessageValueType>
void PageRankCL<VertexValueType, MessageValueType>::checkErrorFileLine(int errNum, int expected, const char *file, const int lineNumber)
{
    if (errNum != expected)
    {
        std::cout << "\nCheck Error:" << std::endl;
        // std::cerr << "Line : " << lineNumber << " in File_ " << file << std::endl;
        std::cout << "Line : " << lineNumber << " in File_ " << file << std::endl;
        std::cerr << "Line : " << lineNumber << " in File_ " << file << std::endl;
        exit(1);
    }
}

template <typename VertexValueType, typename MessageValueType>
cl_device_id PageRankCL<VertexValueType, MessageValueType>::getMaxFlopsDev(cl_context cxGPUContext)
{
    size_t szParmDataBytes;
    cl_device_id *cdDevices;

    // get the list of GPU devices associated with context
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL,
                     &szParmDataBytes);
    cdDevices = (cl_device_id *)malloc(szParmDataBytes);
    size_t device_count = szParmDataBytes / sizeof(cl_device_id);

    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes,
                     cdDevices, NULL);

    cl_device_id max_flops_device = cdDevices[0];
    int max_flops = 0;

    size_t current_device = 0;

    // CL_DEVICE_MAX_COMPUTE_UNITS
    cl_uint compute_units;
    clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(compute_units), &compute_units, NULL);

    // CL_DEVICE_MAX_CLOCK_FREQUENCY
    cl_uint clock_frequency;
    clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_CLOCK_FREQUENCY,
                    sizeof(clock_frequency), &clock_frequency, NULL);

    max_flops = compute_units * clock_frequency;
    ++current_device;

    while (current_device < device_count)
    {
        // CL_DEVICE_MAX_COMPUTE_UNITS
        cl_uint compute_units;
        clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(compute_units), &compute_units, NULL);

        // CL_DEVICE_MAX_CLOCK_FREQUENCY
        cl_uint clock_frequency;
        clGetDeviceInfo(cdDevices[current_device],
                        CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency),
                        &clock_frequency, NULL);

        int flops = compute_units * clock_frequency;
        if (flops > max_flops)
        {
            max_flops = flops;
            max_flops_device = cdDevices[current_device];
        }
        ++current_device;
    }

    free(cdDevices);

    return max_flops_device;
}