#include "BellmanFordCL.h"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
//#include <pthread.h>
#include <float.h>
#include <ctime>

#define NULLMSG -1
#define Check_Err(m, n) checkErrorFileLine(a, b, __FILE__, __LINE__)
#define GetMaxPerDev(clcontext) getMaxFlopsDev(clcontext)

int roundWorkSize(int, int);

int roundWorkSize(int group_size, int total_size)
{
    int rem = total_size % group_size;
    if (rem == 0)
        return total_size;
    else
        return total_size + group_size - rem;
}

template <typename VertexValueType, typename MessageValueType>
BellmanFordCL<VertexValueType, MessageValueType>::BellmanFordCL()
{
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFordCL<VertexValueType, MessageValueType>::
    loadAndBuildProgram(cl_context context, const char *file_name)
{
    std::ifstream kernelFile(file_name, std::ios::in);
    Check_Err(kernelFile.is_open(), true);

    std::ostream osm;
    osm << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *sourceFile = srcStdStr.c_str();
    Check_Err((sourceFile != NULL), true);
    this->program = clCreateProgramWithSource(context, 1, (const char **)sourceFile, NULL, &errNum);
    clBuildProgram(this->program, 0, NULL, NULL, NULL, NULL);
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFordCL<VertexValueType, MessageValueType>::
    Init(int vCount, int eCount, int numOfInitV)
{
    BellmanFord<VertexValueType, MessageValueType>::Init(vCount, eCount, numOfInitV);

    this->vertexLimit = VERTEXSCALEINGPU;
    this->mPerMSGSet = MSGSCALEINGPU;
    this->ePerEdgeSet = EDGESCALEINGPU;

    cl_int numPlatforms;
    errNum = clGetPlatformIDs(1, &this->platform, &numPlatforms);
    std::cout << "Number of OpenCL Platforms: " << numPlatforms << std::endl;

    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cout << "Failed to find any OpenCL platforms." << std::endl;
        return 1;
    }

    gpu_context = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
    // cpu_contxt = clCreateContextFromType(0, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
    if (this->GPU_isOrNot)
    {
        if (this->MutliGPU_isOrNot)
        {
            size_t deviceBytes;
            cl_int errNum = clGetDeviceInfo(gpu_context, CL_CONTEXT_DEVICES, 0, null, &deviceBytes);
            Check_Err(errNum, CL_SUCCESS);
            this->device_count = (cl_uint)deviceBytes / sizeof(cl_device_id);
            auto this->cl_device_array = new CL_DEVICE[this->device_count];
            //
        }
        else
        {
            cl_device_id device_id = GetMaxPerDev(gpu_context);
            this->comman_queue = clCreateCommandQueue(gpu_context, device_id, 0, &errNum);
            loadAndBuildProgram(gpu_context, "kernel_src/BellmanFordCL_kernel.cl");
            size_t max_workgroup_size;
            clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup_size, NULL);
            std::cout << "max_group_size : " << max_workgroup_size << std::endl;

            size_t local_work_size = max_workgroup_size;
            size_t global_work_size = roundWorkSize(local_work_size, vCount);
        }
    }
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFordCL<VertexValueType, MessageValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList)
{
    BellmanFord<VertexValueType, MessageValueType>::GraphInit(g, activeVertices, initVList);
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFordCL<VertexValueType, MessageValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{
    BellmanFord<VertexValueType, MessageValueType>::Deploy(vCount, eCount, numOfInitV);

    kernel = clCreateKernel(this->program, "BellmanFordCLKernel", &errNum);
    Check_Err(errNum, CL_SUCCESS);

    // errNum = CL_SUCCESS;

    this->initVSet = new int[numOfInitV];
    this->vValueSet = new VertexValueType[vCount * this->numOfInitV];
    int mSize = std::max(this->numOfInitV * ePerEdgeSet, mPerMSGSet);
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFordCL<VertexValueType, MessageValueType>::Buffer_alloc(const Vertex *vSet, const Edge *eSet,
                                                                    int numOfInitV, const int *initVSet,
                                                                    const VertexValueType *vValues,
                                                                    MessageValueType *mValues, const int vcount,
                                                                    const int ecount)
{

    hostVSet = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                              sizeof(Vertex) * vcount, vSet1, &errNum);
    Check_Err(errNum, CL_SUCCESS);
    hostESet = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                              sizeof(Edge) * ecount, eSet1, &errNum);
    Check_Err(errNum, CL_SUCCESS);
    host_initVSet = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                   sizeof(int), initVSet1, &errNum);
    Check_Err(errNum, CL_SUCCESS);
    hostVValues = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                 sizeof(VertexValueType) * numOfInitV * vcount, vValues1, &errNum);
    Check_Err(errNum, CL_SUCCESS);
    hostMValues = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                 sizeof(MessageValueType) * numOfInitV * vcount, mValues1, &errNum);
    Check_Err(errNum, CL_SUCCESS);

    vSet = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(Vertex) * vcount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    eSet = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, sizoef(Edge) * ecount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    initVSet = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, sizeof(int), NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    vValues = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(VertexValueType) * numOfInitV * vcount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);
    mValues = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(MessageValueType) * numOfInitV * vcount, NULL, &errNum);
    checkError(errNum, CL_SUCCESS);

    //

    errNum = clEnqueueCopyBuffer(comman_queue, hostVSet, vSet, 0, 0,
                                 sizeof(Vertex) * vcount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);

    errNum = clEnqueueCopyBuffer(comman_queue, hostESet, eSet, 0, 0,
                                 sizeof(Edge) * ecount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueCopyBuffer(comman_queue, host_initVSet, 0, 0,
                                 sizeof(int), NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueCopyBuffer(comman_queue, hostVValues, vValues, 0, 0,
                                 sizeof(VertexValueType) * numOfInitV * vcount, 0, NULL, NULL);
    checkError(errNum, CL_SUCCESS);
    errNum = clEnqueueCopyBuffer(comman_queue, hostMValues, 0, 0,
                                 sizeof(MessageValueType) * numOfInitV * vcount, NULL, NULL);
    checkError(errNum, CL_SUCCESS);

    //clReleaseMemObject();
}

template <typename VertexValueType, typename MessageValueType>
int BellmanFordCL<VertexValueType, MessageValueType>::MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice, const MessageSet<MessageValueType> &mSet)
{
    activeVertice.clear();
    if (g.vCount <= 0)
        reutn 0;
    MessageValueType *mValues = new MessageValueType[g.vCount * this->numOfInitV];

    for (int i = 0; i < g.vCount * this->numOfInitV; i++)
        mValues[i] = (MessageValueType)INVALID_MASSAGE;
    for (int i = 0; i < mSet.mSet.size(); i++)
    {
        auto &mv = mValues[mSet.mSet.at(i).dst * this->numOfInitV + g.vList.at(mSet.mSet.at(i).src).initVIndex];
        if (mv > mSet.mSet.at(i).value)
            mv = mSet.mSet.at(i).value;
    }

    cl_kernel MSGApply_array_kernel;
    MSGApply_array_kernel = clCreateKernel(program, "MSGApply_array_kernel", &errNum);
    Check_Err(errNum, CL_SUCCESS);

    errNum |= clSetKernelArg(MSGApply_array_kernel, 0, sizeof(cl_mem), &this->vSet);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 1, sizeof(cl_mem), &this->eSet);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 2, sizeof(cl_mem), &this->initVSet);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 3, sizeof(cl_mem), &this->vValues);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 4, sizeof(cl_mem), &this->mValues);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 5, sizeof(int), &g.vCount);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 6, sizeof(int), &g.eCount);
    checkError(errNum, CL_SUCCESS);

    errNum = clEnqueueNDRangeKernel(comman_queue, MSGApply_array_kernel,
                                    1, 0, &1, &1, 0, NULL, NULL);
    Check_Err(errNum, CL_SUCCESS);
    //    //array form computation
    //     MSGApply_array(g.vCount, g.eCount, &g.vList[0], this->numOfInitV, &initVSet[0], &g.verticesValue[0], mValues);
    errNum = clEnqueueReadBuffer(comman_queue, this->vValues, CL_FALSE, 0,
                                 sizeof(VertexValueType) * this->numOfInitV * g.vCount,
                                 &g.verticesValue[0], 0, NULL, &readDone);
    Check_Err(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
    errNum = clEnqueueReadBuffer(comman_queue, this->vSet, CL_FALSE, 0,
                                 sizeof(MessageValueType) * g.vCount,
                                 &g.vList[0], 0, NULL, &readDone);
    Check_Err(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);

    //Active vertices set assembly
    for (int i = 0; i < g.vCount; i++)
    {
        if (g.vList.at(i).isActive)
            activeVertice.insert(i);
    }

    free(mValues);

    return activeVertice.size();
}
template <typename VertexValueType, typename MessageValueType>
int BellmanFordCL<VertexValueType, MessageValueType>::MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<MessageValueType> &mSet)
{
    auto &mSet = mMergedSet;
    if (g.vCount <= 0)
        break;
    MessageValueType *mValues = new MessageValueType[g.vCount * this->numOfInitV];

    Buffer_alloc(&g.vList[0], &g.eList[0], this->numOfInitV, &initVSet[0], &g.verticesValue[0], mValues, g.vCount, g.eCount);
    //  void Buffer_alloc(const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, MessageValueType *mValues);

    cl_kernel MSGGenMerge_array_CL_kernel;
    MSGGenMerge_array_CL_kernel = clCreateKernel(program, "MSGGenMerge_array_CL", &errNum);
    Check_Err(errNum, CL_SUCCESS);

    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 0, sizeof(cl_mem), &this->vSet);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 1, sizeof(cl_mem), &this->eSet);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 2, sizeof(cl_mem), &this->initVSet);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 3, sizeof(cl_mem), &this->vValues);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 4, sizeof(cl_mem), &this->mValues);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 5, sizeof(int), &g.vCount);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 6, sizeof(int), &g.eCount);
    checkError(errNum, CL_SUCCESS);
    //MSGGenMerge_array(g.vCount, g.eCount, &g.vList[0], &g.eList[0], this->numOfInitV, &initVSet[0], &g.verticesValue[0], mValues)
    errNum = clEnqueueNDRangeKernel(comman_queue, MSGGenMerge_array_CL_kernel,
                                    1, 0, &1, &1, 0, NULL, NULL);
    Check_Err(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);

    /*
    errNum = clEnqueueReadBuffer(comman_queue, this->mValues, CL_FALSE, 0,
                                 sizeof(MessageValueType) * numOfInitV * g.vCount,
                                 &this->hostMValues, 0, NULL, &readDone);
                                
   Check_Err(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
*/

    errNum = clEnqueueReadBuffer(comman_queue, this->mValues, CL_FALSE, 0,
                                 sizeof(MessageValueType) * this->numOfInitV * g.vCount,
                                 &mValues, 0, NULL, &readDone);
    Check_Err(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);

    for (int i = 0; i < g.vCount * this->numOfInitV; i++)
    {
        if (mValues[i] != (MessageValueType)INVALID_MASSAGE)
        {
            int dst = i / this->numOfInitV;
            int initV = initVSet[i % this->numOfInitV];
            mSet.insertMsg(Message<MessageValueType>(initV, dst, mValues[i]));
        }
    }
    free(mValues);
    return mSet.mSet.size();
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFordCL<VertexValueType, MessageValueType>::Free()
{
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFordCL<VertexValueType, MessageValueType>::ApplyD_CL(Graph<VertexValueType> &g, const std::vector<int> &initVList, int partitionCount)
{
    std::set<int> activeVertices = {};
    std::vector<std::set<int>> AVSet = {};
    auto mGenSetSet = std::vector<MessageSet<MessageValueType>>();
    auto mMergedSetSet = std::vector<MessageSet<MessageValueType>>();
    for (int i = 0; i < partitionCount; i++)
    {
        AVSet.push_back(std::set<int>());
        mGenSetSet.push_back(MessageSet<MessageValueType>());
        mMergedSetSet.push_back(MessageSet<MessageValueType>());
    }
    Init(g.vCount, g.eCount, initVList.size());
    GraphInit(g, activeVertices, initVList);
    Deploy(g.vCount, g.eCount, initVList.size());

    int itCount = 0;
    while (activeVertices.size() > 0)
    {
        //Test
        std::cout << ++iterCount << ":" << clock() << std::endl;
        //Test end

        auto subGraphSet = this->DivideGraphByEdge(g, partitionCount);

        for (auto &elem : AVSet)
        {
            elem.clear();
            elem = activeVertices;
        }

        //Test
        std::cout << "GDivide:" << clock() << std::endl;
        //Test end

        for (int i = 0; i < partitionCount; i++)
        {
        ApplyStep:
            // ApplyStep(subGraphSet.at(i), initVList, AVSet.at(i));
            auto &g = subGraphSet.at(i);
            auto &activeVertices = AVSet.at(i);
            auto mGenSet = MessageSet<MessageValueType>();
            auto mMergedSet = MessageSet<MessageValueType>();
            mMergedSet.mSet.clear();

            MSGGenMerge(g, initVSet, activeVertices, mMergedSet);

            //Test
            std::cout << "MGenMerge:" << clock() << std::endl;
            //Test end
            activeVertices.clear();

            MSGApply(g, initVSet, activeVertices, mMergedSet);

            //Test
            std::cout << "Apply:" << clock() << std::endl;
            //Test end
        }

        activeVertices.clear();

        MergeGraph(g, subGraphSet, activeVertices, AVSet, initVList);
        //Test
        std::cout << "GMerge:" << clock() << std::endl;
        //Test end
    }
    Free();

    //Test
    std::cout << "end:" << clock() << std::endl;
    //Test end
}

void checkErrorFileLine(int errNum, int expected, const char *file, const int lineNumber)
{
    if (errNum != expected)
    {
        std::cerr << "Line : " << lineNumber << " in File_ " << std::endl;
        exit(1);
    }
}

cl_device_id getMaxFlopsDev(cl_context cxGPUContext)
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
