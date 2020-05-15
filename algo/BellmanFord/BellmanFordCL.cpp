#include "BellmanFordCL.h"

#define NULLMSG -1
#define Check_Err(m, n) checkErrorFileLine(m, n, __FILE__, __LINE__)
#define GetMaxPerDev(clcontext) getMaxFlopsDev(clcontext)

#define DEBUG_INFO std::cout << "DEBUG_INFO::Line : " << __LINE__ << " in File : " << __FILE__ << std::endl;

#define LOG(message) logError(__LINE__, message);

int logError(int line, const std::string &message)
{
    std::cerr << "[" << line << "]" << message << std::endl;
}

int roundWorkSize(int group_size, int total_size)
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
BellmanFordCL<VertexValueType, MessageValueType>::BellmanFordCL()
{
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFordCL<VertexValueType, MessageValueType>::
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
void BellmanFordCL<VertexValueType, MessageValueType>::
    Init(int vCount, int eCount, int numOfInitV)
{
    BellmanFord<VertexValueType, MessageValueType>::Init(vCount, eCount, numOfInitV);

    this->vertexLimit = VERTEXSCALEINGPU;
    this->mPerMSGSet = MSGSCALEINGPU;
    this->ePerEdgeSet = EDGESCALEINGPU;

    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        LOG("Failed to find any OpenCL platforms.")
    }
    std::cout << "\tNumber of OpenCL Platforms: \t" << numPlatforms << std::endl;

    platformIds = (cl_platform_id *)alloca(sizeof(cl_platform_id) * numPlatforms);
    errNum = clGetPlatformIDs(numPlatforms, platformIds, NULL);

    // errNum = clGetPlatformIDs(1, &platform, &numPlatforms);

    for (cl_uint i = 0; i < numPlatforms; i++)
    {
        displayPlatformInfo(platformIds[i], CL_PLATFORM_PROFILE, "CL_PLATFORM_PROFILE");
        displayPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, "CL_PLATFORM_VERSION");
        displayPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");
        displayPlatformInfo(platformIds[i], CL_PLATFORM_EXTENSIONS, "CL_PLATFORM_EXTENSIONS");
    }

    errNum = clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, 0, NULL, &device_count);
    devices = (cl_device_id *)alloca(sizeof(cl_device_id) * device_count);
    errNum = clGetDeviceIDs(platformIds[0], CL_DEVICE_TYPE_ALL, device_count, devices, NULL);
    std::cout << "\tNumber of devices in platform[0]: \t" << device_count << std::endl;
    for (cl_uint j = 0; j < device_count; j++)
    {
        displayDeviceInfo<cl_device_type>(devices[j], CL_DEVICE_TYPE, "CL_DEVICE_TYPE");
    }

    cl_context_properties context_properties[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIds[0],
        (cl_context_properties)NULL};

    // gpu_context = clCreateContext(0, 1, device_count, NULL, NULL, &errNum);
    gpu_context = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        LOG("No GPU devices found.")
    }

    if (this->GPU_isOrNot)
    {
        if (this->MutliGPU_isOrNot)
        {
            size_t deviceBytes;
            cl_int errNum = clGetContextInfo(gpu_context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBytes);
            Check_Err(errNum, CL_SUCCESS);
            this->device_count = (cl_uint)deviceBytes / sizeof(cl_device_id);
        }
        else
        {
            cl_device_id device_id = GetMaxPerDev(gpu_context);
            this->comman_queue = clCreateCommandQueue(gpu_context, device_id, 0, &errNum);

            loadAndBuildProgram(gpu_context, "../Graph_Algo/algo/BellmanFord/BellmanFordCL_kernel.cl");
            size_t max_workgroup_size;
            clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup_size, NULL);
            std::cout << "max_group_size : " << max_workgroup_size << std::endl;

            //local_work_size = max_workgroup_size;
            // modified:  apply to some small numbers.

            //local_work_size = max_workgroup_size;
            //global_work_size = roundWorkSize(local_work_size, vCount);
            local_work_size = 1;
            global_work_size = 1;
        }
    }
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFordCL<VertexValueType, MessageValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices,
                                                                 const std::vector<int> &initVList)
{
    BellmanFord<VertexValueType, MessageValueType>::GraphInit(g, activeVertices, initVList);
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFordCL<VertexValueType, MessageValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{
    BellmanFord<VertexValueType, MessageValueType>::Deploy(vCount, eCount, numOfInitV);
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFordCL<VertexValueType, MessageValueType>::Buffer_alloc(Vertex *vSet1, Edge *eSet1,
                                                                    int numOfInitV,
                                                                    VertexValueType *vValues1,
                                                                    MessageValueType *mValues1, int vcount,
                                                                    int ecount, bool isMidNot)
{
    std::cout << "vSET:" << std::endl;
    for (int i = 0; i < vcount; i++)
    {
        std::cout << "vset[" << i << "]:" << vSet1[i].initVIndex
                  << "," << vSet1[i].isActive << "," << vSet1[i].vertexID << std::endl;
    }

    std::cout << "eSET:" << std::endl;
    for (int i = 0; i < ecount; i++)
    {
        std::cout << "eset[" << i << "]:" << eSet1[i].dst << ","
                  << eSet1[i].src << "," << eSet1[i].weight << std::endl;
    }
    std::cout << "vValues:" << std::endl;
    for (int i = 0; i < numOfInitV * vcount; i++)
    {
        std::cout << "vValues[" << i << "]:" << vValues1[i] << std::endl;
    }
    std::cout << "vcount:" << vcount << std::endl;
    std::cout << "ecount:" << ecount << std::endl;
    std::cout << "isMidOrNot:" << isMidNot << std::endl;

    DEBUG_INFO
    hostVSet = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                              sizeof(Vertex) * vcount, vSet1, &errNum);
    Check_Err(errNum, CL_SUCCESS);
    hostESet = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                              sizeof(Edge) * ecount, eSet1, &errNum);
    Check_Err(errNum, CL_SUCCESS);
    hostVValues = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                 sizeof(VertexValueType) * numOfInitV * vcount, vValues1, &errNum);
    Check_Err(errNum, CL_SUCCESS);
    hostMValues = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                 sizeof(MessageValueType) * numOfInitV * vcount, mValues1, &errNum);
    Check_Err(errNum, CL_SUCCESS);
    DEBUG_INFO
    std::cout << "\nvSet1: " << vSet << std::endl;
    if (!isMidNot)
    {
        DEBUG_INFO
        vSet = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(Vertex) * vcount, NULL, &errNum);
        //vSet = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(Vertex) * global_work_size, NULL);
        Check_Err(errNum, CL_SUCCESS);
        eSet = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, sizeof(Edge) * ecount, NULL, &errNum);
        Check_Err(errNum, CL_SUCCESS);
        vValues = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(VertexValueType) * numOfInitV * vcount, NULL,
                                 &errNum);
        Check_Err(errNum, CL_SUCCESS);
        mValues = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(MessageValueType) * numOfInitV * vcount, NULL,
                                 &errNum);
        Check_Err(errNum, CL_SUCCESS);
    }
    std::cout << "\nvSet2: " << vSet << std::endl;
    //
    DEBUG_INFO
    errNum = clEnqueueCopyBuffer(comman_queue, hostVSet, vSet, 0, 0,
                                 sizeof(Vertex) * vcount, 0, NULL, NULL);
    Check_Err(errNum, CL_SUCCESS);
    // opencl sizeof == sizeof(cpp class)
    DEBUG_INFO
    errNum = clEnqueueCopyBuffer(comman_queue, hostESet, eSet, 0, 0,
                                 sizeof(Edge) * ecount, 0, NULL, NULL);
    Check_Err(errNum, CL_SUCCESS);
    DEBUG_INFO
    errNum = clEnqueueCopyBuffer(comman_queue, hostVValues, vValues, 0, 0,
                                 sizeof(VertexValueType) * numOfInitV * vcount, 0, NULL, NULL);
    Check_Err(errNum, CL_SUCCESS);
    DEBUG_INFO
    errNum = clEnqueueCopyBuffer(comman_queue, hostMValues, mValues, 0, 0,
                                 sizeof(MessageValueType) * numOfInitV * vcount, 0, NULL, &copyDone);
    Check_Err(errNum, CL_SUCCESS);
    clWaitForEvents(1, &copyDone);
    DEBUG_INFO
}

template <typename VertexValueType, typename MessageValueType>
int BellmanFordCL<VertexValueType, MessageValueType>::MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet,
                                                               std::set<int> &activeVertice,
                                                               const MessageSet<MessageValueType> &mSet)
{
    activeVertice.clear();
    if (g.vCount <= 0)
        return 0;
    auto mValues = new MessageValueType[g.vCount * this->numOfInitV];

    for (int i = 0; i < g.vCount * this->numOfInitV; i++)
        mValues[i] = (MessageValueType)INVALID_MASSAGE;
    for (int i = 0; i < mSet.mSet.size(); i++)
    {
        int mValues_ind = mSet.mSet.at(i).dst * this->numOfInitV + g.vList.at(mSet.mSet.at(i).src).initVIndex;
        auto &mv = mValues[mValues_ind];
        if (mv > mSet.mSet.at(i).value)
            mv = mSet.mSet.at(i).value;
    }
    std::cout << "MSGAPPLY:\n"
              << std::endl;
    Buffer_alloc(g.vList.data(), g.eList.data(), this->numOfInitV, g.verticesValue.data(), mValues, g.vCount,
                 g.eCount, true);

    MSGApply_array_kernel = clCreateKernel(program, "MSGApply_array", &errNum);
    Check_Err(errNum, CL_SUCCESS);

    errNum |= clSetKernelArg(MSGApply_array_kernel, 0, sizeof(cl_mem), &this->vSet);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 1, sizeof(cl_mem), &this->eSet);
    //  errNum |= clSetKernelArg(MSGApply_array_kernel, 2, sizeof(cl_mem), &this->initVSet);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 2, sizeof(cl_mem), &this->vValues);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 3, sizeof(cl_mem), &this->mValues);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 4, sizeof(int), &g.vCount);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 5, sizeof(int), &g.eCount);
    errNum |= clSetKernelArg(MSGApply_array_kernel, 6, sizeof(int), &this->numOfInitV);
    Check_Err(errNum, CL_SUCCESS);

    errNum = clEnqueueNDRangeKernel(comman_queue, MSGApply_array_kernel,
                                    1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    Check_Err(errNum, CL_SUCCESS);
    //    //array form computation
    //     MSGApply_array(g.vCount, g.eCount, &g.vList[0], this->numOfInitV, &initVSet[0], &g.verticesValue[0], mValues);
    errNum = clEnqueueReadBuffer(comman_queue, this->vValues, CL_FALSE, 0,
                                 sizeof(VertexValueType) * this->numOfInitV * g.vCount,
                                 g.verticesValue.data(), 0, NULL, &readDone);
    Check_Err(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
    errNum = clEnqueueReadBuffer(comman_queue, this->vSet, CL_FALSE, 0,
                                 sizeof(MessageValueType) * g.vCount,
                                 g.vList.data(), 0, NULL, &readDone);
    Check_Err(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);

    //Active vertices set assembly
    for (int i = 0; i < g.vCount; i++)
    {
        if (g.vList.at(i).isActive)
            activeVertice.insert(i);
    }

    delete[] mValues;

    return activeVertice.size();
}

template <typename VertexValueType, typename MessageValueType>
int BellmanFordCL<VertexValueType, MessageValueType>::MSGGenMerge_CL(Graph<VertexValueType> &g,
                                                                     std::vector<int> &initVSet,
                                                                     std::set<int> &activeVertice,
                                                                     MessageSet<MessageValueType> &mSet)
{

    if (g.vCount <= 0)
        return 0;
    MessageValueType *mValues1 = new MessageValueType[g.vCount * this->numOfInitV];
    DEBUG_INFO
    std::cout << "MSGGENMERGE_CL:\n"
              << std::endl;
    Buffer_alloc(g.vList.data(), g.eList.data(), this->numOfInitV, g.verticesValue.data(), mValues1, g.vCount,
                 g.eCount, false);
    //  void Buffer_alloc(const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, MessageValueType *mValues);

    DEBUG_INFO
    MSGGenMerge_array_CL_kernel = clCreateKernel(program, "MSGGenMerge_array_CL", &errNum);
    Check_Err(errNum, CL_SUCCESS);

    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 0, sizeof(cl_mem), &this->vSet);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 1, sizeof(cl_mem), &this->eSet);
    //  errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 2, sizeof(cl_mem), &this->initVSet);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 2, sizeof(cl_mem), &this->vValues);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 3, sizeof(cl_mem), &this->mValues);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 4, sizeof(int), &g.vCount);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 5, sizeof(int), &g.eCount);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 6, sizeof(int), &this->numOfInitV);
    Check_Err(errNum, CL_SUCCESS);
    DEBUG_INFO
    //MSGGenMerge_array(g.vCount, g.eCount, &g.vList[0], &g.eList[0], this->numOfInitV, &initVSet[0], &g.verticesValue[0], mValues)
    errNum = clEnqueueNDRangeKernel(comman_queue, MSGGenMerge_array_CL_kernel,
                                    1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    Check_Err(errNum, CL_SUCCESS);
    //   clWaitForEvents(1, &readDone);

    /*
    errNum = clEnqueueReadBuffer(comman_queue, this->mValues, CL_FALSE, 0,
                                 sizeof(MessageValueType) * numOfInitV * g.vCount,
                                 &this->hostMValues, 0, NULL, &readDone);
                                
   Check_Err(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
*/
    DEBUG_INFO
    errNum = clEnqueueReadBuffer(comman_queue, this->mValues, CL_FALSE, 0,
                                 sizeof(MessageValueType) * this->numOfInitV * g.vCount,
                                 mValues1, 0, NULL, &readDone);
    clWaitForEvents(1, &readDone);
    DEBUG_INFO
    Check_Err(errNum, CL_SUCCESS);

    for (int i = 0; i < g.vCount * this->numOfInitV; i++)
    {
        std::cout << "mValues1[" << i << "]:" << mValues1[i] << std::endl;
        //  if (mValues1[i] != (MessageValueType)INVALID_MASSAGE)
        if (mValues1[i] < (MessageValueType)(INVALID_MASSAGE - 1))
        // accuracy warning
        {
            DEBUG_INFO
            int dst = i / this->numOfInitV;
            int initV = initVSet[i % this->numOfInitV];
            mSet.insertMsg(Message<MessageValueType>(initV, dst, mValues1[i]));
            std::cout << "dst:" << dst << "\tinitV:" << initV << std::endl;
        }
    }
    DEBUG_INFO
    delete[] mValues1;
    return mSet.mSet.size();
}
template <typename VertexValueType, typename MessageValueType>
void BellmanFordCL<VertexValueType, MessageValueType>::Buffer_alloc1(Vertex *vSet1, Edge *eSet1,
                                                                     int numOfInitV,
                                                                     VertexValueType *vValues1,
                                                                     MessageValueType *mValues1, int vcount,
                                                                     int ecount, int flag)
{
    std::cout << "vSET:" << std::endl;
    for (int i = 0; i < vcount; i++)
    {
        std::cout << "vset[" << i << "]:" << vSet1[i].initVIndex
                  << "," << vSet1[i].isActive << "," << vSet1[i].vertexID << std::endl;
    }

    std::cout << "eSET:" << std::endl;
    for (int i = 0; i < ecount; i++)
    {
        std::cout << "eset[" << i << "]:" << eSet1[i].dst << ","
                  << eSet1[i].src << "," << eSet1[i].weight << std::endl;
    }
    std::cout << "vValues:" << std::endl;
    for (int i = 0; i < numOfInitV * vcount; i++)
    {
        std::cout << "vValues[" << i << "]:" << vValues1[i] << std::endl;
    }
    std::cout << "vcount:" << vcount << std::endl;
    std::cout << "ecount:" << ecount << std::endl;
    std::cout << "isMidOrNot:" << isMidNot << std::endl;
    if (flag == 0)
    {
        DEBUG_INFO
        hostVSet = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(Vertex) * vcount, vSet1, &errNum);
        Check_Err(errNum, CL_SUCCESS);
        hostESet = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                  sizeof(Edge) * ecount, eSet1, &errNum);
        Check_Err(errNum, CL_SUCCESS);
        hostVValues = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                     sizeof(VertexValueType) * numOfInitV * vcount, vValues1, &errNum);
        Check_Err(errNum, CL_SUCCESS);
        hostMValues = clCreateBuffer(this->gpu_context, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                     sizeof(MessageValueType) * numOfInitV * vcount, mValues1, &errNum);
        Check_Err(errNum, CL_SUCCESS);
        DEBUG_INFO

        std::cout << "\nvSet1: " << vSet << std::endl;
        DEBUG_INFO
        vSet = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(Vertex) * vcount, NULL, &errNum);
        //vSet = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(Vertex) * global_work_size, NULL);
        Check_Err(errNum, CL_SUCCESS);
        eSet = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, sizeof(Edge) * ecount, NULL, &errNum);
        Check_Err(errNum, CL_SUCCESS);
        vValues = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(VertexValueType) * numOfInitV * vcount, NULL,
                                 &errNum);
        Check_Err(errNum, CL_SUCCESS);
        mValues = clCreateBuffer(gpu_context, CL_MEM_READ_WRITE, sizeof(MessageValueType) * numOfInitV * vcount, NULL,
                                 &errNum);
        Check_Err(errNum, CL_SUCCESS);

        std::cout << "\nvSet2: " << vSet << std::endl;
        //
        DEBUG_INFO
        errNum = clEnqueueCopyBuffer(comman_queue, hostVSet, vSet, 0, 0,
                                     sizeof(Vertex) * vcount, 0, NULL, NULL);
        Check_Err(errNum, CL_SUCCESS);
        // opencl sizeof == sizeof(cpp class)
        DEBUG_INFO
        errNum = clEnqueueCopyBuffer(comman_queue, hostESet, eSet, 0, 0,
                                     sizeof(Edge) * ecount, 0, NULL, NULL);
        Check_Err(errNum, CL_SUCCESS);
        DEBUG_INFO
        errNum = clEnqueueCopyBuffer(comman_queue, hostVValues, vValues, 0, 0,
                                     sizeof(VertexValueType) * numOfInitV * vcount, 0, NULL, NULL);
        Check_Err(errNum, CL_SUCCESS);
        DEBUG_INFO
        errNum = clEnqueueCopyBuffer(comman_queue, hostMValues, mValues, 0, 0,
                                     sizeof(MessageValueType) * numOfInitV * vcount, 0, NULL, NULL);
        Check_Err(errNum, CL_SUCCESS);
        DEBUG_INFO
    }
    else if (flag == 1)
    {
        cl_mem hostMValues_temp = clCreateBuffer(this->gpu_context,
                                                 CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
                                                 sizeof(MessageValueType) * numOfInitV * vcount,
                                                 mValues1, &errNum);
        Check_Err(errNum, CL_SUCCESS);
        errNum = clEnqueueCopyBuffer(comman_queue, hostMValues_temp, mValues, 0, 0,
                                     sizeof(MessageValueType) * numOfInitV * vcount, 0, NULL, NULL);
        Check_Err(errNum, CL_SUCCESS);
        clReleaseMemObject(hostMValues_temp);
    }
}

template <typename VertexValueType, typename MessageValueType>
int BellmanFordCL<VertexValueType, MessageValueType>::MSGApply1(Graph<VertexValueType> &g, const std::vector<int> &initVSet,
                                                                std::set<int> &activeVertice,
                                                                const MessageSet<MessageValueType> &mSet)
{
    activeVertice.clear();
    if (g.vCount <= 0)
        return 0;
    auto mValues = new MessageValueType[g.vCount * this->numOfInitV];

    for (int i = 0; i < g.vCount * this->numOfInitV; i++)
        mValues[i] = (MessageValueType)INVALID_MASSAGE;
    for (int i = 0; i < mSet.mSet.size(); i++)
    {
        int mValues_ind = mSet.mSet.at(i).dst * this->numOfInitV + g.vList.at(mSet.mSet.at(i).src).initVIndex;
        auto &mv = mValues[mValues_ind];
        if (mv > mSet.mSet.at(i).value)
            mv = mSet.mSet.at(i).value;
    }
    std::cout << "MSGAPPLY:\n"
              << std::endl;
    Buffer_alloc1(g.vList.data(), g.eList.data(), this->numOfInitV, g.verticesValue.data(), mValues, g.vCount,
                  g.eCount, 1);

    MSGApply_array_kernel = clCreateKernel(program, "MSGApply_array", &errNum);
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
    //    //array form computation
    //     MSGApply_array(g.vCount, g.eCount, &g.vList[0], this->numOfInitV, &initVSet[0], &g.verticesValue[0], mValues);
    errNum = clEnqueueReadBuffer(comman_queue, this->vValues, CL_FALSE, 0,
                                 sizeof(VertexValueType) * this->numOfInitV * g.vCount,
                                 g.verticesValue.data(), 0, NULL, &readDone);
    Check_Err(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
    errNum = clEnqueueReadBuffer(comman_queue, this->vSet, CL_FALSE, 0,
                                 sizeof(MessageValueType) * g.vCount,
                                 g.vList.data(), 0, NULL, &readDone);
    Check_Err(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);

    //Active vertices set assembly
    for (int i = 0; i < g.vCount; i++)
    {
        if (g.vList.at(i).isActive)
            activeVertice.insert(i);
    }

    delete[] mValues;

    return activeVertice.size();
}

template <typename VertexValueType, typename MessageValueType>
int BellmanFordCL<VertexValueType, MessageValueType>::MSGGenMerge_CL1(Graph<VertexValueType> &g,
                                                                      std::vector<int> &initVSet,
                                                                      std::set<int> &activeVertice,
                                                                      MessageSet<MessageValueType> &mSet)
{

    if (g.vCount <= 0)
        return 0;
    MessageValueType *mValues1 = new MessageValueType[g.vCount * this->numOfInitV];
    DEBUG_INFO
    std::cout << "MSGGENMERGE_CL:\n"
              << std::endl;
    Buffer_alloc1(g.vList.data(), g.eList.data(), this->numOfInitV, g.verticesValue.data(), mValues1, g.vCount,
                  g.eCount, 0);
    //  void Buffer_alloc(const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, MessageValueType *mValues);

    DEBUG_INFO
    MSGInitial_array_kernel = clCreateKernel(program, "MSGInitial_array", &errNum);
    MSGGenMerge_array_CL_kernel = clCreateKernel(program, "MSGGenMerge_array_CL", &errNum);
    Check_Err(errNum, CL_SUCCESS);

    errNum |= clSetKernelArg(MSGInitial_array_kernel, 0, sizeof(cl_mem), &this->vSet);
    errNum |= clSetKernelArg(MSGInitial_array_kernel, 1, sizeof(cl_mem), &this->eSet);
    //  errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 2, sizeof(cl_mem), &this->initVSet);
    errNum |= clSetKernelArg(MSGInitial_array_kernel, 2, sizeof(cl_mem), &this->vValues);
    errNum |= clSetKernelArg(MSGInitial_array_kernel, 3, sizeof(cl_mem), &this->mValues);
    errNum |= clSetKernelArg(MSGInitial_array_kernel, 4, sizeof(int), &g.vCount);
    errNum |= clSetKernelArg(MSGInitial_array_kernel, 5, sizeof(int), &g.eCount);
    errNum |= clSetKernelArg(MSGInitial_array_kernel, 6, sizeof(int), &this->numOfInitV);
    Check_Err(errNum, CL_SUCCESS);

    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 0, sizeof(cl_mem), &this->vSet);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 1, sizeof(cl_mem), &this->eSet);
    //  errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 2, sizeof(cl_mem), &this->initVSet);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 2, sizeof(cl_mem), &this->vValues);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 3, sizeof(cl_mem), &this->mValues);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 4, sizeof(int), &g.vCount);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 5, sizeof(int), &g.eCount);
    errNum |= clSetKernelArg(MSGGenMerge_array_CL_kernel, 6, sizeof(int), &this->numOfInitV);
    Check_Err(errNum, CL_SUCCESS);

    DEBUG_INFO
    errNum = clEnqueueNDRangeKernel(comman_queue, MSGInitial_array_kernel, 1,
                                    1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    Check_Err(errNum, CL_SUCCESS);
    //MSGGenMerge_array(g.vCount, g.eCount, &g.vList[0], &g.eList[0], this->numOfInitV, &initVSet[0], &g.verticesValue[0], mValues)
    errNum = clEnqueueNDRangeKernel(comman_queue, MSGGenMerge_array_CL_kernel,
                                    1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    Check_Err(errNum, CL_SUCCESS);
    //   clWaitForEvents(1, &readDone);

    /*
    errNum = clEnqueueReadBuffer(comman_queue, this->mValues, CL_FALSE, 0,
                                 sizeof(MessageValueType) * numOfInitV * g.vCount,
                                 &this->hostMValues, 0, NULL, &readDone);
                                
   Check_Err(errNum, CL_SUCCESS);
    clWaitForEvents(1, &readDone);
*/
    DEBUG_INFO
    errNum = clEnqueueReadBuffer(comman_queue, this->mValues, CL_FALSE, 0,
                                 sizeof(MessageValueType) * this->numOfInitV * g.vCount,
                                 mValues1, 0, NULL, &readDone);
    clWaitForEvents(1, &readDone);
    DEBUG_INFO
    Check_Err(errNum, CL_SUCCESS);

    for (int i = 0; i < g.vCount * this->numOfInitV; i++)
    {
        std::cout << "mValues1[" << i << "]:" << mValues1[i] << std::endl;
        //  if (mValues1[i] != (MessageValueType)INVALID_MASSAGE)
        if (mValues1[i] < (MessageValueType)(INVALID_MASSAGE - 1))
        // accuracy warning
        {
            DEBUG_INFO
            int dst = i / this->numOfInitV;
            int initV = initVSet[i % this->numOfInitV];
            mSet.insertMsg(Message<MessageValueType>(initV, dst, mValues1[i]));
            std::cout << "dst:" << dst << "\tinitV:" << initV << std::endl;
        }
    }
    DEBUG_INFO
    delete[] mValues1;
    return mSet.mSet.size();
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFordCL<VertexValueType, MessageValueType>::Free()
{
    /*
    
    clReleaseMemObject(hostESet);
    //   clReleaseMemObject(host_initVSet);
    clReleaseMemObject(hostMValues);
    clReleaseMemObject(hostVSet);
    clReleaseMemObject(hostVValues);

    clReleaseMemObject(vSet);
    clReleaseMemObject(eSet);
    clReleaseMemObject(mValues);
    // clReleaseMemObject(initVSet);
    clReleaseMemObject(vValues);
*/
    errNum |= clReleaseEvent(readDone);
    // errNum |= clReleaseEvent(copyDone);
    errNum |= clReleaseKernel(MSGApply_array_kernel);

    errNum |= clReleaseKernel(MSGGenMerge_array_CL_kernel);

    errNum != clReleaseKernel(MSGInitial_array_kernel);

    errNum |= clReleaseCommandQueue(comman_queue);

    errNum |= clReleaseProgram(program);
    Check_Err(errNum, CL_SUCCESS);
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFordCL<VertexValueType, MessageValueType>::Free_little(bool isMidNot)
{
    errNum = CL_SUCCESS;

    errNum |= clReleaseMemObject(hostESet);

    errNum |= clReleaseMemObject(hostMValues);

    errNum |= clReleaseMemObject(hostVSet);

    errNum |= clReleaseMemObject(hostVValues);

    if (!isMidNot)
    {
        errNum |= clReleaseMemObject(vSet);

        errNum |= clReleaseMemObject(eSet);

        errNum |= clReleaseMemObject(mValues);

        errNum |= clReleaseMemObject(vValues);
    }
    Check_Err(errNum, CL_SUCCESS);
    DEBUG_INFO
}

template <typename VertexValueType, typename MessageValueType>
void BellmanFordCL<VertexValueType, MessageValueType>::ApplyD_CL(Graph<VertexValueType> &g,
                                                                 std::vector<int> &initVList,
                                                                 int partitionCount)
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
    std::cout << "\nglobal_work_size:" << global_work_size << std::endl;
    std::cout << "\nlocal_work_size:" << local_work_size << std::endl;
    int iterCount = 0;
    while (activeVertices.size() > 0)
    {
        std::cout << "\nactiveVertices.size:" << activeVertices.size() << std::endl;
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
            //  ApplyStep:
            // ApplyStep(subGraphSet.at(i), initVList, AVSet.at(i));
            auto &g = subGraphSet.at(i);
            auto &activeVertices = AVSet.at(i);
            auto &initVSet = initVList;
            auto mGenSet = MessageSet<MessageValueType>();
            auto mMergedSet = MessageSet<MessageValueType>();
            mMergedSet.mSet.clear();
            DEBUG_INFO
            std::cout << "\n\ni :" << i << "\n\n"
                      << std::endl;
            //  MSGGenMerge_CL(g, initVSet, activeVertices, mMergedSet);
            MSGGenMerge_CL1(g, initVSet, activeVertices, mMergedSet);
            DEBUG_INFO
            //Test
            std::cout << "MGenMerge:" << clock() << std::endl;
            //Test end
            activeVertices.clear();
            //   Free_little(true);
            // MSGApply(g, initVSet, activeVertices, mMergedSet);
            MSGApply1(g, initVSet, activeVertices, mMergedSet);
            //Test
            std::cout << "Apply:" << clock() << std::endl;
            //Test end

            Free_little(false);
        }

        activeVertices.clear();

        BellmanFord<VertexValueType, MessageValueType>::MergeGraph(g, subGraphSet, activeVertices, AVSet, initVList);
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
        std::cout << "\nCheck Error:" << std::endl;
        // std::cerr << "Line : " << lineNumber << " in File_ " << file << std::endl;
        std::cout << "Line : " << lineNumber << " in File_ " << file << std::endl;
        std::cerr << "Line : " << lineNumber << " in File_ " << file << std::endl;
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

template <typename T>
void displayDeviceInfo(
    cl_device_id id,
    cl_device_info name,
    std::string str)
{
    cl_int errNUm;
    std::size_t paramValueSize;
    errNUm = clGetDeviceInfo(id, name, 0, NULL, &paramValueSize);
    if (errNUm != CL_SUCCESS)
        std::cout << "Failed to find opencl device info" << std::endl;

    T *info = (T *)alloca(sizeof(T) * paramValueSize);
    errNUm = clGetDeviceInfo(id, name, paramValueSize, info, NULL);
    if (errNUm != CL_SUCCESS)
        std::cout << "Failed to find opencl device info" << std::endl;

    //  since c++20, use generic lambda
    using T1 = cl_device_type;
    auto appendStr = [](T1 info, T1 value, std::string name, std::string &str) {
        if (info & value)
        {
            if (str.length() > 0)
                str.append("|");
            str.append(name);
        }
    };

    switch (name)
    {
    case CL_DEVICE_TYPE:
    {

        std::string deviceType;
        appendStr(
            *(reinterpret_cast<cl_device_type *>(info)),
            CL_DEVICE_TYPE_CPU,
            "CL_DEVICE_TYPE_CPU",
            deviceType);
        appendStr(
            *(reinterpret_cast<cl_device_type *>(info)),
            CL_DEVICE_TYPE_GPU,
            "CL_DEVICE_TYPE_GPU",
            deviceType);
        appendStr(
            *(reinterpret_cast<cl_device_type *>(info)),
            CL_DEVICE_TYPE_ACCELERATOR,
            "CL_DEVICE_TYPE_ACCELERATOR",
            deviceType);
        appendStr(
            *(reinterpret_cast<cl_device_type *>(info)),
            CL_DEVICE_TYPE_DEFAULT,
            "CL_DEVICE_TYPE_DEFAULT",
            deviceType);
        std::cout << "\t" << str << ":\t" << *info << std::endl;
        std::cout << "\tCL_DEVICE_TYPE_INFO:\t" << deviceType << std::endl;

        break;
    }
    default:
        std::cout << "\t\t" << str << ":\t" << *info << std::endl;
    }
}

void displayPlatformInfo(
    cl_platform_id id,
    cl_platform_info name,
    std::string str)
{
    cl_int errNum;
    std::size_t paramValueSize;
    errNum = clGetPlatformInfo(
        id, name, 0, NULL, &paramValueSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL platform " << str << "." << std::endl;
    }
    auto info = (char *)alloca(sizeof(char) * paramValueSize);
    errNum = clGetPlatformInfo(
        id, name, paramValueSize, info, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL platform " << str << "." << std::endl;
    }
    std::cout << "\t" << str << ":\t" << info << std::endl;
}
