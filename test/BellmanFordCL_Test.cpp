
#include "../algo/BellmanFord/BellmanFordCL.h"

#include <iostream>
#include <fstream>

int main()
{

    cl_platform_id platform;
    cl_context gpuContext;
    cl_context cpuContext;
    cl_int errNum;

    //Read the Graph
    std::ifstream Gin("testGraph.txt");
    if (!Gin.is_open())
    {
        std::cout << "Error! File testGraph.txt not found!" << std::endl;
        return 1;
    }

    int vCount, eCount;
    Gin >> vCount >> eCount;

    Graph<double> test = Graph<double>(vCount);
    for (int i = 0; i < eCount; i++)
    {
        int src, dst;
        double weight;

        Gin >> src >> dst >> weight;
        test.insertEdge(src, dst, weight);
    }

    Gin.close();

    std::vector<int> initVList = std::vector<int>();
    initVList.push_back(1);
    initVList.push_back(2);
    initVList.push_back(4);

    BellmanFordCL<double, double> executor = BellmanFordCL<double, double>();
    //executor.Apply(test, initVList);
    executor.ApplyD_CL(test, initVList, 4);

    for (int i = 0; i < test.vCount * initVList.size(); i++)
    {
        if (i % initVList.size() == 0)
            std::cout << i / initVList.size() << ": ";
        std::cout << "(" << initVList.at(i % initVList.size()) << " -> " << test.verticesValue.at(i) << ")";
        if (i % initVList.size() == initVList.size() - 1)
            std::cout << std::endl;
    }
}