//
// Created by cave-g-f on 10/19/19.
//
#include "../algo/LabelPropagation/LabelPropagationGPU.h"

#include <iostream>
#include <fstream>
#include <algorithm>
int main()
{
    //Read the Graph
    std::ifstream Gin("../../data/testGraph100000.txt");
    if(!Gin.is_open()) {std::cout << "Error! File testGraph.txt not found!" << std::endl; return 1; }

    int vCount, eCount;
    Gin >> vCount >> eCount;

    std::vector<int> initVList = std::vector<int>();

    Graph<LPA_Value> test = Graph<LPA_Value>(vCount);
    for(int i = 0; i < eCount; i++)
    {
        int src, dst;
        double weight;

        Gin >> src >> dst >> weight;
        test.insertEdgeUpdateInfo(src, dst, weight, i);
    }

    Gin.close();

    auto executor = LabelPropagationGPU<LPA_Value, LPA_MSG>();
    executor.ApplyD(test, initVList, 1);
}
