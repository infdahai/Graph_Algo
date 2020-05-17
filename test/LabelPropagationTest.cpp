//
// Created by cave-g-f on 2019-05-24.
//

#include "../algo/LabelPropagation/LabelPropagation.h"

#include <iostream>
#include <fstream>
#include <algorithm>
int main()
{
    //Read the Graph
    std::ifstream Gin("testGraph.txt");
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

    auto executor = LabelPropagation<LPA_Value, LPA_MSG>();
    executor.ApplyD(test, initVList, 1);
}

