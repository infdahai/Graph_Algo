//
// Created by cave-g-f on 2019-9-23
//

#include "../algo/PageRank/PageRankGPU.h"

#include <iostream>
#include <fstream>

int main()
{
    //Read the Graph
    std::ifstream Gin("../../data/testGraph100000.txt");
    if(!Gin.is_open()) {std::cout << "Error! File testGraph.txt not found!" << std::endl; return 1; }

    int vCount, eCount;
    Gin >> vCount >> eCount;

    Graph<std::pair<double, double>> test = Graph<std::pair<double, double>>(vCount);
    for(int i = 0; i < eCount; i++)
    {
        int src, dst;
        double weight;

        Gin >> src >> dst >> weight;
        test.insertEdgeUpdateInfo(src, dst, weight, i);
    }

    Gin.close();

    std::vector<int> initVList = std::vector<int>();
    initVList.push_back(-1);

    PageRankGPU<std::pair<double, double>, PRA_MSG> executor = PageRankGPU<std::pair<double, double>, PRA_MSG>();
    //executor.Apply(test, initVList);
    executor.ApplyD(test, initVList, 4);
}

