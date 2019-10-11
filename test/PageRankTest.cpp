//
// Created by cave-g-f on 2019-9-23
//

#include "../algo/PageRank/PageRank.h"

#include <iostream>
#include <fstream>

int main()
{
    //Read the Graph
    std::ifstream Gin("testGraph.txt");
    if(!Gin.is_open()) {std::cout << "Error! File testGraph.txt not found!" << std::endl; return 1; }

    int vCount, eCount;
    Gin >> vCount >> eCount;

    Graph<std::pair<double, double>> test = Graph<std::pair<double, double>>(vCount);
    for(int i = 0; i < eCount; i++)
    {
        int src, dst;
        double weight;

        Gin >> src >> dst >> weight;
        test.insertEdgeWithVertexInfo(src, dst, weight);
    }

    Gin.close();

    std::vector<int> initVList = std::vector<int>();
    for(int i = 0; i < vCount; i++)
    {
        initVList.push_back(i);
    }

    PageRank<std::pair<double, double>, PRA_MSG> executor = PageRank<std::pair<double, double>, PRA_MSG>();
    //executor.Apply(test, initVList);
    executor.ApplyD(test, initVList, 4);
}

