
#include "../algo/PageRank/PageRankCL.h"

#include <iostream>
#include <fstream>

int main(int argc, char *argv[])
{
    if(argc != 4)
    {
        std::cout << "Usage:" << std::endl << "./algo_PageRankTest graph_path vcount ecount" << std::endl;
        return 1;
    }

    //Read the Graph
    std::ifstream Gin(argv[1]);
    if(!Gin.is_open()) {std::cout << "Error! File testGraph.txt not found!" << std::endl; return 1; }

    int vCount = atoi(argv[2]);
    int eCount = atoi(argv[3]);

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

    PageRankCL<std::pair<double, double>, PRA_MSG> executor = PageRankCL<std::pair<double, double>, PRA_MSG>();
    //executor.Apply(test, initVList);
    executor.ApplyD_CL(test, initVList, 4);
}

