//
// Created by Thoh Testarossa on 2019-04-06.
//

#include "../core/Graph.h"
#include "../core/GraphUtil.h"
#include "../srv/UtilClient.h"
#include "../srv/UNIX_shm.h"
#include "../srv/UNIX_msg.h"
#include "../algo/PageRank/PageRank.h"

#include <iostream>
#include <fstream>
#include <vector>

#include <future>
#include <cstring>

int optimize = 0;

template <typename VertexValueType, typename MessageValueType>
void testFut(UtilClient<VertexValueType, MessageValueType> *uc, VertexValueType *vValues, Vertex *vSet, int *avSet, int avCount)
{
    uc->connect();
    if(optimize)
        uc->update(vValues, vSet, avSet, avCount);
    else
        uc->update(vValues, vSet);
    uc->request();
    uc->disconnect();
}

int main(int argc, char *argv[])
{
    if(argc != 4 && argc != 5 && argc != 6)
    {
        std::cout << "Usage:" << std::endl << "./UtilClientTest_PageRank vCount eCount numOfInitV [nodeCount] [optimize]" << std::endl;
        return 1;
    }

    int vCount = atoi(argv[1]);
    int eCount = atoi(argv[2]);
    int numOfInitV = atoi(argv[3]);
    int nodeCount = atoi(argv[4]);
    optimize = atoi(argv[5]);

    //Parameter check
    if(vCount <= 0 || eCount <= 0 || numOfInitV <= 0 || nodeCount <= 0)
    {
        std::cout << "Parameter illegal" << std::endl;
        return 3;
    }

    int *initVSet = new int [numOfInitV];
    std::pair<double, double> *vValues = new std::pair<double, double> [vCount];
    bool *filteredV = new bool [vCount];
    int *timestamp = new int [vCount];

    std::vector<Vertex> vSet = std::vector<Vertex>();
    std::vector<Edge> eSet = std::vector<Edge>();

    std::ifstream Gin("../../data/testGraph4000000.txt");
    if(!Gin.is_open())
    {
        std::cout << "Error! File testGraph.txt not found!" << std::endl;
        return 4;
    }

    int tmp;
    Gin >> tmp;
    if(vCount != tmp)
    {
        std::cout << "Graph file doesn't match up UtilClient's parameter" << std::endl;
        return 5;
    }
    Gin >> tmp;
    if(eCount != tmp)
    {
        std::cout << "Graph file doesn't match up UtilClient's parameter" << std::endl;
        return 5;
    }

    //init v index
    std::cout << "init initVSet ..." << std::endl;

    std::cout << "init vSet ..." << std::endl;
    for(int i = 0; i < vCount; i++)
    {
        vSet.emplace_back(i, true, i);
        filteredV[i] = false;
        timestamp[i] = -1;
    }

    initVSet[0] = -1;

    std::cout << "init vValues ..." << std::endl;

    if(initVSet[0] == -1)
    {
        for(int i = 0; i < vCount; i++)
        {
            auto newRank = 0.15;
            vValues[i] = std::pair<double, double>(newRank, newRank);
        }
    }
    else
    {
        for(int i = 0; i < vCount; i++)
        {
            if(vSet[i].initVIndex == INVALID_INITV_INDEX)
            {
                vValues[i] = std::pair<double, double>(0.0, 0.0);
            }
            else
            {
                vValues[i] = std::pair<double, double>(1.0, 1.0);
            }
        }

    }

    std::cout << "init edge ..." << std::endl;

    for(int i = 0; i < eCount; i++)
    {
        int src, dst;
        double weight;
        Gin >> src >> dst >> weight;
        vSet.at(src).outDegree++;
        eSet.emplace_back(src, dst, weight);
    }

    for(int i = 0; i < eCount; i++)
    {
        eSet.at(i).weight = 1.0 / vSet[eSet[i].src].outDegree;
    }

    Gin.close();

    numOfInitV = 1;

    std::cout << "start transfer" << std::endl;

    //Client Init Data Transfer
    auto clientVec = std::vector<UtilClient<std::pair<double, double>, PRA_MSG>>();
    for(int i = 0; i < nodeCount; i++)
        clientVec.push_back(UtilClient<std::pair<double, double>, PRA_MSG>(vCount, ((i + 1) * eCount) / nodeCount - (i * eCount) / nodeCount, numOfInitV, i));
    int chk = 0;

    for(int i = 0; i < nodeCount && chk != -1; i++)
    {
        std::cout << "connect" << std::endl;
        chk = clientVec.at(i).connect();
        std::cout << "connect end" << std::endl;
        if (chk == -1)
        {
            std::cout << "Cannot establish the connection with server correctly" << std::endl;
            return 2;
        }

        chk = clientVec.at(i).transfer(vValues, &vSet[0], &eSet[(i * eCount) / nodeCount], initVSet, filteredV, timestamp);

        if(chk == -1)
        {
            std::cout << "Parameter illegal" << std::endl;
            return 3;
        }

        clientVec.at(i).disconnect();
    }
    int iterCount = 0;

    //Test
    std::cout << "Init finished" << std::endl;
    //Test end

    bool isActive = true;

    int avCount = 0;
    std::vector<int> avSet;
    avSet.reserve(vCount);
    avSet.assign(vCount, 0);

    while(isActive)
    {
        //Test
        std::cout << "Processing at iter " << ++iterCount << std::endl;
        //Test end

        isActive = false;

        auto futList = new std::future<void> [nodeCount];
        for(int i = 0; i < nodeCount; i++)
        {
            std::future<void> tmpFut = std::async(testFut<std::pair<double, double>, PRA_MSG>, &clientVec.at(i), vValues, &vSet[0], &avSet[0], avCount);
            futList[i] = std::move(tmpFut);
        }

        for(int i = 0; i < nodeCount; i++)
            futList[i].get();

        auto start = std::chrono::system_clock::now();

        //Retrieve data
        for(int i = 0; i < nodeCount; i++)
        {
            clientVec.at(i).connect();

            //clear active info
            for(int j = 0; j < vCount; j++)
            {
                vSet[j].isActive = false;
            }

            //Collect data
            for(int j = 0; j < vCount; j++)
            {
                auto &value = clientVec.at(i).vValues[j];
                if(clientVec.at(i).vSet[j].isActive)
                {
                    if(!vSet.at(j).isActive)
                    {
                        vSet.at(j).isActive |= clientVec.at(i).vSet[j].isActive;
                        vValues[j].first = value.first;
                        vValues[j].second = value.second;
                    }
                    else
                    {
                        vValues[j].second += value.second;
                    }
                }
                else if(!vSet.at(j).isActive)
                {
                    vValues[j] = value;
                }
                clientVec.at(i).vSet[j].isActive = false;
            }

            clientVec.at(i).disconnect();
        }

        for(int i = 0; i < vCount; i++)
        {
            if(vSet.at(i).isActive)
            {
                auto oldRank = vValues[i].first;
                vValues[i].first = vValues[i].second + oldRank;
                vValues[i].second = vValues[i].first - oldRank;
            }
        }

        auto mergeEnd = std::chrono::system_clock::now();

        std::cout << "graph merge time: " <<  std::chrono::duration_cast<std::chrono::microseconds>(mergeEnd - start).count() << std::endl;

        avCount = 0;
        for(int i = 0; i < vCount; i++)
        {
            if(vSet[i].isActive)
            {
                avSet.at(avCount) = i;
                avCount++;
                isActive = true;
            }
        }

        std::cout << "avCount : " << avCount << std::endl;

        //test
//        for(int i = 0; i < vCount; i++)
//        {
//            std::cout << i << ":" << vValues[i].first << " " << vValues[i].second << std::endl ;
//        }
    }

    std::cout << "===========result===========" << std::endl;
    //result check
    for(int i = 0; i < vCount; i++)
    {
        std::cout << i << ":" << vValues[i].first << " " << vValues[i].second << std::endl ;
    }

    for(int i = 0; i < nodeCount; i++) clientVec.at(i).shutdown();
}