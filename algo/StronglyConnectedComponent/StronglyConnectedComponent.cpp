//
// Created by Thoh Testarossa on 2019-08-17.
//

#include "StronglyConnectedComponent.h"

#include <iostream>
#include <ctime>

template<typename VertexValueType>
StronglyConnectedComponent_stage_1<VertexValueType>::StronglyConnectedComponent_stage_1()
{

}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_1<VertexValueType>::MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertice, const MessageSet<VertexValueType> &mSet)
{

}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_1<VertexValueType>::MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<VertexValueType> &mSet)
{

}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_1<VertexValueType>::MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, VertexValueType *mValues)
{

}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_1<VertexValueType>::MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, VertexValueType *mValues)
{

}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_1<VertexValueType>::MergeGraph(Graph<VertexValueType> &g, const std::vector<Graph<VertexValueType>> &subGSet, std::set<int> &activeVertices, const std::vector<std::set<int>> &activeVerticeSet, const std::vector<int> &initVList)
{

}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_1<VertexValueType>::Init(int vCount, int eCount, int numOfInitV)
{

}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_1<VertexValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList)
{

}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_1<VertexValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{

}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_1<VertexValueType>::Free()
{

}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_1<VertexValueType>::ApplyStep(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices)
{

}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_1<VertexValueType>::Apply(Graph<VertexValueType> &g, const std::vector<int> &initVList)
{

}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_1<VertexValueType>::ApplyD(Graph<VertexValueType> &g, const std::vector<int> &initVList, int partitionCount)
{

}

template<typename VertexValueType>
StronglyConnectedComponent_stage_2<VertexValueType>::StronglyConnectedComponent_stage_2()
{

}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_2<VertexValueType>::MSGApply(Graph<VertexValueType> &g, const std::vector<int> &initVSet,std::set<int> &activeVertice, const MessageSet<VertexValueType> &mSet)
{
    //Availability check
    if(g.vCount <= 0) return;

    VertexValueType *mValues = new VertexValueType [g.vCount];
    for(int i = 0; i < g.vCount; i++) mValues[i] = INVALID_MASSAGE;
    for(const auto &m : mSet.mSet) mValues[m.dst] = m.value;

    this->MSGApply_array(g.vCount, g.eCount, &g.vList[0], 0, nullptr, &g.verticesValue[0], mValues);

    activeVertice.clear();
    for(const auto &v : g.vList)
    {
        if(v.isActive)
            activeVertice.insert(v.vertexID);
    }
}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_2<VertexValueType>::MSGGenMerge(const Graph<VertexValueType> &g, const std::vector<int> &initVSet, const std::set<int> &activeVertice, MessageSet<VertexValueType> &mSet)
{
    //Availability check
    if(g.vCount <= 0) return;

    VertexValueType *mValues = new VertexValueType [g.vCount];

    this->MSGGenMerge_array(g.vCount, g.eCount, &g.vList[0], &g.eList[0], 0, nullptr, &g.verticesValue[0], mValues);

    //Package mValues into result mSet
    for(int i = 0; i < g.vCount; i++)
    {
        if(mValues[i] != (VertexValueType)INVALID_MASSAGE)
            mSet.insertMsg(Message<VertexValueType>(INVALID_INITV_INDEX, i, mValues[i]));
    }
}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_2<VertexValueType>::MSGApply_array(int vCount, int eCount, Vertex *vSet, int numOfInitV, const int *initVSet, VertexValueType *vValues, VertexValueType *mValues)
{
    //isActive reset
    for(int i = 0; i < vCount; i++) vSet[i].isActive = false;

    for(int i = 0; i < vCount; i++)
    {
        if(vValues[i] > mValues[i])
        {
            vValues[i] = mValues[i];
            vSet[i].isActive = true;
        }
    }
}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_2<VertexValueType>::MSGGenMerge_array(int vCount, int eCount, const Vertex *vSet, const Edge *eSet, int numOfInitV, const int *initVSet, const VertexValueType *vValues, VertexValueType *mValues)
{
    //Invalid MSG init
    for(int i = 0; i < vCount; i++) mValues[i] = (VertexValueType)INVALID_MASSAGE;

    for(int i = 0; i < eCount; i++)
    {
        if(vSet[eSet[i].src].isActive)
        {
            if(mValues[eSet[i].dst] > vValues[eSet[i].src])
                mValues[eSet[i].dst] = vValues[eSet[i].src];
        }
    }
}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_2<VertexValueType>::MergeGraph(Graph<VertexValueType> &g, const std::vector<Graph<VertexValueType>> &subGSet, std::set<int> &activeVertices, const std::vector<std::set<int>> &activeVerticeSet, const std::vector<int> &initVList)
{
    //Init
    activeVertices.clear();
    for(auto &v : g.vList) v.isActive = false;

    for(const auto &subG : subGSet)
    {
        //vSet merge
        for(int i = 0; i < subG.vCount; i++)
            g.vList.at(i).isActive |= subG.vList.at(i).isActive;

        //vValues merge
        for(int i = 0; i < subG.vCount; i++)
        {
            if(g.verticesValue.at(i) > subG.verticesValue.at(i))
                g.verticesValue.at(i) = subG.verticesValue.at(i);
        }
    }

    for(const auto &AVs : activeVerticeSet)
    {
        for(auto av : AVs)
            activeVertices.insert(av);
    }
}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_2<VertexValueType>::Init(int vCount, int eCount, int numOfInitV)
{
    this->totalVValuesCount = vCount;
    this->totalMValuesCount = vCount;
}

template<typename VertexValueType>
void
StronglyConnectedComponent_stage_2<VertexValueType>::GraphInit(Graph<VertexValueType> &g, std::set<int> &activeVertices, const std::vector<int> &initVList)
{
    //v init
    for(auto &v : g.vList)
    {
        v.isActive = true;
        activeVertices.insert(v.vertexID);
    }

    //vValues init
    g.verticesValue.reserve(g.vCount);
    g.verticesValue.assign(g.vCount, -1);
    for(int i = 0; i < g.vCount; i++) g.verticesValue.at(i) = (VertexValueType)i;
}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_2<VertexValueType>::Deploy(int vCount, int eCount, int numOfInitV)
{

}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_2<VertexValueType>::Free()
{

}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_2<VertexValueType>::ApplyStep(Graph<VertexValueType> &g, const std::vector<int> &initVSet, std::set<int> &activeVertices)
{
    MessageSet<VertexValueType> mGenSet = MessageSet<VertexValueType>();
    MessageSet<VertexValueType> mMergedSet = MessageSet<VertexValueType>();

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

template<typename VertexValueType>
void
StronglyConnectedComponent_stage_2<VertexValueType>::Apply(Graph<VertexValueType> &g, const std::vector<int> &initVList)
{
    //Init the Graph
    std::set<int> activeVertices = std::set<int>();
    MessageSet<VertexValueType> mGenSet = MessageSet<VertexValueType>();
    MessageSet<VertexValueType> mMergedSet = MessageSet<VertexValueType>();

    Init(g.vCount, g.eCount, initVList.size());

    GraphInit(g, activeVertices, initVList);

    Deploy(g.vCount, g.eCount, initVList.size());

    while(activeVertices.size() > 0)
        ApplyStep(g, initVList, activeVertices);

    Free();
}

template<typename VertexValueType>
void StronglyConnectedComponent_stage_2<VertexValueType>::ApplyD(Graph<VertexValueType> &g, const std::vector<int> &initVList, int partitionCount)
{
    //Init the Graph
    std::set<int> activeVertices = std::set<int>();

    std::vector<std::set<int>> AVSet = std::vector<std::set<int>>();
    for(int i = 0; i < partitionCount; i++) AVSet.push_back(std::set<int>());
    std::vector<MessageSet<VertexValueType>> mGenSetSet = std::vector<MessageSet<VertexValueType>>();
    for(int i = 0; i < partitionCount; i++) mGenSetSet.push_back(MessageSet<VertexValueType>());
    std::vector<MessageSet<VertexValueType>> mMergedSetSet = std::vector<MessageSet<VertexValueType>>();
    for(int i = 0; i < partitionCount; i++) mMergedSetSet.push_back(MessageSet<VertexValueType>());

    Init(g.vCount, g.eCount, initVList.size());

    GraphInit(g, activeVertices, initVList);

    //Test
    //std::cout << 1 << std::endl;

    Deploy(g.vCount, g.eCount, initVList.size());

    int iterCount = 0;

    while(activeVertices.size() > 0)
    {
        //Test
        std::cout << ++iterCount << ":" << clock() << std::endl;
        //Test end

        auto subGraphSet = this->DivideGraphByEdge(g, partitionCount);

        for(int i = 0; i < partitionCount; i++)
        {
            AVSet.at(i).clear();
            AVSet.at(i) = activeVertices;
        }

        //Test
        std::cout << "GDivide:" << clock() << std::endl;
        //Test end

        for(int i = 0; i < partitionCount; i++)
            ApplyStep(subGraphSet.at(i), initVList, AVSet.at(i));

        activeVertices.clear();
        MergeGraph(g, subGraphSet, activeVertices, AVSet, initVList);
        //Test
        std::cout << "GMerge:" << clock() << std::endl;
        //Test end
    }

    Free();

    //Test
    std::cout << "end" << ":" << clock() << std::endl;
    //Test end
}


