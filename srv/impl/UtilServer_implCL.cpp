//
// Created by infdahai on 2020-5-21.
//

#include "../UtilServer.cpp"

#include "../../algo/BellmanFord/BellmanFordCL.cpp"
#include "../../algo/ConnectedComponent/ConnectedComponentCL.cpp"
#include "../../algo/PageRank/PageRankCL.cpp"
#include "../../algo/LabelPropagation/LabelPropagationCL.cpp"

template class UtilServer<BellmanFordCL<double, double>, double, double>;
template class UtilServer<ConnectedComponentCL<int, int>, int, int>;
template class UtilServer<LabelPropagationCL<LPA_Value, LPA_MSG>, LPA_Value, LPA_MSG>;
template class UtilServer<PageRankCL<std::pair<double, double>, PRA_MSG>, std::pair<double, double>, PRA_MSG>;
