//
// Created by Thoh Testarossa on 2019-05-25.
//

#include "../MessageSet.cpp"
#include "../../algo/PageRank/PageRank.h"

template class Message<double>;
template class Message<int>;
template class Message<std::pair<int, int>>;
template class Message<PRA_MSG>;

template class MessageSet<double>;
template class MessageSet<int>;
template class MessageSet<std::pair<int, int>>;
template class MessageSet<PRA_MSG>;