
typedef struct PRA_MSG {
  int destVId;
  double rank;
} PRA_MSG;

typedef struct PAIR {
  double first;
  double second;
} PAIR;

typedef struct Vertex {
  int vertexID;
  bool isActive;
  int initVIndex;
  bool isMaster;
  int outDegree;
  int inDegree;
} Vertex;

typedef struct Edge {
  int src;
  int dst;
  double weight;
  int originIndex;
} Edge;

__kernel void MSGApply_array_CL(__global Vertex *vSet, __global Edge *eSet,
                                __global PAIR *vValues,
                                __global PRA_MSG *mValues, int vCount,
                                int eCount, int numOfInitV) {
  int tid = get_global_id(0);
  if ((tid >= 0) && (tid < vCount)) {
    int destVId = mValues[tid].destVId;
    if (destVId == -1 || !vSet[destVId].isMaster) {
      return;
    }
    vSet[destVId].isActive = true;
    vValues[destVId].first += mValues[tid].rank;
    vValues[destVId].second = mValues[tid].rank;
  }
}

__kernel void MSGGenMerge_array_CL(__global Vertex *vSet, __global Edge *eSet,
                                   __global PAIR *vValues,
                                   __global PRA_MSG *mValues, int vCount,
                                   int eCount, int numOfInitV,
                                   int deltaThreshold, int resetProb) {
  int tid = get_global_id(0);
  if ((tid >= 0) && (tid < eCount)) {
    int srcVId = eSet[tid].src;
    PRA_MSG msgValue = {-1, -1};
    if (vSet[srcVId].isActive && vValues[srcVId].second > deltaThreshold) {
      msgValue.destVId = eSet[tid].dst;
      msgValue.rank =
          vValues[eSet[tid].src].second * eSet[tid].weight * (1 - resetProb);
      mValues[msgValue.destVId].destVId = msgValue.destVId;
      mValues[msgValue.destVId].rank += msgValue.rank;
    }
  }
}

__kernel void MSGInitial_array_kernel_1(__global PRA_MSG *mValues, int vCount) {
  int tid = get_global_id(0);
  if ((tid >= 0) && (tid < vCount)) {
    mValues[tid].destVId = -1;
    mValues[tid].rank = 0;
  }
}

__kernel void MSGInitial_array_kernel_2(__global Vertex *vSet, int vCount) {
  int tid = get_global_id(0);
  if ((tid >= 0) && (tid < vCount)) {
    vSet[i].isActive = false;
  }
}