#define INT_MAX32 2147483647

typedef struct LPA_Value {
  int destVId;
  int label;
  int labelCnt;
} LPA_Value;

typedef struct LPA_MSG {
  int destVId;
  int edgeOriginIndex;
  int label;
} LPA_MSG;

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

__kernel void MSGGenMerge_array_CL(__global Vertex *vSet, __global Edge *eSet,
                                   __global LPA_Value *vValues,
                                   __global LPA_MSG *mValues, int vCount,
                                   int eCount, int numOfInitV) {
  int tid = get_global_id(0);
  if ((tid >= 0) && (tid < eCount)) {
    mValues[tid].destVId = eSet[tid].dst;
    mValues[tid].edgeOriginIndex = eSet[tid].originIndex;
    mValues[tid].label = vValues[eSet[tid].src].label;
  }
}

__kernel void MSGApply_array_CL(__global Vertex *vSet,
                                __global LPA_Value *vValues,
                                __global LPA_MSG *mValues, int vCount,
                                int eCount, int numOfInitV) {
  int tid = get_global_id(0);
  if ((tid >= 0) && (tid < vCount)) {
    int vid = mValues[tid].destVId;
    if (vid == -1) {
      return;
    }
    if ((vid == -1) || (!vSet[vid].isMaster)) {
      return;
    }
    vValues[tid].label = mValues[tid].label;
    vSet[vid].isActive = true;
  }
}
