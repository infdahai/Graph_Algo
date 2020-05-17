#define INT_MAX32 2147483647

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

__kernel void MSGInitial_array_1(__global int *mValues, int vCount) {
  int tid = get_global_id(0);
  if ((tid >= 0) && (tid < vCount)) {
    mValues[tid] = INT_MAX32;
  }
}

__kernel void MSGInitial_array_2(__global Vertex *vSet, int vCount) {
  int tid = get_global_id(0);
  if ((tid >= 0) && (tid < vCount)) {
    vSet[tid].isActive = false;
  }
}

__kernel void MSGApply_array_CL(__global Vertex *vSet, __global Edge *eSet,
                                __global int *vValues,
                                __global int *mValues, int vCount,
                                int eCount, int numOfInitV) {
  int tid = get_global_id(0);
  if ((tid >= 0) && (tid < vCount)) {
    if (vValues[tid] > mValues[tid]) {
      vValues[tid] = mValues[tid];
      if (!vSet[tid].isActive) {
        vSet[tid].isActive = true;
      }
    }
  }
}

__kernel void MSGGenMerge_array_CL(__global Vertex *vSet, __global Edge *eSet,
                                   __global int *vValues,
                                   __global int *mValues, int vCount,
                                   int eCount, int numOfInitV) {
  int tid = get_global_id(0);
  if ((tid >= 0) && (tid < eCount)) {
    if (vSet[eSet[tid].src].isActive) {
      if (mValues[eSet[tid].dst] > vValues[eSet[tid].src])
        mValues[eSet[tid].dst] = vValues[eSet[i].src];
    }
  }
}