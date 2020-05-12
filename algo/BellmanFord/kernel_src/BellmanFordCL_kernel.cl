#define INT_MAX32 2147483647

typedef struct Vertex {
  int vertexID;
  bool isActive;
  int initVIndex;
} Vertex;

typedef struct Edge {
  int src;
  int dst;
  double weight;
} Edge;

/*
__kernel void MSGApply_kernel(__global Vertex *vSet, int numOfInitV,
                              int *initVSet, double *vValues, int numOfMsg,
                              int *mDstSet, int *mInitVIndexSet,
                              double *mValueSet) {
  int tid = get_global_id(0);
  if (tid < numOfMsg) {
    int vID = mDstSet[tid];
    int vInitIndex = mInitVIndexSet[tid];

    if (vInitIndex != -1) {
      int ttid = vID * numOfInitV + vInitIndex;
      if (vValues[ttid] > mValueSet[tid]) {
        vValues[ttid] = mValueSet[tid];
        vSet[vID].isActive = true;
      }
    }
  }
}
*/

__kernel void MSGApply_array(__global Vertex *vSet, __global Edge *eSet,
                             __global double *vValues, __global double *mValues,
                             int vCount, int eCount, int numOfInitV) {
  int avCount = 0;
  for (int i = 0; i < vCount; i++) {
    vSet[i].isActive = false;
  }
  for (int i = 0; i < vCount * numOfInitV; i++) {
    if (vValues[i] > mValues[i]) {
      vValues[i] = mValues[i];
      if (!vSet[i / numOfInitV].isActive) {
        vSet[i / numOfInitV].isActive = true;
        avCount = avCount + 1;
      }
    }
  }
//  printf("acCount:%d\n", avCount);
}

__kernel void MSGGenMerge_array_CL(__global Vertex *vSet, __global Edge *eSet,
                                   __global double *vValues,
                                   __global double *mValues, int vCount,
                                   int eCount, numOfInitV) {
  for (int i = 0; i < vCount * numOfInitV; i++) {
    mValues[i] = INT_MAX32;
  }
  for (int i = 0; i < eCount; i++) {
    if (vSet[eSet[i].src].isActive) {
      for (int j = 0; j < numOfInitV; j++) {
        if (mValues[eSet[i].dst * numOfInitV + j] >
            vValues[eSet[i].src * numOfInitV + j] + eSet[i].weight)
          mValues[eSet[i].dst * numOfInitV + j] =
              vValues[eSet[i].src * numOfInitV + j] + eSet[i].weight;
      }
    }
  }
}
