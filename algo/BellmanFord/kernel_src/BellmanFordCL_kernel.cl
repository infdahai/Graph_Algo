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

__kernel void MSGApply_array1(__global Vertex *vSet, __global Edge *eSet,
                              __global double *vValues,
                              __global double *mValues, int vCount, int eCount,
                              int numOfInitV) {
  int tid = get_global_id(0);
  if ((tid >= 0) && (tid < vCount * numOfInitV)) {
    if (vValues[tid] > mValues[tid]) {
      vValues[tid] = mValues[tid];
      if (!vSet[tid / numOfInitV].isActive) {
        vSet[tid / numOfInitV].isActive = true;
      }
    }
  }
}

__kernel void MSGGenMerge_array_CL1(__global Vertex *vSet, __global Edge *eSet,
                                    __global double *vValues,
                                    __global double *mValues, int vCount,
                                    int eCount, int numOfInitV) {
  int tid = get_global_id(0);
  if ((tid >= 0) && (tid < eCount)) {
    if (vSet[eSet[tid].src].isActive) {
      for (int j = 0; j < numOfInitV; j++) {
        double val_T =
            vValues[eSet[tid].src * numOfInitV + j] + eSet[tid].weight;
        //      printf("val_T:%f\n",val_T);
        //      printf("mValues[eSet[%d].dst*numofInitV+j]:%f\n",tid,mValues[eSet[tid].dst
        //      * numOfInitV + j]);
        if (mValues[eSet[tid].dst * numOfInitV + j] > val_T)
          mValues[eSet[tid].dst * numOfInitV + j] =
              vValues[eSet[tid].src * numOfInitV + j] + eSet[tid].weight;
      }
    }
  }
}
__kernel void MSGInitial_array_2(__global Vertex *vSet, int vCount) {
  int tid = get_global_id(0);
  if ((tid >= 0) && (tid < vCount)) {
    vSet[tid].isActive = false;
  }
}
__kernel void MSGInitial_array_1(__global double *mValues, int vCount,
                                 int numOfInitV) {
  int tid = get_global_id(0);
  if ((tid >= 0) && (tid < vCount * numOfInitV)) {
    mValues[tid] = INT_MAX32;
    //   printf("mValues[%d]=%f\n",tid,mValues[tid]);
  }
}