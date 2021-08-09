#pragma once
#include "CpuGpuMat.h"

#ifdef __cplusplus
extern "C"
#endif // __cplusplus


void gpuMatrixMultiplication(CpuGpuMat* Mat1, CpuGpuMat* Mat2, CpuGpuMat* Mat3, int inStartIndex, int resStartIndex);	
