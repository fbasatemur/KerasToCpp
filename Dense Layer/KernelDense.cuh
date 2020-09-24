#pragma once
#include "CpuGpuMat.h"

#ifdef __cplusplus									
extern "C"
#endif // __cplusplus


void gpuMatrixMultiplication(CpuGpuMat* input, CpuGpuMat* kernel, CpuGpuMat* result);