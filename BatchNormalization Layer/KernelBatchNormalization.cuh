#pragma once
#include "CpuGpuMat.h"

#ifdef __cplusplus									
extern "C"
#endif // __cplusplus

void gpuBatchNormalization(CpuGpuMat* result, CpuGpuMat* beta, CpuGpuMat* gamma, CpuGpuMat* movingMean, CpuGpuMat* movingVariance, float epsilon);