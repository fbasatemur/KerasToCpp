#pragma once
#include "CpuMat.h"

void cpuMatrixConv2D(CpuMat* input, int filters, CpuMat* kernel, CpuMat* result);

void cpuMatrixMultiplication(CpuMat* Mat1, CpuMat* Mat2, CpuMat* Mat3);

void cpuNormalizeAndShift(CpuMat* Mat, float mean, float variance, float epsilon, float beta, float gama);

void cpuRelu(CpuMat* Mat);

void cpuSigmoid(CpuMat* Mat);

void cpuBatchNormalization(CpuMat* result, CpuMat* beta, CpuMat* gamma, CpuMat* movingMean, CpuMat* movingVariance, float epsilon);