
#include "device_launch_parameters.h"
#include "CpuGpuMat.h"
#include "KernelBatchNormalization.cuh"
#include <math.h>


__global__ void gpuBatchNorm(float* gpuResult, float* gpuBeta, float* gpuGamma, float* gpuMovingMean, float* gpuMovingVar, float epsilon, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < size) {
		gpuResult[id] = (gpuResult[id] - gpuMovingMean[id]) / sqrt(gpuMovingVar[id] + epsilon) * gpuGamma[id] + gpuBeta[id];
	}
}


void gpuBatchNormalization(CpuGpuMat* result, CpuGpuMat* beta, CpuGpuMat* gamma, CpuGpuMat* movingMean, CpuGpuMat* movingVariance, float epsilon) {

	int threadsPerBlock = 32;
	int blocksPerGrid = ceil(double(beta->Size) / double(threadsPerBlock));

	gpuBatchNorm << < blocksPerGrid, threadsPerBlock >> > ((float*)result->GpuP, (float*)beta->GpuP, (float*)gamma->GpuP, (float*)movingMean->GpuP, (float*)movingVariance->GpuP, epsilon, beta->Size);
}