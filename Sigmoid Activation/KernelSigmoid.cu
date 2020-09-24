#include "device_launch_parameters.h"
#include "CpuGpuMat.h"
#include "KernelSigmoid.cuh"
#include <math.h>


__global__ void gpuSigmoidActivation(float* GpuP, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < size) {
		GpuP[id] = (float)(1.0 / (1.0 + exp(-1.0 * (double)GpuP[id])));
	}
}


void gpuSigmoid(CpuGpuMat* Mat) {

	int threadsPerBlock = 32;
	int blocksPerGrid = ceil(double(Mat->Size) / double(threadsPerBlock));

	gpuSigmoidActivation << < blocksPerGrid, threadsPerBlock >> > ((float*)Mat->GpuP, Mat->Size);
}
