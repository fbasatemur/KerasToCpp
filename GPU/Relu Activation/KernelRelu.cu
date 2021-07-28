#include "device_launch_parameters.h"
#include "CpuGpuMat.h"
#include "KernelRelu.cuh"
#include <math.h>


__global__ void gpuReluActivation(float* GpuP, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < size) {
		GpuP[id] = GpuP[id] > 0 ? GpuP[id] : 0;
	}
}


void gpuRelu(CpuGpuMat* Mat)
{
	int threadsPerBlock = 32;
	int blocksPerGrid = ceil(double(Mat->Size) / double(threadsPerBlock));

	gpuReluActivation << < blocksPerGrid, threadsPerBlock >> > ((float*)Mat->GpuP, Mat->Size);
}