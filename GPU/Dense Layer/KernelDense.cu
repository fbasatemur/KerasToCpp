#include "device_launch_parameters.h"
#include "CpuGpuMat.h"
#include "KernelDense.cuh"
#include <math.h>


__global__ void gpuMatrixMult(float* gpuMat1, float* gpuMat2, float* gpuMat3, int m1Rows, int m1Cols, int m2Cols, int inStartIndex, int resStartIndex)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0;

	if (row < m1Rows && col < m2Cols) {
		for (int i = 0; i < m1Cols; i++) {

			sum += gpuMat1[inStartIndex + row * m1Cols + i] * gpuMat2[i * m2Cols + col];
		}
		gpuMat3[resStartIndex + row * m2Cols + col] = sum;
	}
}


void gpuMatrixMultiplication(CpuGpuMat* Mat1, CpuGpuMat* Mat2, CpuGpuMat* Mat3, int inStartIndex, int resStartIndex)
{
	int threadsPerBlock = 32;

	int gridCols = ceil(double(Mat2->Cols) / double(threadsPerBlock));
	int gridRows = ceil(double(Mat1->Rows) / double(threadsPerBlock));

	dim3 gridDim(gridCols, gridRows);
	dim3 blockDim(threadsPerBlock, threadsPerBlock);

	gpuMatrixMult << < gridDim, blockDim >> > ((float*)Mat1->GpuP, (float*)Mat2->GpuP, (float*)Mat3->GpuP, Mat1->Rows, Mat1->Cols, Mat2->Cols, inStartIndex, resStartIndex);
}
