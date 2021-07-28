#include "CpuGpuMat.h"
#include "cuda_runtime.h"		
#include <assert.h>
#include <stdlib.h>

CpuGpuMat::CpuGpuMat() {}

CpuGpuMat::CpuGpuMat(const int& rows, const int& cols, bool useBias) {
	this->Rows = rows;
	this->Cols = useBias ? cols + 1 : cols;
	this->Size = this->Rows * this->Cols;

	cpuGpuAlloc();

	if (useBias) {
		float* biasValue = (float*)this->CpuP;
		biasValue[this->Size - 1] = 1.0F;
	}
}

CpuGpuMat::~CpuGpuMat() {

	cudaError_t result = cudaFree(this->GpuP);
	assert(result == cudaSuccess);

	free(this->CpuP);
}

void CpuGpuMat::host2Device() {
	cudaError_t result = cudaMemcpy(this->GpuP, this->CpuP, getAllocationSize(), cudaMemcpyHostToDevice);
	assert(result == cudaSuccess);
}

void CpuGpuMat::device2Host() {
	cudaError_t result = cudaMemcpy(this->CpuP, this->GpuP, getAllocationSize(), cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);
}

void CpuGpuMat::cpuGpuAlloc() {

	this->CpuP = (void*)malloc(getAllocationSize());

	cudaError_t result = cudaMalloc(&this->GpuP, getAllocationSize());
	assert(result == cudaSuccess);
}

int CpuGpuMat::getAllocationSize() {

	return this->Size * sizeof(float);
}