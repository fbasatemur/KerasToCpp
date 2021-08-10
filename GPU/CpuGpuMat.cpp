#include "CpuGpuMat.h"
#include "cuda_runtime.h"		
#include <assert.h>
#include <stdlib.h>

CpuGpuMat::CpuGpuMat() {}

// constructor for one sample input
CpuGpuMat::CpuGpuMat(int rows, int cols, bool useMemPin) {
	this->Rows = rows;
	this->Cols = cols + 1;						// input + bias memory size	
	this->Size = this->Rows * this->Cols;
	this->MemPinned = useMemPin;

	CpuGpuAlloc();

	float* biasValue = (float*)this->CpuP;
	biasValue[this->Size - 1] = 1.0F;			// bias value setted
}

// constructor for multi-sample input
CpuGpuMat::CpuGpuMat(const float* inputs, int inputRow, int inputCol, int numberInputs, bool useMemPin) {

	this->Rows = inputRow;
	this->Cols = inputCol + 1;						// input + bias memory size
	this->Size = this->Rows * this->Cols * numberInputs;
	this->MemPinned = useMemPin;

	CpuGpuAlloc();

	int inputSize = inputRow * inputCol;
	float* nativeP = (float*)this->CpuP;
	int inputStep = 0;

	int i = 0;
	for (int input = 0; input < numberInputs; input++)
	{
		inputStep = input * this->Rows * this->Cols;

		for (i = 0; i < inputSize; i++)
			nativeP[inputStep + i] = inputs[input * inputSize + i];

		nativeP[inputStep + i] = 1.0F;			// bias value setted
	}
}

CpuGpuMat::CpuGpuMat(CpuGpuMat& result, int numberOutputs, bool useMemPin) {

	this->Rows = result.Rows;
	this->Cols = result.Cols;
	this->Size = this->Rows * this->Cols * numberOutputs;
	this->MemPinned = useMemPin;

	CpuGpuAlloc();

	result.CpuP = this->CpuP;
	result.GpuP = this->GpuP;
	this->deallocCpuP = &result.CpuP;
	this->deallocGpuP = &result.GpuP;
}

CpuGpuMat::~CpuGpuMat() {

	if (!this->deallocCpuP && !this->deallocGpuP && this->CpuP && this->GpuP) {
		assert(cudaFree(this->GpuP) == cudaSuccess);

		if (this->MemPinned)
			assert(cudaFreeHost(this->CpuP) == cudaSuccess);
		else free(this->CpuP);
	}
}

void CpuGpuMat::Host2Device() {
	cudaError_t result = cudaMemcpy(this->GpuP, this->CpuP, GetAllocationSize(), cudaMemcpyHostToDevice);
	assert(result == cudaSuccess);
}

void CpuGpuMat::Device2Host() {
	cudaError_t result = cudaMemcpy(this->CpuP, this->GpuP, GetAllocationSize(), cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);
}

void CpuGpuMat::CpuGpuAlloc() {

	cudaError_t result;
	if (this->MemPinned) {
		result = cudaHostAlloc(&this->CpuP, GetAllocationSize(), 0);
		assert(result == cudaSuccess);
	}
	else this->CpuP = (void*)malloc(GetAllocationSize());

	result = cudaMalloc(&this->GpuP, GetAllocationSize());
	assert(result == cudaSuccess);
}

int CpuGpuMat::GetAllocationSize() {

	return this->Size * sizeof(float);
}

