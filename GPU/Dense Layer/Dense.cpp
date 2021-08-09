#include "Dense.h"
#include <fstream>
#include "KernelDense.cuh"


Dense::Dense(int neurons, int inputRows, int inputCols, bool isEndLayer, bool isMemPin)
{
	this->neurons = neurons;
	this->endLayer = isEndLayer;

	bool useBias = !isEndLayer;

	Kernel.Rows = inputCols + 1;				// for bias value
	Kernel.Cols = neurons;
	Kernel.Size = Kernel.Rows * Kernel.Cols;
	Kernel.MemPinned = isMemPin;

	Result.Rows = inputRows;
	Result.Cols = useBias ? neurons + 1 : neurons;
	Result.Size = Result.Rows * Result.Cols;
	Result.MemPinned = isMemPin;

	Kernel.CpuGpuAlloc();

	if (useBias) {

		Result.CpuGpuAlloc();

		float* biasValue = (float*)Result.CpuP;
		biasValue[Result.Size - 1] = 1.0F;
	}
}

Dense::Dense(int neurons, CpuGpuMat& denseResult, bool isEndLayer, bool isMemPin)
{
	this->neurons = neurons;
	this->endLayer = isEndLayer;

	bool useBias = !isEndLayer;

	Kernel.Rows = denseResult.Cols;
	Kernel.Cols = neurons;
	Kernel.Size = Kernel.Rows * Kernel.Cols;
	Kernel.MemPinned = isMemPin;

	Result.Rows = denseResult.Rows;
	Result.Cols = useBias ? neurons + 1 : neurons;
	Result.Size = Result.Rows * Result.Cols;
	Result.MemPinned = isMemPin;

	Kernel.CpuGpuAlloc();

	if (useBias) {

		Result.CpuGpuAlloc();		// isEndLayer == false

		float* biasValue = (float*)Result.CpuP;
		biasValue[Result.Size - 1] = 1.0F;
	}
}

void Dense::Load(std::string& kernelFilename, std::string& biasFilename)
{
	kernelLoad(kernelFilename);
	biasLoad(biasFilename);
}

void Dense::Apply(CpuGpuMat* input, int inputIndex, int resultIndex)
{
	gpuMatrixMultiplication(input, &this->Kernel, &this->Result, inputIndex * input->Cols * input->Rows, resultIndex);
}

void Dense::Host2Device() {
	Kernel.Host2Device();
	if (!endLayer)	Result.Host2Device();
}

void Dense::kernelLoad(std::string& filename)
{
	float* cpuFloatP1 = (float*)Kernel.CpuP;
	std::ifstream file;
	file.open(filename);
	std::string value;

	for (size_t i = 0; i < Kernel.Size - neurons; i++) {
		file >> value;
		cpuFloatP1[i] = std::stof(value);
	}

	file.close();
}

void Dense::biasLoad(std::string& filename)
{
	float* cpuFloatP1 = (float*)Kernel.CpuP;
	std::ifstream file;
	file.open(filename);
	std::string value;

	int biasPos = Kernel.Size - this->neurons;

	for (size_t i = 0; i < this->neurons; i++) {
		file >> value;
		cpuFloatP1[biasPos + i] = std::stof(value);
	}

	file.close();
}
