#include "Dense.h"
#include <fstream>
#include "KernelDense.cuh"

Dense::Dense(int neurons, int inputRows, int inputCols, bool useBias)
{
	this->neurons = neurons;

	Kernel.Rows = inputCols;
	Kernel.Cols = neurons;
	Kernel.Size = Kernel.Rows * Kernel.Cols;

	Result.Rows = inputRows;
	Result.Cols = useBias ? neurons + 1 : neurons;
	Result.Size = Result.Rows * Result.Cols;

	Kernel.cpuGpuAlloc();
	Result.cpuGpuAlloc();

	if (useBias) {
		float* biasValue = (float*)Result.CpuP;
		biasValue[Result.Size - 1] = 1.0F;
	}
}

void Dense::load(std::string& kernelFilename, std::string& biasFilename)
{
	kernelLoad(kernelFilename);
	biasLoad(biasFilename);
}

void Dense::apply(CpuGpuMat* input)
{
	gpuMatrixMultiplication(input, &Kernel, &Result);
}

void Dense::host2Device() {
	Kernel.host2Device();
	Result.host2Device();
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