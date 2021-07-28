#include "Dense.h"
#include <fstream>
#include "KernelCpu.h"
#include <stdio.h>
Dense::Dense(int neurons, int inputRows, int inputCols, bool useBias) 
{
	this->neurons = neurons;

	Kernel = new CpuMat(inputCols, neurons, 1, false);
	Result = new CpuMat(inputRows, neurons, 1, useBias);
}

void Dense::load(std::string& kernelFilename, std::string& biasFilename)
{
	kernelLoad(kernelFilename);
	biasLoad(biasFilename);
}

void Dense::apply(CpuMat* input)
{
	cpuMatrixMultiplication(input, this->Kernel, this->Result);
}

void Dense::kernelLoad(std::string& filename)
{
	float* cpuFloatP = (float*)Kernel->CpuP;

	std::string* buffer = ReadTxtToBuffer(filename);
	char* pch = strtok((char*)buffer->c_str(), " \n");

	for (size_t i = 0; i < Kernel->Size - neurons; i++) 
	{	
		cpuFloatP[i] = std::stof(pch);
		pch = strtok(NULL, " \n");
	}

	delete buffer;
	delete[] pch;
}

void Dense::biasLoad(std::string& filename)
{
	float* cpuFloatP = (float*)Kernel->CpuP;
	
	std::string* buffer = ReadTxtToBuffer(filename);
	char* pch = strtok((char*)buffer->c_str(), " \n");

	int biasPos = Kernel->Size - this->neurons;

	for (size_t i = 0; i < this->neurons; i++) 
	{
		cpuFloatP[biasPos + i] = std::stof(pch);
		pch = strtok(NULL, " \n");
	}

	delete buffer;
	delete[] pch;
}