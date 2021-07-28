#include "KernelCpu.h"
#include <fstream>
#include "Conv2D.h"

Conv2D::Conv2D(size_t filters, size_t filterRows, size_t filterCols, CpuMat* input, bool useBias)
{
	Kernel = new CpuMat[filters];
	int biasOffset = useBias ? 1 : 0;
	this->filters = filters;

	for (size_t i = 0; i < filters; i++)
	{
		Kernel[i].Rows = filterRows;
		Kernel[i].Cols = filterCols;
		Kernel[i].Depth = input->Depth;
		Kernel[i].Size = filterRows * filterCols * input->Depth + biasOffset;
		Kernel[i].MemAlloc();
	}

	Result = new CpuMat(input ->Rows - filterRows + 1, input->Cols - filterCols + 1, filters, false);
}

void Conv2D::load(std::string& kernelFilename, std::string& biasFilename)
{
	kernelLoad(kernelFilename);
	biasLoad(biasFilename);
}


void Conv2D::apply(CpuMat* input)
{
	cpuMatrixConv2D(input, this->filters, this->Kernel, this->Result);
}


void Conv2D::kernelLoad(std::string& filename)
{
	
	std::string* buffer = ReadTxtToBuffer(filename);
	char* pch = strtok((char*)buffer->c_str(), " \n");

	float* cpuFloatP;
	size_t size3D = Kernel->Size - 1;

	for (size_t f = 0; f < this->filters; f++) 
	{
		cpuFloatP = (float*)Kernel[f].CpuP;
		for (size_t i = 0; i < size3D; i++)
		{
			cpuFloatP[i] = std::stof(pch);
			pch = strtok(NULL, " \n");
		}
	}

	delete buffer;
	delete[] pch;
}

void Conv2D::biasLoad(std::string& filename)
{
	std::string* buffer = ReadTxtToBuffer(filename);
	char* pch = strtok((char*)buffer->c_str(), " \n");

	float* cpuFloatP;
	int biasPos = Kernel->Size - 1;

	for (size_t f = 0; f < this->filters; f++) 
	{
		cpuFloatP = (float*)Kernel[f].CpuP;
		cpuFloatP[biasPos] = std::stof(pch);
		pch = strtok(NULL, " \n");
	}

	delete buffer;
	delete[] pch;
}