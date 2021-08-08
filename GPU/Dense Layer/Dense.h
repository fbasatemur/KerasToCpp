#pragma once
#include <string>
#include "CpuGpuMat.h"


class Dense{
private:
	void kernelLoad(std::string& filename);
	void biasLoad(std::string& filename);

public:
	Dense(int neurons, int inputRows, int inputCols, bool isEndLayer = false, bool isMemPin = false);
	Dense(int neurons, CpuGpuMat& denseResult, bool isEndLayer = false, bool isMemPin = false);

	void Load(std::string& kernelFilename, std::string& biasFilename);
	void Apply(CpuGpuMat* input, int inputIndex = 0, int resultIndex = 0);
	void Host2Device();

	int neurons;
	bool endLayer;
	CpuGpuMat Kernel;
	CpuGpuMat Result;
};