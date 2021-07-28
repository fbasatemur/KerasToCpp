#pragma once
#include <string>
#include "CpuGpuMat.h"

class Dense {
private:
	void kernelLoad(std::string& filename);
	void biasLoad(std::string& filename);

public:
	Dense(int neurons, int inputRows, int inputCols, bool useBias = true);

	void load(std::string& kernelFilename, std::string& biasFilename);
	void apply(CpuGpuMat* input);
	void host2Device();

	int neurons;
	CpuGpuMat Kernel;
	CpuGpuMat Result;
};