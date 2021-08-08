#pragma once

#include <string>
#include "CpuMat.h"

class Dense {
private:
	void kernelLoad(std::string& filename);
	void biasLoad(std::string& filename);

public:
	Dense(int neurons, int inputRows, int inputCols, bool useBias = true);

	void Load(std::string& kernelFilename, std::string& biasFilename);
	void Apply(CpuMat* input);

	int neurons;
	CpuMat* Kernel;
	CpuMat* Result;
};
