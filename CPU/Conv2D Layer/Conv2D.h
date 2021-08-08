#pragma once

#include <string>
#include "CpuMat.h"

class Conv2D {
private:
	void kernelLoad(std::string& filename);
	void biasLoad(std::string& filename);

public:
	Conv2D(size_t filters, size_t filterRows, size_t filterCols, CpuMat* input, bool useBias = true);

	void Load(std::string& kernelFilename, std::string& biasFilename);
	void Apply(CpuMat* input);

	size_t filters;
	CpuMat* Kernel;
	CpuMat* Result;
};
