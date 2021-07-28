#pragma once

#include <string>
#include "CpuMat.h"

class Flatten {

public:
	Flatten(CpuMat* input, bool useBias = true);
	void apply(CpuMat* input);
	CpuMat* Result;
};