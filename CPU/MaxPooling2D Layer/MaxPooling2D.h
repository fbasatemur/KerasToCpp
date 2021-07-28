#pragma once

#include <string>
#include "CpuMat.h"

class MaxPooling2D {

public:
	MaxPooling2D(CpuMat* input, size_t poolX, size_t poolY, size_t strideX, size_t strideY);
	~MaxPooling2D();
	void apply(CpuMat* input);

	size_t poolX, poolY, strideX, strideY;
	CpuMat* Result;
};