#include "MaxPooling2D.h"
#include <math.h>

MaxPooling2D::MaxPooling2D(CpuMat* input, size_t poolX, size_t poolY, size_t strideX, size_t strideY)
{
	Result = new CpuMat();
	Result->Cols = floor((input->Cols - poolX) / strideX) + 1;
	Result->Rows = floor((input->Rows - poolY) / strideY) + 1;
	Result->Depth = input->Depth;
	Result->Size = Result->Cols * Result->Rows * Result->Depth;
	Result->MemAlloc();

	this->poolX = poolX;
	this->poolY = poolY;
	this->strideX = strideX;
	this->strideY = strideY;
}

void MaxPooling2D::apply(CpuMat* input)
{
	size_t inputSize2D = input->Cols * input->Rows;
	size_t resultSize2D = Result->Cols * Result->Rows;
	float* resultP, * inputP;
	float max;

	for (size_t d = 0; d < input->Depth; d++)
	{
		inputP = (float*)input->CpuP + d * inputSize2D;
		resultP = (float*)Result->CpuP + d * resultSize2D;

		for (size_t r = 0; r < input->Rows; r += this->strideY)
		{
			for (size_t c = 0; c < input->Cols; c += this->strideX)
			{
				max = -FLT_MAX;
				for (size_t row = 0; row < this->poolY; row++)
				{
					for (size_t col = 0; col < this->poolX; col++)
					{
						if (inputP[(r + row) * input->Cols + c + col] > max)
							max = inputP[(r + row) * input->Cols + c + col];
					}
				}
				resultP[(r / this->strideY) * Result->Cols + (c / this->strideX)] = max;
			}
		}
	}
}

MaxPooling2D::~MaxPooling2D(){
	delete[] Result->CpuP;
}

